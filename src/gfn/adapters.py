from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, cast
from inspect import signature

import torch
from torch.distributions import Distribution

from gfn.estimators import Estimator
from gfn.states import States
from gfn.utils.handlers import check_cond_forward


class EstimatorAdapter(ABC):
    """Adapter interface for estimator-specific policy behavior.

    This abstract base class defines the minimal interface the Sampler relies on,
    allowing us to keep one generic sampling loop while plugging in different estimator
    behaviors (e.g., non‑recurrent, recurrent with carry, tempered variants)
    without modifying the Sampler.

    The adapter owns an opaque RolloutContext object. The Sampler never inspects
    it and simply passes it back to the adapter at each step. The adapter is
    responsible for:
      - initializing the context in `init_context`.
      - compute the action distribution, while updating any internal state (e.g.,
        recurrent `carry`)
      - compute the log probabilities of the actions in `log_probs`.
      - recording per‑step artifacts in `record` (e.g., log_probs,
        estimator outputs), typically with mask-aware padding.
      - output trajectory-length artifacts via `finalize(ctx)`

    The context should be allocated once per rollout. Masking should be applied inside
    the adapter (via `step_mask`) when slicing conditioning or padding per‑step
    tensors back to full batch size. The Sampler can therefore be oblivious to
    estimator details (conditioning, carry, etc.).
    """

    @property
    def is_backward(self) -> bool:
        ...  # fmt: skip

    @property
    def is_vectorized(self) -> bool:
        ...  # fmt: skip

    @abstractmethod
    def init_context(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: Optional[torch.Tensor] = None,
    ) -> Any:
        ...  # fmt: skip

    @abstractmethod
    def compute_dist(
        self,
        states_active: States,
        ctx: Any,
        step_mask: Optional[torch.Tensor] = None,
        **policy_kwargs: Any,
    ) -> tuple[Distribution, Any]:
        ...  # fmt: skip

    @abstractmethod
    def log_probs(
        self,
        actions_active: torch.Tensor,
        dist: Distribution,
        ctx: Any,
        step_mask: Optional[torch.Tensor] = None,
        vectorized: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        ...  # fmt: skip

    @abstractmethod
    def record(
        self,
        ctx: Any,
        step_mask: torch.Tensor,
        sampled_actions: torch.Tensor,
        dist: Distribution,
        log_probs: Optional[torch.Tensor],
        save_estimator_outputs: bool,
    ) -> None:
        ...  # fmt: skip

    # Optional helper for `sample_actions` BC
    def get_current_estimator_output(self, ctx: Any) -> Optional[torch.Tensor]:
        ...  # fmt: skip


class RolloutContext:
    """Structured, mutable context owned by adapters.

    Uses fixed attributes for core fields and an `extras` dict for adapter-
    specific extensions without changing the class shape. This keeps most
    accesses fast and typed while preserving flexibility similar to dicts.
    """

    __slots__ = (
        "batch_size",
        "device",
        "conditioning",
        "carry",
        "trajectory_log_probs",
        "trajectory_estimator_outputs",
        "current_estimator_output",
        "extras",
    )

    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: Optional[torch.Tensor] = None,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.conditioning = conditioning
        self.carry = None
        self.trajectory_log_probs: List[torch.Tensor] = []
        self.trajectory_estimator_outputs: List[torch.Tensor] = []
        self.current_estimator_output: Optional[torch.Tensor] = None
        self.extras: Dict[str, Any] = {}


class DefaultEstimatorAdapter(EstimatorAdapter):
    """Adapter for non-recurrent estimators (current default behavior).

    Overview
    --------
    This adapter bridges the generic sampling loop and is used throughout the codebase.
    It exposes the minimal interface required by the `EstimatorAdapter` abstract base
    class while keeping the sampler loop estimator-agnostic.

    - If conditioning is provided, the estimator accepts `(states, conditioning)`;
      otherwise it accepts `(states)`.
    - The estimator provides `to_probability_distribution(states, est_out, **kw)`
      returning a torch Distribution over actions for the masked states.

    The adapter owns an opaque rollout context `ctx` which the Sampler never reads. The
    context (a TensorDict) is created once per rollout and mutated in place:

    - init_context: stores rollout invariants and optional conditioning. Also prepares
         per‑step buffers for artifacts trajectory-level artifacts (log_probs,
        estimator_outputs).

    - compute:
      1) Selects the appropriate estimator call signature depending on whether
         conditioning is present. If conditioning is present, the adapter slices
         it with `step_mask` so shapes match `states_active`.
      2) Calls the estimator forward pass to obtain the raw `est_out`.
      3) Converts `est_out` into a torch Distribution with
         `to_probability_distribution`.
      4) Saves `est_out` into `ctx.current_estimator_output`.

    - record:
      Materializes optional per‑step artifacts into context‑managed buffers with
      mask‑aware padding back to the full rollout batch size:
      * Log‑probs: computes `dist.log_prob(sampled_actions)` for active rows only,
        then writes into a 1D tensor of length batch_size filled with zeros and masked
        assignment for active positions. Appends this to a list (one tensor per time step).
      * Estimator outputs: if requested, pads the last estimator output
        (`ctx.current_estimator_output`) to shape `(batch_size, ...)` using `-inf` for
        inactive rows and appends to a list (one tensor per time step).

    - finalize: Stacks recorded per‑step lists along the time dimension into tensors of
        shape `(trajectory_legnth, batch_size, ...)` suitable for `Trajectories`.
        Returns `None` for any artifact that was never recorded.

    Masking & Shapes
    ----------------
    - `states_active` always corresponds to `states[~dones]` inside the sampler.
    - The adapter receives `step_mask` (shape `(N,)`) to slice any step‑dependent
      inputs (e.g., conditioning) and to pad per‑step outputs to the full batch.
    - Padded tensors use `0.0` for log‑probs and `-inf` for estimator outputs to
      maintain compatibility with downstream code.

    Backward/Forward Direction
    --------------------------
    - `is_backward` is forwarded from the underlying estimator so the sampler can
      choose the appropriate environment transition (forward vs backward).

    Vectorized Probability Path
    --------------------------
    - `is_vectorized` is used by the Sampler to choose the appropriate probability path.
    - Vectorized adapters always use faster paths in probability calculators.
      Non-vectorized adapters (e.g., recurrent) use per-step paths with masking and
      alignment identical to the legacy reference.

    Performance Notes
    -----------------
    - `ctx` is allocated once per rollout and mutated in place to avoid per‑step
      overhead.
    - If you know trajectory length bounds, you can extend this adapter to
      pre‑allocate fixed‑size storage in `init_context` rather than appending to
      Python lists.
    """

    def __init__(self, estimator: Estimator) -> None:
        """Initialize the adapter with a non-recurrent estimator.

        The estimator must expose `to_probability_distribution(states, est_out, **kw)`
        and optionally accept conditioning via `estimator(states, conditioning)`.
        """
        self._estimator = estimator

    @property
    def is_backward(self) -> bool:
        """Whether the wrapped estimator samples in the backward direction."""
        return getattr(self._estimator, "is_backward", False)

    @property
    def is_vectorized(self) -> bool:
        """Used for vectorized probability calculations."""
        return True

    def init_context(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: Optional[torch.Tensor] = None,
    ) -> RolloutContext:
        """Create a new per-rollout context.

        Stores rollout invariants (batch size, device, optional conditioning) and
        initializes empty buffers for per-step artifacts.
        """
        return RolloutContext(
            batch_size=batch_size, device=device, conditioning=conditioning
        )

    def compute_dist(
        self,
        states_active: States,
        ctx: Any,
        step_mask: Optional[torch.Tensor] = None,
        **policy_kwargs: Any,
    ) -> tuple[Distribution, Any]:
        """Run the estimator for active rows and build an action Distribution.

        - Uses `step_mask` to slice conditioning to the active subset.
        - Saves the raw estimator output in `ctx.current_estimator_output` for
          optional recording in `record_step`.
        """
        cond_active = None
        if ctx.conditioning is not None:
            if step_mask is None:
                cond_active = ctx.conditioning
            else:
                cond_active = ctx.conditioning[step_mask]

        estimator_outputs = check_cond_forward(
            self._estimator, "estimator", states_active, cond_active
        )

        dist = self._estimator.to_probability_distribution(
            states_active, estimator_outputs, **policy_kwargs
        )

        # TODO: Make optional.
        ctx.current_estimator_output = estimator_outputs

        return dist, ctx

    def log_probs(
        self,
        actions_active: torch.Tensor,
        dist: Distribution,
        ctx: Any,
        step_mask: Optional[torch.Tensor] = None,
        vectorized: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        """Compute log-probs, optionally padding back to full batch when non-vectorized."""
        lp = dist.log_prob(actions_active)
        if torch.any(torch.isinf(lp)):
            raise RuntimeError("Log probabilities are inf. This should not happen.")

        if vectorized:
            return lp, ctx

        assert step_mask is not None, "step_mask is required when vectorized=False"
        step_lp = torch.full((ctx.batch_size,), 0.0, device=ctx.device, dtype=lp.dtype)
        step_lp[step_mask] = lp

        return step_lp, ctx

    def record(
        self,
        ctx: Any,
        step_mask: torch.Tensor,
        sampled_actions: torch.Tensor,
        dist: Distribution,
        log_probs: Optional[torch.Tensor],
        save_estimator_outputs: bool,
    ) -> None:
        """Record per-step artifacts into the context's trajectory-level lists."""
        if log_probs is not None:
            ctx.trajectory_log_probs.append(log_probs)

        if save_estimator_outputs and ctx.current_estimator_output is not None:
            estimator_outputs = ctx.current_estimator_output
            padded = torch.full(
                (ctx.batch_size,) + estimator_outputs.shape[1:], -float("inf"), device=ctx.device
            )
            padded[step_mask] = estimator_outputs
            ctx.trajectory_estimator_outputs.append(padded)

    def finalize(self, ctx: Any) -> dict[str, Optional[torch.Tensor]]:
        """Stack recorded per-step artifacts along time into trajectory-level tensors."""
        log_probs = (
            torch.stack(ctx.trajectory_log_probs, dim=0)
            if ctx.trajectory_log_probs
            else None
        )
        estimator_outputs = (
            torch.stack(ctx.trajectory_estimator_outputs, dim=0)
            if ctx.trajectory_estimator_outputs
            else None
        )

        return {"log_probs": log_probs, "estimator_outputs": estimator_outputs}

    def get_current_estimator_output(self, ctx: Any) -> Optional[torch.Tensor]:
        """Expose the most recent per-step estimator output saved during `compute`."""
        return getattr(ctx, "current_estimator_output", None)


class RecurrentEstimatorAdapter(DefaultEstimatorAdapter):
    """Adapter for recurrent estimators that maintain a rollout carry (hidden state).

    Differences from `DefaultEstimatorAdapter`:
    - is_vectorized = False: runs sequential, per‑step probability calculators for
      PF/PB. PB aligns action at t with state at t+1, t==0 skipped).
    - Rollout context manages `ctx.carry` which contains the hidden state of the
      recurrent estimator. It is initialized once via `estimator.init_carry(batch_size,
      device)` and updated every step.
    - `compute(states, ctx, ...)` calls `estimator(states, ctx.carry) ->
      (estimator_outputs, new_carry)`, updates `ctx.carry`, then builds the
      Distribution from `estimator_outputs`.
    - recording mirrors the default but pads per‑step tensors to batch size and
      stacks into `(T, N, ...)`.
    - No action‑id mask indexing; illegal actions are handled by the Distribution.

    Requires the estimator to implement:
    - `init_carry(batch_size: int, device: torch.device)`
    - a recurrent forward returning `(estimator_outputs, new_carry)`.
    """

    def __init__(self, estimator: Estimator) -> None:
        # Validate that the estimator presents a recurrent interface
        # We check for the presence of `init_carry` and a callable that accepts (states, carry).
        init_carry = getattr(estimator, "init_carry", None)
        if not callable(init_carry):
            raise TypeError(
                "RecurrentEstimatorAdapter requires an estimator implementing "
                "init_carry(batch_size: int, device: torch.device)."
            )
        super().__init__(estimator)

    # TODO: Need to support vectorized probability calculations with Transformers.
    @property
    def is_vectorized(self) -> bool:
        return False

    def init_context(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: Optional[torch.Tensor] = None,
    ) -> RolloutContext:
        """Create context and initialize recurrent carry, (estimator hidden state).

        Differs from the default adapter by allocating `ctx.carry` via
        `estimator.init_carry(batch_size, device)`.
        """
        init_carry = getattr(self._estimator, "init_carry", None)
        if not callable(init_carry):
            raise TypeError(
                "RecurrentEstimatorAdapter requires an estimator that implements "
                "init_carry(batch_size: int, device: torch.device).\n"
                "A) Recurrent estimators must expose an `init_carry` method.\n"
                "B) RecurrentEstimatorAdapter is only compatible with estimators that "
                "expose `init_carry`."
            )
        ctx = super().init_context(batch_size, device, conditioning)
        init_carry_fn = cast(Callable[[int, torch.device], Any], init_carry)
        ctx.carry = init_carry_fn(batch_size, device)

        return ctx

    def compute_dist(
        self,
        states_active: States,
        ctx: Any,
        step_mask: Optional[torch.Tensor] = None,
        **policy_kwargs: Any,
    ) -> tuple[Distribution, Any]:
        """Run estimator with carry and update it.

        Differs from the default adapter by calling
        `estimator(states_active, ctx.carry) -> (est_out, new_carry)`, storing the
        updated carry and saving `current_estimator_output` before building the
        Distribution.
        """
        estimator_outputs, new_carry = self._estimator(states_active, ctx.carry)
        ctx.carry = new_carry
        dist = self._estimator.to_probability_distribution(
            states_active,
            estimator_outputs,
            **policy_kwargs,
        )

        # TODO: Make optional.
        ctx.current_estimator_output = estimator_outputs

        return dist, ctx


def maybe_instantiate_adapter(
    estimator: Estimator,
    adapter: Callable[[Estimator], EstimatorAdapter] | EstimatorAdapter | None,
) -> EstimatorAdapter:
    """Maybe instantiate an adapter for a given estimator.

    Args:
        estimator: The estimator to instantiate an adapter for.
        adapter: An adapter class instance or callable to use for sampling actions
            and computing probability distributions. If None, the default adapter class
            for the estimator will be used.

    Returns:
        An adapter instance.
    """
    # If no adapter is provided, use the default adapter class for the estimator,
    # which we need to retrieve and instantiate here.
    if adapter is None:
        adapter_cls = estimator.default_adapter_class
        assert (
            adapter_cls is not None
        ), "Estimator has no default adapter class and no adapter was provided"
        adapter_cls = cast(Callable[[Estimator], EstimatorAdapter], adapter_cls)
        return adapter_cls(estimator)

    # If an adapter class is provided, instantiate it with the estimator.
    elif isinstance(adapter, type) and issubclass(adapter, EstimatorAdapter):

        # We have to assume that the adapter class accepts exactly 1 argument
        # (estimator).
        sig = signature(adapter)

        # Count parameters excluding 'self'
        params = [p for p in sig.parameters.values() if p.name != "self"]
        if len(params) != 1:
            raise TypeError(
                f"Adapter class {adapter.__name__} must accept exactly 1 argument "
                f"(estimator) to use automatic adapter instantiation, "
                f"but has {len(params)} parameters: {[p.name for p in params]},"
                f"You can provide an adapter instance to the GFlowNet instead."
            )

        adapter_factory = cast(Callable[[Estimator], EstimatorAdapter], adapter)
        return adapter_factory(estimator)

    # If an adapter instance is provided, use it.
    elif isinstance(adapter, EstimatorAdapter):
        return adapter

    else:
        raise ValueError(f"Invalid adapter type: {type(adapter)}")
