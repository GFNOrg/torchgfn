from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

import torch
from torch.distributions import Distribution

from gfn.states import States
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)

if TYPE_CHECKING:
    from gfn.estimators import Estimator


class EstimatorAdapter(ABC):
    """Adapter interface for estimator-specific policy behavior.

    Keeps the sampling loop generic; estimator-specific logic lives here.

    Responsibilities:
    - init_context(batch_size, device, conditioning): allocate rollout context.
    - compute_dist(states_active, ctx, step_mask, save_estimator_outputs, **kw):
      run estimator on active rows, return a torch Distribution, and update `ctx`
      if needed (e.g., carry, cached outputs).
    - log_probs(actions_active, dist, ctx, step_mask, vectorized, save_logprobs):
      compute log-probabilities for active rows; when ``vectorized=False`` return a
      batch-sized tensor padded via ``step_mask``.

    Notes:
    - The sampler never inspects `ctx`; masking and padding happen inside the
      adapter.
    - ``is_backward`` selects forward vs backward environment steps.
    - ``is_vectorized`` selects fast vectorized vs per‑step probability paths.
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
        save_estimator_outputs: bool = False,
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
        save_logprobs: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        ...  # fmt: skip

    # Optional helper for `sample_actions` BC
    def get_current_estimator_output(self, ctx: Any) -> Optional[torch.Tensor]:
        ...  # fmt: skip


class RolloutContext:
    """Structured per‑rollout state owned by adapters.

    Holds rollout invariants and optional per‑step buffers; use ``extras`` for
    adapter‑specific fields without changing the class shape.
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
    """Adapter for non‑recurrent estimators (default).

    Overview
    --------
    This adapter bridges the generic sampling loop and is used throughout the codebase.
    It exposes the minimal interface required by the `EstimatorAdapter` abstract base
    class while keeping the sampler loop estimator-agnostic.

    Workflow with RolloutContext:
    - ``init_context(batch_size, device, conditioning)``: store invariants and
      allocate per‑step buffers.
    - ``compute_dist(states_active, ctx, step_mask, **kw)``: slice conditioning
      by ``step_mask`` when provided, run the estimator on active rows, cache
      ``est_out`` in ``ctx.current_estimator_output``, and return a Distribution.
    - ``log_probs(actions_active, dist, ctx, step_mask, vectorized)``: compute
      log‑probs for active rows; when ``vectorized=False``, return a batch‑padded
      tensor using ``step_mask``. Per‑step artifacts are recorded when flags are set.

    Masking and path selection
    - ``states_active == states[~dones]``; ``step_mask`` has shape ``(N,)``.
    - ``is_backward`` is forwarded from the estimator.
    - ``is_vectorized == True`` enables vectorized probability calculators when
      available.

    Performance
    - One context per rollout; mutate in place. If trajectory length bounds are
      known, pre‑allocation of those buffers is possible.
    """

    def __init__(self, estimator: "Estimator") -> None:
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
        save_estimator_outputs: bool = False,
        **policy_kwargs: Any,
    ) -> tuple[Distribution, Any]:
        """Run the estimator for active rows and build an action Distribution.

        - Uses `step_mask` to slice conditioning to the active subset. When `step_mask`
          is None, the estimator running in a vectorized context.
        - Saves the raw estimator output in `ctx.current_estimator_output` for
          optional recording in `record_step`.
        """
        precopmputed_estimator_outputs = getattr(ctx, "current_estimator_output", None)

        # Reuse precomputed outputs only in vectorized contexts (no step_mask).
        if step_mask is None and precopmputed_estimator_outputs is not None:
            expected_bs = states_active.batch_shape[0]
            if precopmputed_estimator_outputs.shape[0] != expected_bs:
                raise RuntimeError(
                    "current_estimator_output batch size does not match active states. "
                    f"Got {precopmputed_estimator_outputs.shape[0]}, expected {expected_bs}. "
                    "This indicates stale cache reuse; ensure per-step masking when setting "
                    "ctx.current_estimator_output and clear it when not valid."
                )
            estimator_outputs = precopmputed_estimator_outputs

        # Otherwise, compute the estimator outputs.
        else:
            cond_active = None
            if ctx.conditioning is not None:
                if step_mask is None:
                    cond_active = ctx.conditioning
                else:
                    cond_active = ctx.conditioning[step_mask]

            # Call estimator with or without conditioning.
            if cond_active is not None:
                with has_conditioning_exception_handler("estimator", self._estimator):
                    estimator_outputs = self._estimator(states_active, cond_active)
            else:
                with no_conditioning_exception_handler("estimator", self._estimator):
                    estimator_outputs = self._estimator(states_active)

        # Build the distribution.
        dist = self._estimator.to_probability_distribution(
            states_active, estimator_outputs, **policy_kwargs
        )

        # Save current estimator output only when requested.
        if save_estimator_outputs:
            ctx.current_estimator_output = estimator_outputs

            # If we are in a non-vectorized path (masked), append a padded copy to trajectory.
            if step_mask is not None:
                padded = torch.full(
                    (ctx.batch_size,) + estimator_outputs.shape[1:],
                    -float("inf"),
                    device=ctx.device,
                )
                padded[step_mask] = estimator_outputs
                ctx.trajectory_estimator_outputs.append(padded)

        else:
            ctx.current_estimator_output = None

        return dist, ctx

    def log_probs(
        self,
        actions_active: torch.Tensor,
        dist: Distribution,
        ctx: Any,
        step_mask: Optional[torch.Tensor] = None,
        vectorized: bool = False,
        save_logprobs: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        """Compute log-probs, optionally padding back to full batch when non-vectorized."""
        lp = dist.log_prob(actions_active)

        if vectorized:
            if save_logprobs:
                ctx.trajectory_log_probs.append(lp)
            return lp, ctx

        # Non-vectorized path strict check. None of these should be -inf after masking.
        if torch.any(torch.isinf(lp)):
            raise RuntimeError("Log probabilities are inf. This should not happen.")

        assert step_mask is not None, "step_mask is required when vectorized=False"
        step_lp = torch.full((ctx.batch_size,), 0.0, device=ctx.device, dtype=lp.dtype)
        step_lp[step_mask] = lp

        if save_logprobs:
            ctx.trajectory_log_probs.append(step_lp)

        return step_lp, ctx

    def get_current_estimator_output(self, ctx: Any) -> Optional[torch.Tensor]:
        """Expose the most recent per-step estimator output saved during `compute`."""
        return getattr(ctx, "current_estimator_output", None)


class RecurrentEstimatorAdapter(DefaultEstimatorAdapter):
    """Adapter for recurrent estimators with rollout carry (hidden state).

    - Requires ``estimator.init_carry(batch_size, device)`` and a forward that
      returns ``(estimator_outputs, new_carry)``.
    - Maintains ``ctx.carry`` across steps and updates it each call.
    - ``is_vectorized=False``; probability calculators use the per‑step path
      with legacy masks/alignment.
    """

    def __init__(self, estimator: "Estimator") -> None:
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
        save_estimator_outputs: bool = False,
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

        # Save current estimator output only when requested.
        if save_estimator_outputs:
            ctx.current_estimator_output = estimator_outputs

            if step_mask is not None:
                padded = torch.full(
                    (ctx.batch_size,) + estimator_outputs.shape[1:],
                    -float("inf"),
                    device=ctx.device,
                )
                padded[step_mask] = estimator_outputs
                ctx.trajectory_estimator_outputs.append(padded)
        else:
            ctx.current_estimator_output = None

        return dist, ctx


def maybe_instantiate_adapter(
    estimator: "Estimator",
    adapter: Callable[["Estimator"], EstimatorAdapter] | EstimatorAdapter | None,
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
        adapter_cls = cast(Callable[["Estimator"], EstimatorAdapter], adapter_cls)
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

        adapter_factory = cast(Callable[["Estimator"], EstimatorAdapter], adapter)
        return adapter_factory(estimator)

    # If an adapter instance is provided, use it.
    elif isinstance(adapter, EstimatorAdapter):
        return adapter

    else:
        raise ValueError(f"Invalid adapter type: {type(adapter)}")
