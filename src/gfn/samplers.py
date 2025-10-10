from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, cast

import torch
from torch.distributions import Distribution

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.estimators import Estimator
from gfn.states import GraphStates, States
from gfn.utils.common import ensure_same_device
from gfn.utils.graphs import graph_states_share_storage
from gfn.utils.handlers import check_cond_forward
from gfn.utils.prob_calculations import get_trajectory_pbs, get_trajectory_pfs


class EstimatorAdapter(Protocol):
    """Adapter interface for estimator-specific policy behavior.

    Purpose
    -------
    This Protocol defines the minimal interface the Sampler relies on, allowing
    us to keep one generic sampling loop while plugging in different estimator
    behaviors (e.g., non‑recurrent, recurrent with carry, tempered variants)
    without modifying the Sampler. We use a Protocol (structural typing) so any
    class that implements these members is accepted; no inheritance is required.

    Opaque context (ctx)
    --------------------
    The adapter owns an opaque context object (``ctx``). The Sampler never
    inspects it and simply passes it back to the adapter at each step. The
    adapter is responsible for:
      - initializing ``ctx`` once per rollout in ``init_context``
      - updating any internal state (e.g., recurrent ``carry``) during ``compute``
      - recording per‑step artifacts in ``record`` (e.g., log_probs,
        estimator outputs), typically with mask-aware padding
    Finalization is handled by the rollout context itself.

    Guidance
    --------
    - Allocate ``ctx`` once per rollout; mutate it in place for performance.
    - Apply masking inside the adapter (``step_mask``) when slicing conditioning
      or padding per‑step tensors back to full batch size.
    - Keep Sampler oblivious to estimator details (conditioning, carry, etc.).
    """

    @property
    def is_backward(self) -> bool:
        ...  # fmt: skip

    @property
    def is_vectorized(self) -> bool:
        ...  # fmt: skip

    def init_context(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: Optional[torch.Tensor] = None,
    ) -> Any:
        ...  # fmt: skip

    def compute(
        self,
        states_active: States,
        ctx: Any,
        step_mask: torch.Tensor,
        **policy_kwargs: Any,
    ) -> tuple[Distribution, Any]:
        ...  # fmt: skip

    def record(
        self,
        ctx: Any,
        step_mask: torch.Tensor,
        sampled_actions: torch.Tensor,
        dist: Distribution,
        save_logprobs: bool,
        save_estimator_outputs: bool,
    ) -> None:
        ...  # fmt: skip

    def log_prob_of_actions(
        self,
        states_active: States,
        actions_active: torch.Tensor,
        ctx: Any,
        step_mask: torch.Tensor,
        **policy_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
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

    def append_step(
        self,
        step_mask: torch.Tensor,
        sampled_actions: torch.Tensor,
        dist: Distribution,
        save_logprobs: bool,
        save_estimator_outputs: bool,
    ) -> None:
        """Record per-step artifacts into trajectory-level buffers owned by the context."""
        N = self.batch_size
        device = self.device

        if save_logprobs:
            lp_masked = dist.log_prob(sampled_actions)
            if torch.any(torch.isinf(lp_masked)):
                raise RuntimeError("Log probabilities are inf. This should not happen.")
            step_lp = torch.full((N,), 0.0, device=device)
            step_lp[step_mask] = lp_masked
            self.trajectory_log_probs.append(step_lp)

        if save_estimator_outputs and self.current_estimator_output is not None:
            est_out = self.current_estimator_output
            padded = torch.full((N,) + est_out.shape[1:], -float("inf"), device=device)
            padded[step_mask] = est_out
            self.trajectory_estimator_outputs.append(padded)

    def finalize(self) -> dict[str, Optional[torch.Tensor]]:
        """Stack recorded per-step artifacts along time into trajectory-level tensors."""
        log_probs = (
            torch.stack(self.trajectory_log_probs, dim=0)
            if self.trajectory_log_probs
            else None
        )
        estimator_outputs = (
            torch.stack(self.trajectory_estimator_outputs, dim=0)
            if self.trajectory_estimator_outputs
            else None
        )

        return {"log_probs": log_probs, "estimator_outputs": estimator_outputs}


class DefaultEstimatorAdapter:
    """Adapter for non-recurrent estimators (current default behavior).

    Overview
    --------
    This adapter bridges the generic sampling loop and the "classic" non‑recurrent
    estimators already used throughout the codebase. It exposes the minimal
    interface required by the `EstimatorAdapter` Protocol while keeping the
    sampler loop estimator-agnostic.

    Assumptions
    -----------
    - The wrapped estimator is non‑recurrent (no carry between steps).
    - If conditioning is provided, the estimator accepts `(states, conditioning)`;
      otherwise it accepts `(states)`.
    - The estimator provides `to_probability_distribution(states, est_out, **kw)`
      returning a torch Distribution over actions for the masked states.

    Context Lifecycle (opaque to the Sampler)
    ----------------------------------------
    The adapter owns an opaque rollout context `ctx` (see `RolloutContext`) which
    the Sampler never reads. The context is created once per rollout and mutated
    in place:

    - init_context(batch_size, device, conditioning) -> ctx
      Stores rollout invariants and optional conditioning. Also prepares per‑step
      buffers for artifacts that may be recorded (log_probs, estimator_outputs).

    - compute(states_active, ctx, step_mask, **policy_kwargs) -> (dist, ctx)
      1) Selects the appropriate estimator call signature depending on whether
         conditioning is present. If conditioning is present, the adapter slices
         it with `step_mask` so shapes match `states_active`.
      2) Calls the estimator forward pass to obtain the raw `est_out`.
      3) Converts `est_out` into a torch Distribution with
         `to_probability_distribution`.
      4) Saves `est_out` into `ctx.current_estimator_output` so it can optionally be
         recorded by `record` or exposed to callers that need it.

    - record(ctx, step_mask, sampled_actions, dist, save_logprobs, save_estimator_outputs)
      Materializes optional per‑step artifacts into context‑managed buffers with
      mask‑aware padding back to the full rollout batch size `N`:
      * Log‑probs: computes `dist.log_prob(sampled_actions)` for active rows only,
        then writes into a 1D tensor of shape `(N,)` filled with zeros and masked
        assignment for active positions. Appends this to a list (one tensor per time step).
      * Estimator outputs: if requested, pads the last estimator output
        (`ctx.current_estimator_output`) to shape `(N, ...)` using `-inf` for inactive rows
        and appends to a list (one tensor per time step).

    Finalization is performed by the context itself:
    - ctx.finalize() -> {"log_probs": Tensor | None, "estimator_outputs": Tensor | None}
      Stacks recorded per‑step lists along the time dimension into tensors of shape
      `(T, N, ...)` suitable for `Trajectories`. Returns `None` for any artifact
      that was never recorded.

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

    def compute(
        self,
        states_active: States,
        ctx: Any,
        step_mask: torch.Tensor,
        **policy_kwargs: Any,
    ) -> tuple[Distribution, Any]:
        """Run the estimator for active rows and build an action Distribution.

        - Uses `step_mask` to slice conditioning to the active subset.
        - Saves the raw estimator output in `ctx.current_estimator_output` for
          optional recording in `record_step`.
        """
        conditioning = ctx.conditioning  # type: ignore[attr-defined]
        cond_active = conditioning[step_mask] if conditioning is not None else None
        est_out = check_cond_forward(
            self._estimator, "estimator", states_active, cond_active
        )

        dist = self._estimator.to_probability_distribution(
            states_active, est_out, **policy_kwargs
        )
        ctx.current_estimator_output = est_out  # type: ignore[attr-defined]

        return dist, ctx

    def record(
        self,
        ctx: Any,
        step_mask: torch.Tensor,
        sampled_actions: torch.Tensor,
        dist: Distribution,
        save_logprobs: bool,
        save_estimator_outputs: bool,
    ) -> None:
        """Record per-step artifacts into the context's trajectory-level lists."""
        # Delegate recording to the rollout context
        ctx.append_step(  # type: ignore[attr-defined]
            step_mask=step_mask,
            sampled_actions=sampled_actions,
            dist=dist,
            save_logprobs=save_logprobs,
            save_estimator_outputs=save_estimator_outputs,
        )

    def log_prob_of_actions(
        self,
        states_active: States,
        actions_active: torch.Tensor,
        ctx: Any,
        step_mask: torch.Tensor,
        **policy_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        # Optional fast path: use caller-provided estimator outputs
        precomputed = policy_kwargs.pop("precomputed_estimator_output", None)
        if precomputed is not None:
            est_out = precomputed
        else:
            conditioning = ctx.conditioning  # type: ignore[attr-defined]
            cond_active = conditioning[step_mask] if conditioning is not None else None
            est_out = check_cond_forward(
                self._estimator, "estimator", states_active, cond_active
            )
        dist = self._estimator.to_probability_distribution(
            states_active, est_out, **policy_kwargs
        )
        ctx.current_estimator_output = est_out  # type: ignore[attr-defined]

        N = ctx.batch_size  # type: ignore[attr-defined]
        device = ctx.device  # type: ignore[attr-defined]
        lp_masked = dist.log_prob(actions_active)
        if torch.any(torch.isinf(lp_masked)):
            raise RuntimeError("Log probabilities are inf. This should not happen.")
        step_lp = torch.full((N,), 0.0, device=device)
        step_lp[step_mask] = lp_masked
        return step_lp, ctx

    # Vectorized helper to mirror legacy probability calculations without per-step context
    def log_prob_vectorized(
        self,
        states: States,
        actions_tensor: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        **policy_kwargs: Any,
    ) -> torch.Tensor:
        """Compute log_prob for a batch of (state, action) pairs in a vectorized way.

        This mirrors the legacy vectorized path used in probability utils and uses
        the adapter's estimator and distribution construction, including policy kwargs.
        """
        est_out = check_cond_forward(self._estimator, "estimator", states, conditioning)
        dist = self._estimator.to_probability_distribution(
            states, est_out, **policy_kwargs
        )
        return dist.log_prob(actions_tensor)

    def get_current_estimator_output(self, ctx: Any) -> Optional[torch.Tensor]:
        """Expose the most recent per-step estimator output saved during `compute`."""
        return getattr(ctx, "current_estimator_output", None)


class RecurrentEstimatorAdapter(DefaultEstimatorAdapter):
    """Adapter for recurrent estimators that require and update a carry.

    Overview
    --------
    This adapter extends the default (non‑recurrent) behavior to handle models that
    maintain a recurrent state ("carry"). It exposes the same surface as
    `DefaultEstimatorAdapter`, with the following differences:

    - `is_vectorized = False`: Probability calculators will use a non‑vectorized
      (per‑step) path that mirrors the legacy reference exactly (including masks and
      state/action alignment), since a recurrent carry must be updated sequentially.
    - The rollout `ctx` stores a `carry` that is initialized once via
      `estimator.init_carry(batch_size, device)` and updated at every call to
      `compute`/`log_prob_of_actions`.

    Context Lifecycle (opaque to the Sampler)
    ----------------------------------------
    The adapter owns an opaque rollout context `ctx` (see `RolloutContext`) which the
    Sampler never reads. The context is created once per rollout and mutated in place:

    - init_context(batch_size, device, conditioning) -> ctx
      Stores rollout invariants and optional conditioning. Initializes recurrent
      `carry` via `estimator.init_carry`. Also prepares per‑step buffers for optional
      artifacts (log_probs, estimator_outputs).

    - compute(states_active, ctx, step_mask, **policy_kwargs) -> (dist, ctx)
      1) Calls the recurrent estimator as `(states_active, ctx.carry) -> (est_out, new_carry)`
         and stores `new_carry` back into `ctx.carry`.
      2) Converts `est_out` into a torch Distribution with
         `to_probability_distribution(states_active, est_out, **policy_kwargs)`.
      3) Saves `est_out` into `ctx.current_estimator_output` for optional recording.

    - record(ctx, step_mask, sampled_actions, dist, save_logprobs, save_estimator_outputs)
      Materializes optional per‑step artifacts into context‑managed buffers with
      mask‑aware padding back to the full rollout batch size `N`:
      * Log‑probs: computes `dist.log_prob(sampled_actions)` for active rows only,
        then writes into a 1D tensor of shape `(N,)` filled with zeros and masked
        assignment for active positions. Appends this to a list (one tensor per time step).
      * Estimator outputs: if requested, pads `ctx.current_estimator_output` to shape
        `(N, ...)` using `-inf` for inactive rows and appends to a list.

    Finalization is performed by the context itself:
    - ctx.finalize() -> {"log_probs": Tensor | None, "estimator_outputs": Tensor | None}
      Stacks recorded per‑step lists along the time dimension into tensors of shape
      `(T, N, ...)` suitable for `Trajectories`. Returns `None` for any artifact
      that was never recorded.

    Probability Calculators
    -----------------------
    Since `is_vectorized = False`, the PF/PB probability calculators use the
    non‑vectorized, per‑step path that matches the legacy reference:
    - Trajectory PF: `step_mask = ~states.is_sink_state[t] & ~actions.is_dummy[t]`.
    - Trajectory PB: actions at time `t` are aligned with states at time `t+1`, and
      `step_mask = ~states.is_sink_state[t+1] & ~states.is_initial_state[t+1]
                  & ~actions.is_dummy[t] & ~actions.is_exit[t]` with `t==0` skipped.
    - Transitions: the same legacy masks are used, and a single adapter call is made
      per batch.
    No mask indexing with action ids is used; distributions handle illegal actions.
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
        # Expect estimator to implement init_carry(batch_size, device)
        init_carry_fn = cast(Callable[[int, torch.device], Any], init_carry)
        ctx.carry = init_carry_fn(batch_size, device)

        return ctx

    def compute(
        self,
        states_active: States,
        ctx: Any,
        step_mask: torch.Tensor,
        **policy_kwargs: Any,
    ) -> tuple[Distribution, Any]:
        """Run estimator with carry and update it.

        Differs from the default adapter by calling
        `estimator(states_active, ctx.carry) -> (est_out, new_carry)`, storing the
        updated carry and saving `current_estimator_output` before building the
        Distribution.
        """
        # Recurrent estimators are expected to accept (states, carry) -> (out, new_carry)
        est_out, new_carry = self._estimator(states_active, ctx.carry)  # type: ignore[attr-defined]
        ctx.carry = new_carry  # type: ignore[attr-defined]
        dist = self._estimator.to_probability_distribution(
            states_active, est_out, **policy_kwargs
        )
        ctx.current_estimator_output = est_out  # type: ignore[attr-defined]

        return dist, ctx

    def log_prob_of_actions(
        self,
        states_active: States,
        actions_active: torch.Tensor,
        ctx: Any,
        step_mask: torch.Tensor,
        **policy_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        # Recurrent estimators are expected to accept (states, carry) -> (out, new_carry)
        est_out, new_carry = self._estimator(states_active, ctx.carry)  # type: ignore[attr-defined]
        ctx.carry = new_carry  # type: ignore[attr-defined]
        dist = self._estimator.to_probability_distribution(
            states_active, est_out, **policy_kwargs
        )
        ctx.current_estimator_output = est_out  # type: ignore[attr-defined]

        N = ctx.batch_size  # type: ignore[attr-defined]
        device = ctx.device  # type: ignore[attr-defined]
        lp_masked = dist.log_prob(actions_active)
        if torch.any(torch.isinf(lp_masked)):
            raise RuntimeError("Log probabilities are inf. This should not happen.")
        step_lp = torch.full((N,), 0.0, device=device)
        step_lp[step_mask] = lp_masked
        return step_lp, ctx


class Sampler:
    """Wrapper for a PolicyEstimator that enables sampling from GFlowNet environments.

    A Sampler encapsulates a PolicyEstimator and provides methods to sample individual
    actions or complete trajectories from GFlowNet environments. It can be used for
    both forward and backward sampling, depending on the estimator's configuration.

    Attributes:
        estimator: The PolicyEstimator used for sampling actions and computing
            probability distributions.
    """

    def __init__(
        self, estimator: Estimator, adapter: Optional[EstimatorAdapter] = None
    ) -> None:
        """Initializes a Sampler with a PolicyEstimator.

        Args:
            estimator: The PolicyEstimator to use for sampling actions and computing
                probability distributions.
        """
        self.estimator = estimator
        self.adapter = (
            adapter if adapter is not None else DefaultEstimatorAdapter(estimator)
        )

    def sample_actions(
        self,
        env: Env,
        states: States,
        conditioning: torch.Tensor | None = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = False,
        ctx: Any | None = None,
        **policy_kwargs: Any,
    ) -> Tuple[Actions, torch.Tensor | None, torch.Tensor | None]:
        """Samples actions from the given states using the policy estimator.

        This method samples actions from the probability distribution defined by the
        policy estimator.

        When sampling off-policy, ensure to set `save_logprobs=False`. Log probabilities
        for off-policy actions should be calculated separately during GFlowNet training.

        Args:
            env: The environment where the states and actions are defined.
            states: A batch of states to sample actions from.
            conditioning: Optional tensor of conditioning information for conditional
                policies. If provided, the estimator must support conditional sampling.
            save_estimator_outputs: If True, returns the raw outputs from the estimator
                before conversion to probability distributions. This is useful for
                off-policy training with tempered policies.
            save_logprobs: If True, calculates and returns the log probabilities of
                the sampled actions under the policy distribution. This is useful for
                on-policy training.
            **policy_kwargs: Keyword arguments passed to the estimator's
                `to_probability_distribution` method. Common parameters include:
                - `temperature`: Scalar to divide logits by before softmax
                - `epsilon`: Probability of choosing random actions (exploration)
                - `sf_bias`: Bias to apply to exit action logits

        Returns:
            A tuple containing:
            - An Actions object with the sampled actions
            - Optional tensor of log probabilities (if save_logprobs=True)
            - Optional tensor of estimator outputs (if save_estimator_outputs=True)
        """
        if ctx is None:
            ctx = self.adapter.init_context(
                batch_size=states.batch_shape[0],
                device=states.device,
                conditioning=conditioning,
            )

        step_mask = torch.ones(
            states.batch_shape[0], dtype=torch.bool, device=states.device
        )
        dist, ctx = self.adapter.compute(states, ctx, step_mask, **policy_kwargs)

        with torch.no_grad():
            actions_tensor = dist.sample()

        if save_logprobs:
            log_probs = dist.log_prob(actions_tensor)
            if torch.any(torch.isinf(log_probs)):
                raise RuntimeError("Log probabilities are inf. This should not happen.")
        else:
            log_probs = None

        # Allow adapter to record per-step artifacts for callers that reuse ctx.
        self.adapter.record(
            ctx=ctx,
            step_mask=step_mask,
            sampled_actions=actions_tensor,
            dist=dist,
            save_logprobs=save_logprobs,
            save_estimator_outputs=save_estimator_outputs,
        )

        actions = env.actions_from_tensor(actions_tensor)

        estimator_output = None
        if save_estimator_outputs and hasattr(
            self.adapter, "get_current_estimator_output"
        ):
            estimator_output = self.adapter.get_current_estimator_output(ctx)

        assert log_probs is None or log_probs.shape == actions.batch_shape

        return actions, log_probs, estimator_output

    def sample_trajectories(
        self,
        env: Env,
        n: Optional[int] = None,
        states: Optional[States] = None,
        conditioning: Optional[torch.Tensor] = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Samples complete trajectories from the environment.

        This method samples trajectories by sequentially sampling actions from the
        policy estimator. It supports both forward and backward sampling, depending on
        the estimator's `is_backward` flag. If forward sampling, it samples until all
        trajectories reach the sink state. If backward sampling, it samples until all
        trajectories reach the initial state.

        Args:
            env: The environment to sample trajectories from.
            n: Number of trajectories to sample, all starting from s0. Must be
                provided if `states` is None.
            states: Initial states to start trajectories from. It should have batch_shape
                of length 1 (no trajectory dim). If `None`, `n` must be provided and we
                initialize `n` trajectories with the environment's initial state.
            conditioning: Optional tensor of conditioning information for conditional
                policies. Must match the batch shape of states.
            save_estimator_outputs: If True, saves the estimator outputs for each
                step. Useful for off-policy training with tempered policies.
            save_logprobs: If True, calculates and saves the log probabilities of
                sampled actions. Useful for on-policy training.
            **policy_kwargs: Keyword arguments passed to the policy estimator.
                See `sample_actions` for details.

        Returns:
            A Trajectories object containing the sampled trajectories with batch_shape
            (max_length+1, n_trajectories) for states and (max_length, n_trajectories)
            for actions.

        Note:
            For backward trajectories, the reward is computed at the initial state
            (s0) rather than the terminal state (sf).
        """
        if self.adapter.is_backward:
            # [ASSUMPTION] When backward sampling, all provided states are the
            # terminating states (can be passed to log_reward fn)
            assert (
                states is not None
            ), "When backward sampling, `states` must be provided"
            # assert states in env.terminating_states # This assert would be useful,
            # unfortunately, not every environment implements this.
        else:
            if states is None:
                assert n is not None, "Either kwarg `states` or `n` must be specified"
                states = env.reset(batch_shape=(n,))
            else:
                assert (
                    len(states.batch_shape) == 1
                ), "States should have a batch_shape of length 1, w/ no trajectory dim!"

        n_trajectories = states.batch_shape[0]
        device = states.device

        if conditioning is not None:
            assert states.batch_shape == conditioning.shape[: len(states.batch_shape)]
            ensure_same_device(states.device, conditioning.device)

        dones = (
            states.is_initial_state if self.adapter.is_backward else states.is_sink_state
        )

        # Define dummy actions to avoid errors when stacking empty lists.
        trajectories_states: List[States] = [states]
        trajectories_actions: List[Actions] = [
            env.actions_from_batch_shape((n_trajectories,))
        ]
        # Placeholder kept for backward-compatibility of shapes; logprobs are
        # recorded and stacked by the adapter.
        trajectories_terminating_idx = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )

        step = 0
        ctx = self.adapter.init_context(n_trajectories, device, conditioning)

        while not all(dones):
            actions = env.actions_from_batch_shape((n_trajectories,))
            step_mask = ~dones

            # Compute distribution on active rows
            dist, ctx = self.adapter.compute(
                states[step_mask], ctx, step_mask, **policy_kwargs
            )

            # Sample actions for active rows
            with torch.no_grad():
                valid_actions_tensor = dist.sample()
            valid_actions = env.actions_from_tensor(valid_actions_tensor)

            # Let adapter record artifacts
            self.adapter.record(
                ctx=ctx,
                step_mask=step_mask,
                sampled_actions=valid_actions_tensor,
                dist=dist,
                save_logprobs=save_logprobs,
                save_estimator_outputs=save_estimator_outputs,
            )

            actions[step_mask] = valid_actions

            trajectories_actions.append(actions)

            if self.adapter.is_backward:
                new_states = env._backward_step(states, actions)  # type: ignore[attr-defined]
            else:
                new_states = env._step(states, actions)  # type: ignore[attr-defined]

            # Ensure that the new state is a distinct object from the old state.
            assert new_states is not states
            assert isinstance(new_states, States)
            if isinstance(new_states, GraphStates) and isinstance(states, GraphStates):
                # Asserts that there exists no shared storage between the two
                # GraphStates.
                assert not graph_states_share_storage(new_states, states)
            else:
                # Asserts that there exists no shared storage between the two
                # States.
                assert new_states.tensor.data_ptr() != states.tensor.data_ptr()

            # Increment the step, determine which trajectories are finished, and eval
            # rewards.
            step += 1

            # new_dones means those trajectories that just finished. Because we
            # pad the sink state to every short trajectory, we need to make sure
            # to filter out the already done ones.
            new_dones = (
                new_states.is_initial_state
                if self.adapter.is_backward
                else new_states.is_sink_state
            ) & ~dones
            trajectories_terminating_idx[new_dones] = step

            states = new_states
            dones = dones | new_dones
            trajectories_states.append(states)

        # Stack all states and actions
        stacked_states = env.States.stack(trajectories_states)
        stacked_actions = env.Actions.stack(trajectories_actions)[
            1:
        ]  # Drop dummy action
        # Finalize stacked trajectory artifacts from the context (already shaped (T, N, ...))
        trajectory_artifacts = ctx.finalize()  # type: ignore[attr-defined]
        stacked_logprobs = trajectory_artifacts.get("log_probs", None)
        stacked_estimator_outputs = trajectory_artifacts.get("estimator_outputs", None)

        if stacked_logprobs is not None and len(stacked_logprobs) == 0:
            stacked_logprobs = None
        if stacked_estimator_outputs is not None and len(stacked_estimator_outputs) == 0:
            stacked_estimator_outputs = None

        # Broadcast conditioning tensor to match states batch shape if needed
        if conditioning is not None:
            # The states have batch shape (max_length, n_trajectories)
            # The conditioning tensor should have shape (n_trajectories,) or (n_trajectories, 1)
            # We need to broadcast it to (max_length, n_trajectories, 1) for the estimator
            if len(conditioning.shape) == 1:
                # conditioning has shape (n_trajectories,)
                conditioning = (
                    conditioning.unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(stacked_states.batch_shape[0], -1, 1)
                )
            elif len(conditioning.shape) == 2 and conditioning.shape[1] == 1:
                # conditioning has shape (n_trajectories, 1)
                conditioning = conditioning.unsqueeze(0).expand(
                    stacked_states.batch_shape[0], -1, -1
                )

        trajectories = Trajectories(
            env=env,
            states=stacked_states,
            conditioning=conditioning,
            actions=stacked_actions,
            terminating_idx=trajectories_terminating_idx,
            is_backward=self.adapter.is_backward,
            log_rewards=None,  # will be calculated later
            log_probs=stacked_logprobs,
            estimator_outputs=stacked_estimator_outputs,
        )

        return trajectories


class LocalSearchSampler(Sampler):
    """Sampler equipped with local search capabilities.

    The LocalSearchSampler extends the basic Sampler with local search functionality
    based on the back-and-forth heuristic. This approach was first proposed by
    [Zhang et al. 2022](https://arxiv.org/abs/2202.01361) and further explored by
    [Kim et al. 2023](https://arxiv.org/abs/2310.02710).

    The local search process involves:
    1. Taking a trajectory and performing K backward steps using a backward policy
    2. Reconstructing the trajectory from the junction state using the forward policy
    3. Optionally applying Metropolis-Hastings acceptance criterion

    Attributes:
        estimator: The forward policy estimator (inherited from Sampler).
        backward_sampler: A Sampler instance with the backward policy estimator.
    """

    def __init__(
        self,
        pf_estimator: Estimator,
        pb_estimator: Estimator,
    ):
        """Initializes a LocalSearchSampler with forward and backward estimators.

        Args:
            pf_estimator: The forward policy estimator for sampling and reconstructing
                trajectories.
            pb_estimator: The backward policy estimator for backtracking trajectories.
        """
        super().__init__(pf_estimator)
        self.backward_sampler = Sampler(pb_estimator)

    def local_search(
        self,
        env: Env,
        trajectories: Trajectories,
        conditioning: torch.Tensor | None = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = False,
        back_steps: torch.Tensor | None = None,
        back_ratio: float | None = None,
        use_metropolis_hastings: bool = True,
        debug: bool = False,
        **policy_kwargs: Any,
    ) -> tuple[Trajectories, torch.Tensor]:
        """Performs local search on a batch of trajectories.

        This method implements the local search algorithm by:
        1. For each trajectory, performing K backward steps to reach a junction state
        2. Reconstructing the trajectory from the junction state using the forward policy
        3. Optionally applying Metropolis-Hastings acceptance criterion to decide whether
           to accept the new trajectory.

        Args:
            env: The environment to sample trajectories from.
            trajectories: The batch of trajectories to perform local search on.
            conditioning: Optional tensor of conditioning information for conditional
                policies. Must match the batch shape of states.
            save_estimator_outputs: If True, saves the estimator outputs for each
                step. Useful for off-policy training with tempered policies.
            save_logprobs: If True, calculates and saves the log probabilities of
                sampled actions. Useful for on-policy training.
            back_steps: Number of backward steps to take. Must be provided if
                `back_ratio` is None.
            back_ratio: Ratio of trajectory length to use for backward steps.
                Must be provided if `back_steps` is None.
            use_metropolis_hastings: If True, applies Metropolis-Hastings acceptance
                criterion. If False, accepts new trajectories if they have higher
                rewards.
            debug: If True, performs additional validation checks for debugging.
            **policy_kwargs: Keyword arguments passed to the policy estimator.
                See `sample_actions` for details.

        Returns:
            A tuple containing:
            - A Trajectories object refined by local search
            - A boolean tensor indicating which trajectories were updated
        """
        # TODO: Implement local search for GraphStates.
        if isinstance(env.States, GraphStates):
            raise NotImplementedError("Local search is not implemented for GraphStates.")

        save_logprobs = save_logprobs or use_metropolis_hastings

        # K-step backward sampling with the backward estimator,
        # where K is the number of backward steps used in https://arxiv.org/abs/2202.01361.
        if back_steps is None:
            assert (
                back_ratio is not None and 0 < back_ratio <= 1
            ), "Either kwarg `back_steps` or `back_ratio` must be specified"
            K = torch.ceil(back_ratio * (trajectories.terminating_idx - 1)).long()
        else:
            K = torch.where(
                back_steps > trajectories.terminating_idx,
                trajectories.terminating_idx,
                back_steps,
            )

        prev_trajectories = self.backward_sampler.sample_trajectories(
            env,
            states=trajectories.terminating_states,
            conditioning=conditioning,
            save_estimator_outputs=save_estimator_outputs,
            save_logprobs=save_logprobs,
            **policy_kwargs,
        )

        # By reversing the backward trajectories, obtain the forward trajectories.
        # This is called `prev_trajectories` since they are the trajectories before
        # the local search. The `new_trajectories` will be obtained by performing local
        # search on them.
        prev_trajectories = prev_trajectories.reverse_backward_trajectories()
        assert prev_trajectories.log_rewards is not None

        # Reconstructing with self.estimator
        n_prevs = prev_trajectories.terminating_idx - K - 1
        junction_states_tsr = torch.gather(
            prev_trajectories.states.tensor,
            0,
            (n_prevs).view(1, -1, 1).expand(-1, -1, *trajectories.states.state_shape),
        ).squeeze(0)
        recon_trajectories = super().sample_trajectories(
            env,
            states=env.states_from_tensor(junction_states_tsr),
            conditioning=conditioning,
            save_estimator_outputs=save_estimator_outputs,
            save_logprobs=save_logprobs,
            **policy_kwargs,
        )

        # Calculate the log probabilities as needed.
        prev_trajectories_log_pf = (
            get_trajectory_pfs(pf=self.estimator, trajectories=prev_trajectories)
            if save_logprobs
            else None
        )
        recon_trajectories_log_pf = (
            get_trajectory_pfs(pf=self.estimator, trajectories=recon_trajectories)
            if save_logprobs
            else None
        )
        prev_trajectories_log_pb = (
            get_trajectory_pbs(
                pb=self.backward_sampler.estimator, trajectories=prev_trajectories
            )
            if use_metropolis_hastings
            else None
        )
        recon_trajectories_log_pb = (
            get_trajectory_pbs(
                pb=self.backward_sampler.estimator, trajectories=recon_trajectories
            )
            if use_metropolis_hastings
            else None
        )

        (
            new_trajectories,
            new_trajectories_log_pf,
            new_trajectories_log_pb,
        ) = self._combine_prev_and_recon_trajectories(
            n_prevs=n_prevs,
            prev_trajectories=prev_trajectories,
            recon_trajectories=recon_trajectories,
            prev_trajectories_log_pf=prev_trajectories_log_pf,
            recon_trajectories_log_pf=recon_trajectories_log_pf,
            prev_trajectories_log_pb=prev_trajectories_log_pb,
            recon_trajectories_log_pb=recon_trajectories_log_pb,
            debug=debug,
        )

        if use_metropolis_hastings:
            assert (
                prev_trajectories_log_pb is not None
                and new_trajectories_log_pf is not None
                and new_trajectories_log_pb is not None
                and prev_trajectories_log_pf is not None
                and new_trajectories.log_rewards is not None
            )

            # The acceptance ratio is: min(1, R(x')p(x->s'->x') / R(x)p(x'->s'-> x))
            # Also, note this:
            # p(x->s'->x') / p(x'->s'-> x))
            # = p_B(x->s')p_F(s'->x') / p_B(x'->s')p_F(s'->x)
            # = p_B(x->s'->s0)p_F(s0->s'->x') / p_B(x'->s'->s0)p_F(s0->s'->x)
            # = p_B(tau|x)p_F(tau') / p_B(tau'|x')p_F(tau)
            log_accept_ratio = torch.clamp_max(
                new_trajectories.log_rewards
                + prev_trajectories_log_pb.sum(0)
                + new_trajectories_log_pf.sum(0)
                - prev_trajectories.log_rewards
                - new_trajectories_log_pb.sum(0)
                - prev_trajectories_log_pf.sum(0),
                0.0,
            )
            is_updated = torch.rand(
                new_trajectories.n_trajectories, device=log_accept_ratio.device
            ) < torch.exp(log_accept_ratio)
        else:
            assert prev_trajectories.log_rewards is not None
            assert new_trajectories.log_rewards is not None
            is_updated = prev_trajectories.log_rewards <= new_trajectories.log_rewards

        return new_trajectories, is_updated

    def sample_trajectories(
        self,
        env: Env,
        n: Optional[int] = None,
        states: Optional[States] = None,
        conditioning: Optional[torch.Tensor] = None,
        save_estimator_outputs: bool = False,  # FIXME: currently not work if this is True
        save_logprobs: bool = False,  # FIXME: currently not work if this is True
        n_local_search_loops: int = 0,
        back_steps: torch.Tensor | None = None,
        back_ratio: float | None = None,
        use_metropolis_hastings: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Samples trajectories with optional local search operations.

        This method extends the basic trajectory sampling with local search operations.
        After sampling initial trajectories, it performs multiple rounds of local search
        to potentially improve the trajectory quality in terms of the reward.

        Args:
            env: The environment to sample trajectories from.
            n: Number of trajectories to sample, all starting from s0. Must be
                provided if `states` is None.
            states: Initial states to start trajectories from. It should have batch_shape
                of length 1 (no trajectory dim). If `None`, `n` must be provided and we
                initialize `n` trajectories with the environment's initial state.
            conditioning: Optional tensor of conditioning information for conditional
                policies. Must match the batch shape of states.
            save_estimator_outputs: If True, saves the estimator outputs for each
                step. Useful for off-policy training with tempered policies.
            save_logprobs: If True, calculates and saves the log probabilities of
                sampled actions. Useful for on-policy training.
            n_local_search_loops: Number of local search loops to perform after
                initial sampling. Each loop creates additional trajectories.
            back_steps: Number of backward steps to take. Must be provided if
                `back_ratio` is None.
            back_ratio: Ratio of trajectory length to use for backward steps.
                Must be provided if `back_steps` is None.
            use_metropolis_hastings: If True, applies Metropolis-Hastings acceptance
                criterion. If False, accepts new trajectories if they have higher
                rewards.
            **policy_kwargs: Keyword arguments passed to the policy estimator.
                See `sample_actions` for details.

        Returns:
            A Trajectories object representing the batch of sampled trajectories,
            where the number of trajectories is n * (1 + n_local_search_loops).

        Note:
            The final trajectories container contains both the initial trajectories
            and the improved trajectories from local search.
        """
        trajectories = super().sample_trajectories(
            env,
            n,
            states,
            conditioning,
            save_estimator_outputs,
            save_logprobs or use_metropolis_hastings,
            **policy_kwargs,
        )

        if n is None:
            n = int(trajectories.n_trajectories)

        search_indices = torch.arange(n, device=trajectories.states.device)

        for it in range(n_local_search_loops):
            # Search phase
            ls_trajectories, is_updated = self.local_search(
                env,
                trajectories[search_indices],
                conditioning,
                save_estimator_outputs,
                save_logprobs,
                back_steps,
                back_ratio,
                use_metropolis_hastings,
                **policy_kwargs,
            )
            trajectories.extend(ls_trajectories)

            last_indices = torch.arange(
                n * it, n * (it + 1), device=trajectories.states.device
            )
            search_indices[is_updated] = last_indices[is_updated]

        return trajectories

    @staticmethod
    def _combine_prev_and_recon_trajectories(  # noqa: C901
        n_prevs: torch.Tensor,
        prev_trajectories: Trajectories,
        recon_trajectories: Trajectories,
        prev_trajectories_log_pf: torch.Tensor | None = None,
        recon_trajectories_log_pf: torch.Tensor | None = None,
        prev_trajectories_log_pb: torch.Tensor | None = None,
        recon_trajectories_log_pb: torch.Tensor | None = None,
        debug: bool = False,
    ) -> tuple[Trajectories, torch.Tensor | None, torch.Tensor | None]:
        """Combines previous and reconstructed trajectories to create new trajectories.

        This static method combines two trajectory segments: `prev_trajectories` and
        `recon_trajectories` to create `new_trajectories`. Specifically,
        `new_trajectories` is constructed by replacing certain portion of the
        `prev_trajectories` with `recon_trajectories`. See self.local_search for how
        to generate `prev_trajectories` and `recon_trajectories`.

        Args:
            n_prevs: Tensor indicating how many steps to take from prev_trajectories
                for each trajectory in the batch.
            prev_trajectories: Trajectories obtained from backward sampling.
            recon_trajectories: Trajectories obtained from forward reconstruction.
            prev_trajectories_log_pf: Optional log probabilities for forward policy
                on `prev_trajectories`.
            recon_trajectories_log_pf: Optional log probabilities for forward policy
                on `recon_trajectories`.
            prev_trajectories_log_pb: Optional log probabilities for backward policy
                on `prev_trajectories`.
            recon_trajectories_log_pb: Optional log probabilities for backward policy
                on `recon_trajectories`.
            debug: If True, performs additional validation checks for debugging.

        Returns:
            A tuple containing:
            - the `new_trajectories` Trajectories object with the combined trajectories
            - the `new_trajectories_log_pf` tensor of combined forward log probabilities
            - the `new_trajectories_log_pb` tensor of combined backward log probabilities

        Note:
            This method performs complex tensor operations to efficiently combine
            trajectory segments. The debug mode compares the vectorized approach
            with a for-loop implementation to ensure correctness.
        """
        new_trajectories_log_pf = None
        new_trajectories_log_pb = None

        bs = prev_trajectories.n_trajectories
        device = prev_trajectories.states.device
        env = prev_trajectories.env

        # Obtain full trajectories by concatenating the backward and forward parts.
        max_n_prev = n_prevs.max()
        n_recons = recon_trajectories.terminating_idx
        max_n_recon = n_recons.max()

        new_trajectories_log_rewards = recon_trajectories.log_rewards  # Episodic reward
        new_trajectories_dones = n_prevs + n_recons
        max_traj_len = int(new_trajectories_dones.max().item())

        # Create helper indices and masks
        idx = torch.arange(max_traj_len + 1).unsqueeze(1).expand(-1, bs).to(n_prevs)

        prev_mask = idx < n_prevs
        state_recon_mask = (idx >= n_prevs) * (idx <= n_prevs + n_recons)
        state_recon_mask2 = idx[: max_n_recon + 1] <= n_recons
        action_recon_mask = (idx[:-1] >= n_prevs) * (idx[:-1] <= n_prevs + n_recons - 1)
        action_recon_mask2 = idx[:max_n_recon] <= n_recons - 1

        # Transpose for easier indexing
        prev_trajectories_states_tsr = prev_trajectories.states.tensor.transpose(0, 1)
        prev_trajectories_actions_tsr = prev_trajectories.actions.tensor.transpose(0, 1)
        recon_trajectories_states_tsr = recon_trajectories.states.tensor.transpose(0, 1)
        recon_trajectories_actions_tsr = recon_trajectories.actions.tensor.transpose(
            0, 1
        )
        prev_mask = prev_mask.transpose(0, 1)
        state_recon_mask = state_recon_mask.transpose(0, 1)
        state_recon_mask2 = state_recon_mask2.transpose(0, 1)
        action_recon_mask = action_recon_mask.transpose(0, 1)
        action_recon_mask2 = action_recon_mask2.transpose(0, 1)

        # Prepare the new states and actions
        # Note that these are initialized in transposed shapes
        new_trajectories_states_tsr = env.sf.repeat(bs, max_traj_len + 1, 1).to(
            prev_trajectories.states.tensor
        )
        new_trajectories_actions_tsr = env.dummy_action.repeat(bs, max_traj_len, 1).to(
            prev_trajectories.actions.tensor
        )

        # Assign the first part (backtracked from backward policy) of the trajectory
        prev_mask_truc = prev_mask[:, :max_n_prev]
        new_trajectories_states_tsr[prev_mask] = prev_trajectories_states_tsr[
            :, :max_n_prev
        ][prev_mask_truc]
        new_trajectories_actions_tsr[prev_mask[:, :-1]] = prev_trajectories_actions_tsr[
            :, :max_n_prev
        ][prev_mask_truc]

        # Assign the second part (reconstructed from forward policy) of the trajectory
        new_trajectories_states_tsr[state_recon_mask] = recon_trajectories_states_tsr[
            state_recon_mask2
        ]
        new_trajectories_actions_tsr[action_recon_mask] = recon_trajectories_actions_tsr[
            action_recon_mask2
        ]

        # Transpose back
        new_trajectories_states_tsr = new_trajectories_states_tsr.transpose(0, 1)
        new_trajectories_actions_tsr = new_trajectories_actions_tsr.transpose(0, 1)

        # Similarly, combine log_pf and log_pb if needed
        if (
            prev_trajectories_log_pf is not None
            and recon_trajectories_log_pf is not None
        ):
            prev_trajectories_log_pf = prev_trajectories_log_pf.transpose(0, 1)
            recon_trajectories_log_pf = recon_trajectories_log_pf.transpose(0, 1)
            new_trajectories_log_pf = torch.full((bs, max_traj_len), 0.0).to(
                device=device, dtype=prev_trajectories_log_pf.dtype  # type: ignore
            )
            new_trajectories_log_pf[prev_mask[:, :-1]] = prev_trajectories_log_pf[  # type: ignore
                :, :max_n_prev
            ][
                prev_mask_truc
            ]
            new_trajectories_log_pf[action_recon_mask] = recon_trajectories_log_pf[  # type: ignore
                action_recon_mask2
            ]
            new_trajectories_log_pf = new_trajectories_log_pf.transpose(0, 1)
        if (
            prev_trajectories_log_pb is not None
            and recon_trajectories_log_pb is not None
        ):
            prev_trajectories_log_pb = prev_trajectories_log_pb.transpose(0, 1)
            recon_trajectories_log_pb = recon_trajectories_log_pb.transpose(0, 1)
            new_trajectories_log_pb = torch.full((bs, max_traj_len), 0.0).to(
                device=device, dtype=prev_trajectories_log_pb.dtype  # type: ignore
            )
            new_trajectories_log_pb[prev_mask[:, :-1]] = prev_trajectories_log_pb[  # type: ignore
                :, :max_n_prev
            ][
                prev_mask_truc
            ]
            new_trajectories_log_pb[action_recon_mask] = recon_trajectories_log_pb[  # type: ignore
                action_recon_mask2
            ]
            new_trajectories_log_pb = new_trajectories_log_pb.transpose(0, 1)

        # ------------------------------ DEBUG ------------------------------
        # If `debug` is True (expected only when testing), compare the
        # vectorized approach's results (above) to the for-loop results (below).
        if debug:
            _new_trajectories_states_tsr = env.sf.repeat(max_traj_len + 1, bs, 1).to(
                prev_trajectories.states.tensor
            )
            _new_trajectories_actions_tsr = env.dummy_action.repeat(
                max_traj_len, bs, 1
            ).to(prev_trajectories.actions.tensor)

            if (
                prev_trajectories_log_pf is not None
                and recon_trajectories_log_pf is not None
            ):
                _new_trajectories_log_pf = torch.full((max_traj_len, bs), 0.0).to(
                    device=device, dtype=prev_trajectories_log_pf.dtype
                )
                prev_trajectories_log_pf = prev_trajectories_log_pf.transpose(0, 1)
                recon_trajectories_log_pf = recon_trajectories_log_pf.transpose(0, 1)

            if (
                prev_trajectories_log_pb is not None
                and recon_trajectories_log_pb is not None
            ):
                _new_trajectories_log_pb = torch.full((max_traj_len, bs), 0.0).to(
                    device=device, dtype=prev_trajectories_log_pb.dtype
                )
                prev_trajectories_log_pb = prev_trajectories_log_pb.transpose(0, 1)
                recon_trajectories_log_pb = recon_trajectories_log_pb.transpose(0, 1)

            for i in range(bs):
                _n_prev = n_prevs[i]

                # Backward part
                _new_trajectories_states_tsr[: _n_prev + 1, i] = (
                    prev_trajectories.states.tensor[: _n_prev + 1, i]
                )
                _new_trajectories_actions_tsr[:_n_prev, i] = (
                    prev_trajectories.actions.tensor[:_n_prev, i]
                )

                # Forward part
                _len_recon = recon_trajectories.terminating_idx[i]
                _new_trajectories_states_tsr[
                    _n_prev + 1 : _n_prev + _len_recon + 1, i
                ] = recon_trajectories.states.tensor[1 : _len_recon + 1, i]
                _new_trajectories_actions_tsr[_n_prev : _n_prev + _len_recon, i] = (
                    recon_trajectories.actions.tensor[:_len_recon, i]
                )

                if (
                    prev_trajectories_log_pf is not None
                    and recon_trajectories_log_pf is not None
                ):
                    _new_trajectories_log_pf[:_n_prev, i] = prev_trajectories_log_pf[
                        :_n_prev, i
                    ]
                    _new_trajectories_log_pf[_n_prev : _n_prev + _len_recon, i] = (
                        recon_trajectories_log_pf[:_len_recon, i]
                    )
                if (
                    prev_trajectories_log_pb is not None
                    and recon_trajectories_log_pb is not None
                ):
                    _new_trajectories_log_pb[:_n_prev, i] = prev_trajectories_log_pb[
                        :_n_prev, i
                    ]
                    _new_trajectories_log_pb[_n_prev : _n_prev + _len_recon, i] = (
                        recon_trajectories_log_pb[:_len_recon, i]
                    )

            assert torch.all(_new_trajectories_states_tsr == new_trajectories_states_tsr)
            assert torch.all(
                _new_trajectories_actions_tsr == new_trajectories_actions_tsr
            )
            if (
                prev_trajectories_log_pf is not None
                and recon_trajectories_log_pf is not None
            ):
                assert torch.all(_new_trajectories_log_pf == new_trajectories_log_pf)
            if (
                prev_trajectories_log_pb is not None
                and recon_trajectories_log_pb is not None
            ):
                assert torch.all(_new_trajectories_log_pb == new_trajectories_log_pb)

        new_trajectories = Trajectories(
            env=env,
            states=env.states_from_tensor(new_trajectories_states_tsr),
            conditioning=prev_trajectories.conditioning,
            actions=env.actions_from_tensor(new_trajectories_actions_tsr),
            terminating_idx=new_trajectories_dones,
            is_backward=False,
            log_rewards=new_trajectories_log_rewards,
            log_probs=new_trajectories_log_pf,
        )

        return new_trajectories, new_trajectories_log_pf, new_trajectories_log_pb
