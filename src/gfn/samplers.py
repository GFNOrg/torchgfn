from typing import Any, List, Optional, Tuple, cast

import torch

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.estimators import Estimator, PolicyEstimatorProtocol
from gfn.states import GraphStates, States
from gfn.utils.common import ensure_same_device
from gfn.utils.graphs import graph_states_share_storage
from gfn.utils.prob_calculations import get_trajectory_pbs, get_trajectory_pfs


class Sampler:
    """Estimator‑driven sampler for GFlowNet environments.

    The estimator builds action distributions, computes step log‑probs, and records
    artifacts into a rollout context via method flags. Direction (forward/backward)
    is determined by ``estimator.is_backward``.

    Attributes:
        estimator: The underlying policy estimator. Must expose the methods contained
            in the `PolicyMixin` mixin.
    """

    def __init__(self, estimator: Estimator) -> None:
        """Initializes a Sampler with a PolicyEstimator."""
        self.estimator = estimator
        # TODO: Assert that the estimator exposes the methods contained in the `PolicyMixin` mixin.

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
        """Sample one step from ``states`` via the estimator.

        Initializes or reuses a rollout context with ``estimator.init_context``,
        builds a Distribution with ``estimator.compute_dist``, and optionally computes
        log‑probs with ``estimator.log_probs``. Per‑step artifacts are recorded by
        the estimator when the corresponding flags are set.

        Args:
            env: Environment providing action/state conversion utilities.
            states: Batch of states to act on.
            conditioning: Optional conditioning for conditional policies.
            save_estimator_outputs: If True, return the raw estimator outputs
                cached by the PolicyMixin for this step. Useful for off-policy training
                with tempered policies.
            save_logprobs: If True, return per‑step log‑probs padded to batch.
                Useful for on-policy training.
            **policy_kwargs: Extra kwargs forwarded to
                ``to_probability_distribution``.

        Returns:
            ``(Actions, log_probs | None, estimator_outputs | None)``. The
            estimator outputs come from
            ``PolicyMixin.get_current_estimator_output(ctx)`` when requested.
        """
        # NOTE: Explicitly cast to the policy protocol so static analyzers know
        # the estimator exposes the mixin methods (init_context/compute_dist/log_probs).
        policy_estimator = cast(PolicyEstimatorProtocol, self.estimator)
        # Runtime guard: ensure the estimator actually implements the required protocol methods.
        # This keeps helpful error messages when a non‑policy estimator is supplied.
        for required in ("init_context", "compute_dist", "log_probs"):
            if not hasattr(policy_estimator, required):
                raise TypeError(
                    f"Estimator is not policy-capable (missing PolicyMixin method: {required})"
                )

        if ctx is None:
            ctx = policy_estimator.init_context(
                batch_size=states.batch_shape[0],
                device=states.device,
                conditioning=conditioning,
            )

        step_mask = torch.ones(
            states.batch_shape[0], dtype=torch.bool, device=states.device
        )
        dist, ctx = policy_estimator.compute_dist(
            states,
            ctx,
            step_mask,
            save_estimator_outputs=save_estimator_outputs,
            **policy_kwargs,
        )

        with torch.no_grad():
            actions_tensor = dist.sample()

        if save_logprobs:
            # Use estimator to compute step log-probs and pad to batch.
            log_probs, ctx = policy_estimator.log_probs(
                actions_tensor,
                dist,
                ctx,
                step_mask,
                vectorized=False,
                save_logprobs=True,
            )
        else:
            log_probs = None

        actions = env.actions_from_tensor(actions_tensor)

        estimator_output = None
        if save_estimator_outputs:
            if not hasattr(policy_estimator, "get_current_estimator_output"):
                raise TypeError(
                    "Estimator does not support get_current_estimator_output and save_estimator_outputs is True!"
                )
            estimator_output = policy_estimator.get_current_estimator_output(ctx)
            assert estimator_output is not None

        assert log_probs is None or log_probs.shape == actions.batch_shape

        return actions, log_probs, estimator_output

    # TODO: How to avoid "Sampler.sample_trajectories' is too complex" error?
    def sample_trajectories(  # noqa: C901
        self,
        env: Env,
        n: Optional[int] = None,
        states: Optional[States] = None,
        conditioning: Optional[torch.Tensor] = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Roll out complete trajectories using the estimator.

        Reuses a single rollout context across steps, calling
        ``compute_dist`` & ``log_probs`` each iteration. Uses
        ``estimator.is_backward`` to choose the environment step function.

        Args:
            env: Environment to sample in.
            n: Number of trajectories if ``states`` is None.
            states: Starting states (batch shape length 1) or ``None``.
            conditioning: Optional conditioning aligned with the batch.
            save_estimator_outputs: If True, store per‑step estimator outputs. Useful
                for off-policy training with tempered policies.
            save_logprobs: If True, store per‑step log‑probs.  Useful for on-policy
                training.
            **policy_kwargs: Extra kwargs forwarded to the policy.

        Returns:
            A ``Trajectories`` with stacked states/actions and any artifacts.

        Note:
            For backward trajectories, the reward is computed at the initial state
            (s0) rather than the terminal state (sf).
        """
        # NOTE: Cast to the policy protocol for static typing across mixin methods/properties.
        policy_estimator = cast(PolicyEstimatorProtocol, self.estimator)
        # Runtime guard: ensure the estimator actually implements the required protocol
        # method and raises an error when a non‑policy estimator is supplied.
        for required in ("init_context", "compute_dist", "log_probs"):
            if not hasattr(policy_estimator, required):
                raise TypeError(
                    f"Estimator is not policy-capable (missing PolicyMixin method: {required})"
                )

        if policy_estimator.is_backward:
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

        if policy_estimator.is_backward:
            dones = states.is_initial_state
        else:
            dones = states.is_sink_state

        # Define dummy actions to avoid errors when stacking empty lists.
        trajectories_states: List[States] = [states]
        trajectories_actions: List[Actions] = [
            env.actions_from_batch_shape((n_trajectories,))
        ]
        # Placeholder kept for backward-compatibility of shapes; logprobs are
        # recorded and stacked by the estimator via the context.
        trajectories_terminating_idx = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )

        step = 0
        if not hasattr(policy_estimator, "init_context"):
            raise TypeError("Estimator is not policy-capable (missing PolicyMixin)")
        ctx = policy_estimator.init_context(n_trajectories, device, conditioning)

        while not all(dones):
            actions = env.actions_from_batch_shape((n_trajectories,))
            step_mask = ~dones

            # Compute distribution on active rows
            dist, ctx = policy_estimator.compute_dist(
                states[step_mask],
                ctx,
                step_mask,
                save_estimator_outputs=save_estimator_outputs,
                **policy_kwargs,
            )

            # Sample actions for active rows
            with torch.no_grad():
                valid_actions_tensor = dist.sample()
            valid_actions = env.actions_from_tensor(valid_actions_tensor)

            if save_logprobs:
                # Use estimator to compute step log-probs and pad to batch (recorded in ctx).
                _, ctx = policy_estimator.log_probs(
                    valid_actions_tensor,
                    dist,
                    ctx,
                    step_mask,
                    vectorized=False,
                    save_logprobs=True,
                )

            actions[step_mask] = valid_actions
            trajectories_actions.append(actions)

            if policy_estimator.is_backward:
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
                if policy_estimator.is_backward
                else new_states.is_sink_state
            ) & ~dones
            trajectories_terminating_idx[new_dones] = step

            states = new_states
            dones = dones | new_dones
            trajectories_states.append(states)

        # Stack all states and actions.
        stacked_states = env.States.stack(trajectories_states)

        # Stack actions, drop dummy action.
        stacked_actions = env.Actions.stack(trajectories_actions)[1:]

        # Get trajectory artifacts from the context (already shaped (T, N, ...))
        stacked_logprobs = (
            torch.stack(ctx.trajectory_log_probs, dim=0)
            if ctx.trajectory_log_probs
            else None
        )
        stacked_estimator_outputs = (
            torch.stack(ctx.trajectory_estimator_outputs, dim=0)
            if ctx.trajectory_estimator_outputs
            else None
        )

        # Stacked logprobs and estimator outputs are only None if there are no
        # valid trajectories.
        if stacked_logprobs is not None:
            if len(stacked_logprobs) == 0:
                stacked_logprobs = None

        if stacked_estimator_outputs is not None:
            if len(stacked_estimator_outputs) == 0:
                stacked_estimator_outputs = None

        # Broadcast conditioning tensor to match states batch shape if needed
        if conditioning is not None:
            # The states have batch shape (max_length, n_trajectories). The
            # conditioning tensor should have shape (n_trajectories,) or
            # (n_trajectories, 1). We need to broadcast it to (max_length,
            # n_trajectories, 1) for the estimator
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
            is_backward=policy_estimator.is_backward,
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
    ) -> None:
        """Initializes a LocalSearchSampler with forward and backward estimators.

        Args:
            pf_estimator: The forward policy estimator for sampling and reconstructing
                trajectories.
            pb_estimator: The backward policy estimator for backtracking trajectories.
        """
        super().__init__(pf_estimator)
        self.backward_sampler = Sampler(pb_estimator)

    @staticmethod
    def _compute_back_steps(
        terminating_idx: torch.Tensor,
        back_steps: torch.Tensor | None,
        back_ratio: float | None,
    ) -> torch.Tensor:
        """Compute per-trajectory backtrack length K with validation and clamping.

        This centralizes the logic for deriving K used in local search.
        The behavior mirrors the inline implementation:
        - When ``back_steps`` is None, require ``0 < back_ratio <= 1`` and set
          ``K = ceil(back_ratio * (terminating_idx - 1))``.
        - Otherwise, clamp provided ``back_steps`` to ``terminating_idx``.
        """
        if back_steps is None:
            assert (
                back_ratio is not None and 0 < back_ratio <= 1
            ), "Either kwarg `back_steps` or `back_ratio` must be specified"
            return torch.ceil(back_ratio * (terminating_idx - 1)).long()
        return torch.where(
            back_steps > terminating_idx,
            terminating_idx,
            back_steps,
        )

    def _reconstruct_from_junctions(
        self,
        env: Env,
        prev_trajectories: Trajectories,
        n_prevs: torch.Tensor,
        conditioning: torch.Tensor | None,
        save_estimator_outputs: bool,
        save_logprobs: bool,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Gather junction states and reconstruct forward suffixes with self.estimator.

        This isolates the PolicyMixin-dependent sampling.
        """
        # Derive junction positions and gather the junction states (one per traj).
        junction_states_tsr = torch.gather(
            prev_trajectories.states.tensor,
            0,
            (n_prevs)
            .view(1, -1, 1)
            .expand(-1, -1, *prev_trajectories.states.state_shape),
        ).squeeze(0)

        # Reconstruct forward suffixes starting from the junction states using the
        # forward policy estimator owned by `self`.
        recon_trajectories = super().sample_trajectories(
            env,
            states=env.states_from_tensor(junction_states_tsr),
            conditioning=conditioning,
            save_estimator_outputs=save_estimator_outputs,
            save_logprobs=save_logprobs,
            **policy_kwargs,
        )

        return recon_trajectories

    @staticmethod
    def _metropolis_hastings_accept(
        prev_log_rewards: torch.Tensor,
        new_log_rewards: torch.Tensor,
        prev_log_pf: torch.Tensor,
        new_log_pf: torch.Tensor,
        prev_log_pb: torch.Tensor,
        new_log_pb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MH acceptance mask for candidate trajectories.

        The acceptance probability is
        min{1, R(x') p_B(x->s') p_F(s'->x') / [R(x) p_B(x'->s') p_F(s'->x)]}.
        """
        log_accept_ratio = torch.clamp_max(
            new_log_rewards
            + prev_log_pb.sum(0)
            + new_log_pf.sum(0)
            - prev_log_rewards
            - new_log_pb.sum(0)
            - prev_log_pf.sum(0),
            0.0,
        )
        return torch.rand(
            new_log_rewards.shape[0], device=log_accept_ratio.device
        ) < torch.exp(log_accept_ratio)

    @staticmethod
    def _splice_pf(
        n_prevs: torch.Tensor,
        prev_log_pf: torch.Tensor,
        n_recons: torch.Tensor,
        recon_log_pf: torch.Tensor,
        T_new: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Splice per-step PF log-probs of prefix and reconstructed suffix.

        Args:
            n_prevs: Number of prefix steps kept from prev trajectories (N,).
            prev_log_pf: Per-step PF for prev trajectories (T_prev, N).
            n_recons: Number of reconstruction steps per trajectory (N,).
            recon_log_pf: Per-step PF for reconstructed trajectories (T_recon, N).
            T_new: Maximum trajectory length of the spliced trajectories.
            device: Torch device for the output tensor.

        Returns:
            Spliced per-step PF log-probs of shape (T_new, N).
        """
        bs = int(n_prevs.shape[0])

        # Determine maxima for mask construction
        max_n_prev = n_prevs.max()
        max_n_recon = n_recons.max()

        # Build masks over states time (T_new + 1), then adapt to per-step PF (T_new)
        idx_states = (
            torch.arange(T_new + 1, device=n_prevs.device).unsqueeze(1).expand(-1, bs)
        )
        prev_mask = (idx_states < n_prevs).transpose(0, 1)  # (bs, T_new+1)
        action_recon_mask = (
            (idx_states[:-1] >= n_prevs) & (idx_states[:-1] <= (n_prevs + n_recons - 1))
        ).transpose(
            0, 1
        )  # (bs, T_new)
        action_recon_mask2 = (idx_states[:max_n_recon] <= (n_recons - 1)).transpose(
            0, 1
        )  # (bs, max_n_recon)

        # Transpose PF tensors to (bs, time)
        prev_pf_t = prev_log_pf.transpose(0, 1)
        recon_pf_t = recon_log_pf.transpose(0, 1)

        # Allocate and fill spliced PF
        new_pf_t = torch.full((bs, T_new), 0.0).to(device=device, dtype=prev_pf_t.dtype)
        prev_mask_trunc = prev_mask[:, :max_n_prev]
        new_pf_t[prev_mask[:, :-1]] = prev_pf_t[:, :max_n_prev][prev_mask_trunc]
        new_pf_t[action_recon_mask] = recon_pf_t[action_recon_mask2]

        return new_pf_t.transpose(0, 1)

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
        2. Reconstructing the trajectory from the junction state using the forward
           policy estimator.
        3. Optionally applying Metropolis-Hastings acceptance criterion to decide
           whether to accept the new trajectory.

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
        # High-level outline:
        # 1) Choose the backtrack length K per-trajectory (either from `back_steps` or
        #    via a ratio of current lengths). This determines how much prefix to keep.
        # 2) Sample backward trajectories from terminal states using the backward
        #    policy estimator, then reverse them into forward-time to obtain the prefix
        #    trajectories.
        # 3) Extract the junction states at step `n_prevs = L - K - 1` and reconstruct
        #    forward suffixes using the forward policy starting from those junctions.
        # 4) Optionally compute PF/PB per-step log-probabilities for MH acceptance.
        # 5) Splice prefix and suffix into candidate new trajectories.
        # 6) Accept/reject (MH or greedy by reward), return the candidates and update
        #    mask.

        # TODO: Implement local search for GraphStates.
        # Guard against graph-based states; not yet supported.
        if issubclass(env.States, GraphStates):
            raise NotImplementedError("Local search is not implemented for GraphStates.")

        # Ensure PF/PB log-probabilities are computed when MH acceptance is requested.
        save_logprobs = save_logprobs or use_metropolis_hastings

        # 1) K-step backward sampling with the backward estimator, where K is the
        # number of backward steps. When specified via `back_ratio`, K is proportional
        # to the previous trajectory length; otherwise clamp the provided `back_steps`
        # to valid bounds. This is used in https://arxiv.org/abs/2202.01361.

        # Compute per-trajectory backtrack length K.
        K = self._compute_back_steps(
            trajectories.terminating_idx, back_steps, back_ratio
        )

        # 1) Backward sampling from terminal states (PolicyMixin-driven Sampler).
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
        # Convert backward trajectories to forward-time ordering (s0 -> ... -> sf).
        prev_trajectories = prev_trajectories.reverse_backward_trajectories()
        assert prev_trajectories.log_rewards is not None

        # Reconstruct suffixes from junction states using self.estimator.
        n_prevs = prev_trajectories.terminating_idx - K - 1
        recon_trajectories = self._reconstruct_from_junctions(
            env,
            prev_trajectories,
            n_prevs,
            conditioning,
            save_estimator_outputs,
            save_logprobs,
            **policy_kwargs,
        )

        # Calculate the log probabilities as needed.
        # 4) PF on prefix and reconstructed suffix (needed for MH or for logging).
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
        # 5) PB on prefix and reconstructed suffix (needed only for MH acceptance).
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

        # 6) Splice prefix and suffix into candidate trajectories.
        new_trajectories = prev_trajectories.splice_from_reconstruction(
            n_prevs=n_prevs,
            recon=recon_trajectories,
            debug=debug,
        )

        # Build PF by splicing prev/recon PF log-probs.
        if save_logprobs:
            assert prev_trajectories_log_pf is not None
            assert recon_trajectories_log_pf is not None
            T_new = int(new_trajectories.max_length)
            n_recons = recon_trajectories.terminating_idx
            new_trajectories.log_probs = self._splice_pf(
                n_prevs=n_prevs,
                prev_log_pf=prev_trajectories_log_pf,
                n_recons=n_recons,
                recon_log_pf=recon_trajectories_log_pf,
                T_new=T_new,
                device=new_trajectories.states.device,
            )

        # Compute PF/PB sums for MH without building full per-step spliced tensors.
        if use_metropolis_hastings:
            assert prev_trajectories_log_pb is not None
            assert prev_trajectories_log_pf is not None
            assert recon_trajectories_log_pb is not None
            assert recon_trajectories_log_pf is not None

            # Sum over prefix/suffix per trajectory using n_prevs and recon lengths.
            # Prefix sums: [0:n_prev)
            sum_prev_pf = prev_trajectories_log_pf.cumsum(0)
            sum_prev_pb = prev_trajectories_log_pb.cumsum(0)
            prefix_idx = (n_prevs - 1).clamp_min(0).view(1, -1)
            prefix_pf = sum_prev_pf.gather(0, prefix_idx).squeeze(0)
            prefix_pb = sum_prev_pb.gather(0, prefix_idx).squeeze(0)
            zero_prefix = torch.zeros_like(prefix_pf)
            prefix_pf = torch.where(n_prevs > 0, prefix_pf, zero_prefix)
            prefix_pb = torch.where(n_prevs > 0, prefix_pb, zero_prefix)

            # Suffix sums from recon: [0:n_recon)
            n_recons = recon_trajectories.terminating_idx
            sum_recon_pf = recon_trajectories_log_pf.cumsum(0)
            sum_recon_pb = recon_trajectories_log_pb.cumsum(0)
            suffix_idx = (n_recons - 1).clamp_min(0).view(1, -1)
            suffix_pf = sum_recon_pf.gather(0, suffix_idx).squeeze(0)
            suffix_pb = sum_recon_pb.gather(0, suffix_idx).squeeze(0)
            zero_suffix = torch.zeros_like(suffix_pf)
            suffix_pf = torch.where(n_recons > 0, suffix_pf, zero_suffix)
            suffix_pb = torch.where(n_recons > 0, suffix_pb, zero_suffix)

        # 7) Accept/reject. With MH, accept with probability:
        #    min\{1, R(x') p_B(x->s') p_F(s'->x') / [R(x) p_B(x'->s') p_F(s'->x)]\}.
        #    Without MH, accept when the episodic reward improves (ties accepted).
        if use_metropolis_hastings:
            assert prev_trajectories_log_pb is not None
            assert prev_trajectories_log_pf is not None
            assert recon_trajectories_log_pb is not None
            assert recon_trajectories_log_pf is not None
            assert prev_trajectories.log_rewards is not None
            assert new_trajectories.log_rewards is not None

            # The acceptance ratio is: min(1, R(x')p(x->s'->x') / R(x)p(x'->s'-> x))
            # Also, note this:
            # p(x->s'->x') / p(x'->s'-> x))
            # = p_B(x->s')p_F(s'->x') / p_B(x'->s')p_F(s'->x)
            # = p_B(x->s'->s0)p_F(s0->s'->x') / p_B(x'->s'->s0)p_F(s0->s'->x)
            # = p_B(tau|x)p_F(tau') / p_B(tau'|x')p_F(tau)
            # Combine episodic reward and log-prob sums, clamp at 0 (min with 1 in prob
            # space).
            prev_total_pf = prev_trajectories_log_pf.sum(0)
            prev_total_pb = prev_trajectories_log_pb.sum(0)
            assert isinstance(prefix_pf, torch.Tensor)
            assert isinstance(prefix_pb, torch.Tensor)
            new_total_pf = prefix_pf + suffix_pf
            new_total_pb = prefix_pb + suffix_pb
            log_accept_ratio = torch.clamp_max(
                new_trajectories.log_rewards
                + prev_total_pb
                + new_total_pf
                - prev_trajectories.log_rewards
                - new_total_pb
                - prev_total_pf,
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
        # Roll out an initial batch with the forward policy, then perform
        # `n_local_search_loops` rounds of refinement. Each round appends
        # one candidate per original seed trajectory to the container.
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

        # Indices referring to the current seed trajectories within the container.
        # Initially these are the first `n` entries (the initial batch).
        search_indices = torch.arange(n, device=trajectories.states.device)

        for it in range(n_local_search_loops):
            # Run a single local-search refinement on the current seeds.
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
            # Append refined candidates; they occupy a new contiguous block at the end.
            trajectories.extend(ls_trajectories)

            # Map accepted seeds to the indices of the just-appended block so that
            # the next round uses the latest accepted candidates as seeds.
            last_indices = torch.arange(
                n * it, n * (it + 1), device=trajectories.states.device
            )
            search_indices[is_updated] = last_indices[is_updated]

        return trajectories
