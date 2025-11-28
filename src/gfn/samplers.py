import warnings
from typing import Any, Callable, List, Optional, Tuple, cast

import torch

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import Env, EnvFastPathMixin
from gfn.estimators import Estimator, FastPolicyMixin, PolicyEstimatorProtocol
from gfn.states import GraphStates, States
from gfn.utils.common import ensure_same_device
from gfn.utils.graphs import graph_states_share_storage
from gfn.utils.prob_calculations import get_trajectory_pbs, get_trajectory_pfs


def _mark_cudagraph_step() -> None:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(marker):
        marker()


def _fill_like_reference(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    fill = value.to(device=reference.device, dtype=reference.dtype)
    while fill.ndim < reference.ndim:
        fill = fill.unsqueeze(0)
    return fill.expand_as(reference).clone()


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
        conditions: torch.Tensor | None = None,
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
            conditions: Optional condition vector for conditional policies.
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
                conditions=conditions,
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
        conditions: Optional[torch.Tensor] = None,
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
            conditions: Optional condition tensor for conditional environments,
                with shape (n_trajectories, condition_vector_dim); each row is the
                condition vector for each trajectory.
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

        if conditions is not None:
            assert states.batch_shape == conditions.shape[: len(states.batch_shape)]
            ensure_same_device(states.device, conditions.device)

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
        ctx = policy_estimator.init_context(n_trajectories, device, conditions)

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

        # Broadcast condition tensor to match states batch shape if needed
        if conditions is not None:
            # The states have batch shape (max_length, n_trajectories). The
            # conditions tensor should have shape (n_trajectories,) or
            # (n_trajectories, 1). We need to broadcast it to (max_length,
            # n_trajectories, 1) for the estimator
            if len(conditions.shape) == 1:
                # conditions has shape (n_trajectories,)
                conditions = (
                    conditions.unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(stacked_states.batch_shape[0], -1, 1)
                )
            elif len(conditions.shape) == 2 and conditions.shape[1] == 1:
                # conditions has shape (n_trajectories, 1)
                conditions = conditions.unsqueeze(0).expand(
                    stacked_states.batch_shape[0], -1, -1
                )

        trajectories = Trajectories(
            env=env,
            states=stacked_states,
            conditions=conditions,
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

    def local_search(
        self,
        env: Env,
        trajectories: Trajectories,
        conditions: torch.Tensor | None = None,
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
            conditions: Optional tensor of conditions information for conditional
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
            conditions=conditions,
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
            conditions=conditions,
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
        conditions: Optional[torch.Tensor] = None,
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
            conditions: Optional tensor of conditions information for conditional
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
            conditions,
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
                conditions,
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
            conditions=prev_trajectories.conditions,
            actions=env.actions_from_tensor(new_trajectories_actions_tsr),
            terminating_idx=new_trajectories_dones,
            is_backward=False,
            log_rewards=new_trajectories_log_rewards,
            log_probs=new_trajectories_log_pf,
        )

        return new_trajectories, new_trajectories_log_pf, new_trajectories_log_pb


class CompiledChunkSampler(Sampler):
    """Chunked tensor sampler that stays on the fast path for torch.compile."""

    def __init__(
        self,
        estimator: Estimator,
        *,
        chunk_size: int = 32,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        super().__init__(estimator)
        self.chunk_size = int(chunk_size)
        self.compile_mode = compile_mode
        self._compiled_chunk_cache: dict[tuple[int, str], Callable] = {}

    def sample_trajectories(
        self,
        env: Env,
        n: Optional[int] = None,
        states: Optional[States] = None,
        conditions: Optional[torch.Tensor] = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:

        # Log-probs: we’d need to store each chunk’s dist (or sampled actions) plus a
        # boolean mask of which rows were active, then call policy.fast_distribution
        # (...).log_prob(...) during or after the chunk loop. Because done rows get
        # forced to the exit action, we’d have to mask those out when accumulating
        # log-probs so the padded semantics match Trajectories.log_probs. That means
        # keeping per-step tensors shaped (chunk_len, batch, action_dim) and writing
        # them into the context at the end.

        # Estimator outputs: same idea—capture the raw tensor returned by policy.
        # fast_features/fast_distribution (whatever we consider the “estimator output”)
        # for active rows, pad them back to batch size, and append to a list per chunk
        # so we can stack them like the legacy sampler.
        if save_estimator_outputs or save_logprobs:
            raise NotImplementedError(
                "CompiledChunkSampler does not yet record log-probs or estimator outputs."
            )

        if not isinstance(env, EnvFastPathMixin):
            raise TypeError(
                "CompiledChunkSampler requires environments implementing EnvFastPathMixin."
            )

        if not isinstance(self.estimator, FastPolicyMixin):
            raise TypeError(
                "CompiledChunkSampler requires estimators implementing FastPolicyMixin."
            )

        assert self.chunk_size > 0, "chunk_size must be positive"

        policy = cast(FastPolicyMixin, self.estimator)

        if states is None:
            assert n is not None, "Either `n` or `states` must be provided."
            states_obj = env.reset(batch_shape=(n,))
        else:
            states_obj = states
        assert len(states_obj.batch_shape) == 1, "States batch must be 1-D."

        batch = states_obj.batch_shape[0]
        device = states_obj.device

        if conditions is not None:
            assert (
                conditions.shape[0] == batch
            ), "Conditions batch dimension must match states batch size."
            ensure_same_device(device, conditions.device)

        curr_states = states_obj.tensor
        done = states_obj.is_sink_state.clone()
        exit_action_value = env.exit_action.to(device=curr_states.device)
        dummy_action_value = env.dummy_action.to(device=curr_states.device)

        # `step_actions_seq` keeps the raw sampled actions (with exits injected for
        # finished rows) so we can exactly replay the tensor-forward environment when
        # rebuilding the state stack after the chunk loop. `recorded_actions_seq`
        # mirrors those actions but rewrites already-finished rows with the env dummy
        # action so that downstream Trajectories consumers (DBG/SubTB losses) never see
        # transitions originating from sink states.
        recorded_actions_seq: List[torch.Tensor] = []
        step_actions_seq: List[torch.Tensor] = []
        sink_seq: List[torch.Tensor] = []

        chunk_size = int(policy_kwargs.pop("chunk_size", self.chunk_size))

        def _chunk_loop(current_states: torch.Tensor, done_mask: torch.Tensor) -> tuple[
            torch.Tensor,
            torch.Tensor,
            List[torch.Tensor],
            List[torch.Tensor],
            List[torch.Tensor],
            torch.Tensor,
        ]:
            """
            This function is the core of the chunked sampler. It is responsible for
            sampling actions for a chunk of states. It is called in a loop until all
            states are done. It returns the current states, a boolean mask indicating
            which states are done, the actions sampled for the chunk, and a boolean
            mask indicating which states are sinks.

            The purpose of this function is to serve as a torch.compile-ed function to
            speed up the sampling process. It is called in a loop until all states are
            done.

            Args:
                current_states: The current states to sample actions for.
                done_mask: A boolean mask indicating which states are done.

            Returns:
            """
            local_step_actions: List[torch.Tensor] = []
            local_recorded_actions: List[torch.Tensor] = []
            local_sinks: List[torch.Tensor] = []
            step_template: torch.Tensor | None = None
            record_template: torch.Tensor | None = None
            steps_taken = 0

            for _ in range(chunk_size):
                if bool(done_mask.all().item()):
                    assert step_template is not None and record_template is not None
                    pad_step = _fill_like_reference(step_template, exit_action_value)
                    pad_record = _fill_like_reference(record_template, dummy_action_value)
                    local_step_actions.append(pad_step)
                    local_recorded_actions.append(pad_record)
                    local_sinks.append(done_mask.clone())
                    continue

                state_view = current_states
                features = policy.fast_features(
                    state_view,
                    forward_masks=None,
                    backward_masks=None,
                    conditions=conditions,
                )
                dist = policy.fast_distribution(
                    features,
                    forward_masks=None,
                    backward_masks=None,
                    states_tensor=state_view,
                    **policy_kwargs,
                )

                actions_tensor = dist.sample()

                if done_mask.any():
                    # Broadcast the boolean mask and the per-env exit/dummy templates so
                    # they match the estimator's sampled action tensor shape (covers
                    # both scalar Discrete actions and potential multi-dim action
                    # heads).
                    mask = done_mask
                    while mask.ndim < actions_tensor.ndim:
                        mask = mask.unsqueeze(-1)

                    exit_fill = exit_action_value.to(
                        device=actions_tensor.device, dtype=actions_tensor.dtype
                    )
                    while exit_fill.ndim < actions_tensor.ndim:
                        exit_fill = exit_fill.unsqueeze(0)

                    dummy_fill = dummy_action_value.to(
                        device=actions_tensor.device, dtype=actions_tensor.dtype
                    )
                    while dummy_fill.ndim < actions_tensor.ndim:
                        dummy_fill = dummy_fill.unsqueeze(0)

                    step_actions = torch.where(mask, exit_fill, actions_tensor)
                    record_actions = torch.where(mask, dummy_fill, actions_tensor)
                else:
                    step_actions = actions_tensor
                    record_actions = actions_tensor

                # Only the step actions (exit-padded) are used to advance the tensor
                # env. The recorded actions (dummy-padded) are used to reconstruct the
                # state stack after the chunk loop.
                step_res = env.step_tensor(current_states, step_actions)
                current_states = step_res.next_states
                sinks = step_res.is_sink_state
                if sinks is None:
                    sinks = env.states_from_tensor(current_states).is_sink_state

                done_mask = done_mask | sinks
                local_step_actions.append(step_actions)
                local_recorded_actions.append(record_actions)
                step_template = step_actions.detach()
                record_template = record_actions.detach()
                local_sinks.append(sinks)
                steps_taken += 1

            return (
                current_states,
                done_mask,
                local_step_actions,
                local_recorded_actions,
                local_sinks,
                torch.tensor(steps_taken, device=current_states.device),
            )

        chunk_fn: Callable = _chunk_loop
        chunk_fn_compiled = False
        device_type = curr_states.device.type
        compile_allowed = (
            hasattr(torch, "compile") and device_type in ("cuda", "cpu") and conditions is None and not policy_kwargs
        )
        cache_key = (id(env), device_type)
        if compile_allowed:
            cached = self._compiled_chunk_cache.get(cache_key)
            if cached is not None:
                chunk_fn = cached
                chunk_fn_compiled = True
            else:
                try:
                    compiled = torch.compile(_chunk_loop, mode=self.compile_mode)  # type: ignore[arg-type]
                    self._compiled_chunk_cache[cache_key] = compiled
                    chunk_fn = compiled
                    chunk_fn_compiled = True
                except Exception:
                    warnings.warn(
                        "Compilation of chunk_loop failed, using non-compiled version.",
                        stacklevel=2,
                    )
                    chunk_fn = _chunk_loop

        # Main loop: call the compiled function until all states are done.
        while not bool(done.all().item()):
            if chunk_fn_compiled:
                _mark_cudagraph_step()
            (
                curr_states,
                done,
                step_actions_chunk,
                recorded_actions_chunk,
                sinks_chunk,
                steps_taken_tensor,
            ) = chunk_fn(curr_states, done)
            steps_taken = int(steps_taken_tensor.item())
            if steps_taken:
                step_actions_seq.extend(step_actions_chunk[:steps_taken])
                recorded_actions_seq.extend(recorded_actions_chunk[:steps_taken])
                sink_seq.extend(sinks_chunk[:steps_taken])

        if recorded_actions_seq:
            actions_tsr = torch.stack(recorded_actions_seq, dim=0)
            T = actions_tsr.shape[0]

            s = states_obj.tensor
            states_stack = [s]
            for t in range(T):
                # Re-simulate using the true step actions so reconstructed states match
                # the chunk rollout exactly even though padded (recorded) actions may
                # differ.
                step = env.step_tensor(s, step_actions_seq[t])
                s = step.next_states
                states_stack.append(s)
            states_tsr = torch.stack(states_stack, dim=0)

            sinks_tsr = torch.stack(sink_seq, dim=0)
            first_sink = torch.argmax(sinks_tsr.to(torch.long), dim=0)
            never_sink = ~sinks_tsr.any(dim=0)
            first_sink = torch.where(
                never_sink,
                torch.tensor(T - 1, device=device),
                first_sink,
            )
            terminating_idx = first_sink + 1
        else:
            states_tsr = states_obj.tensor.unsqueeze(0)
            actions_tsr = env.actions_from_batch_shape((0, batch)).tensor
            terminating_idx = torch.zeros(batch, dtype=torch.long, device=device)

        # Ensure the stacked (dummy-padded) actions respect the environment's action
        # shape before wrapping them into an Actions container (e.g., discrete envs
        # expect (..., 1)). Without this guard DB/SubTB estimators would fail when the
        # chunk sampler returns rank-1 tensors.
        action_shape = getattr(env, "action_shape", None)
        if action_shape:
            tail_shape = tuple(actions_tsr.shape[-len(action_shape) :])
            if tail_shape != tuple(action_shape):
                if tuple(action_shape) == (1,):
                    actions_tsr = actions_tsr.unsqueeze(-1)
                else:
                    raise ValueError(
                        "CompiledChunkSampler produced actions with shape "
                        f"{actions_tsr.shape}, expected trailing dims {action_shape}."
                    )

        trajectories = Trajectories(
            env=env,
            states=env.states_from_tensor(states_tsr),
            conditions=conditions,
            actions=env.actions_from_tensor(actions_tsr),
            terminating_idx=terminating_idx,
            is_backward=policy.is_backward,
            log_rewards=None,
            log_probs=None,
            estimator_outputs=None,
        )
        return trajectories
