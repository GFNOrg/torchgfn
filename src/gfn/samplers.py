from copy import deepcopy
from typing import Any, List, Optional, Tuple

import torch

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.modules import GFNModule
from gfn.states import States
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)
from gfn.utils.prob_calculations import get_trajectory_pbs, get_trajectory_pfs


class Sampler:
    """`Sampler is a container for a PolicyEstimator.

    Can be used to either sample individual actions, sample trajectories from $s_0$,
    or complete a batch of partially-completed trajectories from a given batch states.

    Attributes:
        estimator: the submitted PolicyEstimator.
    """

    def __init__(self, estimator: GFNModule) -> None:
        self.estimator = estimator

    def sample_actions(
        self,
        env: Env,
        states: States,
        conditioning: torch.Tensor | None = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = False,
        **policy_kwargs: Any,
    ) -> Tuple[Actions, torch.Tensor | None, torch.Tensor | None]:
        """Samples actions from the given states.

        Args:
            env: The environment to sample actions from.
            states: A batch of states.
            conditioning: An optional tensor of conditioning information.
            save_estimator_outputs: If True, the estimator outputs will be returned.
            save_logprobs: If True, calculates and saves the log probabilities of sampled
                actions.
            policy_kwargs: keyword arguments to be passed to the
                `to_probability_distribution` method of the estimator. For example, for
                DiscretePolicyEstimators, the kwargs can contain the `temperature`
                parameter, `epsilon`, and `sf_bias`. In the continuous case these
                kwargs will be user defined. This can be used to, for example, sample
                off-policy.

        When sampling off policy, ensure to `save_estimator_outputs` and not
            `calculate logprobs`. Log probabilities are instead calculated during the
            computation of `PF` as part of the `GFlowNet` class, and the estimator
            outputs are required for estimating the logprobs of these off policy
            actions.

        Returns:
            A tuple of tensors containing:
             - An Actions object containing the sampled actions.
             - An optional tensor of shape `batch_shape` containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
             - An optional tensor of shape `batch_shape` containing the estimator outputs
        """
        # TODO: Should estimators instead ignore None for the conditioning vector?
        if conditioning is not None:
            with has_conditioning_exception_handler("estimator", self.estimator):
                estimator_output = self.estimator(states, conditioning)
        else:
            with no_conditioning_exception_handler("estimator", self.estimator):
                estimator_output = self.estimator(states)

        dist = self.estimator.to_probability_distribution(
            states, estimator_output, **policy_kwargs
        )

        with torch.no_grad():
            actions = dist.sample()

        if save_logprobs:
            log_probs = dist.log_prob(actions)
            if torch.any(torch.isinf(log_probs)):
                raise RuntimeError("Log probabilities are inf. This should not happen.")
        else:
            log_probs = None

        actions = env.actions_from_tensor(actions)
        if not save_estimator_outputs:
            estimator_output = None

        assert log_probs is None or log_probs.shape == actions.batch_shape
        # assert estimator_output is None or estimator_output.shape == actions.batch_shape  TODO: check expected shape
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
        """Sample trajectories sequentially.

        Args:
            env: The environment to sample trajectories from.
            n: If given, a batch of n_trajectories will be sampled all
                starting from the environment's s_0.
            states: If given, trajectories would start from such states. Otherwise,
                trajectories are sampled from $s_o$ and n_trajectories must be provided.
            conditioning: An optional tensor of conditioning information.
            save_estimator_outputs: If True, the estimator outputs will be returned. This
                is useful for off-policy training with tempered policy.
            save_logprobs: If True, calculates and saves the log probabilities of sampled
                actions. This is useful for on-policy training.
            policy_kwargs: keyword arguments to be passed to the
                `to_probability_distribution` method of the estimator. For example, for
                DiscretePolicyEstimators, the kwargs can contain the `temperature`
                parameter, `epsilon`, and `sf_bias`. In the continuous case these
                kwargs will be user defined. This can be used to, for example, sample
                off-policy.

        Returns: A Trajectories object representing the batch of sampled trajectories.

        Raises:
            AssertionError: When both states and n_trajectories are specified.
            AssertionError: When states are not linear.
        """

        if states is None:
            assert n is not None, "Either kwarg `states` or `n` must be specified"
            states = env.reset(batch_shape=(n,))
            n_trajectories = n
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should have len(states.batch_shape) == 1, w/ no trajectory dim!"
            n_trajectories = states.batch_shape[0]

        if conditioning is not None:
            assert states.batch_shape == conditioning.shape[: len(states.batch_shape)]

        device = states.tensor.device

        dones = (
            states.is_initial_state
            if self.estimator.is_backward
            else states.is_sink_state
        )

        # Define dummy actions to avoid errors when stacking empty lists.
        dummy_actions = env.actions_from_batch_shape((n_trajectories,))
        dummy_logprobs = torch.full(
            (n_trajectories,), fill_value=0, dtype=torch.float, device=device
        )

        trajectories_states: List[States] = [deepcopy(states)]
        trajectories_actions: List[Actions] = [dummy_actions]
        trajectories_logprobs: List[torch.Tensor] = [dummy_logprobs]
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )

        step = 0
        all_estimator_outputs = []

        while not all(dones):
            actions = deepcopy(dummy_actions)
            log_probs = dummy_logprobs.clone()
            # This optionally allows you to retrieve the estimator_outputs collected
            # during sampling. This is useful if, for example, you want to evaluate off
            # policy actions later without repeating calculations to obtain the env
            # distribution parameters.
            if conditioning is not None:
                masked_conditioning = conditioning[~dones]
            else:
                masked_conditioning = None

            valid_actions, actions_log_probs, estimator_outputs = self.sample_actions(
                env,
                states[~dones],
                masked_conditioning,
                save_estimator_outputs=True if save_estimator_outputs else False,
                save_logprobs=save_logprobs,
                **policy_kwargs,
            )
            if estimator_outputs is not None:
                # Place estimator outputs into a stackable tensor. Note that this
                # will be replaced with torch.nested.nested_tensor in the future.
                estimator_outputs_padded = torch.full(
                    (n_trajectories,) + estimator_outputs.shape[1:],
                    fill_value=-float("inf"),
                    dtype=torch.float,
                    device=device,
                )
                estimator_outputs_padded[~dones] = estimator_outputs
                all_estimator_outputs.append(estimator_outputs_padded)

            actions[~dones] = valid_actions
            trajectories_actions.append(actions)
            if save_logprobs:
                # When off_policy, actions_log_probs are None.
                log_probs[~dones] = actions_log_probs
                trajectories_logprobs.append(log_probs)

            if self.estimator.is_backward:
                new_states = env._backward_step(states, actions)
            else:
                new_states = env._step(states, actions)
            sink_states_mask = new_states.is_sink_state

            # Increment the step, determine which trajectories are finisihed, and eval
            # rewards.
            step += 1

            # new_dones means those trajectories that just finished. Because we
            # pad the sink state to every short trajectory, we need to make sure
            # to filter out the already done ones.
            new_dones = (
                new_states.is_initial_state
                if self.estimator.is_backward
                else sink_states_mask
            ) & ~dones
            trajectories_dones[new_dones & ~dones] = step
            try:
                trajectories_log_rewards[new_dones & ~dones] = env.log_reward(
                    states[new_dones & ~dones]
                )
            except NotImplementedError:
                trajectories_log_rewards[new_dones & ~dones] = torch.log(
                    env.reward(states[new_dones & ~dones])
                )
            states = new_states
            dones = dones | new_dones

            trajectories_states.append(deepcopy(states))
        # TODO: do not ignore the next three ignores
        trajectories_states = states.stack_states(trajectories_states)  # pyright: ignore
        trajectories_actions = env.Actions.stack(trajectories_actions)[
            1:  # Drop dummy action
        ]  # pyright: ignore
        trajectories_logprobs = (
            torch.stack(trajectories_logprobs, dim=0)[1:]  # Drop dummy logprob
            if save_logprobs
            else None
        )  # pyright: ignore

        # TODO: use torch.nested.nested_tensor(dtype, device, requires_grad).
        if save_estimator_outputs:
            all_estimator_outputs = torch.stack(all_estimator_outputs, dim=0)
        # TODO: do not ignore the next ignores
        trajectories = Trajectories(
            env=env,
            states=trajectories_states,  # pyright: ignore
            conditioning=conditioning,
            actions=trajectories_actions,  # pyright: ignore
            when_is_done=trajectories_dones,
            is_backward=self.estimator.is_backward,
            log_rewards=trajectories_log_rewards,
            log_probs=trajectories_logprobs,  # pyright: ignore
            estimator_outputs=(
                all_estimator_outputs if save_estimator_outputs else None
            ),  # pyright: ignore
        )

        return trajectories


class LocalSearchSampler(Sampler):
    """Sampler equipped with local search capabilities.
    The local search operation is based on back-and-forth heuristic, first proposed
    by Zhang et al. 2022 (https://arxiv.org/abs/2202.01361) for negative sampling
    and further explored its effectiveness in various applications by Kim et al. 2023
    (https://arxiv.org/abs/2310.02710).

    Attributes:
        pf_estimator: the submitted PolicyEstimator for the forward pass.
        pb_estimator: the PolicyEstimator for the backward pass.
    """

    def __init__(self, pf_estimator: GFNModule, pb_estimator: GFNModule):
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

        Args:
            env: The environment to sample trajectories from.
            trajectories: The batch of trajectories to perform local search on.
            conditioning: An optional tensor of conditioning information.
            save_estimator_outputs: If True, the estimator outputs will be returned. This
                is useful for off-policy training with tempered policy.
            save_logprobs: If True, calculates and saves the log probabilities of sampled
                actions. This is useful for on-policy training.
            back_steps: The number of backward steps.
            back_ratio: The ratio of the number of backward steps to the length of the trajectory.
            use_metropolis_hastings: If True, applies Metropolis-Hastings acceptance criterion.
            policy_kwargs: keyword arguments to be passed to the
                `to_probability_distribution` method of the estimator. For example, for
                DiscretePolicyEstimators, the kwargs can contain the `temperature`
                parameter, `epsilon`, and `sf_bias`. In the continuous case these
                kwargs will be user defined. This can be used to, for example, sample
                off-policy.

        Returns:
            A tuple of Trajectories object and a boolean tensor indicating whether the
            trajectory was updated.
        """
        save_logprobs = save_logprobs or use_metropolis_hastings

        # K-step backward sampling with the backward estimator,
        # where K is the number of backward steps used in https://arxiv.org/abs/2202.01361.
        if back_steps is None:
            assert (
                back_ratio is not None and 0 < back_ratio <= 1
            ), "Either kwarg `back_steps` or `back_ratio` must be specified"
            K = torch.ceil(back_ratio * (trajectories.when_is_done - 1)).long()
        else:
            K = torch.where(
                back_steps > trajectories.when_is_done,
                trajectories.when_is_done,
                back_steps,
            )

        prev_trajectories = self.backward_sampler.sample_trajectories(
            env,
            states=trajectories.last_states,
            conditioning=conditioning,
            save_estimator_outputs=save_estimator_outputs,
            save_logprobs=save_logprobs,
            **policy_kwargs,
        )

        # By reversing the backward trajectories, obtain the forward trajectories.
        # This is called `prev_trajectories` since they are the trajectories before
        # the local search. The `new_trajectories` will be obtained by performing local
        # search on them.
        prev_trajectories = Trajectories.reverse_backward_trajectories(
            prev_trajectories
        )
        assert prev_trajectories.log_rewards is not None

        ### Reconstructing with self.estimator
        n_prevs = prev_trajectories.when_is_done - K - 1
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
            is_updated = prev_trajectories.log_rewards <= new_trajectories.log_rewards

        return new_trajectories, is_updated

    def sample_trajectories(
        self,
        env: Env,
        n: Optional[int] = None,
        states: Optional[States] = None,
        conditioning: Optional[torch.Tensor] = None,
        save_estimator_outputs: bool = False,  # FIXME: currently not work when this is True
        save_logprobs: bool = False,  # TODO: Support save_logprobs=True
        n_local_search_loops: int = 0,
        back_steps: torch.Tensor | None = None,
        back_ratio: float | None = None,
        use_metropolis_hastings: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Sample trajectories sequentially with optional local search.

        Args:
            env: The environment to sample trajectories from.
            n: If given, a batch of n_trajectories will be sampled all
                starting from the environment's s_0.
            states: If given, trajectories would start from such states. Otherwise,
                trajectories are sampled from $s_o$ and n_trajectories must be provided.
            conditioning: An optional tensor of conditioning information.
            save_estimator_outputs: If True, the estimator outputs will be returned. This
                is useful for off-policy training with tempered policy.
            save_logprobs: If True, calculates and saves the log probabilities of sampled
                actions. This is useful for on-policy training.
            n_local_search_loops: The number of local search loops.
            back_steps: The number of backward steps.
            back_ratio: The ratio of the number of backward steps to the length of the trajectory.
            use_metropolis_hastings: If True, applies Metropolis-Hastings acceptance criterion.
            policy_kwargs: keyword arguments to be passed to the
                `to_probability_distribution` method of the estimator. For example, for
                DiscretePolicyEstimators, the kwargs can contain the `temperature`
                parameter, `epsilon`, and `sf_bias`. In the continuous case these
                kwargs will be user defined. This can be used to, for example, sample
                off-policy.

        Returns: A Trajectories object representing the batch of sampled trajectories,
            where the batch size is n * (1 + n_local_search_loops).
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
            n = trajectories.n_trajectories

        search_indices = torch.arange(n, device=trajectories.states.device)
        for it in range(1, n_local_search_loops):  # 0-th loop is the initial sampling
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
        """
        Combine `prev_trajectories` and `recon_trajectories` to obtain `new_trajectories`.
        Specifically, `new_trajectories` is constructed by replacing certain portion of
        the `prev_trajectories` with `recon_trajectories`. See self.local_search for how
        to generate `prev_trajectories` and `recon_trajectories`.
        """

        save_logprobs = (prev_trajectories_log_pf is not None) and (
            recon_trajectories_log_pf is not None
        )

        new_trajectories_log_pf = None
        new_trajectories_log_pb = None

        bs = prev_trajectories.n_trajectories
        device = prev_trajectories.states.device
        env = prev_trajectories.env

        # Obtain full trajectories by concatenating the backward and forward parts.
        max_n_prev = n_prevs.max()
        n_recons = recon_trajectories.when_is_done
        max_n_recon = n_recons.max()

        new_trajectories_log_rewards = recon_trajectories.log_rewards  # Episodic reward
        new_trajectories_dones = n_prevs + n_recons
        max_traj_len = new_trajectories_dones.max()

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
        new_trajectories_actions_tsr[action_recon_mask] = (
            recon_trajectories_actions_tsr[action_recon_mask2]
        )

        # Transpose back
        new_trajectories_states_tsr = new_trajectories_states_tsr.transpose(0, 1)
        new_trajectories_actions_tsr = new_trajectories_actions_tsr.transpose(0, 1)

        # Similarly, combine log_pf and log_pb if needed
        if save_logprobs:
            prev_trajectories_log_pf = (
                prev_trajectories_log_pf.transpose(0, 1)
                if prev_trajectories_log_pf is not None
                else None
            )
            recon_trajectories_log_pf = (
                recon_trajectories_log_pf.transpose(0, 1)
                if recon_trajectories_log_pf is not None
                else None
            )
            new_trajectories_log_pf = torch.full((bs, max_traj_len), 0.0).to(
                device=device, dtype=torch.float
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
        if (prev_trajectories_log_pb is not None) and (
            recon_trajectories_log_pb is not None
        ):
            prev_trajectories_log_pb = prev_trajectories_log_pb.transpose(0, 1)
            recon_trajectories_log_pb = recon_trajectories_log_pb.transpose(0, 1)
            new_trajectories_log_pb = torch.full((bs, max_traj_len), 0.0).to(
                device=device, dtype=torch.float
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

            if save_logprobs:
                prev_trajectories_log_pf = prev_trajectories_log_pf.transpose(0, 1)  # type: ignore
                recon_trajectories_log_pf = recon_trajectories_log_pf.transpose(0, 1)  # type: ignore
                _new_trajectories_log_pf = torch.full((max_traj_len, bs), 0.0).to(
                    device=device, dtype=torch.float
                )
            if use_metropolis_hastings := (prev_trajectories_log_pb is not None) and (
                recon_trajectories_log_pb is not None
            ):
                prev_trajectories_log_pb = prev_trajectories_log_pb.transpose(0, 1)  # type: ignore
                recon_trajectories_log_pb = recon_trajectories_log_pb.transpose(0, 1)  # type: ignore
                _new_trajectories_log_pb = torch.full((max_traj_len, bs), 0.0).to(
                    device=device, dtype=torch.float
                )

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
                _len_recon = recon_trajectories.when_is_done[i]
                _new_trajectories_states_tsr[
                    _n_prev + 1 : _n_prev + _len_recon + 1, i
                ] = recon_trajectories.states.tensor[1 : _len_recon + 1, i]
                _new_trajectories_actions_tsr[_n_prev : _n_prev + _len_recon, i] = (
                    recon_trajectories.actions.tensor[:_len_recon, i]
                )

                if save_logprobs:
                    if prev_trajectories_log_pf is not None:  # Add this check
                        _new_trajectories_log_pf[:_n_prev, i] = (
                            prev_trajectories_log_pf[:_n_prev, i]
                        )
                    if recon_trajectories_log_pf is not None:  # Add this check
                        _new_trajectories_log_pf[_n_prev : _n_prev + _len_recon, i] = (
                            recon_trajectories_log_pf[:_len_recon, i]
                        )
                if use_metropolis_hastings:
                    if prev_trajectories_log_pb is not None:  # Add this check
                        _new_trajectories_log_pb[:_n_prev, i] = (
                            prev_trajectories_log_pb[:_n_prev, i]
                        )
                    if recon_trajectories_log_pb is not None:  # Add this check
                        _new_trajectories_log_pb[_n_prev : _n_prev + _len_recon, i] = (
                            recon_trajectories_log_pb[:_len_recon, i]
                        )

            assert torch.all(
                _new_trajectories_states_tsr == new_trajectories_states_tsr
            )
            assert torch.all(
                _new_trajectories_actions_tsr == new_trajectories_actions_tsr
            )
            if save_logprobs:
                assert torch.all(_new_trajectories_log_pf == new_trajectories_log_pf)  # type: ignore
            if use_metropolis_hastings:
                assert torch.all(_new_trajectories_log_pb == new_trajectories_log_pb)  # type: ignore

        new_trajectories = Trajectories(
            env=env,
            states=env.states_from_tensor(new_trajectories_states_tsr),
            conditioning=prev_trajectories.conditioning,
            actions=env.actions_from_tensor(new_trajectories_actions_tsr),
            when_is_done=new_trajectories_dones,
            is_backward=False,
            log_rewards=new_trajectories_log_rewards,
            log_probs=new_trajectories_log_pf if save_logprobs else None,
        )

        return new_trajectories, new_trajectories_log_pf, new_trajectories_log_pb
