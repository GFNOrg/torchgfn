from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.modules import GFNModule
from gfn.states import States, stack_states


class Sampler:
    """`Sampler is a container for a PolicyEstimator.

    Can be used to either sample individual actions, sample trajectories from $s_0$,
    or complete a batch of partially-completed trajectories from a given batch states.

    Attributes:
        estimator: the submitted PolicyEstimator.
    """

    def __init__(
        self,
        estimator: GFNModule,
    ) -> None:
        self.estimator = estimator

    def sample_actions(
        self,
        env: Env,
        states: States,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = True,
        **policy_kwargs: Optional[dict],
    ) -> Tuple[
        Actions,
        TT["batch_shape", torch.float] | None,
        TT["batch_shape", torch.float] | None,
    ]:
        """Samples actions from the given states.

        Args:
            estimator: A GFNModule to pass to the probability distribution calculator.
            env: The environment to sample actions from.
            states: A batch of states.
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
             - A tensor of shape (*batch_shape,) containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
        """
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

        return actions, log_probs, estimator_output

    def sample_trajectories(
        self,
        env: Env,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = True,
        **policy_kwargs,
    ) -> Trajectories:
        """Sample trajectories sequentially.

        Args:
            env: The environment to sample trajectories from.
            states: If given, trajectories would start from such states. Otherwise,
                trajectories are sampled from $s_o$ and n_trajectories must be provided.
            n_trajectories: If given, a batch of n_trajectories will be sampled all
                starting from the environment's s_0.
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
            assert (
                n_trajectories is not None
            ), "Either states or n_trajectories should be specified"
            states = env.reset(batch_shape=(n_trajectories,))
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]

        device = states.tensor.device

        dones = (
            states.is_initial_state
            if self.estimator.is_backward
            else states.is_sink_state
        )

        trajectories_states: List[States] = [deepcopy(states)]
        trajectories_actions: List[TT["n_trajectories", torch.long]] = []
        trajectories_logprobs: List[TT["n_trajectories", torch.float]] = []
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )

        step = 0
        all_estimator_outputs = []

        while not all(dones):
            actions = env.actions_from_batch_shape((n_trajectories,))  # Dummy actions.
            log_probs = torch.full(
                (n_trajectories,), fill_value=0, dtype=torch.float, device=device
            )
            # This optionally allows you to retrieve the estimator_outputs collected
            # during sampling. This is useful if, for example, you want to evaluate off
            # policy actions later without repeating calculations to obtain the env
            # distribution parameters.
            valid_actions, actions_log_probs, estimator_outputs = self.sample_actions(
                env,
                states[~dones],
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
            if save_logprobs:
                # When off_policy, actions_log_probs are None.
                log_probs[~dones] = actions_log_probs
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]

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

            trajectories_states += [deepcopy(states)]

        trajectories_states = stack_states(trajectories_states)
        trajectories_actions = env.Actions.stack(trajectories_actions)
        trajectories_logprobs = (
            torch.stack(trajectories_logprobs, dim=0) if save_logprobs else None
        )

        # TODO: use torch.nested.nested_tensor(dtype, device, requires_grad).
        if save_estimator_outputs:
            all_estimator_outputs = torch.stack(all_estimator_outputs, dim=0)

        trajectories = Trajectories(
            env=env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.estimator.is_backward,
            log_rewards=trajectories_log_rewards,
            log_probs=trajectories_logprobs,
            estimator_outputs=all_estimator_outputs if save_estimator_outputs else None,
        )

        return trajectories
