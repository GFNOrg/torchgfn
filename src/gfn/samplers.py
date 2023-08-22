from typing import List, Optional, Tuple

import torch
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.modules import GFNModule
from gfn.states import States


class Sampler:
    """`Sampler is a container for a PolicyEstimator.

    Can be used to either sample individual actions, sample trajectories from $s_0$,
    or complete a batch of partially-completed trajectories from a given batch states.

    Attributes:
        estimator: the submitted PolicyEstimator.
        probability_distribution_kwargs: keyword arguments to be passed to the `to_probability_distribution`
             method of the estimator. For example, for DiscretePolicyEstimators, the kwargs can contain
             the `temperature` parameter, `epsilon`, and `sf_bias`.
    """

    def __init__(
        self,
        estimator: GFNModule,
        **probability_distribution_kwargs: Optional[dict],
    ) -> None:
        self.estimator = estimator
        self.probability_distribution_kwargs = probability_distribution_kwargs

    def sample_actions(
        self, env: Env, states: States
    ) -> Tuple[Actions, TT["batch_shape", torch.float]]:
        """Samples actions from the given states.

        Args:
            env: The environment to sample actions from.
            states (States): A batch of states.

        Returns:
            A tuple of tensors containing:
             - An Actions object containing the sampled actions.
             - A tensor of shape (*batch_shape,) containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
        """
        module_output = self.estimator(states)
        dist = self.estimator.to_probability_distribution(
            states, module_output, **self.probability_distribution_kwargs
        )

        with torch.no_grad():
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if torch.any(torch.isinf(log_probs)):
            raise RuntimeError("Log probabilities are inf. This should not happen.")

        return env.Actions(actions), log_probs

    def sample_trajectories(
        self,
        env: Env,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
    ) -> Trajectories:
        """Sample trajectories sequentially.

        Args:
            env: The environment to sample trajectories from.
            states: If given, trajectories would start from such states. Otherwise,
                trajectories are sampled from $s_o$ and n_trajectories must be provided.
            n_trajectories: If given, a batch of n_trajectories will be sampled all
                starting from the environment's s_0.

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

        trajectories_states: List[TT["n_trajectories", "state_shape", torch.float]] = [
            states.tensor
        ]
        trajectories_actions: List[TT["n_trajectories", torch.long]] = []
        trajectories_logprobs: List[TT["n_trajectories", torch.float]] = []
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )

        step = 0

        while not all(dones):
            actions = env.Actions.make_dummy_actions(batch_shape=(n_trajectories,))
            log_probs = torch.full(
                (n_trajectories,), fill_value=0, dtype=torch.float, device=device
            )
            valid_actions, actions_log_probs = self.sample_actions(env, states[~dones])
            actions[~dones] = valid_actions
            log_probs[~dones] = actions_log_probs
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]

            if self.estimator.is_backward:
                new_states = env.backward_step(states, actions)
            else:
                new_states = env.step(states, actions)
            sink_states_mask = new_states.is_sink_state

            step += 1

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

            trajectories_states += [states.tensor]

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = env.States(tensor=trajectories_states)
        trajectories_actions = env.Actions.stack(trajectories_actions)
        trajectories_logprobs = torch.stack(trajectories_logprobs, dim=0)

        trajectories = Trajectories(
            env=env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.estimator.is_backward,
            log_rewards=trajectories_log_rewards,
            log_probs=trajectories_logprobs,
        )

        return trajectories
