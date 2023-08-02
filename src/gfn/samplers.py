from typing import List, Optional, Tuple

import torch
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.modules import GFNModule
from gfn.states import States


# TODO: Environment should not live inside the estimator and here... needs refactor.
class Sampler:
    """`Sampler is a container for a PolicyEstimator.

    Can be used to either sample individual actions, sample trajectories from $s_0$,
    or complete a batch of partially-completed trajectories from a given batch states.

    Attributes:
        estimator: the submitted PolicyEstimator.
        env: the Environment instance inside the PolicyEstimator.
        is_backward: if True, samples trajectories of actions backward (a distribution
            over parents). If True, the estimator must be a ProbabilityDistribution
            over parents.
    """

    def __init__(self, estimator: GFNModule, is_backward: bool = False) -> None:
        self.estimator = estimator
        self.env = estimator.env
        self.is_backward = is_backward  # TODO: take directly from estimator.

    def sample_actions(
        self, states: States
    ) -> Tuple[Actions, TT["batch_shape", torch.float]]:
        """Samples actions from the given states.

        Args:
            states (States): A batch of states.

        Returns:
            A tuple of tensors containing:
             - An Actions object containing the sampled actions.
             - A tensor of shape (*batch_shape,) containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
        """
        module_output = self.estimator(states)
        dist = self.estimator.to_probability_distribution(states, module_output)

        with torch.no_grad():
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if torch.any(torch.isinf(log_probs)):
            raise RuntimeError("Log probabilities are inf. This should not happen.")

        return self.env.Actions(actions), log_probs

    def sample_trajectories(
        self,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
    ) -> Trajectories:
        """Sample trajectories sequentially.

        Args:
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
            states = self.env.reset(batch_shape=(n_trajectories,))
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]

        device = states.tensor.device

        dones = states.is_initial_state if self.is_backward else states.is_sink_state

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
            actions = self.env.Actions.make_dummy_actions(batch_shape=(n_trajectories,))
            log_probs = torch.full(
                (n_trajectories,), fill_value=0, dtype=torch.float, device=device
            )
            valid_actions, actions_log_probs = self.sample_actions(states[~dones])
            actions[~dones] = valid_actions
            log_probs[~dones] = actions_log_probs
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]

            if self.is_backward:
                new_states = self.env.backward_step(states, actions)
            else:
                new_states = self.env.step(states, actions)
            sink_states_mask = new_states.is_sink_state

            step += 1

            new_dones = (
                new_states.is_initial_state if self.is_backward else sink_states_mask
            ) & ~dones
            trajectories_dones[new_dones & ~dones] = step
            try:
                trajectories_log_rewards[new_dones & ~dones] = self.env.log_reward(
                    states[new_dones & ~dones]
                )
            except NotImplementedError:
                # print(states[new_dones & ~dones])
                # print(torch.log(self.env.reward(states[new_dones & ~dones])))
                trajectories_log_rewards[new_dones & ~dones] = torch.log(
                    self.env.reward(states[new_dones & ~dones])
                )
            states = new_states
            dones = dones | new_dones

            trajectories_states += [states.tensor]

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = self.env.States(tensor=trajectories_states)
        trajectories_actions = self.env.Actions.stack(trajectories_actions)
        trajectories_logprobs = torch.stack(trajectories_logprobs, dim=0)

        trajectories = Trajectories(
            env=self.env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.is_backward,
            log_rewards=trajectories_log_rewards,
            log_probs=trajectories_logprobs,
        )

        return trajectories
