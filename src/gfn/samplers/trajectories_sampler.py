from typing import List, Optional

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories
from gfn.envs import Env
from gfn.samplers import ActionsSampler, BackwardActionsSampler
from gfn.states import States

# Typing
StatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
ActionsTensor = TensorType["n_trajectories", torch.long]
LogProbsTensor = TensorType["n_trajectories", torch.float]
DonesTensor = TensorType["n_trajectories", torch.bool]


class TrajectoriesSampler:
    def __init__(
        self,
        actions_sampler: ActionsSampler,
        is_backward: bool = False,
    ):
        """Sample complete trajectories, or completes trajectories from a given batch states, using actions_sampler.

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
            is_backward (bool, optional): Whether to sample trajectories backward.
             If True, the corresponding ActionsSampler's estimator needs to be a ProbabilityDistribution over parents.
             Defaults to False.
        """
        self.env = actions_sampler.env
        self.actions_sampler = actions_sampler
        self.is_backward = is_backward

    def sample_trajectories(
        self,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
    ) -> Trajectories:
        """
        Args:
            states (Optional[States], optional): If given, trajectories would start from such states.
             Otherwise, n_trajectories needs to be given. Defaults to None.
            n_trajectories (Optional[int], optional): If given, a batch of n_trajectories will be sampled all
             starting from the environment's s_0. Defaults to None.

        Returns:
            Trajectories: A Trajectories object representing the batch of sampled trajectories.
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

        trajectories_states: List[StatesTensor] = [states.tensor]
        trajectories_actions: List[ActionsTensor] = []
        trajectories_logprobs: List[LogProbsTensor] = []
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )

        step = 0

        while not all(dones):
            actions = torch.full(
                (n_trajectories,),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
            log_probs = torch.full(
                (n_trajectories,), fill_value=0, dtype=torch.float, device=device
            )
            actions_log_probs, valid_actions = self.actions_sampler.sample(
                states[~dones]
            )
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
        trajectories_states = self.env.States(states_tensor=trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions, dim=0)
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

    def sample(self, n_trajectories: int) -> Trajectories:
        return self.sample_trajectories(n_trajectories=n_trajectories)
