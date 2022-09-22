from typing import List, Optional

import torch
from torchtyping import TensorType

from gfn.containers import States, Trajectories

from .base import TrainingSampler

# Typing
StatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
ActionsTensor = TensorType["n_trajectories", torch.long]
DonesTensor = TensorType["n_trajectories", torch.bool]


class TrajectoriesSampler(TrainingSampler):
    def sample_trajectories(
        self,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
    ) -> Trajectories:
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
        assert states is not None

        dones = states.is_initial_state if self.is_backward else states.is_sink_state

        trajectories_states: List[StatesTensor] = [states.states]
        trajectories_actions: List[ActionsTensor] = []
        trajectories_dones = torch.zeros(n_trajectories, dtype=torch.long)
        trajectories_log_pfs = torch.zeros(n_trajectories, dtype=torch.float)
        trajectories_log_pbs = torch.zeros(n_trajectories, dtype=torch.float)
        step = 0

        while not all(dones):
            actions = torch.full(
                (n_trajectories,),
                fill_value=-1,
                dtype=torch.long,
                device=states.device,
            )
            actions_log_probs, valid_actions = self.actions_sampler.sample(
                states[~dones]
            )
            actions[~dones] = valid_actions
            trajectories_actions += [actions]
            trajectories_log_pfs[~dones] += actions_log_probs

            if self.is_backward:
                new_states = self.env.backward_step(states, actions)
            else:
                new_states = self.env.step(states, actions)
            sink_states_mask = new_states.is_sink_state
            if self.backward_actions_sampler is not None and not self.is_backward:
                non_sink_new_states = new_states[~sink_states_mask]
                _, backward_probs = self.backward_actions_sampler.get_probs(
                    non_sink_new_states
                )
                trajectories_log_pbs[~sink_states_mask] += (
                    backward_probs.gather(-1, actions[~sink_states_mask].unsqueeze(-1))
                    .squeeze(-1)
                    .log()
                )
            step += 1

            new_dones = (
                new_states.is_initial_state if self.is_backward else sink_states_mask
            ) & ~dones
            trajectories_dones[new_dones & ~dones] = step
            states = new_states
            dones = dones | new_dones

            trajectories_states += [states.states]

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = self.env.States(states=trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions, dim=0)

        trajectories = Trajectories(
            env=self.env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.is_backward,
            log_pfs=trajectories_log_pfs,
            log_pbs=trajectories_log_pbs
            if self.backward_actions_sampler is not None and not self.is_backward
            else None,
        )

        return trajectories

    def sample(self, n_objects: int) -> Trajectories:
        return self.sample_trajectories(n_trajectories=n_objects)
