from typing import Optional

import torch

from gfn.containers import States, Transitions

from .actions_samplers import FixedActionsSampler
from .base import TrainingSampler


class TransitionsSampler(TrainingSampler):
    def sample_transitions(
        self, states: Optional[States] = None, n_transitions: Optional[int] = None
    ) -> Transitions:
        if states is None:
            assert (
                n_transitions is not None
            ), "Either states or n_transitions should be specified"
            states = self.env.reset(batch_shape=(n_transitions,))
        assert states is not None and len(states.batch_shape) == 1
        n_transitions = states.batch_shape[0]
        actions = torch.full((n_transitions,), fill_value=-1, dtype=torch.long)

        valid_selector = (
            ~states.is_initial_state if self.is_backward else ~states.is_sink_state
        )
        valid_states = states[valid_selector]
        _, valid_actions = self.actions_sampler.sample(valid_states)
        actions[valid_selector] = valid_actions

        if self.is_backward:
            new_states = self.env.backward_step(states, actions)
        else:
            new_states = self.env.step(states, actions)

        is_done = (
            new_states.is_initial_state
            if self.is_backward
            else new_states.is_sink_state
        )

        if isinstance(self.actions_sampler, FixedActionsSampler):
            self.actions_sampler.actions = self.actions_sampler.actions[
                valid_actions != self.env.n_actions - 1
            ]

        transitions = Transitions(
            env=self.env,
            n_transitions=n_transitions,
            states=states,
            actions=actions,
            next_states=new_states,
            is_done=is_done,
            is_backward=self.is_backward,
        )

        return transitions

    def sample(self, n_objects: int) -> Transitions:
        # TODO: Change `sample_transitions` such that it can take a number of trajectories as input, roll them out, and get transitions from the resulting trajectories (maybe, using `trajectories_sampler.sample_trajectories`)
        return self.sample_transitions(n_transitions=n_objects)
