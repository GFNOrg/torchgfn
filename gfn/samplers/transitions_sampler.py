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

        valid_selector = ~states.is_sink_state
        valid_states = states[valid_selector]
        _, valid_actions = self.actions_sampler.sample(valid_states)
        actions[valid_selector] = valid_actions

        new_states = self.env.step(states, actions)

        is_done = new_states.is_sink_state

        if isinstance(self.actions_sampler, FixedActionsSampler):
            self.actions_sampler.actions = self.actions_sampler.actions[
                valid_actions != self.env.n_actions - 1
            ]

        transitions = Transitions(
            env=self.env,
            states=states[valid_selector],
            actions=actions[valid_selector],
            next_states=new_states[valid_selector],
            is_done=is_done[valid_selector],
        )

        return transitions

    def sample(self, n_objects: int) -> Transitions:
        # TODO: merge with Sub-Trajectories sampling
        """:param: n_objects: number of trajectories to roll-out
        :return: Transitions object corresponding to all transitions in the trajectories.

        An alternative is to use the following code, by defining a trajectories_sampler
        > trajectories = trajectories_sampler.sample_trajectories(
        >     n_trajectories=n_objects
        > )
        > transitions = Transitions.from_trajectories(trajectories)
        > return transitions
        """
        all_transitions = Transitions(env=self.env)
        transitions = self.sample_transitions(n_transitions=n_objects)
        all_transitions.extend(transitions)
        while torch.any(~transitions.is_done):
            transitions = self.sample_transitions(states=transitions.next_states)
            all_transitions.extend(transitions)

        return all_transitions
