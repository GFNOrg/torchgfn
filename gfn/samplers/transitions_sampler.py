from typing import Optional

import torch

from gfn.containers import States, Transitions
from gfn.envs import Env
from gfn.samplers import ActionSampler, BackwardsActionSampler, FixedActions


class TransitionsSampler:
    def __init__(self, env: Env, action_sampler: ActionSampler):
        self.env = env
        self.action_sampler = action_sampler
        self.is_backwards = isinstance(action_sampler, BackwardsActionSampler)

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
            ~states.is_initial_state if self.is_backwards else ~states.is_sink_state
        )
        valid_states = states[valid_selector]
        _, valid_actions = self.action_sampler.sample(valid_states)
        actions[valid_selector] = valid_actions

        if self.is_backwards:
            new_states = self.env.backward_step(states, actions)
        else:
            new_states = self.env.step(states, actions)

        is_done = (
            new_states.is_initial_state
            if self.is_backwards
            else new_states.is_sink_state
        )
        if not self.is_backwards:
            rewards = torch.zeros(
                n_transitions, dtype=torch.float, device=states.device
            )
            rewards[is_done] = self.env.reward(states[is_done])
        else:
            rewards = None

        if isinstance(self.action_sampler, FixedActions):
            self.action_sampler.actions = self.action_sampler.actions[
                valid_actions != self.env.n_actions - 1
            ]

        transitions = Transitions(
            env=self.env,
            n_transitions=n_transitions,
            states=states,
            actions=actions,
            next_states=new_states,
            is_done=is_done,
            rewards=rewards,
            is_backwards=self.is_backwards,
        )

        return transitions


if __name__ == "__main__":
    from gfn.envs import HyperGrid
    from gfn.samplers.action_samplers import UniformActionSampler

    env = HyperGrid(ndim=2, height=8)

    print("---Trying Forward sampling of trajectories---")

    print("Trying the Uniform Action Sampler")
    action_sampler = UniformActionSampler()
    transitions_sampler = TransitionsSampler(env, action_sampler)
    transitions = transitions_sampler.sample_transitions(n_transitions=5)
    print(transitions)
    transitions = transitions_sampler.sample_transitions(states=transitions.next_states)
    print(transitions)

    print("Trying the Fixed Actions Sampler")
    action_sampler = FixedActions(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 0, 2, 1]]
        )
    )
    transitions_sampler = TransitionsSampler(env, action_sampler)
    transitions = transitions_sampler.sample_transitions(n_transitions=5)
    print(transitions)

    transitions = transitions_sampler.sample_transitions(states=transitions.next_states)
    print(transitions)
