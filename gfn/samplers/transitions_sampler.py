import torch
from typing import Union
from gfn.envs import Env, AbstractStatesBatch
from gfn.samplers import ActionSampler, BackwardsActionSampler, FixedActions
from gfn.containers import Transitions


class TransitionsSampler:
    def __init__(self, env: Env, action_sampler: ActionSampler):
        self.env = env
        self.action_sampler = action_sampler
        self.is_backwards = isinstance(action_sampler, BackwardsActionSampler)

    def sample_transitions(
        self, states: Union[None, AbstractStatesBatch] = None
    ) -> Transitions:
        if states is None:
            states = self.env.reset()
        n_transitions = states.shape[0]
        rewards = torch.zeros(
            n_transitions, dtype=torch.float, device=states.states.device
        )

        _, actions = self.action_sampler.sample(states)

        if self.is_backwards:
            new_states, dones = self.env.backward_step(states, actions)
        else:
            # TODO: fix this when states is passed as argument
            new_states, dones = self.env.step(actions)

        rewards[dones] = self.env.reward(new_states.states[dones])

        transitions = Transitions(
            env=self.env,
            n_transitions=n_transitions,
            states=states,
            actions=actions,
            next_states=new_states,
            is_done=dones,
            rewards=rewards,
        )

        return transitions


if __name__ == "__main__":
    from gfn.envs import HyperGrid
    from gfn.samplers.action_samplers import UniformActionSampler

    n_envs = 5
    env = HyperGrid(n_envs=n_envs)

    print("---Trying Forward sampling of trajectories---")

    print("Trying the Uniform Action Sampler")
    action_sampler = UniformActionSampler()
    transitions_sampler = TransitionsSampler(env, action_sampler)
    transitions = transitions_sampler.sample_transitions()
    print(transitions)

    print("Trying the Fixed Actions Sampler")
    action_sampler = FixedActions(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 0, 2, 1]]
        )
    )
    transitions_sampler = TransitionsSampler(env, action_sampler)
    transitions = transitions_sampler.sample_transitions()
    print(transitions)
