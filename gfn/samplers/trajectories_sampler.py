import torch
from torchtyping import TensorType
from typing import Union

from dataclasses import dataclass
from gfn.envs.env import Env, AbstractStatesBatch
from gfn.samplers.action_samplers import ActionSampler, BackwardsActionSampler, FixedActions


@dataclass
class Trajectories:
    """Class for keeping track of multiple COMPLETE trajectories."""
    n_trajectories: int
    # TensorType['max_length', 'n_trajectories', 'shape']
    states: AbstractStatesBatch
    actions: TensorType['max_length', 'n_trajectories',
                        torch.long]
    # logits: TensorType['max_length', 'n_trajectories',
    #                    'n_actions']
    # backwards_masks: TensorType['max_length', 'n_trajectories', 'n_actions-1']
    # The following field mentions how many actions were taken in each trajectory.
    when_is_done: TensorType['n_trajectories', torch.long]
    rewards: TensorType['n_trajectories', torch.float] = None

    def __repr__(self) -> str:
        states = self.states.states
        assert states.ndim == 3
        states = states.transpose(0, 1)
        states_repr = '\n'.join(['-> '.join([str(step.numpy()) for step in traj])
                                 for traj in states])
        return f"Trajectories(n_trajectories={self.n_trajectories}, " \
               f"states={states_repr}, actions={self.actions}, " \
               f"when_is_done={self.when_is_done}, rewards={self.rewards})"

    def get_last_states_raw(self) -> TensorType['n_trajectories', 'shape']:
        return self.states.states[-1]


class TrajectoriesSampler:
    def __init__(self, env: Env, action_sampler: ActionSampler):
        self.env = env
        self.action_sampler = action_sampler
        self.is_backwards = isinstance(action_sampler, BackwardsActionSampler)

    def sample_trajectories(self, states: Union[None, AbstractStatesBatch] = None) -> Trajectories:
        if states is None:
            states = self.env.reset()
        n_trajectories = states.shape[0]
        trajectories_states = []
        trajectories_actions = []
        trajectories_already_dones = []
        # trajectories_backwards_masks = []
        trajectories_dones = (-1) * \
            torch.ones(n_trajectories, dtype=torch.long)
        step = 0
        while not all(states.already_dones):
            logits, actions = self.action_sampler.sample(states)
            trajectories_states += [states.states]
            trajectories_actions += [actions]
            trajectories_already_dones += [states.already_dones]

            if self.is_backwards:
                new_states, dones = self.env.backward_step(states, actions)
            else:
                # TODO: fix this when states is passed as argument
                new_states, dones = self.env.step(actions)
            trajectories_dones[dones & ~states.already_dones] = step
            states = new_states
            step += 1
            if isinstance(self.action_sampler, FixedActions) and self.action_sampler.step >= self.action_sampler.total_steps:
                states.already_dones = torch.ones_like(
                    states.already_dones, dtype=torch.bool)
            # trajectories_backwards_masks += [states.backward_masks]
        # trajectories_dones[trajectories_dones == -1] = step

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_already_dones = torch.stack(
            trajectories_already_dones, dim=0)
        trajectories_states = self.env.StatesBatch(states=trajectories_states)
        trajectories_states.already_dones = trajectories_already_dones
        trajectories_actions = torch.stack(trajectories_actions, dim=0)
        # trajectories_backwards_masks = torch.stack(
        #     trajectories_backwards_masks, dim=0)

        if self.is_backwards:
            trajectories_rewards = None
        else:
            trajectories_rewards = self.env.reward(states)

        trajectories = Trajectories(n_trajectories=n_trajectories,
                                    states=trajectories_states,
                                    actions=trajectories_actions,
                                    # logits=trajectories_logits,
                                    # backwards_masks=trajectories_backwards_masks,
                                    when_is_done=trajectories_dones,
                                    rewards=trajectories_rewards)

        return trajectories


if __name__ == '__main__':
    from gfn.envs import HyperGrid
    from gfn.preprocessors import IdentityPreprocessor, OneHotPreprocessor, KHotPreprocessor
    from gfn.samplers.action_samplers import (UniformActionSampler, UniformBackwardsActionSampler,
                                              ModuleActionSampler, BackwardsModuleActionSampler,
                                              FixedActions)

    n_envs = 5
    env = HyperGrid(n_envs=n_envs)

    print('---Trying Forwars sampling of trajectories---')

    print("Trying the Uniform Action Sample with sf_temperature")
    action_sampler = UniformActionSampler(sf_temperature=2.)
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    print(trajectories)

    print("Trying the Fixed Actions Sampler: ")
    action_sampler = FixedActions(torch.tensor([[0, 1, 2, 0],
                                                [1, 1, 1, 2],
                                                [2, 2, 2, 2],
                                                [1, 0, 1, 2],
                                                [1, 0, 2, 1]]))
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    print(trajectories)

    preprocessors = [IdentityPreprocessor(env=env),
                     OneHotPreprocessor(env=env),
                     KHotPreprocessor(env=env)]

    pfs = [torch.nn.Linear(preprocessor.output_dim, env.n_actions)
           for preprocessor in preprocessors]

    for (preprocessor, pf) in zip(preprocessors, pfs):
        print("Trying the PFAction Sampler with preprocessor {}".format(preprocessor))
        action_sampler = ModuleActionSampler(
            preprocessor=preprocessor, module=pf)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories()
        print(trajectories)

    print('---Trying Backwards sampling of trajectories---')
    states = env.StatesBatch(random=True)
    states.states[0] = torch.zeros(2)
    states.update_masks()
    states.update_the_dones()
    print('Trying the Uniform Backwards Action Sampler with one of the initial states being s_0')
    action_sampler = UniformBackwardsActionSampler()
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories(states)
    print(trajectories)

    pbs = [torch.nn.Linear(preprocessor.output_dim, env.n_actions - 1)
           for preprocessor in preprocessors]

    for (preprocessor, pb) in zip(preprocessors, pbs):
        states = env.StatesBatch(random=True)
        states.update_masks()
        states.update_the_dones()
        print("Trying the PBAction Sampler with preprocessor {}".format(preprocessor))
        action_sampler = BackwardsModuleActionSampler(
            preprocessor=preprocessor, module=pb)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories(states)
        print(trajectories)
