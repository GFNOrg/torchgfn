from abc import ABC, abstractmethod
from typing import List
import torch
from torchtyping import TensorType

from dataclasses import dataclass, field
from gflownet_playground.envs.env import Env, AbstractStatesBatch
from gflownet_playground.preprocessors.base import Preprocessor
from torch.distributions import Categorical


@dataclass
class Trajectories:
    """Class for keeping track of multiple COMPLETE trajectories."""
    n_trajectories: int
    states: TensorType['max_length', 'n_trajectories',
                       'shape'] 
    actions: TensorType['max_length', 'n_trajectories',
                        torch.long] 
    # The following field mentions how many actions were taken in each trajectory.
    when_is_done: TensorType['n_trajectories', torch.long] 
    rewards: TensorType['n_trajectories', torch.float]


class ActionSampler(ABC):
    "Implements a method that samples actions from any given batch of states."
    @abstractmethod
    def get_probs(self, states: AbstractStatesBatch) -> TensorType['batch_size', 'n_actions']:
        pass

    def sample(self, states: AbstractStatesBatch) -> TensorType['batch_size', torch.long]:
        probs = self.get_probs(states)
        return Categorical(probs).sample()


class UniformActionSampler(ActionSampler):
    def get_probs(self, states):
        probs = states.masks.float() / states.masks.float().sum(dim=-1, keepdim=True)
        return probs


class PFActionSampler(ActionSampler):
    def __init__(self, preprocessor: Preprocessor, pf: torch.nn.Module, temperature: float = 1.):
        # pf needs to return logits ! not probs
        self.preprocessor = preprocessor
        self.pf = pf
        self.temperature = temperature

    def get_probs(self, states):
        preprocessed_states = self.preprocessor.preprocess(states)
        logits = self.pf(preprocessed_states)
        logits[~states.masks] = -float('inf')
        return torch.softmax(logits / self.temperature, dim=-1)


class TrajectoriesSampler:
    def __init__(self, env: Env, action_sampler: ActionSampler):
        self.env = env
        self.action_sampler = action_sampler

    def sample_trajectories(self) -> Trajectories:
        states = self.env.reset()
        n_trajectories = self.env.n_envs
        trajectories_states = []
        trajectories_actions = []
        trajectories_dones = (-1) * \
            torch.ones(n_trajectories, dtype=torch.long)
        step = 0
        while not all(states.already_dones):
            actions = self.action_sampler.sample(states)
            trajectories_states += [states.states]
            trajectories_actions += [actions]
            new_states, dones = self.env.step(actions)
            trajectories_dones[dones & ~states.already_dones] = step
            states = new_states
            step += 1
        trajectories_dones[trajectories_dones == -1] = step

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_actions = torch.stack(trajectories_actions, dim=0)

        trajectories_rewards = self.env.reward(states)

        trajectories = Trajectories(n_trajectories=n_trajectories,
                                    states=trajectories_states,
                                    actions=trajectories_actions,
                                    when_is_done=trajectories_dones,
                                    rewards=trajectories_rewards)

        return trajectories


if __name__ == '__main__':
    from gflownet_playground.envs.hypergrid import HyperGrid
    from gflownet_playground.preprocessors import IdentityPreprocessor, OneHotPreprocessor, KHotPreprocessor

    n_envs = 5
    env = HyperGrid(n_envs=n_envs)

    print("Trying the Uniform Action Sampler")
    action_sampler = UniformActionSampler()
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    print(trajectories)

    preprocessors = [IdentityPreprocessor(env=env),
     OneHotPreprocessor(env=env),
      KHotPreprocessor(env=env)]

    pfs = [torch.nn.Linear(preprocessor.output_dim, env.n_actions) for preprocessor in preprocessors]
    
    for (preprocessor, pf) in zip(preprocessors, pfs):
        print("Trying the PFAction Sampler with preprocessor {}".format(preprocessor))
        action_sampler = PFActionSampler(preprocessor=preprocessor, pf=pf)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories()
        print(trajectories)
    