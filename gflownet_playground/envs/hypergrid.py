"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

from abc import ABC, abstractmethod
from re import L
import torch
from torchtyping import TensorType
from torch import Tensor
from typing import Tuple, Type
from dataclasses import dataclass, field
from copy import deepcopy

from gym.spaces import Discrete
from scipy.stats import norm

from gflownet_playground.envs.env import AbstractStatesBatch, Env


@dataclass
class HyperGrid(Env):
    "Hypergrid environment"
    ndim: int = 2
    height: int = 8
    R0: float = 1e-2
    R1: float = .5
    R2: float = 2.
    reward_cos: bool = False

    def __post_init__(self):
        self.state_shape = (self.ndim,)
        self.n_actions = self.ndim + 1
        super().__post_init__()
        self.n_states = self.height ** self.ndim

    def make_state_class(self, batch_size):
        envSelf = self
        bs = batch_size

        @ dataclass
        class StatesBatch(AbstractStatesBatch):
            batch_size: int = bs
            state_dim: Tuple = (envSelf.ndim, )
            shape: Tuple = field(init=False)
            states: TensorType[(batch_size, *state_dim)] = field(init=False)
            masks: TensorType[batch_size,
                              envSelf.n_actions, bool] = field(init=False)
            backward_masks: TensorType[batch_size,
                                       envSelf.n_actions - 1, bool] = field(init=False)
            already_dones: TensorType[batch_size, bool] = field(init=False)

            def __post_init__(self):
                self.shape = (self.batch_size, *self.state_dim)
                self.states = torch.zeros(
                    self.shape, dtype=torch.long, device=envSelf.device)
                self.masks = torch.ones(
                    (self.batch_size, envSelf.n_actions), dtype=torch.bool, device=envSelf.device)
                self.backward_masks = torch.zeros(
                    (self.batch_size, envSelf.n_actions - 1), dtype=torch.bool, device=envSelf.device)
                self.already_dones = torch.zeros(
                    self.batch_size, dtype=torch.bool, device=envSelf.device)

        return StatesBatch

    def step(self, actions):
        not_done_states_masks = self._state.masks[~self._state.already_dones]
        not_done_actions = actions[~self._state.already_dones]
        actions_valid = all(torch.gather(
            not_done_states_masks, 1, not_done_actions.unsqueeze(1)))
        if not actions_valid:
            raise ValueError('Actions are not valid')

        dones = self._state.already_dones | (actions == self.n_actions - 1)
        self._state.already_dones = dones

        not_done_states = self._state.states[~dones]
        not_done_actions = actions[~dones]
        not_done_masks = self._state.masks[~dones]
        not_done_backward_masks = self._state.backward_masks[~dones]

        n_states_to_update = len(not_done_actions)
        not_done_states[torch.arange(
            n_states_to_update), not_done_actions] += 1

        self._state.states[~dones] = not_done_states

        not_done_masks[torch.cat([not_done_states == self.height - 1,
                                  torch.zeros(n_states_to_update, 1, dtype=torch.bool)],
                                 1)
                       ] = False

        self._state.masks[~dones] = not_done_masks

        not_done_backward_masks[torch.arange(
            n_states_to_update), not_done_actions] = True

        self._state.backward_masks[~dones] = not_done_backward_masks

        return deepcopy(self._state), dones

    def reward(self, final_states):
        final_states = final_states.states
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states / (self.height - 1) - 0.5)
        if not self.reward_cos:
            reward = R0 + (0.25 < ax).prod(-1) * R1 + \
                ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
        else:
            pdf_input = ax * 5
            pdf = 1. / (2 * torch.pi) ** 0.5 * torch.exp(-pdf_input ** 2 / 2)
            reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1
        return reward

    def get_states_indices(self, states) -> TensorType['batch_size', torch.long]:
        states = states.states
        canonical_base = self.height ** torch.arange(self.ndim - 1, -1, -1)
        flat_indices = (canonical_base * states).sum(1).long()
        return flat_indices


if __name__ == '__main__':
    print('Testing HyperGrid env with 3 environemnts, and height of 3')
    env = HyperGrid(n_envs=3, height=3)
    env.reset()
    actions = torch.tensor([0, 1, 1], dtype=torch.long)
    states, dones = env.step(actions)
    actions = torch.tensor([0, 2, 1], dtype=torch.long)
    states, dones = env.step(actions)
    actions = torch.tensor([1, 1, 2], dtype=torch.long)
    states, dones = env.step(actions)
    try:
        actions = torch.tensor([0, 1, 1], dtype=torch.long)
        states, dones = env.step(actions)
    except ValueError:
        print('ValueError raised as expected because of invalid actions')
    print(env._state)

# class HyperGrid(Env):
#     def __init__(self, ndim=2, H=8, R0=1e-2, R1=.5, R2=2, reward_cos=False):
#         """
#         > Example : ndim=2 (H=3, ...)
#         ```
#         ---------------------------
#         | (0, 2) | (1, 2) | (2, 2)
#         ---------------------------
#         | (0, 1) | (1, 1) | (2, 1)
#         ---------------------------
#         | (0, 0) | (1, 0) | (2, 0)
#         ---------------------------
#         ```
#         """
#         # We have (H+1)^ndim points, each point being of dimension ndim.
#         self.ndim = ndim
#         self.n_actions = ndim + 1
#         self.H = H
#         grid_shape = (H, ) * ndim + (ndim, )  # (H, ..., H, ndim)
#         self.grid = torch.zeros(grid_shape)
#         for i in range(ndim):
#             grid_i = torch.linspace(start=0, end=H - 1, steps=H)
#             for _ in range(i):
#                 grid_i = grid_i.unsqueeze(1)
#             self.grid[..., i] = grid_i
#         self.reward_cos = reward_cos
#         self.grid_shape = grid_shape
#         self.R0 = R0
#         self.R1 = R1
#         self.R2 = R2

#     @property
#     def n_states(self):
#         return self.H ** self.ndim

#     @property
#     def state_shape(self):
#         return torch.Size((self.ndim, ))

#     @property
#     def state_dim(self):
#         return self.ndim

#     def is_actions_valid(self, states, actions):
#         if any(actions >= self.n_actions):
#             return False
#         mask = actions < self.n_actions - 1
#         non_final_states = states[mask]
#         non_final_actions = actions[mask]
#         k = non_final_states.shape[0]
#         dimensions_to_increase = non_final_states[torch.arange(
#             k), non_final_actions]
#         return all(dimensions_to_increase < self.H - 1)

#     def step(self, states, actions):
#         super().step(states, actions)
#         k = states.shape[0]
#         new_states = deepcopy(states)
#         dones = torch.zeros(k).bool()
#         dones[actions == self.n_actions - 1] = True
#         new_states[torch.arange(k)[~dones], actions[~dones]] += 1

#         assert new_states.max() < self.H, "Something terrible has happened. This should never happen !"

#         return new_states, dones

#     def reward(self, states):
#         R0, R1, R2 = (self.R0, self.R1, self.R2)
#         ax = abs(states / (self.H - 1) - 0.5)
#         if not self.reward_cos:
#             reward = R0 + (0.25 < ax).prod(-1) * R1 + \
#                 ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
#         else:
#             reward = R0 + ((np.cos(ax * 50) + 1) *
#                            norm.pdf(ax * 5)).prod(-1) * R1
#         return reward

#     def mask_maker(self, states):
#         k = states.shape[0]
#         masks = torch.zeros(k, self.n_actions).bool()
#         edges = (states == self.H - 1)
#         at_least_one_edge = edges.long().sum(1).bool()
#         masks[at_least_one_edge, :-1] = edges[at_least_one_edge]

#         return masks.bool()

#     def backward_mask_maker(self, states):
#         k = states.shape[0]
#         masks = torch.zeros(k, self.n_actions - 1).bool()
#         edges = (states == 0)
#         at_least_one_edge = edges.long().sum(1).bool()
#         masks[at_least_one_edge] = edges[at_least_one_edge]

#         return masks.bool()

#     def get_states_indices(self, states):
#         dim = self.state_dim
#         flat_states = states.view(-1, dim)
#         canonical_base = self.H ** torch.arange(dim - 1, -1, -1)
#         flat_indices = (canonical_base * flat_states).sum(1).long()
#         return flat_indices.view(states.shape[:- len(self.state_shape)])


# if __name__ == '__main__':
#     # testing the environment
#     import matplotlib.pyplot as plt
#     import torch

#     ndim = 3
#     H = 8
#     env = HyperGrid(ndim=ndim, H=H, R0=0.1)
#     print('state_shape:', env.state_shape, 'state_dim:',
#           env.state_dim, 'n_states:', env.n_states)

#     all_rewards = env.reward(env.grid)
#     true_Z = all_rewards.sum().item()
#     if ndim == 2:
#         plt.imshow(all_rewards)
#         plt.title('True Z: {}'.format(true_Z))
#         plt.colorbar()
#         plt.show()

#     print("The environment has {} actions.".format(env.n_actions))

#     states = torch.randint(0, H, (5, ndim)).float()
#     print('Testing with the following states: ')
#     print(states)
#     print('Mask of available actions: ')
#     print(env.mask_maker(states))
#     print('Mask of available actions for backward: ')
#     print(env.backward_mask_maker(states))

#     attempt = 0
#     while True:
#         actions = torch.randint(0, ndim + 1, (5, )).long()
#         print('Trying actions: {}'.format(actions))
#         if env.is_actions_valid(states, actions):
#             print('All actions are valid.')
#             new_states, dones = env.step(states, actions)
#             print('Next states:')
#             print(new_states)
#             print('Is it done:')
#             print(dones)
#             break
#         else:
#             attempt += 1
#             print('Invalid actions, retrying...')
#         if attempt > 9:
#             print('10 attempts already ! aborting...')
#             break
