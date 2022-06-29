"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

import torch
import numpy as np
from scipy.stats import norm
from copy import deepcopy
from gflownet_playground.envs.env import Env

class HyperGrid(Env):
    def __init__(self, ndim=2, H=8, R0=1e-2, R1=.5, R2=2, reward_cos=False):
        """
        > Example : ndim=2 (H=3, ...)
        ```
        ---------------------------
        | (0, 2) | (1, 2) | (2, 2) 
        ---------------------------
        | (0, 1) | (1, 1) | (2, 1) 
        ---------------------------
        | (0, 0) | (1, 0) | (2, 0) 
        ---------------------------
        ```
        """
        # We have (H+1)^ndim points, each point being of dimension ndim.
        self.ndim = ndim
        self.n_actions = ndim + 1
        self.H = H
        self.n_states = H ** ndim
        grid_shape = (H, ) * ndim + (ndim, )  # (H, ..., H, ndim)
        self.grid = torch.zeros(grid_shape)
        for i in range(ndim):
            grid_i = torch.linspace(start=0, end=H - 1, steps=H)
            for _ in range(i):
                grid_i = grid_i.unsqueeze(1)
            self.grid[..., i] = grid_i
        self.reward_cos = reward_cos
        self.grid_shape = grid_shape
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2

    def is_actions_valid(self, states, actions):
        if any(actions >= self.n_actions):
            return False
        mask = actions < self.n_actions - 1
        non_final_states = states[mask]
        non_final_actions = actions[mask]
        k = non_final_states.shape[0]
        dimensions_to_increase = non_final_states[torch.arange(k), non_final_actions]
        return all(dimensions_to_increase < self.H - 1)

    def step(self, states, actions):
        super().step(states, actions)
        k = states.shape[0]
        new_states = deepcopy(states)
        dones = torch.zeros(k).bool()
        dones[actions == self.n_actions - 1] = True
        new_states[torch.arange(k)[~dones], actions[~dones]] += 1

        assert new_states.max() < self.H, "Something terrible has happened. This should never happen !"

        return new_states, dones

    def reward(self, states):
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(states / (self.H - 1) - 0.5)
        if not self.reward_cos:
            reward = R0 + (0.25 < ax).prod(-1) * R1 + \
                ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
        else:
            reward = R0 + ((np.cos(ax * 50) + 1) *
                           norm.pdf(ax * 5)).prod(-1) * R1
        return reward

    def mask_maker(self, states):
        k = states.shape[0]
        masks = torch.zeros(k, self.n_actions).bool()
        edges = (states == self.H - 1)
        at_least_one_edge = edges.long().sum(1).bool()
        masks[at_least_one_edge, :-1] = edges[at_least_one_edge]

        return masks.bool()

    def backward_mask_maker(self, states):
        k = states.shape[0]
        masks = torch.zeros(k, self.n_actions - 1).bool()
        edges = (states == 0)
        at_least_one_edge = edges.long().sum(1).bool()
        masks[at_least_one_edge] = edges[at_least_one_edge]

        return masks.bool()


if __name__ == '__main__':
    # testing the environment
    import matplotlib.pyplot as plt
    import torch

    ndim = 3
    H = 8
    env = HyperGrid(ndim=ndim, H=H, R0=0.1)

    all_rewards = env.reward(env.grid)
    true_Z = all_rewards.sum().item()
    if ndim == 2:
        plt.imshow(all_rewards)
        plt.title('True Z: {}'.format(true_Z))
        plt.colorbar()
        plt.show()

    print("The environment has {} actions.".format(env.n_actions))

    states = torch.randint(0, H, (5, ndim)).float()
    print('Testing with the following states: ')
    print(states)
    print('Mask of available actions: ')
    print(env.mask_maker(states))
    print('Mask of available actions for backward: ')
    print(env.backward_mask_maker(states))

    attempt = 0
    while True:
        actions = torch.randint(0, ndim + 1, (5, )).long()
        print('Trying actions: {}'.format(actions))
        if env.is_actions_valid(states, actions):
            print('All actions are valid.')
            new_states, dones = env.step(states, actions)
            print('Next states:')
            print(new_states)
            print('Is it done:')
            print(dones)
            break 
        else:
            attempt += 1
            print('Invalid actions, retrying...')
        if attempt > 9:
            print('10 attempts already ! aborting...')
            break