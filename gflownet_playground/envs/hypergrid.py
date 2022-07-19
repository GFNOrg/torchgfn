"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

import torch
from torchtyping import TensorType
from typing import Tuple
from dataclasses import dataclass, field
from copy import deepcopy


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
            states: TensorType[(batch_size, *state_dim)] = None
            shape: Tuple = field(init=False)
            masks: TensorType[batch_size,
                              envSelf.n_actions, bool] = field(init=False)
            backward_masks: TensorType[batch_size,
                                       envSelf.n_actions - 1, bool] = field(init=False)
            already_dones: TensorType[batch_size, bool] = field(init=False)

            def __post_init__(self):
                if self.states is None:
                    self.shape = (self.batch_size, *self.state_dim)
                    self.states = torch.zeros(
                        self.shape, dtype=torch.long, device=envSelf.device)
                else:
                    assert self.states.ndim == 2 and self.states.shape[1] == envSelf.ndim
                    self.batch_size = self.states.shape[0]
                # self.masks = torch.ones(
                #     (self.batch_size, envSelf.n_actions), dtype=torch.bool, device=envSelf.device)
                self.masks = self.make_masks(self.states)
                self.backward_masks = self.make_backward_masks(self.states)
                # self.backward_masks = torch.zeros(
                #     (self.batch_size, envSelf.n_actions - 1), dtype=torch.bool, device=envSelf.device)
                self.already_dones = torch.zeros(
                    self.batch_size, dtype=torch.bool, device=envSelf.device)

            def __repr__(self):
                return f"StatesBatch(\nstates={self.states},\n masks={self.masks},\n backward_masks={self.backward_masks},\n already_dones={self.already_dones})"

            def make_masks(self, states: TensorType[('bs', *state_dim)]) -> TensorType['bs',
                                                                                       envSelf.n_actions, bool]:
                masks = torch.ones(
                    (states.shape[0], envSelf.n_actions), dtype=torch.bool, device=envSelf.device)
                masks[torch.cat([states == envSelf.height - 1,
                                 torch.zeros(states.shape[0], 1, dtype=torch.bool, device=envSelf.device)],
                                1)
                      ] = False
                return masks

            def make_backward_masks(self, states: TensorType[('bs', *state_dim)]) -> TensorType['bs', envSelf.n_actions - 1, bool]:
                masks = torch.ones(
                    states.shape[0], envSelf.n_actions - 1, dtype=torch.bool, device=envSelf.device)

                masks[states == 0] = False
                return masks

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
        # not_done_masks = self._state.masks[~dones]
        # not_done_backward_masks = self._state.backward_masks[~dones]

        n_states_to_update = len(not_done_actions)
        not_done_states[torch.arange(
            n_states_to_update), not_done_actions] += 1

        self._state.states[~dones] = not_done_states

        self._state.masks = self._state.make_masks(self._state.states)
        self._state.backward_masks = self._state.make_backward_masks(
            self._state.states)

        # not_done_masks[torch.cat([not_done_states == self.height - 1,
        #                           torch.zeros(n_states_to_update, 1, dtype=torch.bool)],
        #                          1)
        #                ] = False

        # self._state.masks[~dones] = not_done_masks

        # not_done_backward_masks[torch.arange(
        #     n_states_to_update), not_done_actions] = True

        # self._state.backward_masks[~dones] = not_done_backward_masks

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
    print('Final rewards:', env.reward(env._state))
