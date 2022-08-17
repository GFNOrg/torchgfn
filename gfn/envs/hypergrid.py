"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

from copy import deepcopy
from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.containers import States
from gfn.envs.env import Env, NonValidActionsError

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]


class HyperGrid(Env):
    def __init__(
        self,
        ndim: int = 2,
        height: int = 8,
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        reward_cos: bool = False,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s_0 = torch.zeros(ndim, device=device)
        s_f = torch.ones(ndim, device=device) * (-1)
        n_actions = ndim + 1
        super().__init__(n_actions, s_0, s_f)
        self.ndim = ndim
        self.height = height
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.reward_cos = reward_cos

    def make_random_states_tensor(self, batch_shape: Tuple[int]) -> StatesTensor:
        return torch.randint(0, self.height, (*batch_shape, *self.state_shape))

    def update_masks(self, states: States) -> None:
        # TODO: probably not the best way to do this
        states.forward_masks[..., :-1] = states.states != self.height - 1
        states.backward_masks = states.states != 0

    def step_no_worry(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), 1, reduce="add")

    def backward_step_no_worry(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), -1, reduce="add")

    def reward(self, final_states: States) -> TensorFloat:
        final_states_raw = final_states.states
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states_raw / (self.height - 1) - 0.5)
        if not self.reward_cos:
            reward = (
                R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
            )
        else:
            pdf_input = ax * 5
            pdf = 1.0 / (2 * torch.pi) ** 0.5 * torch.exp(-(pdf_input**2) / 2)
            reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1
        return reward

    def get_states_indices(self, states):
        states_raw = states.states
        canonical_base = self.height ** torch.arange(self.ndim - 1, -1, -1)
        flat_indices = (canonical_base * states_raw).sum(-1).long()
        return flat_indices


if __name__ == "__main__":

    print("Testing HyperGrid environment")
    env = HyperGrid(ndim=2, height=3)
    print(env)

    print("\nInstantiating a linear batch of initial states")
    states = env.reset(batch_shape=3)
    print("States:", states)

    print("\nTrying the step function starting from 3 instances of s_0")
    actions = torch.tensor([0, 1, 2], dtype=torch.long)
    states = env.step(states, actions)
    print("After one step:", states)
    actions = torch.tensor([2, 0, 1], dtype=torch.long)
    states = env.step(states, actions)
    print("After two steps:", states)
    actions = torch.tensor([2, 0, 1], dtype=torch.long)
    states = env.step(states, actions)
    print("After three steps:", states)
    try:
        actions = torch.tensor([2, 0, 1], dtype=torch.long)
        states = env.step(states, actions)
    except NonValidActionsError:
        print("NonValidActionsError raised as expected because of invalid actions")
    print(states)
    print("Final rewards:", env.reward(states))

    print("\nTrying the backward step function starting from a batch of random states")

    print("\nInstantiating a two-dimensional batch of random states")
    states = env.reset(batch_shape=(2, 3), random_init=True)
    print("States:", states)
    while not all(states.is_initial_state().view(-1)):
        actions = torch.randint(0, env.n_actions - 1, (2, 3), dtype=torch.long)
        print("Actions: ", actions)
        try:
            states = env.backward_step(states, actions)
            print("States:", states)
        except NonValidActionsError:
            print("NonValidActionsError raised as expected because of invalid actions")
