"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

from typing import Literal, Tuple

import torch
from einops import rearrange
from torchtyping import TensorType

from gfn.containers.states import States
from gfn.envs.env import Env

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
        height: int = 4,
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        reward_cos: bool = False,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        device = torch.device(device)
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
        return torch.randint(
            0, self.height, (*batch_shape, *self.state_shape), dtype=torch.float
        )

    def update_masks(self, states: States) -> None:
        # TODO: is this the best way ?
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

    def get_states_indices(self, states: States) -> TensorLong:
        if isinstance(states, States):
            states_raw = states.states
        else:
            states_raw = states
        canonical_base = self.height ** torch.arange(
            self.ndim - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        return indices

    @property
    def n_states(self) -> int:
        return self.height**self.ndim

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        all_states = self.all_states
        assert torch.all(
            self.get_states_indices(all_states)
            == torch.arange(self.n_states, device=self.device)
        )
        true_dist = self.reward(all_states)
        true_dist /= true_dist.sum()
        return true_dist

    @property
    def log_partition(self) -> float:
        grid = self.build_grid()
        rewards = self.reward(grid)
        return rewards.sum().log().item()

    def build_grid(self) -> States:
        "Utility function to build the complete grid"
        H = self.height
        ndim = self.ndim
        grid_shape = (H,) * ndim + (ndim,)  # (H, ..., H, ndim)
        grid = torch.zeros(grid_shape, device=self.device)
        for i in range(ndim):
            grid_i = torch.linspace(start=0, end=H - 1, steps=H)
            for _ in range(i):
                grid_i = grid_i.unsqueeze(1)
            grid[..., i] = grid_i

        rearrange_string = " ".join([f"n{i}" for i in range(1, ndim + 1)])
        rearrange_string += " ndim -> "
        rearrange_string += " ".join([f"n{i}" for i in range(ndim, 0, -1)])
        rearrange_string += " ndim"
        grid = rearrange(grid, rearrange_string)
        return self.States(grid)

    @property
    def all_states(self) -> States:
        grid = self.build_grid()
        flat_grid = rearrange(grid.states, "... ndim -> (...) ndim")
        return self.States(flat_grid)
