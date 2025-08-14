import itertools
from abc import ABC, abstractmethod
from typing import Callable

import torch

EPS = 1e-12


def get_reward_fn(
    reward_fn_str: str, height: int, ndim: int, reward_fn_kwargs: dict | None = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get the reward function for the HyperGrid environment."""
    reward_fn_kwargs = reward_fn_kwargs or {}
    if reward_fn_str == "original":
        return OriginalReward(height, ndim, **reward_fn_kwargs)
    elif reward_fn_str == "cosine":
        return CosineReward(height, ndim, **reward_fn_kwargs)
    elif reward_fn_str == "sparse":
        return SparseReward(height, ndim, **reward_fn_kwargs)
    elif reward_fn_str == "deceptive":
        return DeceptiveReward(height, ndim, **reward_fn_kwargs)
    else:
        raise ValueError(f"Invalid reward function string: {reward_fn_str}")


class GridReward(ABC):
    """Base class for reward functions that can be pickled."""

    def __init__(self, height: int, ndim: int, **kwargs):
        self.height = height
        self.ndim = ndim
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OriginalReward(GridReward):
    """The reward function from the original GFlowNet paper (Bengio et al., 2021;
    https://arxiv.org/abs/2106.04399)."""

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R0 = self.kwargs.get("R0", 0.1)
        R1 = self.kwargs.get("R1", 0.5)
        R2 = self.kwargs.get("R2", 2.0)

        ax = abs(states_tensor / (self.height - 1) - 0.5)
        return (
            R0
            + (0.25 + EPS < ax).prod(-1) * R1
            + ((0.3 + EPS < ax) * (ax < 0.4 + EPS)).prod(-1) * R2
        )


class CosineReward(GridReward):
    """Cosine reward function."""

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R0 = self.kwargs.get("R0", 0.1)
        R1 = self.kwargs.get("R1", 0.5)

        ax = abs(states_tensor / (self.height - 1) - 0.5)
        pdf_input = ax * 5
        pdf = 1.0 / (2 * torch.pi) ** 0.5 * torch.exp(-(pdf_input**2) / 2)
        reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1
        return reward


class SparseReward(GridReward):
    """Sparse reward function from the GAFN paper (Pan et al., 2022;
    https://arxiv.org/abs/2210.03308)."""

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        targets = []
        for number_of_1s in range(ndim):
            targets.extend(
                itertools.permutations(
                    [1] * number_of_1s + [self.height - 2] * (self.ndim - number_of_1s)
                )
            )
        self.targets = torch.tensor(list(set(targets)), dtype=torch.long)

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        self.targets = self.targets.to(states_tensor.device)
        reward = (
            (states_tensor.unsqueeze(1) == self.targets.unsqueeze(0)).prod(-1).sum(-1)
        ) + 1e-12  # Avoid log(0)
        return reward


class DeceptiveReward(GridReward):
    """Deceptive reward function from the Adaptive Teachers paper (Kim et al., 2025;
    https://arxiv.org/abs/2410.01432).

    Note that the reward definition in the paper (eq. (9)) is incorrect, and we follow
    the official implementation (https://github.com/alstn12088/adaptive-teacher/blob/8cfcb2298fce3f46eb36ead03791eeee75b7d066/grid/env.py#L27)
    while modifying it to use EPS = 1e-12 to handle inequalities with floating points.
    """

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R0 = self.kwargs.get("R0", 1e-5)
        R1 = self.kwargs.get("R1", 0.1)
        R2 = self.kwargs.get("R2", 2.0)

        ax = abs(states_tensor / (self.height - 1) - 0.5)
        return (
            R0
            + R1
            - (0.1 + EPS < ax).prod(-1) * R1
            + ((0.3 + EPS < ax) * (ax < 0.4 + EPS)).prod(-1) * R2
        )
