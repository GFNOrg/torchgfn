from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gfn.envs import Env

import torch
from torchtyping import TensorType

# Typing  --- n_transitions is an int
Tensor2D = TensorType["max_length", "n_trajectories", torch.long]
Tensor2D2 = TensorType["n_trajectories", "shape"]
Tensor1D = TensorType["n_trajectories", torch.long]
FloatTensor1D = TensorType["n_trajectories", torch.float]


class Container(ABC):
    "Base class for states containers (states, transitions, or trajectories)"

    @abstractmethod
    def __len__(self) -> int:
        "Returns the number of elements in the container"
        pass

    @abstractmethod
    def __getitem__(self, index: int | Sequence[int]) -> Container:
        "Subsets the container"
        pass

    @abstractmethod
    def extend(self, other: Container) -> None:
        "Extends the current container"
        pass

    def sample(self, n_samples: int) -> Container:
        "Samples a subset of the container"
        return self[torch.randperm(len(self))[:n_samples]]

    def save(self, path: str) -> None:
        "Saves the container to a file"
        for key, val in self.__dict__.items():
            if isinstance(val, Env):
                continue
            elif isinstance(val, Container):
                val.save(os.path.join(path, key))
            elif isinstance(val, torch.Tensor):
                torch.save(val, os.path.join(path, key + ".pt"))
            else:
                raise ValueError(f"Unexpected {key} of type {type(val)}")

    def load(self, path: str) -> None:
        "Loads the container from a file, overwriting the current container"
        for key, val in self.__dict__.items():
            if isinstance(val, Env):
                continue
            elif isinstance(val, Container):
                val.load(os.path.join(path, key))
            elif isinstance(val, torch.Tensor):
                self.__dict__[key] = torch.load(os.path.join(path, key + ".pt"))
            else:
                raise ValueError(f"Unexpected {key} of type {type(val)}")
