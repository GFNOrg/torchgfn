from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gfn.env import Env

import torch


class Container(ABC):
    """Base class for states containers (states, transitions, or trajectories)."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of elements in the container."""

    @abstractmethod
    def __getitem__(self, index: int | Sequence[int]) -> Container:
        """Subsets the container."""

    @abstractmethod
    def extend(self, other: Container) -> None:
        "Extends the current container"

    def sample(self, n_samples: int) -> Container:
        """Samples a subset of the container."""
        return self[torch.randperm(len(self))[:n_samples]]

    def save(self, path: str) -> None:
        """Saves the container to a file."""
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
        """Loads the container from a file, overwriting the current container."""
        for key, val in self.__dict__.items():
            if isinstance(val, Env):
                continue
            elif isinstance(val, Container):
                val.load(os.path.join(path, key))
            elif isinstance(val, torch.Tensor):
                self.__dict__[key] = torch.load(os.path.join(path, key + ".pt"))
            else:
                raise ValueError(f"Unexpected {key} of type {type(val)}")
