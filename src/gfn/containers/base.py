from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence

import torch

from gfn.env import Env


class Container(ABC):
    """Base class for states containers (states, transitions, or trajectories)."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of elements in the container."""

    @abstractmethod
    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> Container:
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

    @property
    def has_log_probs(self) -> bool:
        """Returns whether the trajectories have log probabilities."""
        if not hasattr(self, "log_probs"):
            return False

        return self.log_probs is not None and self.log_probs.nelement() > 0
