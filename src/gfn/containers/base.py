from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gfn.states import States

import torch

from gfn.env import Env


class Container(ABC):
    """Base class for state containers (states, transitions, or trajectories)."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of elements in the container.

        Returns:
            int: The number of elements in the container.
        """

    @abstractmethod
    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> Container:
        """Returns a subset of the container based on the provided index.

        Args:
            index: An integer, slice, tuple, sequence of indices or booleans, or a torch.Tensor
                specifying which elements to select.

        Returns:
            Container: A new container containing the selected elements.
        """

    @abstractmethod
    def extend(self, other: Container) -> None:
        """Extends the current container with elements from another container.

        Args:
            other (Container): The container whose elements will be added.
        """

    def sample(self, n_samples: int) -> Container:
        """Randomly samples a subset of elements from the container.

        Args:
            n_samples (int): The number of elements to sample.

        Returns:
            Container: A new container with the sampled elements.
        """
        return self[torch.randperm(len(self))[:n_samples]]

    def save(self, path: str) -> None:
        """Saves the container and its contents to a directory.

        Args:
            path (str): The directory path where the container will be saved.

        Raises:
            ValueError: If an attribute type is not supported for saving.
        """
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
        """Loads the container's contents from a directory, overwriting current contents.

        Args:
            path (str): The directory path from which to load the container.

        Raises:
            ValueError: If an attribute type is not supported for loading.
        """
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
    @abstractmethod
    def terminating_states(self) -> States:
        """Returns the last (terminating) states of the container.

        Returns:
            States: The terminating states.
        """

    @property
    @abstractmethod
    def log_rewards(self) -> torch.Tensor:
        """Returns the log rewards associated with the container.

        Returns:
            torch.Tensor: The log rewards tensor.
        """

    @property
    def has_log_probs(self) -> bool:
        """Indicates whether the container has log probabilities.

        Returns:
            bool: True if log probabilities are present and non-empty, False otherwise.
        """
        if not hasattr(self, "log_probs"):
            return False

        return self.log_probs is not None and self.log_probs.nelement() > 0
