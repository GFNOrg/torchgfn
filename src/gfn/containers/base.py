from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gfn.states import States

import torch
from tensordict import TensorDict
from tensordict.base import TensorDictBase

from gfn.env import Env


class Container(ABC):
    """Base class for state containers (states, transitions, or trajectories)."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of elements in the container.

        Returns:
            The number of elements in the container.
        """

    @abstractmethod
    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> Container:
        """Returns a subset of the container based on the provided index.

        Args:
            index: An integer, slice, tuple, sequence of indices or booleans,
                or a torch.Tensor specifying which elements to select.

        Returns:
            A new container containing the selected elements and associated data.
        """

    @abstractmethod
    def extend(self, other: Container) -> None:
        """Extends the current container with elements from another container object.

        Args:
            other: The other container whose elements will be added.
        """

    def sample(self, n_samples: int) -> Container:
        """Randomly samples a subset of elements from the container.

        Args:
            n_samples: The number of elements to sample.

        Returns:
            A new container with the sampled elements.
        """
        return self[torch.randperm(len(self))[:n_samples]]

    @abstractmethod
    def to_tensordict(self) -> TensorDictBase:
        """Serialize the container's data into a TensorDict.

        Returns:
            A TensorDict containing all tensor data and scalar metadata.
            The ``env`` reference is not included; it must be supplied
            when reconstructing via :meth:`from_tensordict`.
        """

    @classmethod
    @abstractmethod
    def from_tensordict(cls, env: Env, td: TensorDictBase) -> Container:
        """Reconstruct a container from a TensorDict.

        Args:
            env: The environment needed to reconstruct States/Actions.
            td: The TensorDict produced by :meth:`to_tensordict`.

        Returns:
            A new container instance.
        """

    def save(self, path: str) -> None:
        """Saves the container to a single ``.pt`` file.

        Args:
            path: File path (e.g. ``"trajectories.pt"``).
        """
        torch.save(self.to_tensordict().to_dict(), path)

    @classmethod
    def load(cls, env: Env, path: str) -> Container:
        """Loads a container from a ``.pt`` file saved by :meth:`save`.

        Args:
            env: The environment needed to reconstruct States/Actions.
            path: File path to the saved container.

        Returns:
            A new container instance.
        """
        raw = torch.load(path, weights_only=True, map_location=env.device)
        td = TensorDict(raw, batch_size=[])
        return cls.from_tensordict(env, td)

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device on which the container is stored.

        Returns:
            The device on which the container is stored.
        """

    @property
    @abstractmethod
    def terminating_states(self) -> States:
        """The last (terminating) states of the container.

        Returns:
            The terminating states.
        """

    @property
    @abstractmethod
    def log_rewards(self) -> torch.Tensor:
        """The log rewards associated with the container.

        Returns:
            The log rewards tensor.
        """

    @property
    def has_log_probs(self) -> bool:
        """Whether the container has log probabilities.

        Returns:
            True if log probabilities are present and non-empty, False otherwise.
        """
        if not hasattr(self, "log_probs"):
            return False

        return self.log_probs is not None and self.log_probs.nelement() > 0
