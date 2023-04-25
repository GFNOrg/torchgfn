from __future__ import annotations  # This allows to use the class name in type hints


from abc import ABC
from typing import ClassVar, Sequence

from math import prod
import torch
from torchtyping import TensorType

# Typing
OneActionTensor = TensorType["action_shape"]
ActionsTensor = TensorType["batch_shape", "action_shape"]
BoolTensor = TensorType["batch_shape", torch.bool]


class Actions(ABC):
    """Base class for actions for all GFlowNet environments.
    Each environment needs to subclass this class. A generic subclass for discrete actions
    with integer indices is provided.
    Note that all actions need to have the same shape.
    """

    # The following class variable represents the shape of a single action
    action_shape: ClassVar[tuple[int, ...]]  # all actions need to have the same shape
    # The following class variable is padded to shorter trajectories
    dummy_action: ClassVar[OneActionTensor]  # dummy action for the environment
    # The following class variable corresponds to $s \rightarrow s_f$ transitions
    exit_action: ClassVar[OneActionTensor]  # action to exit the environment

    def __init__(self, tensor: ActionsTensor):
        """Initialize actions from a tensor.
        Args:
            tensor: tensor of actions
        """
        self.tensor = tensor
        self.batch_shape = tuple(self.tensor.shape)[: -len(self.action_shape)]

    @classmethod
    def make_dummy_actions(cls, batch_shape: tuple[int]) -> Actions:
        """Creates an Actions object with the given batch shape, filled with dummy actions."""
        action_ndim = len(cls.action_shape)
        tensor = cls.dummy_action.repeat(*batch_shape, *((1,) * action_ndim))
        return cls(tensor)

    @classmethod
    def make_exit_actions(cls, batch_shape: tuple[int]) -> Actions:
        """Creates an Actions object with the given batch shape, filled with exit actions."""
        n_actions_dim = len(cls.action_shape)
        tensor = cls.exit_action.repeat(*batch_shape, *((1,) * n_actions_dim))
        return cls(tensor)

    def __len__(self) -> int:
        return prod(self.batch_shape)

    def __repr__(self):
        return f"""{self.__class__.__name__} object of batch shape {self.batch_shape}.
          The subclass did not implement __repr__."""

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(self, index: int | Sequence[int] | Sequence[bool]) -> Actions:
        actions = self.tensor[index]
        return self.__class__(actions)

    def extend(self, other: Actions) -> None:
        """Collates to another Actions object of the same batch shape."""
        if len(self.batch_shape) == len(other.batch_shape) == 1:
            self.batch_shape = (self.batch_shape[0] + other.batch_shape[0],)
            self.tensor = torch.cat(
                (self.tensor, other.tensor), dim=0
            )
        elif len(self.batch_shape) == len(other.batch_shape) == 2:
            self.extend_with_dummy_actions(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            other.extend_with_dummy_actions(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            self.batch_shape = (
                self.batch_shape[0],
                self.batch_shape[1] + other.batch_shape[1],
            )
            self.tensor = torch.cat(
                (self.tensor, other.tensor), dim=1
            )
        else:
            raise NotImplementedError(
                "extend is only implemented for bi-dimensional actions."
            )

    def extend_with_dummy_actions(self, required_first_dim: int) -> None:
        """Extends a bi-dimensional Actions object with dummy actions in the first dimension.
        This is used to pad trajectories actions"""
        if len(self.batch_shape) == 2:
            if self.batch_shape[0] >= required_first_dim:
                return
            n = required_first_dim - self.batch_shape[0]
            dummy_actions = self.__class__.make_dummy_actions((n, self.batch_shape[1]))
            self.batch_shape = (self.batch_shape[0] + n, self.batch_shape[1])
            self.tensor = torch.cat(
                (self.tensor, dummy_actions.tensor), dim=0
            )
        else:
            raise NotImplementedError(
                "extend_with_dummy_actions is only implemented for bi-dimensional actions."
            )

    def compare(self, other: ActionsTensor) -> BoolTensor:
        """Compares the actions to a tensor of actions.
        Args:
            other: tensor of actions
        Returns:
            boolean tensor of shape batch_shape indicating whether the actions are equal
        """
        out = self.tensor == other
        action_ndim = len(self.__class__.action_shape)
        for _ in range(action_ndim):
            out = out.all(dim=-1)
        return out

    @property
    def is_dummy(self) -> BoolTensor:
        """Returns a boolean tensor indicating whether the actions are dummy actiosn."""
        dummy_actions_tensor = self.__class__.dummy_action.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.action_shape))
        )
        return self.compare(dummy_actions_tensor)

    @property
    def is_exit(self) -> BoolTensor:
        """Returns a boolean tensor indicating whether the actions are exit actions."""
        exit_actions_tensor = self.__class__.exit_action.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.action_shape))
        )
        return self.compare(exit_actions_tensor)
