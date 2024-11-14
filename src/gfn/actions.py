from __future__ import annotations  # This allows to use the class name in type hints

from abc import ABC
import enum
from math import prod
from typing import ClassVar, Optional, Sequence

import torch


class Actions(ABC):
    """Base class for actions for all GFlowNet environments.

    Each environment needs to subclass this class. A generic subclass for discrete
    actions with integer indices is provided Note that all actions need to have the
    same shape.

    Attributes:
        tensor: a batch of actions with shape (*batch_shape, *actions_ndims).
        batch_shape: the batch_shape from the input tensor.
    """

    # The following class variable represents the shape of a single action.
    action_shape: ClassVar[tuple[int, ...]]  # All actions need to have the same shape.
    # The following class variable is padded to shorter trajectories.
    dummy_action: ClassVar[torch.Tensor]  # Dummy action for the environment.
    # The following class variable corresponds to $s \rightarrow s_f$ transitions.
    exit_action: ClassVar[torch.Tensor]  # Action to exit the environment.

    def __init__(self, tensor: torch.Tensor):
        """Initialize actions from a tensor.

        Args:
            tensor: tensors representing a batch of actions with shape (*batch_shape, *action_shape).
        """
        assert (
            tensor.shape[-len(self.action_shape) :] == self.action_shape
        ), f"Batched actions tensor has shape {tensor.shape}, but the expected action shape is {self.action_shape}."

        self.tensor = tensor
        self.batch_shape = tuple(self.tensor.shape)[: -len(self.action_shape)]

    @classmethod
    def make_dummy_actions(cls, batch_shape: tuple[int]) -> Actions:
        """Creates an Actions object of dummy actions with the given batch shape."""
        action_ndim = len(cls.action_shape)
        tensor = cls.dummy_action.repeat(*batch_shape, *((1,) * action_ndim))
        return cls(tensor)

    @classmethod
    def make_exit_actions(cls, batch_shape: tuple[int]) -> Actions:
        """Creates an Actions object of exit actions with the given batch shape."""
        action_ndim = len(cls.action_shape)
        tensor = cls.exit_action.repeat(*batch_shape, *((1,) * action_ndim))
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

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], actions: Actions
    ) -> None:
        """Set particular actions of the batch."""
        self.tensor[index] = actions.tensor

    @classmethod
    def stack(cls, actions_list: list[Actions]) -> Actions:
        """Stacks a list of Actions objects into a single Actions object.

        The individual actions need to have the same batch shape. An example application
        is when the individual actions represent per-step actions of a batch of
        trajectories (in which case, the common batch_shape would be (n_trajectories,),
        and the resulting Actions object would have batch_shape (n_steps,
        n_trajectories).
        """
        actions_tensor = torch.stack(
            [actions.tensor for actions in actions_list], dim=0
        )
        return cls(actions_tensor)

    def extend(self, other: Actions) -> None:
        """Collates to another Actions object of the same batch shape."""
        if len(self.batch_shape) == len(other.batch_shape) == 1:
            self.batch_shape = (self.batch_shape[0] + other.batch_shape[0],)
            self.tensor = torch.cat((self.tensor, other.tensor), dim=0)
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
            self.tensor = torch.cat((self.tensor, other.tensor), dim=1)
        else:
            raise NotImplementedError(
                "extend is only implemented for bi-dimensional actions."
            )

    def extend_with_dummy_actions(self, required_first_dim: int) -> None:
        """Extends an Actions instance along the first dimension with dummy actions.

        The Actions instance batch_shape must be 2-dimensional. This is used to pad
        trajectories actions.

        Args:
            required_first_dim: the target size of the first dimension post expansion.
        """
        if len(self.batch_shape) == 2:
            if self.batch_shape[0] >= required_first_dim:
                return
            n = required_first_dim - self.batch_shape[0]
            dummy_actions = self.__class__.make_dummy_actions((n, self.batch_shape[1]))
            self.batch_shape = (self.batch_shape[0] + n, self.batch_shape[1])
            self.tensor = torch.cat((self.tensor, dummy_actions.tensor), dim=0)
        else:
            raise NotImplementedError(
                "extend_with_dummy_actions is only implemented for bi-dimensional actions."
            )

    def compare(self, other: torch.Tensor) -> torch.Tensor:
        """Compares the actions to a tensor of actions.

        Args:
            other: tensor of actions to compare, with shape (*batch_shape, *action_shape).

        Returns: boolean tensor of shape batch_shape indicating whether the actions are
            equal.
        """
        assert (
            other.shape == self.batch_shape + self.action_shape
        ), f"Expected shape {self.batch_shape + self.action_shape}, got {other.shape}."
        out = self.tensor == other
        n_batch_dims = len(self.batch_shape)

        # Flattens all action dims, which we reduce all over.
        out = out.flatten(start_dim=n_batch_dims).all(dim=-1)

        assert out.dtype == torch.bool and out.shape == self.batch_shape
        return out

    @property
    def is_dummy(self) -> torch.Tensor:
        """Returns a boolean tensor of shape `batch_shape` indicating whether the actions are dummy actions."""
        dummy_actions_tensor = self.__class__.dummy_action.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.action_shape))
        )
        return self.compare(dummy_actions_tensor)

    @property
    def is_exit(self) -> torch.Tensor:
        """Returns a boolean tensor of shape `batch_shape` indicating whether the actions are exit actions."""
        exit_actions_tensor = self.__class__.exit_action.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.action_shape))
        )
        return self.compare(exit_actions_tensor)


class GraphActionType(enum.Enum):
    EXIT = enum.auto()
    ADD_NODE = enum.auto()
    ADD_EDGE = enum.auto()


class GraphActions:

    nodes_features_dim: ClassVar[int]  # Dim size of the features tensor.
    edge_features_dim: ClassVar[int]  # Dim size of the edge features tensor.

    def __init__(self, action_type: GraphActionType, features: torch.Tensor, edge_index: Optional[torch.Tensor] = None):
        """Initializes a GraphAction object.
    
        Args:
            action: a GraphActionType indicating the type of action.
            features: a tensor of shape (*batch_shape, feature_shape) representing the features of the nodes or of the edges, depending on the action type
            edge_index: an tensor of shape (*batch_shape, 2) representing the edge to add.
                This must defined if and only if the action type is GraphActionType.AddEdge.
        """
        self.action_type = action_type
        batch_dim, features_dim = features.shape
        if self.action_type == GraphActionType.ADD_NODE:
            assert features_dim == self.nodes_features_dim
            assert edge_index is None
        elif self.action_type == GraphActionType.ADD_EDGE:
            assert features_dim == self.edge_features_dim
            assert edge_index is not None
            assert edge_index.shape == (2, batch_dim)
    
        self.features = features
        self.edge_index = edge_index
        self.batch_shape = (batch_dim,)

    def __repr__(self):
        return f"""GraphAction object of type {self.action_type} and features of shape {self.features.shape}."""

    @property
    def device(self) -> torch.device:
        """Returns the device of the features tensor."""
        return self.features.device

    def __len__(self) -> int:
        """Returns the number of actions in the batch."""
        return prod(self.batch_shape)

    def __getitem__(self, index: int | Sequence[int] | Sequence[bool]) -> GraphActions:
        """Get particular actions of the batch."""
        features = self.features[index]
        edge_index = self.edge_index[index] if self.edge_index is not None else None
        return GraphActions(self.action_type, features, edge_index)

    def __setitem__(self, index: int | Sequence[int] | Sequence[bool], action: GraphActions) -> None:
        """Set particular actions of the batch."""
        assert self.action_type == action.action_type
        self.features[index] = action.features
        if self.edge_index is not None:
            self.edge_index[index] = action.edge_index
    
    def compare(self, other: GraphActions) -> torch.Tensor:
        """Compares the actions to another GraphAction object.

        Args:
            other: GraphAction object to compare.

        Returns: boolean tensor of shape batch_shape indicating whether the actions are equal.
        """
        if self.action_type != other.action_type:
            return torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        out = torch.all(self.features == other.features, dim=-1)
        if self.edge_index is not None:
            out &= torch.all(self.edge_index == other.edge_index, dim=-1)
        return out

    @property
    def is_exit(self) -> torch.Tensor:
        """Returns a boolean tensor of shape `batch_shape` indicating whether the actions are exit actions."""
        return torch.full(self.batch_shape, self.action_type == GraphActionType.EXIT, dtype=torch.bool, device=self.device)


        
