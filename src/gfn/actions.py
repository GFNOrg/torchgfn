from __future__ import annotations  # This allows to use the class name in type hints

import enum
from abc import ABC
from math import prod
from typing import ClassVar, List, Sequence

import torch
from tensordict import TensorDict


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

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)[: -len(self.action_shape)]

    @classmethod
    def make_dummy_actions(cls, batch_shape: tuple[int, ...]) -> Actions:
        """Creates an Actions object of dummy actions with the given batch shape."""
        action_ndim = len(cls.action_shape)
        tensor = cls.dummy_action.repeat(*batch_shape, *((1,) * action_ndim))
        return cls(tensor)

    @classmethod
    def make_exit_actions(cls, batch_shape: tuple[int, ...]) -> Actions:
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

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> Actions:
        actions = self.tensor[index]
        return self.__class__(actions)

    def __setitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor,
        actions: Actions,
    ) -> None:
        """Set particular actions of the batch."""
        self.tensor[index] = actions.tensor

    @classmethod
    def stack(cls, actions_list: List[Actions]) -> Actions:
        """Stacks a list of Actions objects into a single Actions object.

        The individual actions need to have the same batch shape. An example application
        is when the individual actions represent per-step actions of a batch of
        trajectories (in which case, the common batch_shape would be (n_trajectories,),
        and the resulting Actions object would have batch_shape (n_steps,
        n_trajectories).
        """
        actions_tensor = torch.stack([actions.tensor for actions in actions_list], dim=0)
        return cls(actions_tensor)

    def extend(self, other: Actions) -> None:
        """Collates to another Actions object of the same batch shape."""
        if len(self.batch_shape) == len(other.batch_shape) == 1:
            self.tensor = torch.cat((self.tensor, other.tensor), dim=0)
        elif len(self.batch_shape) == len(other.batch_shape) == 2:
            self.extend_with_dummy_actions(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            other.extend_with_dummy_actions(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
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
            self.tensor = torch.cat((self.tensor, dummy_actions.tensor), dim=0)
        else:
            raise NotImplementedError(
                "extend_with_dummy_actions is only implemented for bi-dimensional actions."
            )

    def _compare(self, other: torch.Tensor) -> torch.Tensor:
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
        return self._compare(dummy_actions_tensor)

    @property
    def is_exit(self) -> torch.Tensor:
        """Returns a boolean tensor of shape `batch_shape` indicating whether the actions are exit actions."""
        exit_actions_tensor = self.__class__.exit_action.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.action_shape))
        )
        return self._compare(exit_actions_tensor)


class GraphActionType(enum.IntEnum):
    ADD_NODE = 0
    ADD_EDGE = 1
    EXIT = 2
    DUMMY = 3


class GraphActions(Actions):
    """Actions for graph-based environments.

    Each action is one of:
    - ADD_NODE: Add a node with given features
    - ADD_EDGE: Add an edge between two nodes with given features
    - EXIT: Terminate the trajectory

    Attributes:
        num_node_classes: Number of node classes
        num_edge_classes: Number of edge classes
    """

    num_node_classes: ClassVar[int]
    num_edge_classes: ClassVar[int]

    def __init__(self, tensor: TensorDict):
        """Initializes a GraphAction object.

        Args:
            tensor: A TensorDict containing the action type, node features, edge features, and edge index.
        """
        self._batch_shape = tensor["action_type"].shape
        self._tensor_dict = tensor
        self.tensor = self._to_tensor_conversion(tensor)

    @staticmethod
    def _edge_index_to_tensor(edge_index: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                edge_index[0].reshape(*edge_index[0].shape, 1),
                edge_index[1].reshape(*edge_index[1].shape, 1),
            ],
            dim=-1
        )

    @staticmethod
    def _to_tensor_conversion(tensor_dict: TensorDict) -> torch.Tensor:
        action_type = tensor_dict["action_type"].reshape(*tensor_dict["action_type"].shape, 1)
        node_class = tensor_dict["node_class"].reshape(*tensor_dict["node_class"].shape, 1)
        edge_class = tensor_dict["edge_class"].reshape(*tensor_dict["edge_class"].shape, 1)
        edge_index = tensor_dict["edge_index"].reshape(*tensor_dict["edge_index"].shape, 1)
        return torch.cat(
            [
                action_type,
                node_class,
                edge_class,
                edge_index,
            ],
            dim=-1
        )
    
    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape
    
    def __getitem__(self, index: int | slice | tuple | torch.Tensor) -> Actions:
        actions = self._tensor_dict[index]
        return self.__class__(actions)

    def __setitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor,
        actions: Actions,
    ) -> None:
        """Set particular actions of the batch."""
        self._tensor_dict[index] = actions._tensor_dict
        self.tensor = self._to_tensor_conversion(self._tensor_dict)

    def __repr__(self):
        return f"""GraphAction object with {self.batch_shape} actions."""

    @property
    def is_exit(self) -> torch.Tensor:
        """Returns a boolean tensor of shape `batch_shape` indicating whether the actions are exit actions."""
        return self.action_type == GraphActionType.EXIT

    @property
    def is_dummy(self) -> torch.Tensor:
        """Returns a boolean tensor of shape `batch_shape` indicating whether the actions are dummy actions."""
        return self.action_type == GraphActionType.DUMMY

    @property
    def action_type(self) -> torch.Tensor:
        """Returns the action type tensor."""
        return self._tensor_dict["action_type"]

    @property
    def node_class(self) -> torch.Tensor:
        """Returns the node class tensor."""
        return self._tensor_dict["node_class"]

    @property
    def edge_index(self) -> torch.Tensor:
        """Returns the edge index tensor."""
        return self._tensor_dict["edge_index"]

    @property
    def edge_class(self) -> torch.Tensor:
        """Returns the edge class tensor."""
        return self._tensor_dict["edge_class"]

    @classmethod
    def make_dummy_actions(cls, batch_shape: tuple[int]) -> GraphActions:
        """Creates a GraphActions object of dummy actions with the given batch shape."""
        # TODO: make default dtype int32 (even int8 for action_type)
        return cls(
            TensorDict(
                {
                    "action_type": torch.full(batch_shape, fill_value=GraphActionType.DUMMY),
                    "node_class": torch.zeros(batch_shape, dtype=torch.int64),
                    "edge_class": torch.zeros(batch_shape, dtype=torch.int64),
                    "edge_index": torch.zeros(batch_shape, dtype=torch.int64),
                },
                batch_size=batch_shape,
            )
        )

    @classmethod
    def make_exit_actions(cls, batch_shape: tuple[int]) -> Actions:
        """Creates an GraphActions object of exit actions with the given batch shape."""
        return cls(
            TensorDict(
                {
                    "action_type": torch.full(batch_shape, fill_value=GraphActionType.EXIT),
                    "node_class": torch.zeros(batch_shape, dtype=torch.int64),
                    "edge_class": torch.zeros(batch_shape, dtype=torch.int64),
                    "edge_index": torch.zeros(batch_shape, dtype=torch.int64),
                },
                batch_size=batch_shape,
            )
        )

    def extend(self, other: Actions) -> None:
        """Extends an Actions instance with another Actions instance."""
        super().extend(other)
        self._batch_shape = self.tensor.batch_size
