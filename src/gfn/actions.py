from __future__ import annotations  # This allows to use the class name in type hints

import enum
from abc import ABC
from math import prod
from typing import ClassVar, List, Sequence

import torch
from tensordict import TensorDict


class Actions(ABC):
    """Base class for actions, representing edges in the DAG of a GFlowNet.

    Each environment needs to define a subclass of `Actions` to represent its specific
    action space.

    Two useful subclasses of `Actions` are provided:
    - `DiscreteActions` for discrete environments, which represents actions as a tensor
      of shape (*batch_shape, *action_shape).
    - `GraphActions` for graph-based environments, which represents actions as a tensor
      of shape (*batch_shape, 4) containing the action type, node class, edge class,
      and edge index components.

    Attributes:
        tensor: Tensor of shape (*batch_shape, *action_shape) representing a batch of
            actions.
        action_shape: Class variable, a tuple defining the shape of a single action.
        dummy_action: Class variable, a tensor of shape (*action_shape,) representing
            the dummy action for padding shorter trajectories.
        exit_action: Class variable, a tensor of shape (*action_shape,) representing
            the action to transition to the sink state.
    """

    # The following class variable represents the shape of a single action.
    action_shape: ClassVar[tuple[int, ...]]  # All actions need to have the same shape.
    # The following class variable is padded to shorter trajectories.
    dummy_action: ClassVar[torch.Tensor]  # Dummy action for the environment.
    # The following class variable corresponds to $s \rightarrow s_f$ transitions.
    exit_action: ClassVar[torch.Tensor]  # Action to exit the environment.

    def __init__(self, tensor: torch.Tensor, debug: bool = False):
        """Initializes an Actions object with a batch of actions.

        Args:
            tensor: Tensor of shape (*batch_shape, *action_shape) representing a batch of
                actions.
        """
        # Debug-only validation keeps hot paths tensor-only when debug is False.
        if debug:
            if tensor.shape[-len(self.action_shape) :] != self.action_shape:
                raise ValueError(
                    f"Batched actions tensor has shape {tensor.shape}, expected {self.action_shape}."
                )

        self.tensor = tensor
        self.debug = debug

    @property
    def device(self) -> torch.device:
        """The device on which the actions are stored.

        Returns:
            The device of the underlying tensor.
        """
        return self.tensor.device

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the actions.

        Returns:
            The batch shape as a tuple.
        """
        return tuple(self.tensor.shape)[: -len(self.action_shape)]

    @classmethod
    def make_dummy_actions(
        cls,
        batch_shape: tuple[int, ...],
        device: torch.device | None = None,
        debug: bool = False,
    ) -> Actions:
        """Creates an Actions object filled with dummy actions.

        Args:
            batch_shape: Shape of the batch dimensions.
            device: The device to create the actions on.
            debug: Whether to run debug validations on the constructed Actions.

        Returns:
            An Actions object with the specified batch shape filled with dummy actions.
        """
        action_ndim = len(cls.action_shape)
        tensor = cls.dummy_action.repeat(*batch_shape, *((1,) * action_ndim))
        if device is not None:
            tensor = tensor.to(device)
        return cls(tensor, debug=debug)

    @classmethod
    def make_exit_actions(
        cls,
        batch_shape: tuple[int, ...],
        device: torch.device | None = None,
        debug: bool = False,
    ) -> Actions:
        """Creates an Actions object filled with exit actions.

        Args:
            batch_shape: Shape of the batch dimensions.
            device: The device to create the actions on.
            debug: Whether to run debug validations on the constructed Actions.

        Returns:
            An Actions object with the specified batch shape filled with exit actions.
        """
        action_ndim = len(cls.action_shape)
        tensor = cls.exit_action.repeat(*batch_shape, *((1,) * action_ndim))
        if device is not None:
            tensor = tensor.to(device)
        return cls(tensor, debug=debug)

    def __len__(self) -> int:
        """Returns the number of actions in the batch.

        Returns:
            The number of actions.
        """
        return prod(self.batch_shape)

    def __repr__(self):
        """Returns a string representation of the Actions object.

        Returns:
            A string summary of the Actions object.
        """
        parts = [
            f"{self.__class__.__name__}(",
            f"batch={self.batch_shape},",
            f"action={self.action_shape},",
            f"device={self.device})",
        ]
        return " ".join(parts)

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> Actions:
        """Returns a subset of the actions along the batch dimension.

        Args:
            index: Indices to select actions.

        Returns:
            A new Actions object with the selected actions.
        """
        actions = self.tensor[index]
        return self.__class__(actions, debug=self.debug)

    def __setitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor,
        actions: Actions,
    ) -> None:
        """Sets particular actions of the batch to a new Actions object.

        Args:
            index: Indices to set.
            actions: Actions object containing the new actions.
        """
        self.tensor[index] = actions.tensor

    @classmethod
    def stack(cls, actions_list: List[Actions], debug: bool | None = None) -> Actions:
        """Stacks a list of Actions objects along a new dimension (0).

        The individual actions need to have the same batch shape. An example application
        is when the individual actions represent per-step actions of a batch of
        trajectories (in which case, the common batch_shape would be (n_trajectories,),
        and the resulting Actions object would have batch_shape
        (n_steps, n_trajectories)).

        Args:
            actions_list: List of Actions objects to stack.

        Returns:
            A new Actions object with the stacked actions.
        """
        if debug is None:
            # Reuse caller-provided debug setting when available to keep behavior consistent.
            debug = getattr(actions_list[0], "debug", False) if actions_list else False
        debug = bool(debug)

        actions_tensor = torch.stack([actions.tensor for actions in actions_list], dim=0)
        return cls(actions_tensor, debug=debug)

    def extend(self, other: Actions) -> None:
        """Concatenates another Actions object along the final batch dimension.

        Both Actions objects must have the same number of batch dimensions, which
        should be 1 or 2.

        Args:
            other: Actions object to be concatenated to the current Actions object.
        """
        if len(self.batch_shape) == len(other.batch_shape) == 1:
            self.tensor = torch.cat((self.tensor, other.tensor), dim=0)
        elif len(self.batch_shape) == len(other.batch_shape) == 2:
            other = other.clone()  # TODO: Is there a more efficient way?
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
        actions in a batch of trajectories to a common length.

        Args:
            required_first_dim: The target size of the first dimension post expansion.
        """
        if len(self.batch_shape) == 2:
            if self.batch_shape[0] >= required_first_dim:
                return
            n = required_first_dim - self.batch_shape[0]
            dummy_actions = self.__class__.make_dummy_actions(
                (n, self.batch_shape[1]), device=self.device, debug=self.debug
            )
            self.tensor = torch.cat((self.tensor, dummy_actions.tensor), dim=0)
        else:
            raise NotImplementedError(
                "extend_with_dummy_actions is only implemented for bi-dimensional actions."
            )

    def _compare(self, other: torch.Tensor) -> torch.Tensor:
        """Compares the actions to a tensor of actions.

        Args:
            other: Tensor of actions to compare, with shape (*batch_shape, *action_shape).

        Returns:
            A boolean tensor of shape (*batch_shape,) indicating whether the actions are
            equal.
        """
        n_batch_dims = len(self.batch_shape)
        if self.debug:
            if n_batch_dims == 1:
                # other.shape can either have only the action shape, or the
                # flattened batch_shape + action_shape.
                if other.shape not in (
                    self.action_shape,
                    self.batch_shape + self.action_shape,
                ):
                    raise ValueError(
                        (
                            f"Expected shape {self.action_shape} or "
                            f"{self.batch_shape + self.action_shape}, got {other.shape}."
                        )
                    )
            else:
                # other.shape must have the full batch and action shape.
                if other.shape != self.batch_shape + self.action_shape:
                    raise ValueError(
                        (
                            f"Expected shape {self.batch_shape + self.action_shape}, "
                            f"got {other.shape}."
                        )
                    )

        out = self.tensor == other
        if len(self.action_shape) > 1:
            out = out.flatten(start_dim=n_batch_dims)
        out = out.all(dim=-1)

        if self.debug:
            if out.shape != self.batch_shape:
                raise ValueError(
                    f"Comparison output has shape {out.shape}, expected {self.batch_shape}."
                )

        return out

    @property
    def is_dummy(self) -> torch.Tensor:
        """Returns a boolean tensor indicating whether the actions are dummy actions.

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for dummy actions.
        """
        if len(self.batch_shape) == 1:
            dummy_actions_tensor = self.__class__.dummy_action
        else:
            dummy_actions_tensor = self.__class__.dummy_action.repeat(
                *self.batch_shape, *((1,) * len(self.__class__.action_shape))
            )
        return self._compare(dummy_actions_tensor)

    @property
    def is_exit(self) -> torch.Tensor:
        """Returns a boolean tensor indicating whether the actions are exit actions.

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for exit actions.
        """
        if len(self.batch_shape) == 1:
            exit_actions_tensor = self.__class__.exit_action
        else:
            exit_actions_tensor = self.__class__.exit_action.repeat(
                *self.batch_shape, *((1,) * len(self.__class__.action_shape))
            )
        return self._compare(exit_actions_tensor)

    def clone(self) -> Actions:
        """Returns a clone of the Actions object.

        Returns:
            A new Actions object with the same tensor.
        """
        return self.__class__(self.tensor.clone(), debug=self.debug)


class GraphActionType(enum.IntEnum):
    ADD_NODE = 0
    ADD_EDGE = 1
    EXIT = 2
    DUMMY = 3


class GraphActions(Actions):
    """Actions for graph-based environments.

    Each action is one of these types:
    - ADD_NODE: Add a node with given features
    - ADD_EDGE: Add an edge between two nodes with given features
    - EXIT: Terminate the trajectory

    Attributes:
        tensor: Tensor of shape (*batch_shape, 4) containing the action type, node class,
            edge class, and edge index components.
        ACTION_TYPE_KEY: Class variable, key for the action type component.
        NODE_CLASS_KEY: Class variable, key for the node class component.
        EDGE_CLASS_KEY: Class variable, key for the edge class component.
        EDGE_INDEX_KEY: Class variable, key for the edge index component.
        ACTION_INDICES: Class variable, mapping from keys to tensor indices.
    """

    ACTION_TYPE_KEY: ClassVar[str] = "action_type"
    NODE_CLASS_KEY: ClassVar[str] = "node_class"
    NODE_INDEX_KEY: ClassVar[str] = "node_index"
    EDGE_CLASS_KEY: ClassVar[str] = "edge_class"
    EDGE_INDEX_KEY: ClassVar[str] = "edge_index"

    ACTION_INDICES: ClassVar[dict[str, int]] = {
        ACTION_TYPE_KEY: 0,
        NODE_CLASS_KEY: 1,
        NODE_INDEX_KEY: 2,
        EDGE_CLASS_KEY: 3,
        EDGE_INDEX_KEY: 4,
    }

    # Required by the Actions base class for DB/SubTB style algorithms.
    action_shape = (5,)
    dummy_action = torch.tensor(
        [GraphActionType.DUMMY, -2, -2, -2, -2], dtype=torch.long
    )
    exit_action = torch.tensor([GraphActionType.EXIT, -1, -1, -1, -1], dtype=torch.long)

    def __init__(self, tensor: torch.Tensor, debug: bool = False):
        """Initializes a GraphActions object.

        Args:
            tensor: A tensor of shape (*batch_shape, 5) containing the action type,
                node class, edge class, and edge index components.
        """
        if debug:
            if tensor.shape[-1] != 5:
                raise ValueError(
                    f"Expected tensor of shape (*batch_shape, 5), got {tensor.shape}.\n"
                    "The last dimension should contain the action type, node class, node index, edge class, and edge index."
                )
        self.tensor = tensor
        self.debug = debug

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the graph actions.

        Returns:
            The batch shape as a tuple.
        """
        if self.debug:
            assert self.tensor.shape[-1] == 5
        return self.tensor.shape[:-1]

    @classmethod
    def from_tensor_dict(
        cls, tensor_dict: TensorDict, debug: bool = False
    ) -> GraphActions:
        """Creates a GraphActions object from a tensor dict.

        Args:
            tensor_dict: A TensorDict containing the action components with keys
                ACTION_TYPE_KEY, NODE_CLASS_KEY, NODE_INDEX_KEY, EDGE_CLASS_KEY, and EDGE_INDEX_KEY.

        Returns:
            A GraphActions object constructed from the tensor dict.
        """
        batch_shape = tensor_dict[cls.ACTION_TYPE_KEY].shape
        action_type = tensor_dict[cls.ACTION_TYPE_KEY].reshape(*batch_shape, 1)
        node_class = tensor_dict[cls.NODE_CLASS_KEY].reshape(*batch_shape, 1)
        node_index = tensor_dict[cls.NODE_INDEX_KEY].reshape(*batch_shape, 1)
        edge_class = tensor_dict[cls.EDGE_CLASS_KEY].reshape(*batch_shape, 1)
        edge_index = tensor_dict[cls.EDGE_INDEX_KEY].reshape(*batch_shape, 1)

        return cls(
            torch.cat(
                [action_type, node_class, node_index, edge_class, edge_index], dim=-1
            ),
            debug=debug,
        )

    def __repr__(self):
        """Returns a string representation of the GraphActions object.

        Returns:
            A string summary of the GraphActions object.
        """
        return f"""GraphAction object with {self.batch_shape} actions."""

    @property
    def is_exit(self) -> torch.Tensor:
        """Returns a boolean tensor indicating whether the actions are exit actions.

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for exit actions.
        """
        return self.action_type == GraphActionType.EXIT

    @property
    def is_dummy(self) -> torch.Tensor:
        """Returns a boolean tensor indicating whether the actions are dummy actions.

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for dummy actions.
        """
        return self.action_type == GraphActionType.DUMMY

    @property
    def action_type(self) -> torch.Tensor:
        """Returns the action type tensor.

        Returns:
            A tensor of shape (*batch_shape,) containing the action types.
        """
        return self.tensor[..., self.ACTION_INDICES[self.ACTION_TYPE_KEY]]

    @property
    def node_class(self) -> torch.Tensor:
        """Returns the node class tensor.

        Returns:
            A tensor of shape (*batch_shape,) containing the node classes.
        """
        return self.tensor[..., self.ACTION_INDICES[self.NODE_CLASS_KEY]]

    @property
    def node_index(self) -> torch.Tensor:
        """Returns the node index tensor.

        Returns:
            A tensor of shape (*batch_shape,) containing the node indices.
        """
        return self.tensor[..., self.ACTION_INDICES[self.NODE_INDEX_KEY]]

    @property
    def edge_class(self) -> torch.Tensor:
        """Returns the edge class tensor.

        Returns:
            A tensor of shape (*batch_shape,) containing the edge classes.
        """
        return self.tensor[..., self.ACTION_INDICES[self.EDGE_CLASS_KEY]]

    @property
    def edge_index(self) -> torch.Tensor:
        """Returns the edge index tensor.

        Returns:
            A tensor of shape (*batch_shape,) containing the edge indices.
        """
        return self.tensor[..., self.ACTION_INDICES[self.EDGE_INDEX_KEY]]

    @classmethod
    def make_dummy_actions(
        cls,
        batch_shape: tuple[int],
        device: torch.device | None = None,
        debug: bool = False,
    ) -> GraphActions:
        """Creates a GraphActions object filled with dummy actions.

        Args:
            batch_shape: Shape of the batch dimensions.
            device: The device to create the actions on.

        Returns:
            A GraphActions object with the specified batch shape filled with dummy
            actions.
        """
        tensor = torch.zeros(
            batch_shape + (5,),
            dtype=torch.long,
            device=device,
        )
        tensor[..., cls.ACTION_INDICES[cls.ACTION_TYPE_KEY]] = GraphActionType.DUMMY
        return cls(tensor, debug=debug)

    @classmethod
    def make_exit_actions(
        cls,
        batch_shape: tuple[int],
        device: torch.device | None = None,
        debug: bool = False,
    ) -> GraphActions:
        """Creates a GraphActions object filled with exit actions.

        Args:
            batch_shape: Shape of the batch dimensions.
            device: The device to create the actions on.

        Returns:
            A GraphActions object with the specified batch shape filled with exit actions.
        """
        tensor = torch.zeros(
            batch_shape + (5,),
            dtype=torch.long,
            device=device,
        )
        tensor[..., cls.ACTION_INDICES[cls.ACTION_TYPE_KEY]] = GraphActionType.EXIT
        return cls(tensor, debug=debug)

    @classmethod
    def edge_index_action_to_src_dst(
        cls, edge_index_action: torch.Tensor, n_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the edge index action to source and destination node indices."""
        raise NotImplementedError("Not implemented.")
