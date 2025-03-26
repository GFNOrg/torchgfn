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
        features_dim: Dimension of node/edge features
        tensor: TensorDict containing:
            - action_type: Type of action (GraphActionType)
            - features: Features for nodes/edges
            - edge_index: Source/target nodes for edges
    """

    features_dim: ClassVar[int]

    def __init__(self, tensor: TensorDict):
        """Initializes a GraphAction object.

        Args:
            action: a GraphActionType indicating the type of action.
            features: a tensor of shape (batch_shape, feature_shape) representing the features
                of the nodes or of the edges, depending on the action type. In case of EXIT
                action, this can be None.
            edge_index: an tensor of shape (batch_shape, 2) representing the edge to add.
                This must defined if and only if the action type is GraphActionType.AddEdge.
        """
        self._batch_shape = tensor["action_type"].shape

        features = tensor.get("features", None)
        if features is None:
            assert torch.all(
                torch.logical_or(
                    tensor["action_type"] == GraphActionType.EXIT,
                    tensor["action_type"] == GraphActionType.DUMMY,
                )
            )
            features = torch.zeros((*self.batch_shape, self.features_dim))
        edge_index = tensor.get("edge_index", None)
        if edge_index is None:
            assert torch.all(tensor["action_type"] != GraphActionType.ADD_EDGE)
            edge_index = torch.zeros((*self.batch_shape, 2), dtype=torch.long)

        self.tensor = TensorDict(
            {
                "action_type": tensor["action_type"],
                "features": features,
                "edge_index": edge_index,
            },
            batch_size=self.batch_shape,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> GraphActions:
        """Converts the action from discrete space to graph action space.

        This method maps discrete action indices to specific graph operations:
        - GraphActionType.ADD_EDGE: Add an edge between specific nodes
        - GraphActionType.EXIT: Terminate trajectory
        - GraphActionType.DUMMY: No-op action (for padding)

        Args:
            states: Current states batch
            actions: Discrete actions to convert

        Returns:
            Equivalent actions in the GraphActions format
        """
        # TODO: factor out into utility function.
        action_tensor = tensor.squeeze(-1).clone()
        action_type = torch.where(
            action_tensor == self.n_actions - 1,
            GraphActionType.EXIT,
            GraphActionType.ADD_EDGE,
        )
        action_type[action_tensor == self.n_actions] = GraphActionType.DUMMY

        # Convert action indices to source-target node pairs
        ei0, ei1 = cls.get_edge_indices(tensor.shape[0], is_directed, tensor.device)

        # Adds -1 "edge" representing exit, -2 "edge" representing dummy.
        ei0 = torch.cat((ei0, torch.tensor([-1, -2])), dim=0).to(tensor.device)
        ei1 = torch.cat((ei1, torch.tensor([-1, -2])), dim=0).to(tensor.device)

        # Indexes either the second last element (exit) or la
        # action_tensor[action_tensor >= (self.n_actions - 1)] = 0
        ei0, ei1 = ei0[action_tensor], ei1[action_tensor]

        out = GraphActions(
            TensorDict(
                {
                    "action_type": action_type,
                    "features": torch.ones(action_tensor.shape + (1,)),
                    "edge_index": torch.stack([ei0, ei1], dim=-1),
                },
                batch_size=action_tensor.shape,
            ).to(tensor.device)
        )
        return out
    
    @staticmethod
    def get_edge_indices(
        n_nodes: int,
        is_directed: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the source and target node indices for the edges.

        Args:
            n_nodes: The number of nodes in the graph.
            is_directed: Whether the graph is directed.
            device: The device to run the computation on.

        Returns:
            A tuple of two tensors, the source and target node indices.
        """
        if is_directed:
            # Upper triangle.
            i_up, j_up = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
            # Lower triangle.
            i_lo, j_lo = torch.tril_indices(n_nodes, n_nodes, offset=-1, device=device)

            ei0 = torch.cat([i_up, i_lo])  # Combine them
            ei1 = torch.cat([j_up, j_lo])
        else:
            ei0, ei1 = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)

        return ei0, ei1

    @classmethod
    def to_tensor(cls, actions: GraphActions) -> torch.Tensor:
        """Converts the actions to a flat tensor.

        Args:
            actions: The actions to convert.

        Returns:
            The tensor representation of the actions.
        """
        ... # TODO: implement this.

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    def __repr__(self):
        return f"""GraphAction object with {self.batch_shape} actions."""

    def _compare(self, other: GraphActions) -> torch.Tensor:
        """Compares the actions to another GraphAction object.

        Args:
            other: GraphAction object to compare.

        Returns: boolean tensor of shape batch_shape indicating whether the actions are equal.
        """
        action_compare = torch.all(
            self.tensor["action_type"] == other.tensor["action_type"]
        )
        exit_compare = (
            torch.all(self.tensor["features"] == other.tensor["features"])
            | action_compare
            == GraphActionType.EXIT
        )
        edge_compare = (action_compare != GraphActionType.ADD_EDGE) | (
            torch.all(self.tensor["edge_index"] == other.tensor["edge_index"])
        )
        return action_compare & exit_compare & edge_compare

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
        return self.tensor["action_type"]

    @property
    def features(self) -> torch.Tensor:
        """Returns the features tensor."""
        return self.tensor["features"]

    @property
    def edge_index(self) -> torch.Tensor:
        """Returns the edge index tensor."""
        return self.tensor["edge_index"]

    @classmethod
    def make_dummy_actions(cls, batch_shape: tuple[int]) -> GraphActions:
        """Creates a GraphActions object of dummy actions with the given batch shape."""
        return cls(
            TensorDict(
                {
                    "action_type": torch.full(
                        batch_shape, fill_value=GraphActionType.DUMMY
                    ),
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
                    "action_type": torch.full(
                        batch_shape, fill_value=GraphActionType.EXIT
                    ),
                },
                batch_size=batch_shape,
            )
        )

    def extend(self, other: Actions) -> None:
        """Extends an Actions instance with another Actions instance."""
        super().extend(other)
        self._batch_shape = self.tensor.batch_size
