from __future__ import annotations  # This allows to use the class name in type hints

from abc import ABC
from copy import deepcopy
from math import prod
from typing import (
    Callable,
    ClassVar,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData
from torch_geometric.data.data import BaseData

from gfn.actions import GraphActionType


class States(ABC):
    """Base class for states, seen as nodes of the DAG.

    For each environment, a States subclass is needed. A `States` object
    is a collection of multiple states (nodes of the DAG). A tensor representation
    of the states is required for batching. If a state is represented with a tensor
    of shape (*state_shape), a batch of states is represented with a States object,
    with the attribute `tensor` of shape `(*batch_shape, *state_shape)`. Other
    representations are possible (e.g. state as string, numpy array, graph, etc...),
    but these representations cannot be batched.

    If the environment's action space is discrete (i.e. the environment subclasses
    `DiscreteEnv`), then each `States` object is also endowed with a `forward_masks` and
    `backward_masks` boolean attributes representing which actions are allowed at each
    state. This makes it possible to instantly access the allowed actions at each state,
    without having to call the environment's `is_action_valid` method. Put different,
    `is_action_valid` for such environments, directly calls the masks. This is handled
    in the DiscreteState subclass.

    A `batch_shape` attribute is also required, to keep track of the batch dimension.
    A trajectory can be represented by a States object with `batch_shape = (n_states,)`.
    Multiple trajectories can be represented by a States object with
    `batch_shape = (n_states, n_trajectories)`.

    Because multiple trajectories can have different lengths, batching requires
    appending a dummy tensor to trajectories that are shorter than the longest
    trajectory. The dummy state is the $s_f$ attribute of the environment
    (e.g. `[-1, ..., -1]`, or `[-inf, ..., -inf]`, etc...). Which is never processed,
    and is used to pad the batch of states only.

    Attributes:
        tensor: Tensor representing a batch of states.
        _batch_shape: Sizes of the batch dimensions.
        _log_rewards: Stores the log rewards of each state.
    """

    state_shape: ClassVar[tuple[int, ...]]
    s0: ClassVar[torch.Tensor | GeometricData]
    sf: ClassVar[torch.Tensor | GeometricData]

    make_random_states_tensor: Callable = lambda x: (_ for _ in ()).throw(
        NotImplementedError(
            "The environment does not support initialization of random states."
        )
    )

    def __init__(self, tensor: torch.Tensor):
        """Initialize the State container with a batch of states.
        Args:
            tensor: Tensor of shape (*batch_shape, *state_shape) representing a batch of states.
        """
        assert self.s0.shape == self.state_shape
        assert self.sf.shape == self.state_shape
        assert tensor.shape[-len(self.state_shape) :] == self.state_shape

        self.tensor = tensor
        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)[: -len(self.state_shape)]

    @classmethod
    def from_batch_shape(
        cls,
        batch_shape: int | tuple[int, ...],
        random: bool = False,
        sink: bool = False,
    ) -> States | GraphStates:
        """Create a States object with the given batch shape.

        By default, all states are initialized to $s_0$, the initial state. Optionally,
        one can initialize random state, which requires that the environment implements
        the `make_random_states_tensor` class method. Sink can be used to initialize
        states at $s_f$, the sink state. Both random and sink cannot be True at the
        same time.

        Args:
            batch_shape: Shape of the batch dimensions.
            random (optional): Initalize states randomly.
            sink (optional): States initialized with s_f (the sink state).

        Raises:
            ValueError: If both Random and Sink are True.
        """
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        if random and sink:
            raise ValueError("Only one of `random` and `sink` should be True.")

        if random:
            tensor = cls.make_random_states_tensor(batch_shape)
        elif sink:
            tensor = cls.make_sink_states_tensor(batch_shape)
        else:
            tensor = cls.make_initial_states_tensor(batch_shape)
        return cls(tensor)

    @classmethod
    def make_initial_states_tensor(cls, batch_shape: tuple[int, ...]) -> torch.Tensor:
        """Makes a tensor with a `batch_shape` of states consisting of $s_0`$s."""
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        if isinstance(cls.s0, torch.Tensor):
            return cls.s0.repeat(*batch_shape, *((1,) * state_ndim))
        else:
            raise NotImplementedError(
                f"make_initial_states_tensor is not implemented by default for {cls.__name__}"
            )

    @classmethod
    def make_sink_states_tensor(cls, batch_shape: tuple[int, ...]) -> torch.Tensor:
        """Makes a tensor with a `batch_shape` of states consisting of $s_f$s."""
        state_ndim = len(cls.state_shape)
        assert cls.sf is not None and state_ndim is not None
        if isinstance(cls.sf, torch.Tensor):
            return cls.sf.repeat(*batch_shape, *((1,) * state_ndim))
        else:
            raise NotImplementedError(
                f"make_sink_states_tensor is not implemented by default for {cls.__name__}"
            )

    def __len__(self) -> int:
        return prod(self.batch_shape)

    def __repr__(self) -> str:
        parts = [
            f"{self.__class__.__name__}(",
            f"batch={self.batch_shape},",
            f"state={self.state_shape},",
            f"dev={self.device})",
        ]
        return " ".join(parts)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> States:
        """Access particular states of the batch."""
        return self.__class__(self.tensor[index])

    def __setitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor,
        states: States,
    ) -> None:
        """Set particular states of the batch."""
        self.tensor[index] = states.tensor

    def clone(self) -> States:
        """Returns a *detached* clone of the current instance using deepcopy."""
        return deepcopy(self)

    def flatten(self) -> States:
        """Flatten the batch dimension of the states.

        Useful for example when extracting individual states from trajectories.
        """
        states = self.tensor.view(-1, *self.state_shape)
        return self.__class__(states)

    def extend(self, other: States) -> None:
        """Concatenates to another States object along the final batch dimension.

        Both States objects must have the same number of batch dimensions, which
        should be 1 or 2.

        Args:
            other (States): Batch of states to concatenate to.

        Raises:
            ValueError: if `self.batch_shape != other.batch_shape` or if
            `self.batch_shape != (1,) or (2,)`.
        """
        if len(other.batch_shape) == len(self.batch_shape) == 1:
            # This corresponds to adding a state to a trajectory
            self.tensor = torch.cat((self.tensor, other.tensor), dim=0)

        elif len(other.batch_shape) == len(self.batch_shape) == 2:
            # This corresponds to adding a trajectory to a batch of trajectories
            self.extend_with_sf(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            other.extend_with_sf(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            self.tensor = torch.cat((self.tensor, other.tensor), dim=1)
        else:
            raise ValueError(
                f"extend is not implemented for batch shapes {self.batch_shape} and {other.batch_shape}"
            )

    def extend_with_sf(self, required_first_dim: int) -> None:
        """Extends a 2-dimensional batch of states along the first batch dimension.

        Given a batch of states (i.e. of `batch_shape=(a, b)`), extends `a` it to a
        States object of `batch_shape = (required_first_dim, b)`, by adding the
        required number of $s_f$ tensors. This is useful to extend trajectories of
        different lengths.

        Args:
            required_first_dim: The size of the first batch dimension post-expansion.
        """
        if len(self.batch_shape) == 2 and isinstance(self.__class__.sf, torch.Tensor):
            if self.batch_shape[0] >= required_first_dim:
                return
            self.tensor = torch.cat(
                (
                    self.tensor,
                    self.__class__.sf.repeat(
                        required_first_dim - self.batch_shape[0], self.batch_shape[1], 1
                    ),
                ),
                dim=0,
            )
        else:
            raise ValueError(
                f"extend_with_sf is not implemented for graph states nor for batch shapes {self.batch_shape}"
            )

    def _compare(self, other: torch.Tensor) -> torch.Tensor:
        """Computes elementwise equality between state tensor with an external tensor.

        Args:
            other: Tensor with shape (*batch_shape, *state_shape) representing states to compare to.

        Returns a tensor of booleans with shape `batch_shape` indicating whether the states are equal
            to the states in self.
        """
        assert other.shape == self.batch_shape + self.state_shape
        out = self.tensor == other
        state_ndim = len(self.__class__.state_shape)
        for _ in range(state_ndim):
            out = out.all(dim=-1)

        assert out.shape == self.batch_shape
        return out

    @property
    def is_initial_state(self) -> torch.Tensor:
        """Returns a tensor of shape `batch_shape` that is True for states that are $s_0$ of the DAG."""
        if isinstance(self.__class__.s0, torch.Tensor):
            source_states_tensor = self.__class__.s0.repeat(
                *self.batch_shape, *((1,) * len(self.__class__.state_shape))
            )
        else:
            raise NotImplementedError(
                f"is_initial_state is not implemented by default for {self.__class__.__name__}"
            )
        return self._compare(source_states_tensor)

    @property
    def is_sink_state(self) -> torch.Tensor:
        """Returns a tensor of shape `batch_shape` that is True for states that are $s_f$ of the DAG."""
        # TODO: self.__class__.sf == self.tensor -- or something similar?
        if isinstance(self.__class__.sf, torch.Tensor):
            sink_states = self.__class__.sf.repeat(
                *self.batch_shape, *((1,) * len(self.__class__.state_shape))
            ).to(self.tensor.device)
        else:
            raise NotImplementedError(
                f"is_sink_state is not implemented by default for {self.__class__.__name__}"
            )
        return self._compare(sink_states)

    @property
    def log_rewards(self) -> torch.Tensor | None:
        """Returns the log rewards of the states as tensor of shape `batch_shape`."""
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: torch.Tensor) -> None:
        """Sets the log rewards of the states.

        Args:
            log_rewards: Tensor of shape `batch_shape` representing the log rewards of the states.
        """
        assert tuple(log_rewards.shape) == self.batch_shape
        self._log_rewards = log_rewards

    def sample(self, n_samples: int) -> States:
        """Samples a subset of the States object."""
        return self[torch.randperm(len(self))[:n_samples]]

    @classmethod
    def stack(cls, states: Sequence[States]) -> States:
        """Given a list of states, stacks them along a new dimension (0)."""
        state_example = states[0]
        assert all(
            state.batch_shape == state_example.batch_shape for state in states
        ), "All states must have the same batch_shape"

        stacked_states = state_example.from_batch_shape((0, 0))  # Empty.
        stacked_states.tensor = torch.stack([s.tensor for s in states], dim=0)
        if state_example._log_rewards:
            log_rewards = []
            for s in states:
                if s._log_rewards is None:
                    raise ValueError("Some states have no log rewards.")
                log_rewards.append(s._log_rewards)
            stacked_states._log_rewards = torch.stack(log_rewards, dim=0)

        return stacked_states


class DiscreteStates(States, ABC):
    """Base class for states of discrete environments.

    States are endowed with a `forward_masks` and `backward_masks`: boolean attributes
    representing which actions are allowed at each state. This is the mechanism by
    which all elements of the library (including an environment's `is_action_valid`
    method) verifies the allowed actions at each state.

    Attributes:
        forward_masks: A boolean tensor of allowable forward policy actions.
        backward_masks:  A boolean tensor of allowable backward policy actions.
    """

    n_actions: ClassVar[int]
    device: ClassVar[torch.device]

    def __init__(
        self,
        tensor: torch.Tensor,
        forward_masks: Optional[torch.Tensor] = None,
        backward_masks: Optional[torch.Tensor] = None,
    ) -> None:
        """Initalize a DiscreteStates container with a batch of states and masks.
        Args:
            tensor: A tensor with shape (*batch_shape, *state_shape) representing a batch of states.
            forward_masks: Optional boolean tensor tensor with shape (*batch_shape, n_actions) of
                allowable forward policy actions.
            backward_masks: Optional boolean tensor tensor with shape (*batch_shape, n_actions) of
                allowable backward policy actions.
        """
        super().__init__(tensor)
        assert tensor.shape == self.batch_shape + self.state_shape

        # In the usual case, no masks are provided and we produce these defaults.
        # Note: this **must** be updated externally by the env.
        if forward_masks is None:
            forward_masks = torch.ones(
                (*self.batch_shape, self.__class__.n_actions),
                dtype=torch.bool,
                device=self.__class__.device,
            )
        if backward_masks is None:
            backward_masks = torch.ones(
                (*self.batch_shape, self.__class__.n_actions - 1),
                dtype=torch.bool,
                device=self.__class__.device,
            )

        self.forward_masks: torch.Tensor = forward_masks
        self.backward_masks: torch.Tensor = backward_masks
        assert self.forward_masks.shape == (*self.batch_shape, self.n_actions)
        assert self.backward_masks.shape == (*self.batch_shape, self.n_actions - 1)

    def clone(self) -> DiscreteStates:
        """Returns a clone of the current instance."""
        return self.__class__(
            self.tensor.detach().clone(),  # TODO: Are States carrying gradients?
            self.forward_masks,
            self.backward_masks,
        )

    def _check_both_forward_backward_masks_exist(self):
        assert self.forward_masks is not None and self.backward_masks is not None

    def __repr__(self):
        """Returns a detailed string representation of the DiscreteStates object."""
        parts = [
            f"{self.__class__.__name__}(",
            f"batch={self.batch_shape},",
            f"state={self.state_shape},",
            f"actions={self.n_actions},",
            f"dev={self.device},",
            f"masks={tuple(self.forward_masks.shape)})",
        ]
        return " ".join(parts)

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> DiscreteStates:
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        out = self.__class__(states, forward_masks, backward_masks)
        return out

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], states: DiscreteStates
    ) -> None:
        super().__setitem__(index, states)
        self._check_both_forward_backward_masks_exist()
        self.forward_masks[index] = states.forward_masks
        self.backward_masks[index] = states.backward_masks

    def flatten(self) -> DiscreteStates:
        states = self.tensor.view(-1, *self.state_shape)
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks.view(-1, self.forward_masks.shape[-1])
        backward_masks = self.backward_masks.view(-1, self.backward_masks.shape[-1])
        return self.__class__(states, forward_masks, backward_masks)

    def extend(self, other: DiscreteStates) -> None:
        super().extend(other)
        self.forward_masks = torch.cat(
            (self.forward_masks, other.forward_masks), dim=len(self.batch_shape) - 1
        )
        self.backward_masks = torch.cat(
            (self.backward_masks, other.backward_masks), dim=len(self.batch_shape) - 1
        )

    def extend_with_sf(self, required_first_dim: int) -> None:
        """Extends forward and backward masks along the first batch dimension.

        After extending the state along the first batch dimensions with $s_f$ by
        `required_first_dim`, also extends both forward and backward masks with ones
        along the first dimension by `required_first_dim`.

        Args:
            required_first_dim: The size of the first batch dimension post-expansion.
        """
        super().extend_with_sf(required_first_dim)

        def _extend(masks, first_dim):
            return torch.cat(
                (
                    masks,
                    torch.ones(
                        first_dim - masks.shape[0],
                        *masks.shape[1:],
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ),
                dim=0,
            )

        self.forward_masks = _extend(self.forward_masks, required_first_dim)
        self.backward_masks = _extend(self.backward_masks, required_first_dim)

    # The helper methods are convenience functions for common mask operations.
    def set_nonexit_action_masks(self, cond, allow_exit: bool):
        """Masks denoting disallowed actions according to cond, appending the exit mask.

        A convenience function for common mask operations.

        Args:
            cond: a boolean of shape (batch_shape,) + (n_actions - 1,), which
                denotes which actions are *not* allowed. For example, if a state element
                represents action count, and no action can be repeated more than 5
                times, cond might be state.tensor > 5 (assuming count starts at 0).
            allow_exit: sets whether exiting can happen at any point in the
                trajectory - if so, it should be set to True.
        """
        # Resets masks in place to prevent side-effects across steps.
        self.forward_masks[:] = True
        if allow_exit:
            exit_idx = torch.zeros(self.batch_shape + (1,)).to(cond.device)
        else:
            exit_idx = torch.ones(self.batch_shape + (1,)).to(cond.device)
        self.forward_masks[torch.cat([cond, exit_idx], dim=-1).bool()] = False

    def set_exit_masks(self, batch_idx):
        """Sets forward masks such that the only allowable next action is to exit.

        A convenience function for common mask operations.

        Args:
            batch_idx: A Boolean index along the batch dimension, along which to
                enforce exits.
        """
        self.forward_masks[batch_idx, :] = torch.cat(
            [
                torch.zeros([int(torch.sum(batch_idx).item()), *self.s0.shape]),
                torch.ones([int(torch.sum(batch_idx).item()), 1]),
            ],
            dim=-1,
        ).bool()

    def init_forward_masks(self, set_ones: bool = True):
        """Initalizes forward masks.

        A convienience function for common mask operations.

        Args:
            set_ones: if True, forward masks are initalized to all ones. Otherwise,
                they are initalized to all zeros.
        """
        shape = self.batch_shape + (self.n_actions,)
        if set_ones:
            self.forward_masks = torch.ones(shape).bool()
        else:
            self.forward_masks = torch.zeros(shape).bool()

    @classmethod
    def stack(cls, states: List[DiscreteStates]) -> DiscreteStates:
        """Stacks a list of DiscreteStates objects along a new dimension (0)."""
        out = super().stack(states)
        assert isinstance(out, DiscreteStates)
        out.forward_masks = torch.stack([s.forward_masks for s in states], dim=0)
        out.backward_masks = torch.stack([s.backward_masks for s in states], dim=0)
        return out


class GraphStates(States):
    """
    Base class for Graph as a state representation. The `GraphStates` object is a batched
    collection of multiple graph objects. The `GeometricBatch` object is used to
    represent the batch of graph objects as states.
    """

    s0: ClassVar[GeometricData]
    sf: ClassVar[GeometricData]

    def __init__(self, tensor: GeometricBatch):
        """Initialize the GraphStates with a PyG Batch object.

        Args:
            tensor: A PyG Batch object representing a batch of graphs.
        """
        self.tensor = tensor
        if not hasattr(self.tensor, "batch_shape"):
            if isinstance(self.tensor.batch_size, tuple):
                self.tensor.batch_shape = self.tensor.batch_size
            else:
                self.tensor.batch_shape = (self.tensor.batch_size,)

        if tensor.x.size(0) > 0:
            assert tensor.num_graphs == prod(tensor.batch_shape)

        self._log_rewards: Optional[torch.Tensor] = None

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Returns the batch shape as a tuple."""
        return tuple(self.tensor.batch_shape)

    @classmethod
    def make_initial_states_tensor(cls, batch_shape: int | Tuple) -> GeometricBatch:
        """Makes a batch of graphs consisting of s0 states.

        Args:
            batch_shape: Shape of the batch dimensions.

        Returns:
            A PyG Batch object containing copies of the initial state.
        """
        assert cls.s0.edge_attr is not None
        assert cls.s0.x is not None

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = int(np.prod(batch_shape))

        # Create a list of Data objects by copying s0
        data_list = [cls.s0.clone() for _ in range(num_graphs)]

        if len(data_list) == 0:  # If batch_shape is 0, create a single empty graph
            data_list = [
                GeometricData(
                    x=torch.zeros(0, cls.s0.x.size(1)),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, cls.s0.edge_attr.size(1)),
                )
            ]

        # Create a batch from the list
        batch = GeometricBatch.from_data_list(cast(List[BaseData], data_list))

        # Store the batch shape for later reference
        batch.batch_shape = tuple(batch_shape)

        return batch

    @classmethod
    def make_sink_states_tensor(cls, batch_shape: int | Tuple) -> GeometricBatch:
        """Makes a batch of graphs consisting of sf states.

        Args:
            batch_shape: Shape of the batch dimensions.

        Returns:
            A PyG Batch object containing copies of the sink state.
        """
        assert cls.sf.edge_attr is not None
        assert cls.sf.x is not None
        if cls.sf is None:
            raise NotImplementedError("Sink state is not defined")

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = int(np.prod(batch_shape))

        # Create a list of Data objects by copying sf
        data_list = [cls.sf.clone() for _ in range(num_graphs)]
        if len(data_list) == 0:  # If batch_shape is 0, create a single empty graph
            data_list = [
                GeometricData(
                    x=torch.zeros(0, cls.sf.x.size(1)),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, cls.sf.edge_attr.size(1)),
                )
            ]

        # Create a batch from the list
        batch = GeometricBatch.from_data_list(cast(List[BaseData], data_list))

        # Store the batch shape for later reference
        batch.batch_shape = batch_shape

        return batch

    @classmethod
    def make_random_states_tensor(cls, batch_shape: int | Tuple) -> GeometricBatch:
        """Makes a batch of random graph states.

        Args:
            batch_shape: Shape of the batch dimensions.

        Returns:
            A PyG Batch object containing random graph states.
        """
        assert cls.s0.edge_attr is not None
        assert cls.s0.x is not None

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = int(np.prod(batch_shape))
        device = cls.s0.x.device

        data_list = []
        for _ in range(num_graphs):
            # Create a random graph with random number of nodes
            num_nodes = np.random.randint(1, 10)

            # Create random node features
            x = torch.rand(num_nodes, cls.s0.x.size(1), device=device)

            # Create random edges (not all possible edges to keep it sparse)
            num_edges = np.random.randint(0, num_nodes * (num_nodes - 1) // 2 + 1)
            if num_edges > 0 and num_nodes > 1:
                # Generate random source and target nodes
                edge_index = torch.zeros(2, num_edges, dtype=torch.long, device=device)
                for i in range(num_edges):
                    src, dst = np.random.choice(num_nodes, 2, replace=False)
                    edge_index[0, i] = src
                    edge_index[1, i] = dst

                # Create random edge features
                edge_attr = torch.rand(
                    num_edges, cls.s0.edge_attr.size(1), device=device
                )

                data = GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                # No edges
                data = GeometricData(
                    x=x,
                    edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
                    edge_attr=torch.zeros(0, cls.s0.edge_attr.size(1), device=device),
                )

            data_list.append(data)

        if len(data_list) == 0:  # If batch_shape is 0, create a single empty graph
            data_list = [
                GeometricData(
                    x=torch.zeros(0, cls.s0.x.size(1)),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, cls.s0.edge_attr.size(1)),
                )
            ]

        # Create a batch from the list
        batch = GeometricBatch.from_data_list(cast(List[BaseData], data_list))

        # Store the batch shape for later reference
        batch.batch_shape = batch_shape

        return batch

    def __repr__(self):
        """Returns a detailed string representation of the GraphStates object."""
        parts = [
            f"{self.__class__.__name__}(",
            f"batch={self.batch_shape},",
            f"state x={self.tensor.x.shape},",
            f"state edge_index={self.tensor.edge_index.shape},",
            f"state edge_attr={self.tensor.edge_attr.shape},",
            f"actions={self.n_actions},",
            f"dev={self.device},",
            # f"masks={tuple(self.forward_masks.shape)})",  # TODO: file an issue
        ]
        return " ".join(parts)

    def __getitem__(
        self,
        index: Union[int, Sequence[int], slice, torch.Tensor, Literal[1], Tuple],
    ) -> GraphStates:
        """Get a subset of the GraphStates.

        Args:
            index: Index or indices to select.

        Returns:
            A new GraphStates object containing the selected graphs.
        """
        assert (
            self.batch_shape != ()
        ), "We can't index on a Batch with 0-dimensional batch shape."

        # Convert the index to a list of indices.
        tensor_idx = torch.arange(len(self)).view(*self.batch_shape)[index]
        new_shape = tuple(tensor_idx.shape)
        flat_idx = tensor_idx.flatten()

        # Get the selected graphs from the batch
        selected_graphs = self.tensor.index_select(flat_idx)
        if len(selected_graphs) == 0:
            assert np.prod(new_shape) == 0 and len(new_shape) > 0
            selected_graphs = [  # TODO: Is this the best way to create an empty Batch?
                GeometricData(
                    x=torch.zeros(*new_shape, self.tensor.x.size(1)),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(*new_shape, self.tensor.edge_attr.size(1)),
                )
            ]

        # Create a new batch from the selected graphs.
        # TODO: is there any downside to always using GeometricBatch even when the batch dimension is empty.
        new_batch = GeometricBatch.from_data_list(cast(List[BaseData], selected_graphs))
        new_batch.batch_shape = new_shape

        # Create a new GraphStates object
        out = self.__class__(new_batch)

        # Copy log rewards if they exist
        if self._log_rewards is not None:
            out.log_rewards = self._log_rewards[index]

        return out

    def __setitem__(
        self,
        index: Union[int, Sequence[int], slice, torch.Tensor, Literal[1], Tuple],
        graph: GraphStates,
    ) -> None:
        """Set a subset of the GraphStates.

        Args:
            index: Index or indices to set.
            graph: GraphStates object containing the new graphs.
        """
        # Convert the index to a list of indices
        batch_shape = self.batch_shape
        if isinstance(index, int) and len(batch_shape) == 1:
            indices = [index]
        else:
            tensor_idx = torch.arange(len(self)).view(*batch_shape)
            indices = (
                tensor_idx[index].flatten().tolist()
            )  # TODO: is .flatten() necessary?

        assert len(indices) == len(graph)

        # Get the data list from the current batch
        data_list = self.tensor.to_data_list()

        # Get the data list from the new graphs
        new_data_list = graph.tensor.to_data_list()

        # Replace the selected graphs
        for i, idx in enumerate(indices):
            data_list[idx] = new_data_list[i]

        # Create a new batch from the updated data list
        self.tensor = GeometricBatch.from_data_list(data_list)

        # Preserve the batch shape
        self.tensor.batch_shape = batch_shape

    @property
    def device(self) -> torch.device:
        """Returns the device of the tensor."""
        return self.tensor.x.device

    def to(self, device: torch.device) -> GraphStates:
        """Moves the GraphStates to the specified device.

        Args:
            device: The device to move to.

        Returns:
            The GraphStates object on the specified device.
        """
        self.tensor = self.tensor.to(device)
        if self._log_rewards is not None:
            self._log_rewards = self._log_rewards.to(device)
        return self

    @staticmethod
    def _clone_batch(batch: GeometricBatch) -> GeometricBatch:
        """Clones a PyG Batch object.

        Args:
            batch: The Batch object to clone.

        Returns:
            A new Batch object with the same data.
        """
        new_batch = batch.clone()
        # The Batch.clone() changes the type of the batch shape to a list
        # We need to set it back to a tuple
        new_batch.batch_shape = batch.batch_shape
        return new_batch

    def clone(self) -> GraphStates:
        """Returns a detached clone of the current instance.

        Returns:
            A new GraphStates object with the same data.
        """
        # Create a deep copy of the batch
        new_batch = self._clone_batch(self.tensor)

        # Create a new GraphStates object
        out = self.__class__(new_batch)

        # Copy log rewards if they exist
        if self._log_rewards is not None:
            out._log_rewards = self._log_rewards.clone()

        return out

    def extend(self, other: GraphStates):
        """Concatenates to another GraphStates object along the batch dimension.

        Args:
            other: GraphStates object to concatenate with.
        """
        if len(self) == 0:
            # If self is empty, just copy other
            self.tensor = self._clone_batch(other.tensor)
            if other._log_rewards is not None:
                self._log_rewards = other._log_rewards.clone()
            return

        # Get the data lists
        self_data_list = self.tensor.to_data_list()
        other_data_list = other.tensor.to_data_list()

        # Update the batch shape
        if len(self.batch_shape) == 1:
            # Create a new batch
            new_batch_shape = (self.batch_shape[0] + other.batch_shape[0],)
            self.tensor = GeometricBatch.from_data_list(self_data_list + other_data_list)
            self.tensor.batch_shape = new_batch_shape
        else:
            # Handle the case where batch_shape is (T, B)
            # and we want to concatenate along the B dimension
            assert len(self.batch_shape) == 2
            max_len = max(self.batch_shape[0], other.batch_shape[0])

            # We need to extend both batches to the same length T
            if self.batch_shape[0] < max_len:
                self_extension = self.make_sink_states_tensor(
                    (max_len - self.batch_shape[0], self.batch_shape[1])
                )
                self_data_list = self_data_list + self_extension.to_data_list()

            if other.batch_shape[0] < max_len:
                other_extension = other.make_sink_states_tensor(
                    (max_len - other.batch_shape[0], other.batch_shape[1])
                )
                other_data_list = other_data_list + other_extension.to_data_list()

            # Now both have the same length T, we can concatenate along B
            batch_shape = (max_len, self.batch_shape[1] + other.batch_shape[1])
            self.tensor = GeometricBatch.from_data_list(self_data_list + other_data_list)
            self.tensor.batch_shape = batch_shape

        # Combine log rewards if they exist
        if self._log_rewards is not None and other._log_rewards is not None:
            self.log_rewards = torch.cat([self._log_rewards, other._log_rewards], dim=0)
        elif other._log_rewards is not None:
            self.log_rewards = other._log_rewards.clone()

    def _compare(self, other: GeometricData) -> torch.Tensor:
        """Compares the current batch of graphs with another graph.

        Args:
            other: A PyG Data object to compare with.

        Returns:
            A boolean tensor indicating which graphs in the batch are equal to other.
        """
        out = torch.zeros(len(self), dtype=torch.bool, device=self.device)

        # Get the data list from the batch
        data_list = self.tensor.to_data_list()

        assert other.edge_index is not None  # TODO: is allowing None here a good idea?
        assert other.edge_attr is not None  #
        assert other.num_nodes is not None  #

        for i, data in enumerate(data_list):
            # Check if the number of nodes is the same
            if data.num_nodes != other.num_nodes:
                continue

            # Check if node features are the same
            if not torch.all(data.x == other.x):
                continue

            # Check if the number of edges is the same
            if data.edge_index.size(1) != other.edge_index.size(1):
                continue

            # Check if edge indices are the same (this is more complex due to potential reordering)
            # We'll use a simple heuristic: sort edges and compare
            data_edges = data.edge_index.t().tolist()
            other_edges = other.edge_index.t().tolist()
            data_edges.sort()
            other_edges.sort()
            if data_edges != other_edges:
                continue

            # Check if edge attributes are the same (after sorting)
            data_edge_attr = data.edge_attr[
                torch.argsort(data.edge_index[0] * data.num_nodes + data.edge_index[1])
            ]
            other_edge_attr = other.edge_attr[
                torch.argsort(
                    other.edge_index[0] * other.num_nodes + other.edge_index[1]
                )
            ]
            if not torch.all(data_edge_attr == other_edge_attr):
                continue

            # If all checks pass, the graphs are equal
            out[i] = True

        return out.view(self.batch_shape)

    @property
    def is_sink_state(self) -> torch.Tensor:
        """Returns a tensor that is True for states that are sf."""
        return self._compare(self.sf)

    @property
    def is_initial_state(self) -> torch.Tensor:
        """Returns a tensor that is True for states that are s0."""
        return self._compare(self.s0)

    @classmethod
    def stack(cls, states: List[GraphStates]) -> GraphStates:
        """Given a list of states, stacks them along a new dimension (0).

        Args:
            states: List of GraphStates objects to stack.

        Returns:
            A new GraphStates object with the stacked states.
        """
        # Check that all states have the same batch shape
        state_batch_shape = states[0].batch_shape
        assert all(state.batch_shape == state_batch_shape for state in states)

        # Get all data lists
        all_data_lists = [state.tensor.to_data_list() for state in states]

        # Flatten the list of lists
        flat_data_list = [data for data_list in all_data_lists for data in data_list]

        # Create a new batch
        batch = GeometricBatch.from_data_list(flat_data_list)

        # Set the batch shape
        batch.batch_shape = (len(states),) + state_batch_shape

        # Create a new GraphStates object
        out = cls(batch)

        # Stack log rewards if they exist
        if all(state._log_rewards is not None for state in states):
            log_rewards = []
            for state in states:
                log_rewards.append(state._log_rewards)
            out.log_rewards = torch.stack(log_rewards)

        return out

    @property
    def forward_masks(self) -> dict:
        """Returns masks denoting allowed forward actions.

        Returns:
            A dictionary containing masks for different action types.
        """
        # Get the data list from the batch
        data_list = self.tensor.to_data_list()
        N = self.tensor.x.size(0)

        # Initialize masks
        action_type_mask = torch.ones(
            self.batch_shape + (3,), dtype=torch.bool, device=self.device
        )
        features_mask = torch.ones(
            self.batch_shape + (self.tensor.x.size(1),),
            dtype=torch.bool,
            device=self.device,
        )
        edge_index_masks = torch.ones(
            (len(data_list), N, N), dtype=torch.bool, device=self.device
        )

        # For each graph in the batch
        for i, data in enumerate(data_list):
            # Flatten the batch index
            flat_idx = i

            # ADD_NODE is always allowed
            action_type_mask[flat_idx, GraphActionType.ADD_NODE] = True

            # ADD_EDGE is allowed only if there are at least 2 nodes
            assert data.num_nodes is not None
            action_type_mask[flat_idx, GraphActionType.ADD_EDGE] = data.num_nodes > 1

            # EXIT is always allowed
            action_type_mask[flat_idx, GraphActionType.EXIT] = True

        # Create edge_index mask as a dense representation (NxN matrix)
        start_n = 0
        for i, data in enumerate(data_list):
            # For each graph, create a dense mask for potential edges
            n = data.num_nodes
            assert n is not None
            edge_mask = torch.ones((n, n), dtype=torch.bool, device=self.device)
            # Remove self-loops by setting diagonal to False
            edge_mask.fill_diagonal_(False)

            # Exclude existing edges
            if data.edge_index.size(1) > 0:
                for j in range(data.edge_index.size(1)):
                    src, dst = data.edge_index[0, j], data.edge_index[1, j]
                    edge_mask[src, dst] = False

            edge_index_masks[i, start_n : (start_n + n), start_n : (start_n + n)] = (
                edge_mask
            )
            start_n += n

            # Update ADD_EDGE mask based on whether there are valid edges to add
            action_type_mask[flat_idx, GraphActionType.ADD_EDGE] &= edge_mask.any()

        return {
            "action_type": action_type_mask,
            "features": features_mask,
            "edge_index": edge_index_masks,
        }

    @property
    def backward_masks(self) -> dict:
        """Returns masks denoting allowed backward actions.

        Returns:
            A dictionary containing masks for different action types.
        """
        # Get the data list from the batch
        data_list = self.tensor.to_data_list()
        N = self.tensor.x.size(0)

        # Initialize masks
        action_type_mask = torch.ones(
            self.batch_shape + (3,), dtype=torch.bool, device=self.device
        )
        features_mask = torch.ones(
            self.batch_shape + (self.tensor.x.size(1),),
            dtype=torch.bool,
            device=self.device,
        )
        edge_index_masks = torch.zeros(
            (len(data_list), N, N), dtype=torch.bool, device=self.device
        )

        # For each graph in the batch
        for i, data in enumerate(data_list):
            assert data.num_nodes is not None

            # Flatten the batch index
            flat_idx = i

            # ADD_NODE is allowed if there's at least one node (can remove a node)
            action_type_mask[flat_idx, GraphActionType.ADD_NODE] = data.num_nodes >= 1

            # ADD_EDGE is allowed if there's at least one edge (can remove an edge)
            action_type_mask[flat_idx, GraphActionType.ADD_EDGE] = (
                data.edge_index.size(1) > 0
            )

            # EXIT is allowed if there's at least one node
            action_type_mask[flat_idx, GraphActionType.EXIT] = data.num_nodes >= 1

        # Create edge_index mask for backward actions (existing edges that can be removed)
        start_n = 0
        for i, data in enumerate(data_list):
            # For backward actions, we can only remove existing edges
            n = data.num_nodes
            assert n is not None
            edge_mask = torch.zeros((n, n), dtype=torch.bool, device=self.device)

            # Include only existing edges
            if data.edge_index.size(1) > 0:
                for j in range(data.edge_index.size(1)):
                    src, dst = (
                        data.edge_index[0, j].item(),
                        data.edge_index[1, j].item(),
                    )
                    edge_mask[src, dst] = True

            edge_index_masks[i, start_n : (start_n + n), start_n : (start_n + n)] = (
                edge_mask
            )
            start_n += n

        return {
            "action_type": action_type_mask,
            "features": features_mask,
            "edge_index": edge_index_masks,
        }

    def flatten(self) -> None:
        raise NotImplementedError

    def extend_with_sf(self, required_first_dim: int) -> None:
        raise NotImplementedError