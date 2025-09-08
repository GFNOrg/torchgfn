from __future__ import annotations  # This allows to use the class name in type hints

from abc import ABC
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
from tensordict import TensorDict
from torch_geometric.data import Data as GeometricData

from gfn.actions import GraphActions, GraphActionType
from gfn.utils.common import ensure_same_device
from gfn.utils.graphs import GeometricBatch, get_edge_indices


class States(ABC):
    r"""Base class for states, representing nodes in the DAG of a GFlowNet.

    Each environment needs to define a subclass of `States`. A `States` object
    is a collection of multiple states (nodes of the DAG) that supports batching.
    Generally, if a state is represented with a tensor of shape (*state_shape), a batch
    of states is represented with a `States` object, with the attribute `tensor` of shape
    (*batch_shape, *state_shape). Other representations are possible (e.g., state as a
    string, numpy array, graph, etc.), but these may need additional logic to support
    batching (see `GraphStates` below for an example).

    Two useful subclasses of `States` are provided:
    - `DiscreteStates` for discrete environments, which represents discrete states
      with a tensor of shape (*batch_shape, *state_shape).
    - `GraphStates` for graph-based environments, which represents graphs as a numpy
      object array of shape (*batch_shape,) containing `GeometricData` objects.

    A `batch_shape` property keeps track of the batch dimension. A trajectory can be
    represented by a States object with `batch_shape = (n_states,)`. Multiple trajectories
    can be represented by a States object with `batch_shape = (n_states, n_trajectories)`.

    Because multiple trajectories can have different lengths, batching requires
    appending a dummy state ($sf$) to trajectories that are shorter than the longest
    trajectory. This dummy state should never be processed, and is used to pad the
    batch of states only.

    Attributes:
        tensor: Tensor of shape (*batch_shape, *state_shape) representing a batch of
            states.
        _log_rewards: (Optional) tensor of shape (*batch_shape,) storing the log rewards
            of each state.
        state_shape: Class variable, a tuple defining the shape of a single state.
        s0: Class variable, a tensor of shape (*state_shape,) representing the initial
            state.
        sf: Class variable, a tensor of shape (*state_shape,) representing the sink
            state.
        make_random_states: Class variable, a callable that returns a random state.
            This is used to initialize random states.
    """

    state_shape: ClassVar[tuple[int, ...]]
    s0: ClassVar[torch.Tensor | GeometricData]
    sf: ClassVar[torch.Tensor | GeometricData]

    make_random_states: Callable = lambda *x: (_ for _ in ()).throw(
        NotImplementedError(
            "The environment does not support initialization of random states."
        )
    )

    def __init__(self, tensor: torch.Tensor) -> None:
        """Initializes a States object with a batch of states.

        Args:
            tensor: Tensor of shape (*batch_shape, *state_shape) representing a batch of
                states.
        """
        assert self.s0.shape == self.state_shape
        assert self.sf.shape == self.state_shape
        assert tensor.shape[-len(self.state_shape) :] == self.state_shape

        self.tensor = tensor
        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @property
    def device(self) -> torch.device:
        """The device on which the states are stored.

        Returns:
            The device of the underlying tensor.
        """
        return self.tensor.device

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the states.

        Returns:
            The batch shape as a tuple.
        """
        return tuple(self.tensor.shape)[: -len(self.state_shape)]

    @classmethod
    def from_batch_shape(
        cls,
        batch_shape: int | tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        device: torch.device | None = None,
    ) -> States:
        r"""Creates a States object with the given batch shape.

        By default, all states are initialized to $s_0$, the initial state. Optionally,
        one can initialize random state, which requires that the environment implements
        the `make_random_states` class method. Sink can be used to initialize
        states at $s_f$, the sink state. Both random and sink cannot be True at the
        same time.

        Args:
            batch_shape: Shape of the batch dimensions.
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).
            device: The device to create the states on.

        Returns:
            A States object with the specified batch shape and initialization.
        """
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        if random and sink:
            raise ValueError("Only one of `random` and `sink` should be True.")

        if random:
            return cls.make_random_states(batch_shape, device=device)
        elif sink:
            return cls.make_sink_states(batch_shape, device=device)
        else:
            return cls.make_initial_states(batch_shape, device=device)

    @classmethod
    def make_initial_states(
        cls, batch_shape: tuple[int, ...], device: torch.device | None = None
    ) -> States:
        r"""Creates a States object with all states set to $s_0$.

        Args:
            batch_shape: Shape of the batch dimensions.
            device: The device to create the states on.

        Returns:
            A States object with all states set to $s_0$.
        """
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        device = cls.s0.device if device is None else device
        if isinstance(cls.s0, torch.Tensor):
            return cls(cls.s0.repeat(*batch_shape, *((1,) * state_ndim)).to(device))
        else:
            raise NotImplementedError(
                f"make_initial_states is not implemented by default for {cls.__name__}"
            )

    @classmethod
    def make_sink_states(
        cls, batch_shape: tuple[int, ...], device: torch.device | None = None
    ) -> States:
        r"""Creates a States object with all states set to $s_f$.

        Args:
            batch_shape: Shape of the batch dimensions.
            device: The device to create the states on.

        Returns:
            A States object with all states set to $s_f$.
        """
        state_ndim = len(cls.state_shape)
        assert cls.sf is not None and state_ndim is not None
        device = cls.sf.device if device is None else device
        if isinstance(cls.sf, torch.Tensor):
            return cls(cls.sf.repeat(*batch_shape, *((1,) * state_ndim)).to(device))
        else:
            raise NotImplementedError(
                f"make_sink_states is not implemented by default for {cls.__name__}"
            )

    def __len__(self) -> int:
        """Returns the number of states in the batch.

        Returns:
            The number of states.
        """
        return prod(self.batch_shape)

    def __repr__(self) -> str:
        """Returns a string representation of the States object.

        Returns:
            A string summary of the States object.
        """
        parts = [
            f"{self.__class__.__name__}(",
            f"batch={self.batch_shape},",
            f"state={self.state_shape},",
            f"device={self.device})",
        ]
        return " ".join(parts)

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> States:
        """Returns a subset of the states along the batch dimension.

        Args:
            index: Indices to select states.

        Returns:
            A new States object with the selected states.
        """
        return self.__class__(self.tensor[index])

    def __setitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor,
        states: States,
    ) -> None:
        """Sets particular states of the batch to a new States object.

        Args:
            index: Indices to set.
            states: States object containing the new states.
        """
        self.tensor[index] = states.tensor

    def clone(self) -> States:
        """Returns a clone of the current instance.

        Returns:
            A new States object with the same data.
        """
        cloned = self.__class__(self.tensor.clone())
        if self._log_rewards is not None:
            cloned._log_rewards = self._log_rewards.clone()

        return cloned

    def flatten(self) -> States:
        """Flattens the batch dimension of the states.

        Useful for example when extracting individual states from trajectories.

        Returns:
            A new States object with the batch dimension flattened.
        """
        states = self.tensor.view(-1, *self.state_shape)
        return self.__class__(states)

    def extend(self, other: States) -> None:
        """Concatenates another States object along the final batch dimension.

        Both States objects must have the same number of batch dimensions, which
        should be 1 or 2.

        Args:
            other: States object to be concatenated to the current States object.
        """
        if len(other.batch_shape) == len(self.batch_shape) == 1:
            # This corresponds to adding a state to a trajectory
            self.tensor = torch.cat((self.tensor, other.tensor), dim=0)

        elif len(other.batch_shape) == len(self.batch_shape) == 2:
            # This corresponds to adding a trajectory to a batch of trajectories
            self.pad_dim0_with_sf(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            other.pad_dim0_with_sf(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            self.tensor = torch.cat((self.tensor, other.tensor), dim=1)
        else:
            raise ValueError(
                f"extend is not implemented for batch shapes {self.batch_shape} and {other.batch_shape}"
            )

    def pad_dim0_with_sf(self, required_first_dim: int) -> None:
        r"""Extends a 2-dimensional batch of states along the first batch dimension.

        Given a batch of states (i.e. of `batch_shape=(a, b)`), extends `a` to a
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
                f"pad_dim0_with_sf is not implemented for States of type "
                f"{self.__class__.__name__} nor for batch shapes {self.batch_shape}"
            )

    def _compare(self, other: torch.Tensor) -> torch.Tensor:
        """Computes elementwise equality between state tensor and an external tensor.

        Args:
            other: Tensor with shape (*batch_shape, *state_shape) representing states to
            compare to.

        Returns:
            A boolean tensor of shape (*batch_shape,) indicating whether the states are
            equal to `other`.
        """
        n_batch_dims = len(self.batch_shape)
        if n_batch_dims == 1:
            assert (other.shape == self.state_shape) or (
                other.shape == self.batch_shape + self.state_shape
            ), f"Expected shape {self.state_shape} or {self.batch_shape + self.state_shape}, got {other.shape}."
        else:
            assert (
                other.shape == self.batch_shape + self.state_shape
            ), f"Expected shape {self.batch_shape + self.state_shape}, got {other.shape}."

        out = self.tensor == other
        if len(self.__class__.state_shape) > 1:
            out = out.flatten(start_dim=n_batch_dims)
        out = out.all(dim=-1)

        assert out.shape == self.batch_shape
        return out

    @property
    def is_initial_state(self) -> torch.Tensor:
        r"""Returns a boolean tensor indicating which states are initial ($s_0$).

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for initial states.
        """
        if isinstance(self.__class__.s0, torch.Tensor):
            if len(self.batch_shape) == 1:
                source_states_tensor = self.__class__.s0
            else:
                source_states_tensor = self.__class__.s0.repeat(
                    *self.batch_shape, *((1,) * len(self.__class__.state_shape))
                )
        else:
            raise NotImplementedError(
                "is_initial_state is not implemented by default "
                f"for {self.__class__.__name__}"
            )
        return self._compare(source_states_tensor)

    @property
    def is_sink_state(self) -> torch.Tensor:
        r"""Returns a boolean tensor indicating which states are sink ($s_f$).

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for sink states.
        """
        if isinstance(self.__class__.sf, torch.Tensor):
            if len(self.batch_shape) == 1:
                sink_states = self.__class__.sf
            else:
                sink_states = self.__class__.sf.repeat(
                    *self.batch_shape, *((1,) * len(self.__class__.state_shape))
                ).to(self.tensor.device)
        else:
            raise NotImplementedError(
                "is_sink_state is not implemented by default "
                f"for {self.__class__.__name__}"
            )
        return self._compare(sink_states)

    @property
    def log_rewards(self) -> torch.Tensor | None:
        """Returns the log rewards of the states.

        Returns:
            The log rewards tensor of shape (*batch_shape,), or None if not set.
        """
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: torch.Tensor) -> None:
        """Sets the log rewards of the states.

        Args:
            log_rewards: Tensor of shape (*batch_shape,) representing the log rewards of
            the states.
        """
        assert tuple(log_rewards.shape) == self.batch_shape
        self._log_rewards = log_rewards

    def sample(self, n_samples: int) -> States:
        """Randomly samples a subset of states from the batch.

        Args:
            n_samples: The number of states to sample.

        Returns:
            A new States object with the sampled states.
        """
        return self[torch.randperm(len(self))[:n_samples]]

    @classmethod
    def stack(cls, states: Sequence[States]) -> States:
        """Stacks a list of States objects along a new dimension (0).

        Args:
            states: List of States objects to stack.

        Returns:
            A new States object with the stacked states.
        """
        state_example = states[0]
        assert all(
            state.batch_shape == state_example.batch_shape for state in states
        ), "All states must have the same batch_shape"

        stacked_states = state_example.from_batch_shape(
            (0, 0), device=state_example.device
        )  # Empty.
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

    DiscreteStates are endowed with `forward_masks` and `backward_masks`: boolean
    attributes representing which actions are allowed at each state. This is the mechanism
    by which all elements of the library verifies the allowed actions at each state.

    Attributes:
        n_actions: Number of possible actions.
        device: The device on which the states are stored.
        forward_masks: Boolean tensor indicating forward actions allowed at each state.
        backward_masks: Boolean tensor indicating backward actions allowed at each state.
    """

    n_actions: ClassVar[int]
    device: ClassVar[torch.device]

    def __init__(
        self,
        tensor: torch.Tensor,
        forward_masks: Optional[torch.Tensor] = None,
        backward_masks: Optional[torch.Tensor] = None,
    ) -> None:
        """Initializes a DiscreteStates container with a batch of states and masks.

        Args:
            tensor: Tensor of shape (*batch_shape, *state_shape) representing a batch of
                states.
            forward_masks: Optional boolean tensor of shape (*batch_shape, n_actions)
                indicating forward actions allowed at each state.
            backward_masks: Optional boolean tensor of shape (*batch_shape, n_actions - 1)
                indicating backward actions allowed at each state.
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
        """Returns a clone of the current instance.

        Returns:
            A new DiscreteStates object with the same data and masks.
        """
        return self.__class__(
            self.tensor.clone(),
            self.forward_masks.clone(),
            self.backward_masks.clone(),
        )

    def _check_both_forward_backward_masks_exist(self):
        assert self.forward_masks is not None and self.backward_masks is not None

    def __repr__(self) -> str:
        """Returns a detailed string representation of the DiscreteStates object.

        Returns:
            A string summary of the DiscreteStates object.
        """
        parts = [
            f"{self.__class__.__name__}(",
            f"batch={self.batch_shape},",
            f"state={self.state_shape},",
            f"actions={self.n_actions},",
            f"device={self.device},",
            f"masks={tuple(self.forward_masks.shape)})",
        ]
        return " ".join(parts)

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> DiscreteStates:
        """Returns a subset of the discrete states and their masks.

        Args:
            index: Indices to select states.

        Returns:
            A new DiscreteStates object with the selected states and masks.
        """
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        out = self.__class__(states, forward_masks, backward_masks)
        return out

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], states: DiscreteStates
    ) -> None:
        """Sets particular discrete states and their masks.

        Args:
            index: Indices to set.
            states: DiscreteStates object containing the new states and masks.
        """
        super().__setitem__(index, states)
        self._check_both_forward_backward_masks_exist()
        self.forward_masks[index] = states.forward_masks
        self.backward_masks[index] = states.backward_masks

    def flatten(self) -> DiscreteStates:
        """Flattens the batch dimension of the discrete states and their masks.

        Returns:
            A new DiscreteStates object with the batch dimension flattened.
        """
        states = self.tensor.view(-1, *self.state_shape)
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks.view(-1, self.forward_masks.shape[-1])
        backward_masks = self.backward_masks.view(-1, self.backward_masks.shape[-1])
        return self.__class__(states, forward_masks, backward_masks)

    def extend(self, other: DiscreteStates) -> None:
        """Concatenates another DiscreteStates object along the batch dimension.

        Args:
            other: DiscreteStates object to concatenate with.
        """
        super().extend(other)
        self.forward_masks = torch.cat(
            (self.forward_masks, other.forward_masks), dim=len(self.batch_shape) - 1
        )
        self.backward_masks = torch.cat(
            (self.backward_masks, other.backward_masks), dim=len(self.batch_shape) - 1
        )

    def pad_dim0_with_sf(self, required_first_dim: int) -> None:
        r"""Extends forward and backward masks along the first batch dimension.

        After extending the state along the first batch dimensions with $s_f$ by
        `required_first_dim`, also extends both forward and backward masks with ones
        along the first dimension by `required_first_dim`.

        Args:
            required_first_dim: The size of the first batch dimension post-expansion.
        """
        super().pad_dim0_with_sf(required_first_dim)

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

    @classmethod
    def stack(cls, states: Sequence[DiscreteStates]) -> DiscreteStates:
        """Stacks a list of DiscreteStates objects along a new dimension (0).

        Args:
            states: List of DiscreteStates objects to stack.

        Returns:
            A new DiscreteStates object with the stacked states and masks.
        """
        out = super().stack(states)
        assert isinstance(out, DiscreteStates)
        out.forward_masks = torch.stack([s.forward_masks for s in states], dim=0)
        out.backward_masks = torch.stack([s.backward_masks for s in states], dim=0)
        return out

    # The helper methods are convenience functions for common mask operations.
    def set_nonexit_action_masks(
        self,
        cond: torch.Tensor,
        allow_exit: bool,
    ) -> None:
        """Masks denoting disallowed actions according to cond, appending the exit mask.

        A convenience function for common mask operations.

        Args:
            cond: a boolean of shape (*batch_shape,) + (n_actions - 1,), which
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

    def set_exit_masks(self, batch_idx: torch.Tensor) -> None:
        """Sets forward masks such that the only allowable next action is to exit.

        A convenience function for common mask operations.

        Args:
            batch_idx: A boolean index along the batch dimension, along which to
                enforce exits.
        """
        self.forward_masks[batch_idx, :] = torch.cat(
            [
                torch.zeros([int(torch.sum(batch_idx).item()), *self.s0.shape]),
                torch.ones([int(torch.sum(batch_idx).item()), 1]),
            ],
            dim=-1,
        ).bool()

    def init_forward_masks(self, set_ones: bool = True) -> None:
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


class GraphStates(States):
    """Base class for graph-based state representations.

    A `GraphStates` object is a collection of multiple graph objects stored as
    a numpy array of `GeometricData` objects. This supports batched management of graphs.

    Attributes:
        num_node_classes: Number of node classes.
        num_edge_classes: Number of edge classes.
        is_directed: Whether the graph is directed.
        s0: Initial state (graph).
        sf: Final state (graph).
        data: A numpy array of `GeometricData` objects representing individual graphs.
        _log_rewards: Stores the log rewards of each state.
        _device: The device on which the graphs are stored.
    """

    num_node_classes: ClassVar[int]
    num_edge_classes: ClassVar[int]
    is_directed: ClassVar[bool]

    s0: ClassVar[GeometricData]
    sf: ClassVar[GeometricData]

    def __init__(
        self,
        data: np.ndarray,
        categorical_node_features: bool = False,
        categorical_edge_features: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Initializes the GraphStates with a numpy array of `GeometricData` objects.

        Args:
            data: A numpy array of `GeometricData` objects representing individual graphs.
            categorical_node_features: Whether the node features are categorical.
            categorical_edge_features: Whether the edge features are categorical.
            device: The device to store the graphs on (optional).
        """
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        self.categorical_node_features = categorical_node_features
        self.categorical_edge_features = categorical_edge_features
        self.data = data
        self._log_rewards: Optional[torch.Tensor] = None
        self._device = device
        if data.size > 0:
            g = data.flat[0]
            if self._device is None:
                self._device = cast(torch.Tensor, g.x).device
            else:
                ensure_same_device(self._device, cast(torch.Tensor, g.x).device)

    @property
    def tensor(self) -> GeometricBatch:
        """Returns the batch representation of the data as a GeometricBatch.

        Returns:
            A GeometricBatch object representing the batch of graphs.
        """
        if self.data.size == 0:
            dummy_graph = GeometricData(
                x=torch.zeros(
                    0,
                    self.num_node_classes,
                    dtype=(
                        torch.long
                        if self.categorical_node_features
                        else torch.get_default_dtype()
                    ),
                    device=self.device,
                ),
                edge_index=torch.zeros(2, 0, dtype=torch.long, device=self.device),
                edge_attr=torch.zeros(
                    0,
                    self.num_edge_classes,
                    dtype=(
                        torch.long
                        if self.categorical_edge_features
                        else torch.get_default_dtype()
                    ),
                    device=self.device,
                ),
            )
            return GeometricBatch.from_data_list([dummy_graph])

        return GeometricBatch.from_data_list(self.data.flatten().tolist())

    @property
    def device(self) -> torch.device:
        """The device on which the graphs are stored.

        Returns:
            The device of the graphs.
        """
        assert self._device is not None
        return self._device

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the graphs.

        Returns:
            The batch shape as a tuple.
        """
        return self.data.shape

    @classmethod
    def make_initial_states(
        cls, batch_shape: int | Tuple, device: torch.device | None = None
    ) -> GraphStates:
        r"""Creates a numpy array of graphs consisting of initial states ($s_0$).

        Args:
            batch_shape: Shape of the batch dimensions.
            device: Device to create the graphs on.

        Returns:
            A GraphStates object containing copies of the initial state.
        """
        assert cls.s0.edge_attr is not None
        assert cls.s0.x is not None
        device = cls.s0.x.device if device is None else device

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = prod(batch_shape)

        data_array = np.empty(batch_shape, dtype=object)
        # Create a numpy array of Data objects by copying s0
        for i in range(num_graphs):
            data_array.flat[i] = cls.s0.clone()

        return cls(
            data_array,
            categorical_node_features=cls.s0.x.dtype == torch.long,
            categorical_edge_features=cls.s0.edge_attr.dtype == torch.long,
            device=device,
        )

    @classmethod
    def make_sink_states(
        cls, batch_shape: int | Tuple, device: torch.device | None = None
    ) -> GraphStates:
        r"""Creates a numpy array of graphs consisting of sink states ($s_f$).

        Args:
            batch_shape: Shape of the batch dimensions.
            device: Device to create the graphs on.

        Returns:
            A GraphStates object containing copies of the sink state.
        """
        assert cls.sf.edge_attr is not None
        assert cls.sf.x is not None
        device = cls.sf.x.device if device is None else device

        if cls.sf is None:
            raise NotImplementedError("Sink state is not defined")

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = prod(batch_shape)

        # Create a numpy array of Data objects by copying sf
        data_array = np.empty(batch_shape, dtype=object)
        for i in range(num_graphs):
            data_array.flat[i] = cls.sf.clone()

        return cls(
            data_array,
            categorical_node_features=cls.sf.x.dtype == torch.long,
            categorical_edge_features=cls.sf.edge_attr.dtype == torch.long,
            device=device,
        )

    @property
    def forward_masks(self) -> TensorDict:
        """Computes masks for valid forward actions from the current state.

        A forward action is valid if:
            1. The edge doesn't already exist in the graph
            2. The edge connects two distinct nodes

        For directed graphs, all possible src->dst edges are considered.
        For undirected graphs, only the upper triangular portion of the adjacency matrix
        is used.

        Returns:
            TensorDict: Boolean mask where True indicates valid actions.
        """
        # Get max nodes across all graphs, handling None values
        max_nodes = 0
        for graph in self.data.flat:
            if graph.x is not None:
                max_nodes = max(max_nodes, graph.x.size(0))

        if self.is_directed:
            max_possible_edges = max_nodes * (max_nodes - 1)
        else:
            max_possible_edges = max_nodes * (max_nodes - 1) // 2

        edge_masks = torch.ones(
            self.data.size, max_possible_edges, dtype=torch.bool, device=self.device
        )

        for i, graph in enumerate(self.data.flat):
            if graph.x is None:
                continue
            ei0, ei1 = get_edge_indices(graph.x.size(0), self.is_directed, self.device)
            edge_masks[i, len(ei0) :] = False

            if graph.edge_index is not None and graph.edge_index.size(1) > 0:
                edge_idx = torch.logical_and(
                    graph.edge_index[0][..., None] == ei0[None],
                    graph.edge_index[1][..., None] == ei1[None],
                ).to(self.device)

                # Collapse across the edge dimension
                if len(edge_idx.shape) == 2:
                    edge_idx = edge_idx.sum(0).bool()

                edge_masks[i, : len(edge_idx)][edge_idx] = False

        edge_masks = edge_masks.view(*self.batch_shape, max_possible_edges)

        # There are 3 action types: ADD_NODE, ADD_EDGE, EXIT
        action_type = torch.ones(
            *self.batch_shape, 3, dtype=torch.bool, device=self.device
        )
        action_type[..., GraphActionType.ADD_EDGE] = torch.any(edge_masks, dim=-1)
        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.NODE_CLASS_KEY: torch.ones(
                    *self.batch_shape,
                    self.num_node_classes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                GraphActions.NODE_INDEX_KEY: torch.zeros(
                    *self.batch_shape,
                    max_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                GraphActions.EDGE_CLASS_KEY: torch.ones(
                    *self.batch_shape,
                    self.num_edge_classes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                GraphActions.EDGE_INDEX_KEY: edge_masks,
            },
            batch_size=self.batch_shape,
        )

    @property
    def backward_masks(self) -> TensorDict:
        """Computes masks for valid backward actions from the current state.

        A backward action is valid if:
            1. The edge exists in the current graph (i.e., can be removed)

        For directed graphs, all existing edges are considered for removal.
        For undirected graphs, only the upper triangular edges are considered.

        The EXIT action is not included in backward masks.

        Returns:
            TensorDict: Boolean mask where True indicates valid actions.
        """
        # Get max nodes across all graphs, handling None values
        max_nodes = 0
        for graph in self.data.flat:
            if graph.x is not None:
                max_nodes = max(max_nodes, graph.x.size(0))

        if self.is_directed:
            max_possible_edges = max_nodes * (max_nodes - 1)
        else:
            max_possible_edges = max_nodes * (max_nodes - 1) // 2

        # Disallow all actions
        edge_masks = torch.zeros(
            self.data.size, max_possible_edges, dtype=torch.bool, device=self.device
        )
        node_index_masks = torch.zeros(
            self.data.size, max_nodes, dtype=torch.bool, device=self.device
        )

        for i, graph in enumerate(self.data.flat):
            node_idxs = torch.arange(len(graph.x.flatten()))
            has_edge = torch.any(
                node_idxs[:, None] == graph.edge_index.flatten()[None], dim=1
            )
            node_index_masks[i, : len(graph.x)] = ~has_edge
            ei0, ei1 = get_edge_indices(graph.x.size(0), self.is_directed, self.device)

            if graph.edge_index is not None and graph.edge_index.size(1) > 0:
                edge_idx = torch.logical_and(
                    graph.edge_index[0][..., None] == ei0[None],
                    graph.edge_index[1][..., None] == ei1[None],
                )
                # Collapse across the edge dimension
                if len(edge_idx.shape) == 2:
                    edge_idx = edge_idx.sum(0).bool()

                edge_masks[i, : len(edge_idx)][edge_idx] = True

        node_index_masks = node_index_masks.view(*self.batch_shape, max_nodes)
        edge_masks = edge_masks.view(*self.batch_shape, max_possible_edges)

        # There are 3 action types: ADD_NODE, ADD_EDGE, EXIT
        action_type = torch.zeros(
            *self.batch_shape, 3, dtype=torch.bool, device=self.device
        )
        action_type[..., GraphActionType.ADD_NODE] = torch.any(node_index_masks, dim=-1)
        action_type[..., GraphActionType.ADD_EDGE] = torch.any(edge_masks, dim=-1)
        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.NODE_CLASS_KEY: torch.zeros(
                    *self.batch_shape,
                    self.num_node_classes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                GraphActions.NODE_INDEX_KEY: node_index_masks,
                GraphActions.EDGE_CLASS_KEY: torch.ones(
                    *self.batch_shape,
                    self.num_edge_classes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                GraphActions.EDGE_INDEX_KEY: edge_masks,
            },
            batch_size=self.batch_shape,
        )

    def __repr__(self) -> str:
        """Returns a detailed string representation of the GraphStates object.

        Returns:
            A string summary of the GraphStates object.
        """
        parts = [
            f"{self.__class__.__name__}(",
            f"batch={self.batch_shape}, ",
            f"device={self.device})",
            f"categorical_node_features={self.categorical_node_features}, ",
            f"categorical_edge_features={self.categorical_edge_features}",
        ]
        return "".join(parts)

    def __len__(self) -> int:
        """Returns the total number of graphs.

        Returns:
            The number of graphs in the batch.
        """
        return self.data.size

    def __getitem__(
        self,
        index: Union[int, Sequence[int], slice, torch.Tensor, Literal[1], Tuple],
    ) -> GraphStates:
        """Returns a subset of the GraphStates.

        Args:
            index: Index or indices to select.

        Returns:
            A new GraphStates object containing the selected graphs.
        """
        index_np = self._get_index_np(index)
        selected_graphs = self.data[index_np]
        if not isinstance(selected_graphs, np.ndarray):
            selected_graphs_array = np.empty(1, dtype=object)
            selected_graphs_array[0] = selected_graphs
            selected_graphs = selected_graphs_array.squeeze()

        out = self.__class__(selected_graphs, device=self.device)

        if self._log_rewards is not None:
            out._log_rewards = self._log_rewards[index]
        return out

    def __setitem__(
        self,
        index: Union[int, Sequence[int], slice, torch.Tensor, Tuple],
        graph: GraphStates,
    ) -> None:
        """Sets a subset of the GraphStates.

        Args:
            index: Index or indices to set.
            graph: GraphStates object containing the new graphs.
        """
        index_np = self._get_index_np(index)
        len_dst = np.empty(self.batch_shape)[index_np].size
        len_src = prod(graph.batch_shape)
        assert (
            len_dst == len_src
        ), "Index and graph must have the same length, but got {} and {}".format(
            len_dst, len_src
        )
        self.data[index_np] = graph.data

        if self._log_rewards is not None and graph._log_rewards is not None:
            self._log_rewards[index] = graph._log_rewards
        else:
            self._log_rewards = None

    def _get_index_np(
        self, index: Union[int, Sequence[int], slice, torch.Tensor, Tuple]
    ) -> Union[int, Sequence[int], slice, np.ndarray, Tuple]:
        """Converts a tensor-based index to a numpy index.

        Args:
            index: The index to convert.

        Returns:
            The converted index.
        """
        if isinstance(index, torch.Tensor):
            return index.cpu().numpy()
        elif isinstance(index, Tuple):
            return tuple(
                idx.cpu().numpy() if isinstance(idx, torch.Tensor) else idx
                for idx in index
            )
        else:
            return index

    def to(self, device: torch.device) -> GraphStates:
        """Moves the GraphStates to the specified device.

        Args:
            device: The device to move to.

        Returns:
            The GraphStates object on the specified device.
        """
        for graph in self.data.flat:
            graph.to(str(device))
        if self._log_rewards is not None:
            self._log_rewards = self._log_rewards.to(device)
        return self

    def clone(self) -> GraphStates:
        """Returns a detached clone of the current instance.

        Returns:
            A new GraphStates object with the same data.
        """
        cloned_graphs = np.empty(self.data.shape, dtype=object)
        for i, graph in enumerate(self.data.flat):
            cloned_graphs.flat[i] = graph.clone()

        out = self.__class__(cloned_graphs, device=self.device)
        if self._log_rewards is not None:
            out._log_rewards = self._log_rewards.clone()
        return out

    def extend(self, other: GraphStates):
        """Concatenates another GraphStates object along the batch dimension.

        Args:
            other: GraphStates object to concatenate with.
        """
        if len(self.batch_shape) == len(other.batch_shape) == 1:
            self.data = np.concatenate([self.data, other.data])

        elif len(self.batch_shape) == len(other.batch_shape) == 2:
            max_batch_shape = max(self.batch_shape[0], other.batch_shape[0])

            # Extend self with sink states if needed
            if self.batch_shape[0] < max_batch_shape:
                self_sf = self.make_sink_states(
                    (max_batch_shape - self.batch_shape[0], self.batch_shape[1])
                )
                self.data = np.concatenate([self.data, self_sf.data])

            # Extend other with sink states if needed
            if other.batch_shape[0] < max_batch_shape:
                other_sf = other.make_sink_states(
                    (max_batch_shape - other.batch_shape[0], other.batch_shape[1])
                )
                other.data = np.concatenate([other.data, other_sf.data])

            self.data = np.concatenate([self.data, other.data], axis=1)

            # We don't have log rewards of sf states
            self._log_rewards = None
        else:
            raise ValueError(
                f"Cannot extend GraphStates with batch shape {other.batch_shape}"
            )

        # Combine log rewards if they exist
        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = torch.cat([self._log_rewards, other._log_rewards], dim=0)
        else:
            self._log_rewards = None

    def _compare(self, other: GeometricData) -> torch.Tensor:
        """Compares the current batch of graphs with another graph.

        Args:
            other: A `GeometricData` object to compare with.

        Returns:
            A boolean tensor of shape (*batch_shape,) indicating which graphs in the
            batch are equal to `other`.
        """
        out = torch.zeros(self.data.size, dtype=torch.bool, device=self.device)

        assert other.x is not None
        assert other.edge_index is not None
        assert other.edge_attr is not None

        other_edges = sorted(other.edge_index.t().tolist())
        other_edge_attr = other.edge_attr[
            torch.argsort(other.edge_index[0] * other.x.size(0) + other.edge_index[1])
        ]

        for i, graph in enumerate(self.data.flat):
            if graph.x is None or graph.edge_index is None or graph.edge_attr is None:
                continue

            if graph.x.size(0) != other.x.size(0):
                continue

            # FIXME: What if the nodes are not sorted?
            if not torch.all(graph.x == other.x):
                continue

            # Check if the number of edges is the same
            if graph.edge_index.size(1) != other.edge_index.size(1):
                continue

            # Check if edge indices are the same (this is more complex due to potential reordering)
            # We'll use a simple heuristic: sort edges and compare
            self_edges = sorted(graph.edge_index.t().tolist())
            if self_edges != other_edges:
                continue

            # Check if the number of edge attributes is the same
            if graph.edge_attr.size(0) != other.edge_attr.size(0):
                continue

            # Check if edge attributes are the same (after sorting)
            graph_edge_attr = graph.edge_attr[
                torch.argsort(
                    graph.edge_index[0] * graph.x.size(0) + graph.edge_index[1]
                )
            ]
            if not torch.all(graph_edge_attr == other_edge_attr):
                continue

            # If all checks pass, the graphs are equal
            out[i] = True

        return out.view(self.batch_shape)

    @property
    def is_sink_state(self) -> torch.Tensor:
        r"""Returns a boolean tensor indicating which graphs are sink states ($s_f$).

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for sink states.
        """
        return self._compare(self.sf)

    @property
    def is_initial_state(self) -> torch.Tensor:
        r"""Returns a boolean tensor indicating which graphs are initial states ($s_0$).

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for initial states.
        """
        return self._compare(self.s0)

    @classmethod
    def stack(cls, states: List[GraphStates]) -> GraphStates:
        """Stacks a list of GraphStates objects along a new dimension (0).

        Args:
            states: List of GraphStates objects to stack.

        Returns:
            A new GraphStates object with the stacked graphs.
        """
        # Check that all states have the same batch shape
        state_batch_shape = states[0].batch_shape
        assert all(state.batch_shape == state_batch_shape for state in states)

        graphs_list = [state.data for state in states]
        stacked_graphs = np.stack(graphs_list)

        out = cls(stacked_graphs, device=states[0].device)

        # Handle log rewards
        log_rewards = []
        save_log_rewards = True
        for state in states:
            save_log_rewards &= state._log_rewards is not None
            if save_log_rewards:
                log_rewards.append(state._log_rewards)

        if save_log_rewards:
            out._log_rewards = torch.stack(log_rewards)

        return out
