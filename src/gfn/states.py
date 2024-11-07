from __future__ import annotations  # This allows to use the class name in type hints

from abc import ABC
from copy import deepcopy
from math import prod
from typing import Callable, ClassVar, List, Optional, Sequence

import torch
from torch_geometric.data import Batch, Data


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
    without having to call the environment's `validate_actions` method. Put different,
    `validate_actions` for such environments, directly calls the masks. This is handled
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
        batch_shape: Sizes of the batch dimensions.
        _log_rewards: Stores the log rewards of each state.
    """

    state_shape: ClassVar[tuple[int, ...]]  # Shape of one state
    s0: ClassVar[torch.Tensor]  # Source state of the DAG
    sf: ClassVar[torch.Tensor]  # Dummy state, used to pad a batch of states
    make_random_states_tensor: Callable = lambda x: (_ for _ in ()).throw(
        NotImplementedError(
            "The environment does not support initialization of random states."
        )
    )

    def __init__(self, tensor: torch.Tensor):
        """Initalize the State container with a batch of states.
        Args:
            tensor: Tensor of shape (*batch_shape, *state_shape) representing a batch of states.
        """
        assert self.s0.shape == self.state_shape
        assert self.sf.shape == self.state_shape
        assert tensor.shape[-len(self.state_shape) :] == self.state_shape

        self.tensor = tensor
        self.batch_shape = tuple(self.tensor.shape)[: -len(self.state_shape)]
        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @classmethod
    def from_batch_shape(
        cls, batch_shape: tuple[int], random: bool = False, sink: bool = False
    ) -> States:
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
    def make_initial_states_tensor(cls, batch_shape: tuple[int]) -> torch.Tensor:
        """Makes a tensor with a `batch_shape` of states consisting of $s_0`$s.

        Args:
            batch_shape: Shape of the batch dimensions.

        Returns a tensor of shape (*batch_shape, *state_shape) with all states equal to $s_0$.
        """
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        return cls.s0.repeat(*batch_shape, *((1,) * state_ndim))

    @classmethod
    def make_sink_states_tensor(cls, batch_shape: tuple[int]) -> torch.Tensor:
        """Makes a tensor with a `batch_shape` of states consisting of $s_f$s.

        Args:
            batch_shape: Shape of the batch dimensions.

        Returns a tensor of shape (*batch_shape, *state_shape) with all states equal to $s_f$.
        """
        state_ndim = len(cls.state_shape)
        assert cls.sf is not None and state_ndim is not None
        return cls.sf.repeat(*batch_shape, *((1,) * state_ndim))

    def __len__(self):
        return prod(self.batch_shape)

    def __repr__(self):
        return f"{self.__class__.__name__} object of batch shape {self.batch_shape} and state shape {self.state_shape}"

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(
        self, index: int | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> States:
        """Access particular states of the batch."""
        out = self.__class__(
            self.tensor[index]
        )  # TODO: Inefficient - this might make a copy of the tensor!
        if self._log_rewards is not None:
            out.log_rewards = self._log_rewards[index]
        return out

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], states: States
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
        other_batch_shape = other.batch_shape
        if len(other_batch_shape) == len(self.batch_shape) == 1:
            # This corresponds to adding a state to a trajectory
            self.batch_shape = (self.batch_shape[0] + other_batch_shape[0],)
            self.tensor = torch.cat((self.tensor, other.tensor), dim=0)
            if self._log_rewards is not None:
                assert other._log_rewards is not None
                self._log_rewards = torch.cat(
                    (self._log_rewards, other._log_rewards), dim=0
                )

        elif len(other_batch_shape) == len(self.batch_shape) == 2:
            # This corresponds to adding a trajectory to a batch of trajectories
            self.extend_with_sf(
                required_first_dim=max(self.batch_shape[0], other_batch_shape[0])
            )
            other.extend_with_sf(
                required_first_dim=max(self.batch_shape[0], other_batch_shape[0])
            )
            self.batch_shape = (
                self.batch_shape[0],
                self.batch_shape[1] + other_batch_shape[1],
            )
            self.tensor = torch.cat((self.tensor, other.tensor), dim=1)
        else:
            raise ValueError(
                f"extend is not implemented for batch shapes {self.batch_shape} and {other_batch_shape}"
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
        if len(self.batch_shape) == 2:
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
            self.batch_shape = (required_first_dim, self.batch_shape[1])
        else:
            raise ValueError(
                f"extend_with_sf is not implemented for batch shapes {self.batch_shape}"
            )

    def compare(self, other: torch.tensor) -> torch.Tensor:
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
        source_states_tensor = self.__class__.s0.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        )
        return self.compare(source_states_tensor)

    @property
    def is_sink_state(self) -> torch.Tensor:
        """Returns a tensor of shape `batch_shape` that is True for states that are $s_f$ of the DAG."""
        # TODO: self.__class__.sf == self.tensor -- or something similar?
        sink_states = self.__class__.sf.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        ).to(self.tensor.device)
        return self.compare(sink_states)

    @property
    def log_rewards(self) -> torch.Tensor:
        """Returns the log rewards of the states as tensor of shape `batch_shape`."""
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: torch.Tensor) -> None:
        """Sets the log rewards of the states.

        Args:
            log_rewards: Tensor of shape `batch_shape` representing the log rewards of the states.
        """
        assert log_rewards.shape == self.batch_shape
        self._log_rewards = log_rewards

    def sample(self, n_samples: int) -> States:
        """Samples a subset of the States object."""
        return self[torch.randperm(len(self))[:n_samples]]


class DiscreteStates(States, ABC):
    """Base class for states of discrete environments.

    States are endowed with a `forward_masks` and `backward_masks`: boolean attributes
    representing which actions are allowed at each state. This is the mechanism by
    which all elements of the library (including an environment's `validate_actions`
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

    def clone(self) -> States:
        """Returns a clone of the current instance."""
        return self.__class__(
            self.tensor.detach().clone(),
            self.forward_masks,
            self.backward_masks,
        )

    def _check_both_forward_backward_masks_exist(self):
        assert self.forward_masks is not None and self.backward_masks is not None

    def __getitem__(
        self, index: int | Sequence[int] | Sequence[bool]
    ) -> DiscreteStates:
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        out = self.__class__(states, forward_masks, backward_masks)
        if self._log_rewards is not None:
            log_probs = self._log_rewards[index]
            out.log_rewards = log_probs
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

    def extend(self, other: States) -> None:
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
                torch.zeros((torch.sum(batch_idx),) + self.s0.shape),
                torch.ones((torch.sum(batch_idx),) + (1,)),
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


def stack_states(states: List[States]):
    """Given a list of states, stacks them along a new dimension (0)."""
    state_example = states[0]  # We assume all elems of `states` are the same.

    stacked_states = state_example.from_batch_shape((0, 0))  # Empty.
    stacked_states.tensor = torch.stack([s.tensor for s in states], dim=0)
    if state_example._log_rewards:
        stacked_states._log_rewards = torch.stack(
            [s._log_rewards for s in states], dim=0
        )

    # We are dealing with a list of DiscretrStates instances.
    if hasattr(state_example, "forward_masks"):
        stacked_states.forward_masks = torch.stack(
            [s.forward_masks for s in states], dim=0
        )
        stacked_states.backward_masks = torch.stack(
            [s.backward_masks for s in states], dim=0
        )

    # Adds the trajectory dimension.
    stacked_states.batch_shape = (
        stacked_states.tensor.shape[0],
    ) + state_example.batch_shape

    return stacked_states


class GraphStates(ABC):
    """
    Base class for Graph as a state representation. The `GraphStates` object is a batched collection of
    multiple graph objects. The `Batch` object from PyTorch Geometric is used to represent the batch of
    graph objects as states.
    """

    s0: ClassVar[Data]
    sf: ClassVar[Data]
    node_feature_dim: ClassVar[int]
    edge_feature_dim: ClassVar[int]
    make_random_states_graph: Callable = lambda x: (_ for _ in ()).throw(
        NotImplementedError(
            "The environment does not support initialization of random Graph states."
        )
    )

    def __init__(self, graphs: Batch):
        self.data: Batch = graphs
        self.batch_shape: int = self.data.num_graphs
        self._log_rewards: float = None

    @classmethod
    def from_batch_shape(
        cls, batch_shape: int, random: bool = False, sink: bool = False
    ) -> GraphStates:
        if random and sink:
            raise ValueError("Only one of `random` and `sink` should be True.")
        if random:
            data = cls.make_random_states_graph(batch_shape)
        elif sink:
            data = cls.make_sink_states_graph(batch_shape)
        else:
            data = cls.make_initial_states_graph(batch_shape)
        return cls(data)

    @classmethod
    def make_initial_states_graph(cls, batch_shape: int) -> Batch:
        data = Batch.from_data_list([cls.s0 for _ in range(batch_shape)])
        return data

    @classmethod
    def make_sink_states_graph(cls, batch_shape: int) -> Batch:
        data = Batch.from_data_list([cls.sf for _ in range(batch_shape)])
        return data

    # @classmethod
    # def make_random_states_graph(cls, batch_shape: int) -> Batch:
    #     data = Batch.from_data_list([cls.make_random_states_graph() for _ in range(batch_shape)])
    #     return data

    def __len__(self):
        return self.data.batch_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__} object of batch shape {self.batch_shape} and "
            f"node feature dim {self.node_feature_dim} and edge feature dim {self.edge_feature_dim}"
        )

    def __getitem__(self, index: int | Sequence[int] | slice) -> GraphStates:
        if isinstance(index, int):
            out = self.__class__(Batch.from_data_list([self.data[index]]))
        elif isinstance(index, (Sequence, slice)):
            out = self.__class__(Batch.from_data_list(self.data.index_select(index)))
        else:
            raise NotImplementedError(
                "Indexing with type {} is not implemented".format(type(index))
            )

        if self._log_rewards is not None:
            out._log_rewards = self._log_rewards[index]

        return out

    def __setitem__(self, index: int | Sequence[int], graph: GraphStates):
        """
        Set particular states of the Batch
        """
        data_list = self.data.to_data_list()
        if isinstance(index, int):
            assert (
                len(graph) == 1
            ), "GraphStates must have a batch size of 1 for single index assignment"
            data_list[index] = graph.data[0]
            self.data = Batch.from_data_list(data_list)
        elif isinstance(index, Sequence):
            assert len(index) == len(
                graph
            ), "Index and GraphState must have the same length"
            for i, idx in enumerate(index):
                data_list[idx] = graph.data[i]
            self.data = Batch.from_data_list(data_list)
        elif isinstance(index, slice):
            assert index.stop - index.start == len(
                graph
            ), "Index slice and GraphStates must have the same length"
            data_list[index] = graph.data.to_data_list()
            self.data = Batch.from_data_list(data_list)
        else:
            raise NotImplementedError(
                "Setters with type {} is not implemented".format(type(index))
            )

    @property
    def device(self) -> torch.device:
        return self.data.get_example(0).x.device

    def to(self, device: torch.device) -> GraphStates:
        """
        Moves and/or casts the graph states to the specified device
        """
        if self.device != device:
            self.data = self.data.to(device)
        return self

    def clone(self) -> GraphStates:
        """Returns a *detached* clone of the current instance using deepcopy."""
        return deepcopy(self)

    def extend(self, other: GraphStates):
        """Concatenates to another GraphStates object along the batch dimension"""
        self.data = Batch.from_data_list(
            self.data.to_data_list() + other.data.to_data_list()
        )
        if self._log_rewards is not None:
            assert other._log_rewards is not None
            self._log_rewards = torch.cat(
                (self._log_rewards, other._log_rewards), dim=0
            )

    @property
    def log_rewards(self) -> torch.Tensor:
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: torch.Tensor) -> None:
        self._log_rewards = log_rewards
