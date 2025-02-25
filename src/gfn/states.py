from __future__ import annotations  # This allows to use the class name in type hints

from abc import ABC
from copy import deepcopy
from math import prod
from typing import Callable, ClassVar, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tensordict import TensorDict

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
        _batch_shape: Sizes of the batch dimensions.
        _log_rewards: Stores the log rewards of each state.
    """

    state_shape: ClassVar[tuple[int, ...]]  # Shape of one state
    s0: ClassVar[torch.Tensor | TensorDict]  # Source state of the DAG
    sf: ClassVar[
        torch.Tensor | TensorDict
    ]  # Dummy state, used to pad a batch of states
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
        self._batch_shape = tuple(self.tensor.shape)[: -len(self.state_shape)]
        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @batch_shape.setter
    def batch_shape(self, batch_shape: tuple[int, ...]) -> None:
        self._batch_shape = batch_shape

    @classmethod
    def from_batch_shape(
        cls, batch_shape: tuple[int, ...], random: bool = False, sink: bool = False
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
    def make_initial_states_tensor(cls, batch_shape: tuple[int, ...]) -> torch.Tensor:
        """Makes a tensor with a `batch_shape` of states consisting of $s_0`$s."""
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        if isinstance(cls.s0, torch.Tensor):
            return cls.s0.repeat(*batch_shape, *((1,) * state_ndim))
        else:
            raise NotImplementedError(
                "make_initial_states_tensor is not implemented by default for TensorDicts"
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
                "make_sink_states_tensor is not implemented by default for TensorDicts"
            )

    def __len__(self):
        return prod(self.batch_shape)

    def __repr__(self):
        return f"{self.__class__.__name__} object of batch shape {self.batch_shape} and state shape {self.state_shape}"

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> States:
        """Access particular states of the batch."""
        out = self.__class__(
            self.tensor[index]
        )  # TODO: Inefficient - this might make a copy of the tensor!
        if self._log_rewards is not None:
            out.log_rewards = self._log_rewards[index]
        return out

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
            self.batch_shape = (required_first_dim, self.batch_shape[1])
        else:
            raise ValueError(
                f"extend_with_sf is not implemented for graph states nor for batch shapes {self.batch_shape}"
            )

    def compare(self, other: torch.Tensor) -> torch.Tensor:
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
                "is_initial_state is not implemented by default for TensorDicts"
            )
        return self.compare(source_states_tensor)

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
                "is_sink_state is not implemented by default for TensorDicts"
            )
        return self.compare(sink_states)

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
        assert log_rewards.shape == self.batch_shape
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

        # Adds the trajectory dimension.
        stacked_states.batch_shape = (
            stacked_states.tensor.shape[0],
        ) + state_example.batch_shape

        return stacked_states


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
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> DiscreteStates:
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        out = self.__class__(states, forward_masks, backward_masks)
        if self._log_rewards is not None:
            log_rewards = self._log_rewards[index]
            out.log_rewards = log_rewards
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
        # TODO: do not ignore the next three ignores
        self.forward_masks[batch_idx, :] = torch.cat(
            [
                torch.zeros((torch.sum(batch_idx),) + self.s0.shape),  # pyright: ignore
                torch.ones((torch.sum(batch_idx),) + (1,)),  # pyright: ignore
            ],
            dim=-1,
        ).bool()  # pyright: ignore

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
    Base class for Graph as a state representation. The `GraphStates` object is a batched collection of
    multiple graph objects. The `Batch` object from PyTorch Geometric is used to represent the batch of
    graph objects as states.
    """

    s0: ClassVar[TensorDict]
    sf: ClassVar[TensorDict]

    _next_node_index = 0

    def __init__(self, tensor: TensorDict):
        REQUIRED_KEYS = {
            "node_feature",
            "node_index",
            "edge_feature",
            "edge_index",
            "batch_ptr",
            "batch_shape",
        }
        if not all(key in tensor for key in REQUIRED_KEYS):
            raise ValueError(
                f"TensorDict must contain all required keys: {REQUIRED_KEYS}"
            )

        assert tensor["node_index"].unique().numel() == len(tensor["node_index"])
        self.tensor = tensor
        self.node_features_dim = tensor["node_feature"].shape[-1]
        self.edge_features_dim = tensor["edge_feature"].shape[-1]
        self._log_rewards: Optional[torch.Tensor] = None

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor["batch_shape"].tolist())

    @classmethod
    def from_batch_shape(
        cls, batch_shape: int | Tuple, random: bool = False, sink: bool = False
    ) -> GraphStates:
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
    def make_initial_states_tensor(cls, batch_shape: int | Tuple) -> TensorDict:
        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        nodes = cls.s0["node_feature"].repeat(np.prod(batch_shape), 1)

        return TensorDict(
            {
                "node_feature": nodes,
                "node_index": GraphStates.unique_node_indices(nodes.shape[0]),
                "edge_feature": cls.s0["edge_feature"].repeat(np.prod(batch_shape), 1),
                "edge_index": cls.s0["edge_index"].repeat(np.prod(batch_shape), 1),
                "batch_ptr": torch.arange(
                    int(np.prod(batch_shape)) + 1, device=cls.s0.device
                )
                * cls.s0["node_feature"].shape[0],
                "batch_shape": torch.tensor(batch_shape, device=cls.s0.device),
            }
        )

    @classmethod
    def make_sink_states_tensor(cls, batch_shape: int | Tuple) -> TensorDict:
        if cls.sf is None:
            raise NotImplementedError("Sink state is not defined")

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        nodes = cls.sf["node_feature"].repeat(np.prod(batch_shape), 1)
        out = TensorDict(
            {
                "node_feature": nodes,
                "node_index": GraphStates.unique_node_indices(nodes.shape[0]),
                "edge_feature": cls.sf["edge_feature"].repeat(np.prod(batch_shape), 1),
                "edge_index": cls.sf["edge_index"].repeat(np.prod(batch_shape), 1),
                "batch_ptr": torch.arange(
                    int(np.prod(batch_shape)) + 1, device=cls.sf.device
                )
                * cls.sf["node_feature"].shape[0],
                "batch_shape": torch.tensor(batch_shape, device=cls.sf.device),
            }
        )
        return out

    @classmethod
    def make_random_states_tensor(cls, batch_shape: int | Tuple) -> TensorDict:
        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)

        num_nodes = np.random.randint(10)
        num_edges = np.random.randint(num_nodes * (num_nodes - 1) // 2)
        node_features_dim = cls.s0["node_feature"].shape[-1]
        edge_features_dim = cls.s0["edge_feature"].shape[-1]
        device = cls.s0.device
        return TensorDict(
            {
                "node_feature": torch.rand(
                    int(np.prod(batch_shape)) * num_nodes,
                    node_features_dim,
                    device=device,
                ),
                "node_index": GraphStates.unique_node_indices(
                    int(np.prod(batch_shape)) * num_nodes
                ),
                "edge_feature": torch.rand(
                    int(np.prod(batch_shape)) * num_edges,
                    edge_features_dim,
                    device=device,
                ),
                "edge_index": torch.randint(
                    num_nodes,
                    size=(int(np.prod(batch_shape)) * num_edges, 2),
                    device=device,
                ),
                "batch_ptr": torch.arange(int(np.prod(batch_shape)) + 1, device=device)
                * num_nodes,
                "batch_shape": torch.tensor(batch_shape),
            }
        )

    def __len__(self) -> int:
        return int(np.prod(self.batch_shape))

    def __repr__(self):
        return (
            f"{self.__class__.__name__} object of batch shape {self.tensor['batch_shape']} and "
            f"node feature dim {self.node_features_dim} and edge feature dim {self.edge_features_dim}"
        )

    def __getitem__(
        self, index: int | Sequence[int] | slice | torch.Tensor
    ) -> GraphStates:
        tensor_idx = torch.arange(len(self)).view(*self.batch_shape)
        new_shape = tensor_idx[index].shape
        idx = tensor_idx[index].flatten()

        if torch.any(idx >= len(self.tensor["batch_ptr"]) - 1):
            raise ValueError("Graph index out of bounds")

        # TODO: explain batch_ptr and node_index semantics
        start_ptrs = self.tensor["batch_ptr"][:-1][idx]
        end_ptrs = self.tensor["batch_ptr"][1:][idx]

        node_features = [torch.empty(0, self.node_features_dim)]
        node_indices = [torch.empty(0, dtype=torch.long)]
        edge_features = [torch.empty(0, self.edge_features_dim)]
        edge_indices = [torch.empty(0, 2, dtype=torch.long)]
        batch_ptr = [0]

        for start, end in zip(start_ptrs, end_ptrs):
            node_features.append(self.tensor["node_feature"][start:end])
            node_indices.append(self.tensor["node_index"][start:end])
            batch_ptr.append(batch_ptr[-1] + end - start)

            # Find edges for this graph
            if self.tensor["node_index"].numel() > 0:
                edge_mask = (
                    self.tensor["edge_index"][:, 0] >= self.tensor["node_index"][start]
                ) & (
                    self.tensor["edge_index"][:, 0]
                    <= self.tensor["node_index"][end - 1]
                )
                edge_features.append(self.tensor["edge_feature"][edge_mask])
                edge_indices.append(self.tensor["edge_index"][edge_mask])

        out = self.__class__(
            TensorDict(
                {
                    "node_feature": torch.cat(node_features),
                    "node_index": torch.cat(node_indices),
                    "edge_feature": torch.cat(edge_features),
                    "edge_index": torch.cat(edge_indices),
                    "batch_ptr": torch.tensor(batch_ptr, device=self.tensor.device),
                    "batch_shape": torch.tensor(new_shape, device=self.tensor.device),
                }
            )
        )

        if self._log_rewards is not None:
            out._log_rewards = self._log_rewards[idx]

        assert out.tensor["node_index"].unique().numel() == len(
            out.tensor["node_index"]
        )

        return out

    def __setitem__(self, index: int | Sequence[int], graph: GraphStates):
        """
        Set particular states of the Batch
        """
        # This is to convert index to type int (linear indexing).
        idx = torch.arange(len(self)).view(*self.batch_shape)
        idx = idx[index].flatten()

        # Validate indices
        if torch.any(idx >= len(self.tensor["batch_ptr"]) - 1):
            raise ValueError("Target graph index out of bounds")

        # Source graph details
        source_tensor_dict = graph.tensor
        source_num_graphs = torch.prod(source_tensor_dict["batch_shape"])

        # Validate source and target indices match
        if len(idx) != source_num_graphs:
            raise ValueError(
                "Number of source graphs must match number of target indices"
            )

        for i, graph_idx in enumerate(idx):
            # Get start and end pointers for the current graph
            start_ptr = self.tensor["batch_ptr"][graph_idx]
            end_ptr = self.tensor["batch_ptr"][graph_idx + 1]
            source_start_ptr = source_tensor_dict["batch_ptr"][i]
            source_end_ptr = source_tensor_dict["batch_ptr"][i + 1]

            new_nodes = source_tensor_dict["node_feature"][
                source_start_ptr:source_end_ptr
            ]
            new_nodes = torch.atleast_2d(new_nodes)

            if new_nodes.shape[1] != self.node_features_dim:
                raise ValueError(
                    f"Node features must have dimension {self.node_features_dim}"
                )

            # Concatenate node features
            self.tensor["node_feature"] = torch.cat(
                [
                    self.tensor["node_feature"][
                        :start_ptr
                    ],  # Nodes before the current graph
                    new_nodes,  # New nodes to add
                    self.tensor["node_feature"][
                        end_ptr:
                    ],  # Nodes after the current graph
                ]
            )

            edge_mask = torch.empty(0, dtype=torch.bool)
            if self.tensor["edge_index"].numel() > 0:
                edge_mask = torch.all(
                    self.tensor["edge_index"] > self.tensor["node_index"][end_ptr - 1],
                    dim=-1,
                )
                edge_mask |= torch.all(
                    self.tensor["edge_index"] < self.tensor["node_index"][start_ptr],
                    dim=-1,
                )

            edge_to_add_mask = torch.all(
                source_tensor_dict["edge_index"]
                >= source_tensor_dict["node_index"][source_start_ptr],
                dim=-1,
            )
            edge_to_add_mask &= torch.all(
                source_tensor_dict["edge_index"]
                <= source_tensor_dict["node_index"][source_end_ptr - 1],
                dim=-1,
            )
            self.tensor["edge_index"] = torch.cat(
                [
                    self.tensor["edge_index"][edge_mask],
                    source_tensor_dict["edge_index"][edge_to_add_mask],
                ],
                dim=0,
            )
            self.tensor["edge_feature"] = torch.cat(
                [
                    self.tensor["edge_feature"][edge_mask],
                    source_tensor_dict["edge_feature"][edge_to_add_mask],
                ],
                dim=0,
            )

            self.tensor["node_index"] = torch.cat(
                [
                    self.tensor["node_index"][:start_ptr],
                    source_tensor_dict["node_index"][source_start_ptr:source_end_ptr],
                    self.tensor["node_index"][end_ptr:],
                ]
            )
            # Update batch pointers
            shift = new_nodes.shape[0] - (end_ptr - start_ptr)
            self.tensor["batch_ptr"][graph_idx + 1 :] += shift

        assert self.tensor["node_index"].unique().numel() == len(
            self.tensor["node_index"]
        )

    @property
    def device(self) -> torch.device | None:
        return self.tensor.device

    def to(self, device: torch.device) -> GraphStates:
        """
        Moves and/or casts the graph states to the specified device
        """
        self.tensor = self.tensor.to(device)
        return self

    def clone(self) -> GraphStates:
        """Returns a *detached* clone of the current instance using deepcopy."""
        return deepcopy(self)

    def extend(self, other: GraphStates):
        """Concatenates to another GraphStates object along the batch dimension"""
        # find if there are common node indices
        other_node_index = other.tensor["node_index"].clone()  # Clone to avoid modifying original
        other_edge_index = other.tensor["edge_index"].clone()  # Clone to avoid modifying original
        
        # Always generate new indices for the other state to ensure uniqueness
        new_indices = GraphStates.unique_node_indices(len(other_node_index))
        
        # Update edge indices to match new node indices
        for i, old_idx in enumerate(other_node_index):
            other_edge_index[other_edge_index == old_idx] = new_indices[i]
        
        # Update node indices
        other_node_index = new_indices

        if torch.prod(self.tensor["batch_shape"]) == 0:
            # if self is empty, just copy other
            self.tensor["node_feature"] = other.tensor["node_feature"]
            self.tensor["batch_shape"] = other.tensor["batch_shape"]
            self.tensor["node_index"] = other_node_index
            self.tensor["edge_feature"] = other.tensor["edge_feature"]
            self.tensor["edge_index"] = other_edge_index
            self.tensor["batch_ptr"] = other.tensor["batch_ptr"]

        elif len(self.tensor["batch_shape"]) == 1:
            self.tensor["node_feature"] = torch.cat(
                [self.tensor["node_feature"], other.tensor["node_feature"]], dim=0
            )
            self.tensor["node_index"] = torch.cat(
                [self.tensor["node_index"], other_node_index], dim=0
            )
            self.tensor["edge_feature"] = torch.cat(
                [self.tensor["edge_feature"], other.tensor["edge_feature"]], dim=0
            )
            self.tensor["edge_index"] = torch.cat(
                [self.tensor["edge_index"], other_edge_index],
                dim=0,
            )
            self.tensor["batch_ptr"] = torch.cat(
                [
                    self.tensor["batch_ptr"],
                    other.tensor["batch_ptr"][1:] + self.tensor["batch_ptr"][-1],
                ],
                dim=0,
            )
            self.tensor["batch_shape"] = (
                self.tensor["batch_shape"][0] + other.tensor["batch_shape"][0],
            ) + self.batch_shape[1:]
        else: 
            # Here we handle the case where the batch shape is (T, B)
            # and we want to concatenate along the batch dimension B.
            assert len(self.tensor["batch_shape"]) == 2
            max_len = max(self.tensor["batch_shape"][0], other.tensor["batch_shape"][0])

            node_features = []
            node_indices = []
            edge_features = []
            edge_indices = []
            # Get device from one of the tensors
            device = self.tensor["node_feature"].device
            batch_ptr = [torch.tensor([0], device=device)]
            
            for i in range(max_len):
                # Following the logic of Base class, we want to extend with sink states
                if i >= self.tensor["batch_shape"][0]:
                    self_i = self.make_sink_states_tensor(self.tensor["batch_shape"][1:])
                else:
                    self_i = self[i].tensor
                if i >= other.tensor["batch_shape"][0]:
                    other_i = other.make_sink_states_tensor(other.tensor["batch_shape"][1:])
                else:
                    other_i = other[i].tensor
                
                # Generate new unique indices for both self_i and other_i
                new_self_indices = GraphStates.unique_node_indices(len(self_i["node_index"]))
                new_other_indices = GraphStates.unique_node_indices(len(other_i["node_index"]))

                # Update self_i edge indices
                self_edge_index = self_i["edge_index"].clone()
                for old_idx, new_idx in zip(self_i["node_index"], new_self_indices):
                    mask = (self_edge_index == old_idx)
                    self_edge_index[mask] = new_idx

                # Update other_i edge indices
                other_edge_index = other_i["edge_index"].clone()
                for old_idx, new_idx in zip(other_i["node_index"], new_other_indices):
                    mask = (other_edge_index == old_idx)
                    other_edge_index[mask] = new_idx

                node_features.append(self_i["node_feature"])
                node_indices.append(new_self_indices)  # Use new indices
                edge_features.append(self_i["edge_feature"])
                edge_indices.append(self_edge_index)  # Use updated edge indices
                batch_ptr.append(self_i["batch_ptr"][1:] + batch_ptr[-1][-1])

                node_features.append(other_i["node_feature"])
                node_indices.append(new_other_indices)  # Use new indices
                edge_features.append(other_i["edge_feature"])
                edge_indices.append(other_edge_index)  # Use updated edge indices
                batch_ptr.append(other_i["batch_ptr"][1:] + batch_ptr[-1][-1])

            self.tensor["node_feature"] = torch.cat(node_features, dim=0)
            self.tensor["node_index"] = torch.cat(node_indices, dim=0)
            self.tensor["edge_feature"] = torch.cat(edge_features, dim=0)
            self.tensor["edge_index"] = torch.cat(edge_indices, dim=0)
            self.tensor["batch_ptr"] = torch.cat(batch_ptr, dim=0)

            self.tensor["batch_shape"] = (
                max_len,
                self.tensor["batch_shape"][1] + other.tensor["batch_shape"][1],
            )

        assert self.tensor["node_index"].unique().numel() == len(
            self.tensor["node_index"]
        )
        assert torch.prod(torch.tensor(self.tensor["batch_shape"])) == len(self.tensor["batch_ptr"]) - 1

    @property
    def log_rewards(self) -> torch.Tensor | None:
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: torch.Tensor) -> None:
        self._log_rewards = log_rewards

    def _compare(self, other: TensorDict) -> torch.Tensor:
        out = torch.zeros(len(self.tensor["batch_ptr"]) - 1, dtype=torch.bool)
        for i in range(len(self.tensor["batch_ptr"]) - 1):
            start, end = self.tensor["batch_ptr"][i], self.tensor["batch_ptr"][i + 1]
            if end - start != len(other["node_feature"]):
                out[i] = False
            else:
                out[i] = torch.all(
                    self.tensor["node_feature"][start:end] == other["node_feature"]
                )
                edge_mask = torch.all(
                    (self.tensor["edge_index"] >= self.tensor["node_index"][start])
                    & (self.tensor["edge_index"] <= self.tensor["node_index"][end - 1]),
                    dim=-1,
                )
                edge_index = self.tensor["edge_index"][edge_mask]
                out[i] &= len(edge_index) == len(other["edge_index"]) and torch.all(
                    edge_index == other["edge_index"]
                )
                edge_feature = self.tensor["edge_feature"][edge_mask]
                out[i] &= len(edge_feature) == len(other["edge_feature"]) and torch.all(
                    edge_feature == other["edge_feature"]
                )
        return out.view(self.batch_shape)

    @property
    def is_sink_state(self) -> torch.Tensor:
        return self._compare(self.sf)

    @property
    def is_initial_state(self) -> torch.Tensor:
        return self._compare(self.s0)

    @classmethod
    def stack(cls, states: List[GraphStates]):
        """Given a list of states, stacks them along a new dimension (0)."""
        stacked_states = cls.from_batch_shape(0)
        state_batch_shape = states[0].batch_shape
        assert len(state_batch_shape) == 1
        for state in states:
            assert state.batch_shape == state_batch_shape
            stacked_states.extend(state)

        stacked_states.tensor["batch_shape"] = (len(states),) + state_batch_shape
        assert stacked_states.tensor["node_index"].unique().numel() == len(
            stacked_states.tensor["node_index"]
        )
        return stacked_states

    @property
    def forward_masks(self) -> TensorDict:
        n_nodes = self.tensor["batch_ptr"][1:] - self.tensor["batch_ptr"][:-1]
        ei_mask_shape = (
            len(self.tensor["node_feature"]),
            len(self.tensor["node_feature"]),
        )
        forward_masks = TensorDict(
            {
                "action_type": torch.ones(self.batch_shape + (3,), dtype=torch.bool),
                "features": torch.ones(
                    self.batch_shape + (self.node_features_dim,), dtype=torch.bool
                ),
                "edge_index": torch.zeros(
                    self.batch_shape + ei_mask_shape, dtype=torch.bool
                ),
            }
        )  # TODO: edge_index mask is very memory consuming...
        forward_masks["action_type"][..., GraphActionType.ADD_EDGE] = n_nodes > 1
        forward_masks["action_type"][..., GraphActionType.EXIT] = n_nodes >= 1

        arange = torch.arange(len(self)).view(self.batch_shape)
        arange_nodes = torch.arange(len(self.tensor["node_feature"]))[None, :]
        same_graph_mask = (arange_nodes >= self.tensor["batch_ptr"][:-1, None]) & (
            arange_nodes < self.tensor["batch_ptr"][1:, None]
        )
        edge_index = torch.where(
            self.tensor["edge_index"][..., None] == self.tensor["node_index"]
        )[2].reshape(self.tensor["edge_index"].shape)
        i, j = edge_index[..., 0], edge_index[..., 1]

        for _ in range(len(self.batch_shape)):
            (i, j) = i.unsqueeze(0), j.unsqueeze(0)

        # First allow nodes in the same graph to connect, then disable nodes with existing edges
        forward_masks["edge_index"][
            same_graph_mask[:, :, None] & same_graph_mask[:, None, :]
        ] = True
        torch.diagonal(forward_masks["edge_index"], dim1=-2, dim2=-1).fill_(False)
        forward_masks["edge_index"][arange[..., None], i, j] = False
        forward_masks["action_type"][..., GraphActionType.ADD_EDGE] &= torch.any(
            forward_masks["edge_index"], dim=(-1, -2)
        )
        return forward_masks

    @property
    def backward_masks(self) -> TensorDict:
        n_nodes = self.tensor["batch_ptr"][1:] - self.tensor["batch_ptr"][:-1]
        n_edges = torch.count_nonzero(
            (
                self.tensor["edge_index"][None, :, 0]
                >= self.tensor["batch_ptr"][:-1, None]
            )
            & (
                self.tensor["edge_index"][None, :, 0]
                < self.tensor["batch_ptr"][1:, None]
            )
            & (
                self.tensor["edge_index"][None, :, 1]
                >= self.tensor["batch_ptr"][:-1, None]
            )
            & (
                self.tensor["edge_index"][None, :, 1]
                < self.tensor["batch_ptr"][1:, None]
            ),
            dim=-1,
        )
        ei_mask_shape = (
            len(self.tensor["node_feature"]),
            len(self.tensor["node_feature"]),
        )
        backward_masks = TensorDict(
            {
                "action_type": torch.ones(self.batch_shape + (3,), dtype=torch.bool),
                "features": torch.ones(
                    self.batch_shape + (self.node_features_dim,), dtype=torch.bool
                ),
                "edge_index": torch.zeros(
                    self.batch_shape + ei_mask_shape, dtype=torch.bool
                ),
            }
        )  # TODO: edge_index mask is very memory consuming...
        backward_masks["action_type"][..., GraphActionType.ADD_NODE] = n_nodes >= 1
        backward_masks["action_type"][..., GraphActionType.ADD_EDGE] = n_edges
        backward_masks["action_type"][..., GraphActionType.EXIT] = n_nodes >= 1

        # Allow only existing edges
        arange = torch.arange(len(self)).view(self.batch_shape)
        ei1 = self.tensor["edge_index"][..., 0]
        ei2 = self.tensor["edge_index"][..., 1]
        for _ in range(len(self.batch_shape)):
            (
                ei1,
                ei2,
            ) = ei1.unsqueeze(
                0
            ), ei2.unsqueeze(0)
        backward_masks["edge_index"][arange[..., None], ei1, ei2] = False
        return backward_masks

    @classmethod
    def unique_node_indices(cls, num_new_nodes: int) -> torch.Tensor:
        indices = torch.arange(
            cls._next_node_index, cls._next_node_index + num_new_nodes
        )
        cls._next_node_index += num_new_nodes
        return indices
