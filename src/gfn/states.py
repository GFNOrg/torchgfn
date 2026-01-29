from __future__ import annotations  # This allows to use the class name in type hints

import inspect
import logging
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

logger = logging.getLogger(__name__)


def _assert_factory_accepts_debug(factory: Callable, factory_name: str) -> None:
    """Ensure the factory can accept a debug kwarg (explicit or via **kwargs)."""
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    debug_param = params.get("debug")
    if debug_param is not None:
        return
    raise TypeError(
        f"{factory_name} must accept a `debug` keyword argument (or **kwargs) "
        "to support debug-gated States construction."
    )


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

    Compile-related expectations:
    - Hot paths should be called with tensors already on the target device and with
      correct shapes; debug guards can be enabled during development/tests to validate.
    - Set `debug=False` inside torch.compile regions to avoid Python-side graph breaks;
      enable `debug=True` only when running eager checks.

    Attributes:
        tensor: Tensor of shape (*batch_shape, *state_shape) representing a batch of
            states.
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

    make_random_states: Callable = staticmethod(
        lambda *args, **kwargs: (_ for _ in ()).throw(
            NotImplementedError(
                "The environment does not support initialization of random states."
            )
        )
    )

    def __init__(
        self,
        tensor: torch.Tensor,
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> None:
        """Initializes a States object with a batch of states.

        Args:
            tensor: Tensor of shape (*batch_shape, *state_shape) representing a batch of
                states.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: The device to store the states on.
            debug: If True, keep runtime guards active for safety; keep False in
                compiled regions to avoid graph breaks when using torch.compile.
                Preconditions when debug is False: `tensor` is already on the intended
                device and its trailing dimensions equal `state_shape`.
        """
        if debug:
            # Keep shape validations in debug so compiled graphs avoid Python asserts.
            assert self.s0.shape == self.state_shape
            assert self.sf.shape == self.state_shape
            assert (
                tensor.shape[-len(self.state_shape) :] == self.state_shape
            )  # noqa: E203

        # Per-instance device resolution: prefer explicit device, else infer from tensor
        resolved_device = device if device is not None else tensor.device
        self.tensor = tensor.to(resolved_device)
        self.debug = debug

        # Initialize conditions (for conditional GFlowNets)
        self._conditions: torch.Tensor | None = None
        if conditions is not None:
            assert conditions.shape[:-1] == self.batch_shape, (
                f"Conditions batch shape {conditions.shape[:-1]} doesn't match "
                f"states batch shape {self.batch_shape}"
            )
            # condition should be of default float dtype (since dummy condition is -inf)
            assert conditions.dtype == torch.get_default_dtype()
            ensure_same_device(self.device, conditions.device)
            self.conditions = conditions

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

    @property
    def conditions(self) -> torch.Tensor | None:
        """The conditions attached to these states for conditional GFlowNets.

        Returns:
            Tensor of shape (*batch_shape, condition_dim) or None if no conditions.
        """
        return self._conditions

    @conditions.setter
    def conditions(self, value: torch.Tensor | None) -> None:
        """Sets conditions with batch shape validation.

        Args:
            value: Tensor of shape (*batch_shape, condition_dim) or None.
        """
        if value is not None:
            cond_batch_shape = value.shape[:-1]
            assert cond_batch_shape == self.batch_shape, (
                f"Conditions batch shape {cond_batch_shape} doesn't match "
                f"states batch shape {self.batch_shape}"
            )
            assert value.dtype == torch.get_default_dtype()
            self._conditions = value.to(self.device)
        else:
            self._conditions = None

    @property
    def has_conditions(self) -> bool:
        """Whether conditions are attached to these states."""
        return self._conditions is not None

    @classmethod
    def from_batch_shape(
        cls,
        batch_shape: int | tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
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
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: The device to create the states on.
            debug: If True, keeps compile graph-breaking checks in the logic for safety.

        Returns:
            A States object with the specified batch shape and initialization.
        """
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        if random and sink:
            raise ValueError("Only one of `random` and `sink` should be True.")

        if random:
            _assert_factory_accepts_debug(cls.make_random_states, "make_random_states")
            make_states_fn = cls.make_random_states
        elif sink:
            _assert_factory_accepts_debug(cls.make_sink_states, "make_sink_states")
            make_states_fn = cls.make_sink_states
        else:
            _assert_factory_accepts_debug(cls.make_initial_states, "make_initial_states")
            make_states_fn = cls.make_initial_states
        return make_states_fn(
            batch_shape, conditions=conditions, device=device, debug=debug
        )

    @classmethod
    def make_initial_states(
        cls,
        batch_shape: tuple[int, ...],
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> States:
        r"""Creates a States object with all states set to $s_0$.

        Args:
            batch_shape: Shape of the batch dimensions.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: The device to create the states on.
            debug: If True, keeps compile graph-breaking checks in the logic for safety.

        Returns:
            A States object with all states set to $s_0$.
        """
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        device = cls.s0.device if device is None else device
        if isinstance(cls.s0, torch.Tensor):
            return cls(
                cls.s0.repeat(*batch_shape, *((1,) * state_ndim)).to(device),
                conditions=conditions,
                debug=debug,
            )
        else:
            raise NotImplementedError(
                f"make_initial_states is not implemented by default for {cls.__name__}"
            )

    @classmethod
    def make_sink_states(
        cls,
        batch_shape: tuple[int, ...],
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> States:
        r"""Creates a States object with all states set to $s_f$.

        Args:
            batch_shape: Shape of the batch dimensions.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: The device to create the states on.
            debug: If True, keeps compile graph-breaking checks in the logic for safety.

        Returns:
            A States object with all states set to $s_f$.
        """
        state_ndim = len(cls.state_shape)
        assert cls.sf is not None and state_ndim is not None
        device = cls.sf.device if device is None else device
        if isinstance(cls.sf, torch.Tensor):
            return cls(
                cls.sf.repeat(*batch_shape, *((1,) * state_ndim)).to(device),
                conditions=conditions,
                debug=debug,
            )
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
        ]
        if self.conditions is not None:
            parts.append(f"conditions={self.conditions.shape},")
        parts.append(f"device={self.device})")
        return " ".join(parts)

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> States:
        """Returns a subset of the states along the batch dimension.

        Args:
            index: Indices to select states.

        Returns:
            A new States object with the selected states and conditions.
        """
        conditions = self.conditions[index] if self.conditions is not None else None
        return self.__class__(
            self.tensor[index], conditions=conditions, debug=self.debug
        )

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
        if self.conditions is not None and states.conditions is not None:
            self.conditions[index] = states.conditions
        else:
            if self.conditions is not None or states.conditions is not None:
                logger.warning(
                    "Inconsistent conditions when setting states. Setting to None."
                )
            self.conditions = None

    def clone(self) -> States:
        """Returns a clone of the current instance.

        Returns:
            A new States object with the same data and conditions.
        """
        conditions = self.conditions.clone() if self.conditions is not None else None
        return self.__class__(
            self.tensor.clone(), conditions=conditions, debug=self.debug
        )

    def flatten(self) -> States:
        """Flattens the batch dimension of the states.

        Useful for example when extracting individual states from trajectories.

        Returns:
            A new States object with the batch dimension flattened.
        """
        states = self.tensor.view(-1, *self.state_shape)
        conditions = (
            self.conditions.view(-1, self.conditions.shape[-1])
            if self.conditions is not None
            else None
        )
        return self.__class__(states, conditions=conditions, debug=self.debug)

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
            other = other.clone()  # TODO: Is there a more efficient way?
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

        if self.conditions is not None and other.conditions is not None:
            self.conditions = torch.cat(
                (self.conditions, other.conditions), dim=len(self.batch_shape) - 1
            )
        else:
            if self.conditions is not None or other.conditions is not None:
                logger.warning(
                    "Inconsistent conditions when extending states. Setting to None."
                )
            self.conditions = None

    def pad_dim0_with_sf(self, required_first_dim: int) -> None:
        r"""Extends a 2-dimensional batch of states along the first batch dimension.

        Given a batch of states (i.e. of `batch_shape=(a, b)`), extends `a` to a
        States object of `batch_shape = (required_first_dim, b)`, by adding the
        required number of $s_f$ tensors. This is useful to extend trajectories of
        different lengths.

        Args:
            required_first_dim: The size of the first batch dimension post-expansion.
        """

        if len(self.batch_shape) != 2:
            raise ValueError(
                f"pad_dim0_with_sf is not implemented for States of type "
                f"{self.__class__.__name__} nor for batch shapes {self.batch_shape}"
            )

        if self.batch_shape[0] >= required_first_dim:
            return

        pad_count = required_first_dim - self.batch_shape[0]
        self.tensor = torch.cat(
            (
                self.tensor,
                self.__class__.sf.repeat(pad_count, self.batch_shape[1], 1).to(
                    self.tensor.device
                ),
            ),
            dim=0,
        )
        # Pad conditions with -inf for sf states
        if self.conditions is not None:
            cond_pad = torch.full(
                (pad_count, self.batch_shape[1], self.conditions.shape[-1]),
                -float("inf"),
                device=self.device,
            )
            self.conditions = torch.cat((self.conditions, cond_pad), dim=0)

    def _compare(self, other: torch.Tensor) -> torch.Tensor:
        """Computes elementwise equality between state tensor and an external tensor.

        Note that this does not check if the conditions are equal.

        Args:
            other: Tensor with shape (*batch_shape, *state_shape) representing states to
            compare to.

        Returns:
            A boolean tensor of shape (*batch_shape,) indicating whether the states are
            equal to `other`.
        """
        n_batch_dims = len(self.batch_shape)
        if self.debug:
            full_shape = self.batch_shape + self.state_shape
            if not (
                other.shape == self.state_shape or other.shape == full_shape  # type: ignore[misc]
            ):
                raise ValueError(
                    f"Expected shape {self.state_shape} or {full_shape}, got {other.shape}."
                )

        # Broadcast single-state inputs instead of branching on shape at runtime.
        if other.shape == self.state_shape:
            other_expanded = other.view(
                *((1,) * n_batch_dims), *self.state_shape
            ).expand(*self.batch_shape, *self.state_shape)
        else:
            other_expanded = other

        out = self.tensor == other_expanded
        if len(self.__class__.state_shape) > 1:
            out = out.flatten(start_dim=n_batch_dims)
        out = out.all(dim=-1)

        if self.debug:
            assert out.shape == self.batch_shape

        return out

    @property
    def is_initial_state(self) -> torch.Tensor:
        r"""Returns a boolean tensor indicating which states are initial ($s_0$).

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for initial states.
        """
        if not isinstance(self.__class__.s0, torch.Tensor):
            raise NotImplementedError(
                "is_initial_state is not implemented by default "
                f"for {self.__class__.__name__}"
            )
        # We do not cast devices here to avoid breaking the graph when using
        # torch.compile. We use `ensure_same_device` to catch silent device drift
        # during testing.
        if self.debug:
            ensure_same_device(self.device, self.__class__.s0.device)
            if self.__class__.s0.shape != self.state_shape:
                raise ValueError(
                    f"s0 must have shape {self.state_shape}; got {self.__class__.s0.shape}"
                )

        return self._compare(self.__class__.s0)

    @property
    def is_sink_state(self) -> torch.Tensor:
        r"""Returns a boolean tensor indicating which states are sink ($s_f$).

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for sink states.
        """
        if not isinstance(self.__class__.sf, torch.Tensor):
            raise NotImplementedError(
                "is_sink_state is not implemented by default "
                f"for {self.__class__.__name__}"
            )

        # We do not cast devices here to avoid breaking the graph when using
        # torch.compile. We use `ensure_same_device` to catch silent device drift
        # during testing.
        if self.debug:
            ensure_same_device(self.device, self.__class__.sf.device)
            if self.__class__.sf.shape != self.state_shape:
                raise ValueError(
                    f"sf must have shape {self.state_shape}; got {self.__class__.sf.shape}"
                )

        return self._compare(self.__class__.sf)

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
            A new States object with the stacked states and conditions.
        """
        state_example = states[0]
        assert all(
            state.batch_shape == state_example.batch_shape for state in states
        ), "All states must have the same batch_shape"

        stacked_states = state_example.from_batch_shape(
            (0, 0), device=state_example.device
        )  # Empty.
        stacked_states.tensor = torch.stack([s.tensor for s in states], dim=0)

        # Stack conditions if all states have them
        if all(s.conditions is not None for s in states):
            cond_tensors = cast(list[torch.Tensor], [s.conditions for s in states])
            stacked_states.conditions = torch.stack(cond_tensors, dim=0)

        return stacked_states

    def to(self, device: torch.device) -> States:
        """Moves the States tensor to the specified device in-place.

        Args:
            device: The device to move to.

        Returns:
            The States object on the specified device.
        """
        self.tensor = self.tensor.to(device)
        if self.conditions is not None:
            self.conditions = self.conditions.to(device)
        return self


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

    Compile-related expectations:
    - Inputs (state tensor and masks) should already be on the target device with
      correct shapes; debug can be used to validate during development/tests.
    - Mask helpers reset masks before applying new conditions; rely on this behavior
      to avoid cross-step leakage.
    """

    n_actions: ClassVar[int]

    def __init__(
        self,
        tensor: torch.Tensor,
        forward_masks: Optional[torch.Tensor] = None,
        backward_masks: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> None:
        """Initializes a DiscreteStates container with a batch of states and masks.

        Args:
            tensor: Tensor of shape (*batch_shape, *state_shape) representing a batch of
                states.
            forward_masks: Optional boolean tensor of shape (*batch_shape, n_actions)
                indicating forward actions allowed at each state.
            backward_masks: Optional boolean tensor of shape (*batch_shape, n_actions - 1)
                indicating backward actions allowed at each state.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: The device to store the states on.
            debug: If True, run mask/state validations even in compiled contexts.
        """
        super().__init__(tensor, conditions=conditions, device=device, debug=debug)
        if debug:
            # Keep shape validation in debug to avoid graph breaks in compiled regions.
            assert tensor.shape == self.batch_shape + self.state_shape

        # In the usual case, no masks are provided and we produce these defaults.
        # Note: this **must** be updated externally by the env.
        if forward_masks is None:
            forward_masks = torch.ones(
                (*self.batch_shape, self.__class__.n_actions),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            forward_masks = forward_masks.to(self.device)
        if backward_masks is None:
            backward_masks = torch.ones(
                (*self.batch_shape, self.__class__.n_actions - 1),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            backward_masks = backward_masks.to(self.device)

        self.forward_masks: torch.Tensor = forward_masks
        self.backward_masks: torch.Tensor = backward_masks

        assert self.forward_masks.shape == (*self.batch_shape, self.n_actions)
        assert self.backward_masks.shape == (*self.batch_shape, self.n_actions - 1)

    def clone(self) -> DiscreteStates:
        """Returns a clone of the current instance.

        Returns:
            A new DiscreteStates object with the same data, masks, and conditions.
        """
        cloned = self.__class__(
            tensor=self.tensor.clone(),
            forward_masks=self.forward_masks.clone(),
            backward_masks=self.backward_masks.clone(),
            conditions=self.conditions.clone() if self.conditions is not None else None,
            debug=self.debug,
        )
        return cloned

    def _check_both_forward_backward_masks_exist(self):
        # Only validate in debug to avoid graph breaks in compiled regions.
        if self.debug:
            if not torch.is_tensor(self.forward_masks):
                raise TypeError("forward_masks must be tensors")
            if not torch.is_tensor(self.backward_masks):
                raise TypeError("backward_masks must be tensors")

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
            f"masks={tuple(self.forward_masks.shape)},",
        ]
        if self.conditions is not None:
            parts.append(f"conditions={self.conditions.shape},")
        parts.append(f"device={self.device})")
        return " ".join(parts)

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> DiscreteStates:
        """Returns a subset of the discrete states and their masks.

        Args:
            index: Indices to select states.

        Returns:
            A new DiscreteStates object with the selected states, masks, and conditions.
        """
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        conditions = self.conditions[index] if self.conditions is not None else None
        out = self.__class__(
            states, forward_masks, backward_masks, conditions, debug=self.debug
        )
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
        conditions = (
            self.conditions.view(-1, self.conditions.shape[-1])
            if self.conditions is not None
            else None
        )
        return self.__class__(
            states, forward_masks, backward_masks, conditions, debug=self.debug
        )

    def extend(self, other: DiscreteStates) -> None:
        """Concatenates another DiscreteStates object along the batch dimension.

        Args:
            other: DiscreteStates object to concatenate with.
        """
        assert self.device == other.device, "Devices must match"
        if len(other.batch_shape) == len(self.batch_shape) == 1:
            # This corresponds to adding a state to a trajectory
            self.tensor = torch.cat((self.tensor, other.tensor), dim=0)
        elif len(other.batch_shape) == len(self.batch_shape) == 2:
            # This corresponds to adding a trajectory to a batch of trajectories
            other = other.clone()
            self.pad_dim0_with_sf(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            other.pad_dim0_with_sf(
                required_first_dim=max(self.batch_shape[0], other.batch_shape[0])
            )
            self.tensor = torch.cat((self.tensor, other.tensor), dim=1)

        self.forward_masks = torch.cat(
            (self.forward_masks, other.forward_masks), dim=len(self.batch_shape) - 1
        )
        self.backward_masks = torch.cat(
            (self.backward_masks, other.backward_masks), dim=len(self.batch_shape) - 1
        )

        if self.conditions is not None and other.conditions is not None:
            self.conditions = torch.cat(
                (self.conditions, other.conditions), dim=len(self.batch_shape) - 1
            )
        else:
            # Inconsistent, raise a warning and set to None
            if self.conditions is not None or other.conditions is not None:
                warnings.warn(
                    "Inconsistent conditions when extending states. Setting to None."
                )
            self.conditions = None

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
            A new DiscreteStates object with the stacked states, masks, and conditions.
        """
        out = super().stack(states)
        # Note: conditions are already stacked by parent class
        assert isinstance(out, DiscreteStates)
        out.forward_masks = torch.stack([s.forward_masks for s in states], dim=0).to(
            out.device
        )
        out.backward_masks = torch.stack([s.backward_masks for s in states], dim=0).to(
            out.device
        )
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

        Notes:
            - Always resets `forward_masks` to all True before applying the new mask
              so updates do not leak across steps.
            - Works for 1D or 2D batch shapes; cond must match `batch_shape`.
            - Debug guards validate shape/dtype but should be off in compiled regions.
        """
        if self.debug:
            # Validate mask shape/dtype to catch silent misalignment during testing.
            expected_shape = self.batch_shape + (self.n_actions - 1,)
            if cond.shape != expected_shape:
                raise ValueError(
                    f"cond must have shape {expected_shape}; got {cond.shape}"
                )
            if cond.dtype is not torch.bool:
                raise ValueError(f"cond must be boolean; got {cond.dtype}")

        # Resets masks in place to prevent side-effects across steps.
        self.forward_masks[:] = True
        exit_mask = torch.zeros(
            self.batch_shape + (1,), device=cond.device, dtype=cond.dtype
        )

        if not allow_exit:
            exit_mask.fill_(True)

        # Concatenate and mask in a single tensor op to stay torch.compile friendly.
        # Sets the forward mask to be False where this concatenated mask is True.
        self.forward_masks[torch.cat([cond, exit_mask], dim=-1)] = False

    def set_exit_masks(self, batch_idx: torch.Tensor) -> None:
        """Sets forward masks such that the only allowable next action is to exit.

        A convenience function for common mask operations.

        Args:
            batch_idx: A boolean index along the batch dimension, along which to
                enforce exits.

        Notes:
            - Works for 1D or 2D batch shapes; `batch_idx` must match `batch_shape`.
            - Clears all actions for the selected batch entries, then sets only the
              exit action True via masked_fill to stay torch.compile friendly.
            - Does not move devices; expects masks/tensors already on the target device.
        """
        if self.debug:
            if batch_idx.shape != self.batch_shape:
                raise ValueError(
                    f"batch_idx must have shape {self.batch_shape}; got {batch_idx.shape}"
                )
            if batch_idx.dtype is not torch.bool:
                raise ValueError(f"batch_idx must be boolean; got {batch_idx.dtype}")

        # Avoid Python .item() to stay torch.compile friendly. For any True entry in
        # batch_idx (1D or 2D), zero all actions then set only the exit action True.
        self.forward_masks[batch_idx] = False
        # Use masked_fill on the last action slice to avoid advanced indexing graph breaks.
        self.forward_masks[..., -1].masked_fill_(batch_idx, True)

    def init_forward_masks(self, set_ones: bool = True) -> None:
        """Initalizes forward masks.

        A convienience function for common mask operations.

        Args:
            set_ones: if True, forward masks are initalized to all ones. Otherwise,
                they are initalized to all zeros.
        """
        shape = self.batch_shape + (self.n_actions,)
        if set_ones:
            self.forward_masks = torch.ones(shape).to(self.device).bool()
        else:
            self.forward_masks = torch.zeros(shape).to(self.device).bool()

    def to(self, device: torch.device) -> DiscreteStates:
        """Moves the tensor and masks to the specified device in-place.

        Args:
            device: The device to move to.

        Returns:
            The DiscreteStates object on the specified device.
        """
        self.tensor = self.tensor.to(device)
        self.forward_masks = self.forward_masks.to(device)
        self.backward_masks = self.backward_masks.to(device)
        if self.conditions is not None:
            self.conditions = self.conditions.to(device)
        return self


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
        _device: The device on which the graphs are stored.
    """

    num_node_classes: ClassVar[int]
    num_edge_classes: ClassVar[int]
    is_directed: ClassVar[bool]
    max_nodes: ClassVar[int | None]

    s0: ClassVar[GeometricData]
    sf: ClassVar[GeometricData]

    def __init__(
        self,
        data: np.ndarray,
        categorical_node_features: bool = False,
        categorical_edge_features: bool = False,
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> None:
        """Initializes the GraphStates with a numpy array of `GeometricData` objects.

        Args:
            data: A numpy array of `GeometricData` objects representing individual graphs.
            categorical_node_features: Whether the node features are categorical.
            categorical_edge_features: Whether the edge features are categorical.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: The device to store the graphs on (optional).
            debug: If True, keep runtime validations enabled; stored for parity with
                tensor-based States.
        """
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        self.categorical_node_features = categorical_node_features
        self.categorical_edge_features = categorical_edge_features
        self.data = data
        # Keep a debug flag for interface consistency and future guarded checks.
        self.debug = debug

        # Resolve device per instance: prefer explicit, else infer, else default

        if device is not None:
            resolved_device = device
        else:
            inferred_device: torch.device | None = None

            # Get the device from the first graph in the data array.
            if data.size > 0:
                g = data.flat[0]
                assert isinstance(g, GeometricData)
                assert isinstance(g.x, torch.Tensor)
                inferred_device = g.x.device

            if inferred_device is not None:
                resolved_device = inferred_device
            else:
                resolved_device = torch.empty(0).device

        self._device = resolved_device

        # Move graphs to resolved device.
        if data.size > 0:
            g = data.flat[0]
            assert isinstance(g.x, torch.Tensor)
            if g.x.device != resolved_device:
                for graph in self.data.flat:
                    graph.to(str(resolved_device))

        # Initialize conditions (for conditional GFlowNets)
        self._conditions: torch.Tensor | None = None
        if conditions is not None:
            assert conditions.shape[:-1] == self.batch_shape, (
                f"Conditions batch shape {conditions.shape[:-1]} doesn't match "
                f"states batch shape {self.batch_shape}"
            )
            # condition should be of default float dtype (since dummy condition is -inf)
            assert conditions.dtype == torch.get_default_dtype()
            ensure_same_device(self.device, conditions.device)
            self.conditions = conditions

    @property
    def device(self) -> torch.device:
        """The device on which the states are stored.

        Returns:
            The device of the underlying array of GeometricData.
        """
        assert self._device is not None
        return self._device

    def to(self, device: torch.device) -> GraphStates:
        """Moves the GraphStates to the specified device.

        Args:
            device: The device to move to.

        Returns:
            The GraphStates object on the specified device.
        """
        for graph in self.data.flat:
            graph.to(str(device))
        self._device = device
        if self.conditions is not None:
            self.conditions = self.conditions.to(device)
        return self

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
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the graphs.

        Returns:
            The batch shape as a tuple.
        """
        return self.data.shape

    @classmethod
    def make_initial_states(
        cls,
        batch_shape: int | Tuple,
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> GraphStates:
        r"""Creates a numpy array of graphs consisting of initial states ($s_0$).

        Args:
            batch_shape: Shape of the batch dimensions.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: Device to create the graphs on.
            debug: If True, keeps compile graph-breaking checks in the logic for safety.

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
            conditions=conditions,
            debug=debug,
        )

    @classmethod
    def make_sink_states(
        cls,
        batch_shape: int | Tuple,
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> GraphStates:
        r"""Creates a numpy array of graphs consisting of sink states ($s_f$).

        Args:
            batch_shape: Shape of the batch dimensions.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: Device to create the graphs on.
            debug: If True, keeps compile graph-breaking checks in the logic for safety.

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
            conditions=conditions,
            device=device,
            debug=debug,
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
        curr_max_nodes = 0
        for graph in self.data.flat:
            if graph.x is not None:
                curr_max_nodes = max(curr_max_nodes, graph.x.size(0))

        if self.is_directed:
            max_possible_edges = curr_max_nodes * (curr_max_nodes - 1)
        else:
            max_possible_edges = curr_max_nodes * (curr_max_nodes - 1) // 2

        edge_masks = torch.ones(
            self.data.size, max_possible_edges, dtype=torch.bool, device=self.device
        )
        can_add_node = torch.ones(self.data.size, dtype=torch.bool, device=self.device)
        node_class_masks = torch.ones(
            self.data.size, self.num_node_classes, dtype=torch.bool, device=self.device
        )

        # Adding a node, so node index mask uses curr_max_nodes + 1.
        node_index_masks = torch.zeros(
            self.data.size, curr_max_nodes + 1, dtype=torch.bool, device=self.device
        )
        for i, graph in enumerate(self.data.flat):
            if graph.x is None:
                continue
            if self.max_nodes is not None and graph.x.size(0) >= self.max_nodes:
                can_add_node[i] = False
            node_class_masks[i] = can_add_node[i]

            # One hot encoding: only allow the next index to be added
            node_index_masks[i, graph.x.size(0)] = True

            ei0, ei1 = get_edge_indices(graph.x.size(0), self.is_directed, self.device)
            edge_masks[i, len(ei0) :] = False  # noqa: E203

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
        node_class_masks = node_class_masks.view(
            *self.batch_shape, self.num_node_classes
        )
        node_index_masks = node_index_masks.view(*self.batch_shape, curr_max_nodes + 1)

        # There are 3 action types: ADD_NODE, ADD_EDGE, EXIT
        action_type = torch.ones(
            *self.batch_shape, 3, dtype=torch.bool, device=self.device
        )
        # Ensure shape matches batch shape before assignment (handles stacked shapes)
        can_add_node = can_add_node.view(*self.batch_shape)
        action_type[..., GraphActionType.ADD_NODE] = can_add_node
        action_type[..., GraphActionType.ADD_EDGE] = torch.any(edge_masks, dim=-1)
        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.NODE_CLASS_KEY: node_class_masks,
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
            device=self.device,
        )

    @property
    def backward_masks(self) -> TensorDict:
        """Computes masks for valid backward actions from the current state.

        A backward action is valid if:
            1. The edge exists in the current graph (i.e., can be removed)
            2. The node exists in the current graph and no edges are connected to it

        For directed graphs, all existing edges are considered for removal.
        For undirected graphs, only the upper triangular edges are considered.

        The EXIT action is not included in backward masks.

        Returns:
            TensorDict: Boolean mask where True indicates valid actions.
        """
        # Get max nodes across all graphs, handling None values
        curr_max_nodes = 0
        for graph in self.data.flat:
            if graph.x is not None:
                curr_max_nodes = max(curr_max_nodes, graph.x.size(0))

        if self.is_directed:
            max_possible_edges = curr_max_nodes * (curr_max_nodes - 1)
        else:
            max_possible_edges = curr_max_nodes * (curr_max_nodes - 1) // 2

        # Disallow all actions
        edge_masks = torch.zeros(
            self.data.size, max_possible_edges, dtype=torch.bool, device=self.device
        )

        # Removing a node, so node index mask uses curr_max_nodes.
        node_index_masks = torch.zeros(
            self.data.size, curr_max_nodes, dtype=torch.bool, device=self.device
        )

        for i, graph in enumerate(self.data.flat):
            node_idxs = torch.arange(len(graph.x.flatten()), device=self.device)
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

        node_index_masks = node_index_masks.view(*self.batch_shape, curr_max_nodes)
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
            device=self.device,
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

        conditions = self.conditions[index] if self.conditions is not None else None
        return self.__class__(
            selected_graphs, conditions=conditions, device=self.device, debug=self.debug
        )

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
        if self.conditions is not None and graph.conditions is not None:
            self.conditions[index] = graph.conditions
        else:
            if self.conditions is not None or graph.conditions is not None:
                logger.warning(
                    "Inconsistent conditions when setting states. Setting to None."
                )
            self.conditions = None

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

    def clone(self) -> GraphStates:
        """Returns a detached clone of the current instance.

        Returns:
            A new GraphStates object with the same data.
        """
        cloned_graphs = np.empty(self.data.shape, dtype=object)
        for i, graph in enumerate(self.data.flat):
            cloned_graphs.flat[i] = graph.clone()

        conditions = self.conditions.clone() if self.conditions is not None else None
        return self.__class__(
            cloned_graphs, conditions=conditions, device=self.device, debug=self.debug
        )

    def pad_dim0_with_sf(self, required_first_dim: int) -> None:
        r"""Extends a 2-dimensional batch of graph states along the first batch dimension.

        Given a batch of states (i.e. of `batch_shape=(a, b)`), extends `a` to a
        GraphStates object of `batch_shape = (required_first_dim, b)`, by adding the
        required number of $s_f$ graphs. This is useful to extend trajectories of
        different lengths.

        Args:
            required_first_dim: The size of the first batch dimension post-expansion.
        """
        if len(self.batch_shape) != 2:
            raise ValueError(
                f"pad_dim0_with_sf requires batch_shape of length 2, "
                f"got {self.batch_shape}"
            )

        if self.batch_shape[0] >= required_first_dim:
            return

        pad_count = required_first_dim - self.batch_shape[0]
        sf_states = self.make_sink_states((pad_count, self.batch_shape[1]))
        self.data = np.concatenate([self.data, sf_states.data], axis=0)

        # Pad conditions with -inf for sf states
        if self.conditions is not None:
            cond_pad = torch.full(
                (pad_count, self.batch_shape[1], self.conditions.shape[-1]),
                -float("inf"),
                device=self.device,
            )
            self.conditions = torch.cat((self.conditions, cond_pad), dim=0)

    def extend(self, other: GraphStates):
        """Concatenates another GraphStates object along the batch dimension.

        Args:
            other: GraphStates object to concatenate with.
        """
        if len(self.batch_shape) == len(other.batch_shape) == 1:
            self.data = np.concatenate([self.data, other.data])

        elif len(self.batch_shape) == len(other.batch_shape) == 2:
            max_first_dim = max(self.batch_shape[0], other.batch_shape[0])

            # Pad both to the same first dimension using pad_dim0_with_sf
            self.pad_dim0_with_sf(max_first_dim)
            other.pad_dim0_with_sf(max_first_dim)

            # Concatenate along the second batch dimension
            self.data = np.concatenate([self.data, other.data], axis=1)

        else:
            raise ValueError(
                f"Cannot extend GraphStates with batch shape {other.batch_shape}"
            )

        # Handle conditions for 2D case
        if self.conditions is not None and other.conditions is not None:
            self.conditions = torch.cat(
                (self.conditions, other.conditions), dim=len(self.batch_shape) - 1
            )
        else:
            if self.conditions is not None or other.conditions is not None:
                logger.warning(
                    "Inconsistent conditions when extending states. Setting to None."
                )
            self.conditions = None

    def _compare(self, other: GeometricData) -> torch.Tensor:
        """Compares the current batch of graphs with another graph.

        Note that this does not check if the conditions are equal.

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
        g = self.sf

        if isinstance(g.x, torch.Tensor):
            try:
                ensure_same_device(self.device, cast(torch.Tensor, g.x).device)
                other = g
            except ValueError:
                other = g.clone()
                other.to(str(self.device))
        else:
            other = g

        return self._compare(other)

    @property
    def is_initial_state(self) -> torch.Tensor:
        r"""Returns a boolean tensor indicating which graphs are initial states ($s_0$).

        Returns:
            A boolean tensor of shape (*batch_shape,) that is True for initial states.
        """
        g = self.s0
        if getattr(g, "x", None) is not None:
            try:
                ensure_same_device(self.device, cast(torch.Tensor, g.x).device)  # type: ignore[attr-defined]
                other = g
            except ValueError:
                other = g.clone()
                other.to(str(self.device))
        else:
            other = g
        return self._compare(other)

    @classmethod
    def stack(cls, states: List[GraphStates]) -> GraphStates:
        """Stacks a list of GraphStates objects along a new dimension (0).

        Args:
            states: List of GraphStates objects to stack.

        Returns:
            A new GraphStates object with the stacked graphs and conditions.
        """
        # Check that all states have the same batch shape
        state_batch_shape = states[0].batch_shape
        assert all(state.batch_shape == state_batch_shape for state in states)

        graphs_list = [state.data for state in states]
        stacked_graphs = np.stack(graphs_list)

        # Stack conditions if all states have them
        conditions = None
        if all(s.conditions is not None for s in states):
            cond_tensors = cast(list[torch.Tensor], [s.conditions for s in states])
            conditions = torch.stack(cond_tensors, dim=0)

        return cls(stacked_graphs, conditions=conditions, device=states[0].device)
