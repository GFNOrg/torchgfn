from __future__ import annotations  # This allows to use the class name in type hints

from abc import ABC, abstractmethod
from math import prod
from typing import ClassVar, Optional, Sequence, cast

import torch
from torchtyping import TensorType as TT


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
    in the DiscreteSpace subclass.

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
    s0: ClassVar[TT["state_shape", torch.float]]  # Source state of the DAG
    sf: ClassVar[
        TT["state_shape", torch.float]
    ]  # Dummy state, used to pad a batch of states

    def __init__(self, tensor: TT["batch_shape", "state_shape"]):
        """Initalize the State container with a batch of states.
        Args:
            tensor: Tensor representing a batch of states.
        """
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
    def make_initial_states_tensor(
        cls, batch_shape: tuple[int]
    ) -> TT["batch_shape", "state_shape", torch.float]:
        """Makes a tensor with a `batch_shape` of states consisting of $s_0`$s."""
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        return cls.s0.repeat(*batch_shape, *((1,) * state_ndim))

    @classmethod
    def make_random_states_tensor(
        cls, batch_shape: tuple[int]
    ) -> TT["batch_shape", "state_shape", torch.float]:
        """Makes a tensor with a `batch_shape` of random states, placeholder."""
        raise NotImplementedError(
            "The environment does not support initialization of random states."
        )

    @classmethod
    def make_sink_states_tensor(
        cls, batch_shape: tuple[int]
    ) -> TT["batch_shape", "state_shape", torch.float]:
        """Makes a tensor with a `batch_shape` of states consisting of $s_f$s."""
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

    def __getitem__(self, index: int | Sequence[int] | Sequence[bool]) -> States:
        """Access particular states of the batch."""
        return self.__class__(self.tensor[index])

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], states: States
    ) -> None:
        """Set particular states of the batch."""
        self.tensor[index] = states.tensor

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

    def compare(
        self, other: TT["batch_shape", "state_shape", torch.float]
    ) -> TT["batch_shape", torch.bool]:
        """Computes elementwise equality between state tensor with an external tensor.

        Args:
            other: Tensor of states to compare to.

        Returns: Tensor of booleans indicating whether the states are equal to the
            states in self.
        """
        out = self.tensor == other
        state_ndim = len(self.__class__.state_shape)
        for _ in range(state_ndim):
            out = out.all(dim=-1)
        return out

    @property
    def is_initial_state(self) -> TT["batch_shape", torch.bool]:
        """Return a tensor that is True for states that are $s_0$ of the DAG."""
        source_states_tensor = self.__class__.s0.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        )
        return self.compare(source_states_tensor)

    @property
    def is_sink_state(self) -> TT["batch_shape", torch.bool]:
        """Return a tensor that is True for states that are $s_f$ of the DAG."""
        sink_states = self.__class__.sf.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        ).to(self.tensor.device)
        return self.compare(sink_states)

    @property
    def log_rewards(self) -> TT["batch_shape", torch.float]:
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: TT["batch_shape", torch.float]) -> None:
        self._log_rewards = log_rewards


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
        tensor: TT["batch_shape", "state_shape", torch.float],
        forward_masks: Optional[TT["batch_shape", "n_actions", torch.bool]] = None,
        backward_masks: Optional[TT["batch_shape", "n_actions - 1", torch.bool]] = None,
    ) -> None:
        """Initalize a DiscreteStates container with a batch of states and masks.
        Args:
            tensor: A batch of states.
            forward_masks (optional): Initializes a boolean tensor of allowable forward
                policy actions.
            backward_masks (optional): Initializes a boolean tensor of allowable backward
                policy actions.
        """
        super().__init__(tensor)

        self.forward_masks = torch.ones(
            (*self.batch_shape, self.__class__.n_actions),
            dtype=torch.bool,
            device=self.__class__.device,
        )
        self.backward_masks = torch.ones(
            (*self.batch_shape, self.__class__.n_actions - 1),
            dtype=torch.bool,
            device=self.__class__.device,
        )
        if forward_masks is None and backward_masks is None:
            self.update_masks()
        else:
            self.forward_masks = cast(torch.Tensor, forward_masks)
            self.backward_masks = cast(torch.Tensor, backward_masks)

    @abstractmethod
    def update_masks(self) -> None:
        """Updates the masks, called after each action is taken."""

    def _check_both_forward_backward_masks_exist(self):
        assert self.forward_masks is not None and self.backward_masks is not None

    def __getitem__(
        self, index: int | Sequence[int] | Sequence[bool]
    ) -> DiscreteStates:
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        return self.__class__(states, forward_masks, backward_masks)

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
