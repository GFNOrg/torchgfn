from __future__ import annotations  # This allows to use the class name in type hints

from abc import ABC, abstractmethod
from math import prod
from typing import ClassVar, Optional, Sequence, cast

import torch
from torchtyping import TensorType


class States(ABC):
    """Base class for states, seen as nodes of the DAG.
    For each environment, a States subclass is needed. A `States` object
    is a collection of multiple states (nodes of the DAG). A tensor representation
    of the states is required for batching. If a state is represented with a tensor
    of shape (*state_shape), a batch of states is represented with a States object,
    with the attribute `tensor` of shape (*batch_shape, *state_shape). Other
    representations are possible (e.g. state as string, as numpy array, as graph, etc...),
    but these representations cannot be batched.

    If the environment's action space is discrete (i.e. the environment subclasses `DiscreteEnv`),
    then each States object is also endowed with a `forward_masks` and `backward_masks` boolean attributes
    representing which actions are allowed at each state. This makes it possible to instantly access
    the allowed actions at each state, without having to call the environment's `validate_actions` method.
    Put different, `validate_actions` for such environments, directly calls the masks. This is handled in the
    DiscreteSpace subclass.

    A `batch_shape` attribute is also required, to keep track of the batch dimension.
    A trajectory can be represented by a States object with batch_shape = (n_states,).
    Multiple trajectories can be represented by a States object with batch_shape = (n_states, n_trajectories).

    Because multiple trajectories can have different lengths, batching requires appending a dummy tensor
    to trajectories that are shorter than the longest trajectory. The dummy state is the `s_f`
    attribute of the environment (e.g. [-1, ..., -1], or [-inf, ..., -inf], etc...). Which is never processed,
    and is used to pad the batch of states only.
    """

    state_shape: ClassVar[tuple[int, ...]]  # Shape of one state
    s0: ClassVar[TensorType["state_shape", torch.float]]  # Source state of the DAG
    sf: ClassVar[TensorType["state_shape", torch.float]]  # Dummy state, used to pad a batch of states

    def __init__(self, tensor: TensorType["batch_shape", "state_shape", torch.float]):
        self.tensor = tensor
        self.batch_shape = tuple(self.tensor.shape)[: -len(self.state_shape)]
        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @classmethod
    def from_batch_shape(
        cls, batch_shape: tuple[int], random: bool = False, sink: bool = False
    ) -> States:
        """Create a States object with the given batch shape, all initialized to s_0.
        If random is True, the states are initialized randomly. This requires that
        the environment implements the `make_random_states_tensor` class method.
        If sink is True, the states are initialized to s_f. Both random and sink
        cannot be True at the same time.
        """
        assert not (random and sink)
        if random:
            tensor = cls.make_random_states_tensor(batch_shape)
        elif sink:
            tensor = cls.make_sink_states_tensor(batch_shape)
        else:
            tensor = cls.make_initial_states_tensor(batch_shape)
        return cls(tensor)

    @classmethod
    def make_initial_states_tensor(cls, batch_shape: tuple[int]) -> TensorType["batch_shape", "state_shape", torch.float]:
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        return cls.s0.repeat(*batch_shape, *((1,) * state_ndim))

    @classmethod
    def make_random_states_tensor(cls, batch_shape: tuple[int]) -> TensorType["batch_shape", "state_shape", torch.float]:
        raise NotImplementedError(
            "The environment does not support initialization of random states."
        )

    @classmethod
    def make_sink_states_tensor(cls, batch_shape: tuple[int]) -> TensorType["batch_shape", "state_shape", torch.float]:
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
        # TODO: add more tests for this method
        return self.__class__(self.tensor[index])

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], states: States
    ) -> None:
        """Set particular states of the batch."""
        self.tensor[index] = states.tensor

    def flatten(self) -> States:
        """Flatten the batch dimension of the states.
        This is useful for example when extracting individual states from trajectories.
        """
        states = self.tensor.view(-1, *self.state_shape)
        return self.__class__(states)

    def extend(self, other: States) -> None:
        """Collates to another States object of the same batch shape, which should be 1 or 2.

        Args:
            other (States): Batch of states to collate to.

        Raises:
            ValueError: if self.batch_shape != other.batch_shape or if self.batch_shape != (1,) or (2,)
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
        """Takes a two-dimensional batch of states (i.e. of batch_shape (a, b)),
        and extends it to a States object of batch_shape (required_first_dim, b),
        by adding the required number of `s_f` tensors. This is useful to extend trajectories
        of different lengths."""
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

    def compare(self, other: TensorType["batch_shape", "state_shape", torch.float]) -> TensorType["batch_shape", torch.bool]:
        """Given a tensor of states, returns a tensor of booleans indicating whether the states
        are equal to the states in self.

        Args:
            other: Tensor of states to compare to.

        Returns:
            Tensor of booleans indicating whether the states are equal to the states in self.
        """
        out = self.tensor == other
        state_ndim = len(self.__class__.state_shape)
        for _ in range(state_ndim):
            out = out.all(dim=-1)
        return out

    @property
    def is_initial_state(self) -> TensorType["batch_shape", torch.bool]:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_0$ of the DAG.
        """
        source_states_tensor = self.__class__.s0.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        )
        return self.compare(source_states_tensor)

    @property
    def is_sink_state(self) -> TensorType["batch_shape", torch.bool]:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_f$ of the DAG.
        """
        sink_states = self.__class__.sf.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        )
        return self.compare(sink_states)

    @property
    def log_rewards(self) -> TensorType["batch_shape", torch.float]:
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: TensorType["batch_shape", torch.float]) -> None:
        self._log_rewards = log_rewards


class DiscreteStates(States, ABC):
    """Base class for states of discrete environments.
    States are endowed with a `forward_masks` and `backward_masks` boolean attributes
    representing which actions are allowed at each state. This makes it possible to instantly access
    the allowed actions at each state, without having to call the environment's `validate_actions` method.
    Put different, `validate_actions` for such environments, directly calls the masks.
    """

    n_actions: ClassVar[int]
    device: ClassVar[torch.device]

    def __init__(
        self,
        tensor: TensorType["batch_shape", "state_shape", torch.float],
        forward_masks: Optional[TensorType["batch_shape", "n_actions", torch.bool]] = None,
        backward_masks: Optional[TensorType["batch_shape", "n_actions - 1", torch.bool]] = None,
    ) -> None:
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
    def update_masks(self) -> None:  # TODO: why doesn't it take `states` as input ?
        # TODO: use the previous mask + action in order to get the new mask (for DAG-GFN environment)
        """Update the masks, if necessary.
        This method should be called after each action is taken.
        """
        pass

    def _check_both_forward_backward_masks_exist(self):
        assert self.forward_masks is not None and self.backward_masks is not None

    def __getitem__(self, index: int | Sequence[int] | Sequence[bool]) -> States:
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        return self.__class__(states, forward_masks, backward_masks)

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], states: States
    ) -> None:
        super().__setitem__(index, states)
        self._check_both_forward_backward_masks_exist()
        self.forward_masks[index] = states.forward_masks
        self.backward_masks[index] = states.backward_masks

    def flatten(self) -> States:
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
        super().extend_with_sf(required_first_dim)

        def _extend(masks, first_dim):
            return torch.cat(
                (
                    masks,
                    torch.ones(
                        first_dim - self.batch_shape[0],
                        *masks.shape[1:],
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ),
                dim=0,
            )

        self.forward_masks = _extend(self.forward_masks)
        self.backward_masks = _extend(self.backward_masks)
