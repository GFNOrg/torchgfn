from __future__ import annotations

from abc import ABC
from math import prod
from typing import ClassVar, Sequence, cast

import torch
from torchtyping import TensorType

from gfn.containers.base import Container

# Typing
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
DonesTensor = TensorType["batch_shape", torch.bool]
RewardsTensor = TensorType["batch_shape", torch.float]
OneStateTensor = TensorType["state_shape", torch.float]


def correct_cast(
    forward_masks: ForwardMasksTensor | None,
    backward_masks: BackwardMasksTensor | None,
) -> tuple[ForwardMasksTensor, BackwardMasksTensor]:
    """
    Casts the given masks to the correct type, if they are not None.
    This function is to help with type checking only.
    """
    forward_masks = cast(ForwardMasksTensor, forward_masks)
    backward_masks = cast(BackwardMasksTensor, backward_masks)
    return forward_masks, backward_masks


class States(Container, ABC):
    """Base class for states, seen as nodes of the DAG.
    For each environment, a States subclass is needed. A `States` object
    is a collection of multiple states (nodes of the DAG). A tensor representation
    of the states is required for batching. If a state is represented with a tensor
    of shape (*state_shape), a batch of states is represented with a States object,
    with the attribute `states_tensor` of shape (*batch_shape, *state_shape). Other
    representations are possible (e.g. state as string, as numpy array, as graph, etc...),
    but these representations should not be batched.

    If the environment's action space is discrete, then each States object is also endowed
    with a `forward_masks` and `backward_masks` boolean attributes representing which actions
    are allowed at each state.

    A `batch_shape` attribute is also required, to keep track of the batch dimension.
    A trajectory can be represented by a States object with batch_shape = (n_states,).
    Multiple trajectories can be represented by a States object with batch_shape = (n_states, n_trajectories).

    Because multiple trajectories can have different lengths, batching requires appending a dummy tensor
    to trajectories that are shorter than the longest trajectory. The dummy state is the `s_f`
    attribute of the environment (e.g. [-1, ..., -1], or [-inf, ..., -inf], etc...). Which is never processed,
    and is used to pad the batch of states only.
    """

    state_shape: ClassVar[tuple[int, ...]]  # Shape of one state
    s0: ClassVar[OneStateTensor]  # Source state of the DAG
    sf: ClassVar[OneStateTensor]  # Dummy state, used to pad a batch of states

    def __init__(
        self,
        states_tensor: StatesTensor,
        forward_masks: ForwardMasksTensor | None = None,
        backward_masks: BackwardMasksTensor | None = None,
    ):
        self.states_tensor = states_tensor
        self.batch_shape = tuple(self.states_tensor.shape)[: -len(self.state_shape)]
        if forward_masks is None and backward_masks is None:
            try:
                self.forward_masks, self.backward_masks = self.make_masks()
                self.update_masks()
            except NotImplementedError:
                pass
        else:
            self.forward_masks = forward_masks
            self.backward_masks = backward_masks

        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @classmethod
    def from_batch_shape(cls, batch_shape: tuple[int], random: bool = False) -> States:
        """Create a States object with the given batch shape, all initialized to s_0.
        If random is True, the states are initialized randomly. This requires that
        the environment implements the `make_random_states_tensor` class method.
        """
        if random:
            states_tensor = cls.make_random_states_tensor(batch_shape)
        else:
            states_tensor = cls.make_initial_states_tensor(batch_shape)
        return cls(states_tensor)

    @classmethod
    def make_initial_states_tensor(cls, batch_shape: tuple[int]) -> StatesTensor:
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        return cls.s0.repeat(*batch_shape, *((1,) * state_ndim))

    @classmethod
    def make_random_states_tensor(cls, batch_shape: tuple[int]) -> StatesTensor:
        raise NotImplementedError(
            "The environment does not support initialization of random states."
        )

    def make_masks(self) -> tuple[ForwardMasksTensor, BackwardMasksTensor]:
        """Create the forward and backward masks for the states.
        This method is called only if the masks are not provided at initialization.
        """
        return NotImplementedError(
            "make_masks method not implemented. Your environment must implement it if discrete"
        )

    def update_masks(self) -> None:
        """Update the masks, if necessary.
        This method should be called after each action is taken.
        """
        return NotImplementedError(
            "update_masks method not implemented. Your environment must implement it if discrete"
        )

    def __len__(self):
        return prod(self.batch_shape)

    def __repr__(self):
        return f"{self.__class__.__name__} object of batch shape {self.batch_shape} and state shape {self.state_shape}"

    @property
    def device(self) -> torch.device:
        return self.states_tensor.device

    def __getitem__(self, index: int | Sequence[int] | Sequence[bool]) -> States:
        """Access particular states of the batch."""
        # TODO: add more tests for this method
        states = self.states_tensor[index]
        if self.forward_masks is None and self.backward_masks is None:
            return self.__class__(states)
        else:
            self.forward_masks, self.backward_masks = correct_cast(
                self.forward_masks, self.backward_masks
            )
            forward_masks = self.forward_masks[index]
            backward_masks = self.backward_masks[index]
            return self.__class__(
                states, forward_masks=forward_masks, backward_masks=backward_masks
            )

    def flatten(self) -> States:
        """Flatten the batch dimension of the states.
        This is useful for example when extracting individual states from trajectories.
        """
        states = self.states_tensor.view(-1, *self.state_shape)
        if self.forward_masks is None and self.backward_masks is None:
            return self.__class__(states)
        else:
            self.forward_masks, self.backward_masks = correct_cast(
                self.forward_masks, self.backward_masks
            )
            forward_masks = self.forward_masks.view(-1, self.forward_masks.shape[-1])
            backward_masks = self.backward_masks.view(-1, self.backward_masks.shape[-1])
            return self.__class__(
                states, forward_masks=forward_masks, backward_masks=backward_masks
            )

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
            self.states_tensor = torch.cat(
                (self.states_tensor, other.states_tensor), dim=0
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
            self.states_tensor = torch.cat(
                (self.states_tensor, other.states_tensor), dim=1
            )
        else:
            raise ValueError(
                f"extend is not implemented for batch shapes {self.batch_shape} and {other_batch_shape}"
            )
        if self.forward_masks is not None and self.backward_masks is not None:
            self.forward_masks, self.backward_masks = correct_cast(
                self.forward_masks, self.backward_masks
            )
            other.forward_masks, other.backward_masks = correct_cast(
                other.forward_masks, other.backward_masks
            )
            self.forward_masks = torch.cat(
                (self.forward_masks, other.forward_masks), dim=len(self.batch_shape) - 1
            )
            self.backward_masks = torch.cat(
                (self.backward_masks, other.backward_masks),
                dim=len(self.batch_shape) - 1,
            )

    def extend_with_sf(self, required_first_dim: int) -> None:
        """Takes a two-dimensional batch of states (i.e. of batch_shape (a, b)),
        and extends it to a States object of batch_shape (required_first_dim, b),
        by adding the required number of `s_f` tensors. This is useful to extend trajectories
        of different lengths."""
        if len(self.batch_shape) == 2:
            if self.batch_shape[0] >= required_first_dim:
                return
            self.states_tensor = torch.cat(
                (
                    self.states_tensor,
                    self.__class__.sf.repeat(
                        required_first_dim - self.batch_shape[0], self.batch_shape[1], 1
                    ),
                ),
                dim=0,
            )
            if self.forward_masks is not None and self.backward_masks is not None:
                self.forward_masks, self.backward_masks = correct_cast(
                    self.forward_masks, self.backward_masks
                )
                self.forward_masks = torch.cat(
                    (
                        self.forward_masks,
                        torch.ones(
                            required_first_dim - self.batch_shape[0],
                            *self.forward_masks.shape[1:],
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    ),
                    dim=0,
                )
                self.backward_masks = torch.cat(
                    (
                        self.backward_masks,
                        torch.ones(
                            required_first_dim - self.batch_shape[0],
                            *self.backward_masks.shape[1:],
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    ),
                    dim=0,
                )
            self.batch_shape = (required_first_dim, self.batch_shape[1])
        else:
            raise ValueError(
                f"extend_with_sf is not implemented for batch shapes {self.batch_shape}"
            )

    def compare(self, other: StatesTensor) -> DonesTensor:
        """Given a tensor of states, returns a tensor of booleans indicating whether the states
        are equal to the states in self.

        Args:
            other (StatesTensor): Tensor of states to compare to.

        Returns:
            DonesTensor: Tensor of booleans indicating whether the states are equal to the states in self.
        """
        out = self.states_tensor == other
        state_ndim = len(self.__class__.state_shape)
        for _ in range(state_ndim):
            out = out.all(dim=-1)
        return out

    @property
    def is_initial_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_0$ of the DAG.
        """
        source_states_tensor = self.__class__.s0.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        )
        return self.compare(source_states_tensor)

    @property
    def is_sink_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_f$ of the DAG.
        """
        sink_states = self.__class__.sf.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        )
        return self.compare(sink_states)

    @property
    def log_rewards(self) -> RewardsTensor:
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: RewardsTensor) -> None:
        self._log_rewards = log_rewards
