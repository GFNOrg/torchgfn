from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Generic, Sequence, TypeVar, cast

import torch

from gfn.env import ConditionalEnv

if TYPE_CHECKING:
    from gfn.env import Env
    from gfn.states import States

from gfn.containers.base import Container
from gfn.utils.common import ensure_same_device

StateType = TypeVar("StateType", bound="States")


class StatesContainer(Container, Generic[StateType]):
    """Container for a batch of states (mainly used for FMGFlowNet).

    This class manages a collection of states and their corresponding properties.
    It is mainly used for Flow Matching GFlowNet algorithms.

    Attributes:
        env: The environment where the states are defined.
        states: States with batch_shape (n_states,).
        conditioning: (Optional) Tensor of shape (n_states,) containing the conditioning
            for the states.
        is_terminating: Boolean tensor of shape (n_states,) indicating which states
            are terminating.
        _log_rewards: (Optional) Tensor of shape (n_states,) containing the log rewards
            for terminating states.
    """

    def __init__(
        self,
        env: Env,
        states: StateType | None = None,
        conditioning: torch.Tensor | None = None,
        is_terminating: torch.Tensor | None = None,
        log_rewards: torch.Tensor | None = None,
    ):
        """Initializes a StatesContainer instance.

        Args:
            env: The environment where the states are defined.
            states: States with batch_shape (n_states,). If None, an empty batch is
                created.
            conditioning: Optional tensor of shape (n_states,) containing the conditioning
                for the states.
            is_terminating: Boolean tensor of shape (n_states,) indicating which states
                are terminating. If None, all are set to False.
            log_rewards: Optional tensor of shape (n_states,) containing the log rewards
                for terminating states. If None, computed on the fly when needed.
        """
        self.env = env

        # Assert that all tensors are on the same device as the environment.
        device = self.env.device
        if states is not None:
            ensure_same_device(states.device, device)
        for tensor in [is_terminating, conditioning, log_rewards]:
            ensure_same_device(tensor.device, device) if tensor is not None else True

        self.states = (
            states
            if states is not None
            else cast(StateType, env.states_from_batch_shape((0,)))
        )
        assert len(self.states.batch_shape) == 1
        batch_shape = self.states.batch_shape

        self.conditioning = conditioning
        assert self.conditioning is None or (
            self.conditioning.shape[0] == len(self.states)
            and len(self.conditioning.shape) == 2
        )

        self.is_terminating = (
            is_terminating
            if is_terminating is not None
            else torch.zeros(
                (len(self.states),), dtype=torch.bool, device=self.states.device
            )
        )
        assert (
            self.is_terminating.shape == batch_shape
            and self.is_terminating.dtype == torch.bool
        )

        self._log_rewards = log_rewards
        assert self._log_rewards is None or (
            self._log_rewards.shape == batch_shape
            and self._log_rewards.is_floating_point()
        )

    @property
    def device(self) -> torch.device:
        """The device on which the states are stored.

        Returns:
            The device object of the `self.states`.
        """
        return self.states.device

    @property
    def intermediary_states(self) -> StateType:
        """The intermediary states (not initial states) of the StatesContainer.

        Returns:
            The intermediary states.
        """
        return cast(StateType, self.states[~self.states.is_initial_state])

    @property
    def terminating_states(self) -> StateType:
        """The last (terminating) states of the StatesContainer.

        Returns:
            The terminating states.
        """
        return cast(StateType, self.states[self.is_terminating])

    @property
    def intermediary_conditioning(self) -> torch.Tensor | None:
        """Conditioning for intermediary states.

        Returns:
            The conditioning tensor for intermediary states, or None if not set.
        """
        if self.conditioning is None:
            return None
        return self.conditioning[~self.states.is_initial_state]

    @property
    def terminating_conditioning(self) -> torch.Tensor | None:
        """Conditioning for terminating states.

        Returns:
            The conditioning tensor for terminating states, or None if not set.
        """
        if self.conditioning is None:
            return None
        return self.conditioning[self.is_terminating]

    def __len__(self) -> int:
        """Returns the number of states in the container.

        Returns:
            The number of states.
        """
        return len(self.states)

    def __repr__(self) -> str:
        """Returns a string representation of the StatesContainer.

        Returns:
            A string summary of the container.
        """
        return (
            f"StatesContainer(n_states={len(self.states)}, "
            f"n_terminating={self.is_terminating.sum().item()})"
        )

    @property
    def log_rewards(self) -> torch.Tensor:
        """The log rewards for all states.

        Returns:
            Log rewards tensor of shape (len(self.states),). Intermediate states have
                value -inf.

        Note:
            If not provided at initialization, log rewards are computed on demand for
            terminating states.
        """
        if self._log_rewards is None:
            self._log_rewards = torch.full(
                size=(len(self.states),),
                fill_value=-float("inf"),
                device=self.states.device,
            )
            if isinstance(self.env, ConditionalEnv):
                assert self.conditioning is not None
                log_reward_fn = partial(
                    self.env.log_reward,
                    conditions=self.conditioning[self.is_terminating],
                )
            else:
                log_reward_fn = self.env.log_reward
            self._log_rewards[self.is_terminating] = log_reward_fn(
                self.terminating_states
            )

        assert self._log_rewards.shape == self.states.batch_shape
        return self._log_rewards

    @property
    def terminating_log_rewards(self) -> torch.Tensor:
        """The log rewards for terminating states only.

        Returns:
            The log rewards for terminating states.
        """
        log_rewards = self.log_rewards
        assert log_rewards is not None
        return log_rewards[self.is_terminating]

    def extend(self, other: StatesContainer[StateType]) -> None:
        """Extends this container with another StatesContainer object.

        Args:
            Another StatesContainer to append.
        """
        assert len(self.states.batch_shape) == len(other.states.batch_shape) == 1

        if len(other) == 0:
            return

        if len(self) == 0 and other._log_rewards is not None:
            self._log_rewards = torch.full(
                size=(0,),
                fill_value=-float("inf"),
                device=self.device,
            )

        self.states.extend(other.states)
        self.is_terminating = torch.cat(
            (self.is_terminating, other.is_terminating), dim=0
        )

        # Concatenate conditioning tensors if they exist.
        if self.conditioning is not None and other.conditioning is not None:
            self.conditioning = torch.cat((self.conditioning, other.conditioning), dim=0)
        else:
            self.conditioning = None

        # Concatenate log_rewards of the trajectories if they exist.
        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = torch.cat((self._log_rewards, other._log_rewards), dim=0)
        else:
            self._log_rewards = None

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> StatesContainer[StateType]:
        """Returns a subset of the states along the batch dimension.

        Args:
            index: Indices to select states.

        Returns:
            A new StatesContainer with the selected states and associated data.
        """
        if isinstance(index, int):
            index = [index]

        # Cast the indexed states to maintain their type
        states = cast(StateType, self.states[index])
        is_terminating = self.is_terminating[index]
        conditioning = (
            self.conditioning[index] if self.conditioning is not None else None
        )
        log_rewards = self._log_rewards[index] if self._log_rewards is not None else None

        # We can construct a new StatesContainer with the same StateType
        return StatesContainer[StateType](
            env=self.env,
            states=states,
            conditioning=conditioning,
            is_terminating=is_terminating,
            log_rewards=log_rewards,
        )
