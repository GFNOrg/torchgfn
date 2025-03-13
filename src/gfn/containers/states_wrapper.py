from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Sequence, TypeVar, cast

import torch

if TYPE_CHECKING:
    from gfn.env import Env
    from gfn.states import States

from gfn.containers.base import Container

StateType = TypeVar("StateType", bound="States")


class StatesWrapper(Container, Generic[StateType]):
    """Container for states (mainly used for Flow Matching GFlowNet).

    This container holds states along with optional conditioning tensors.

    Attributes:
        env: Environment instance
        states: Set of states
        is_terminating: Boolean tensor indicating whether the states are terminating
        conditioning: Optional conditioning tensor, same shape as states.tensor
        log_rewards: Optional log rewards for terminating states
    """

    def __init__(
        self,
        env: Env,
        states: StateType | None = None,
        conditioning: torch.Tensor | None = None,
        is_terminating: torch.Tensor | None = None,
        log_rewards: torch.Tensor | None = None,
    ):
        """Initialize a StatesWrapper container.

        Args:
            env: Environment instance
            intermediary_states: First set of states
            terminating_states: Second set of states
            intermediary_conditioning: Optional conditioning for intermediary states
            terminating_conditioning: Optional conditioning for terminating states
        """
        self.env = env

        # Assert that all tensors are on the same device as the environment.
        device = self.env.device
        assert states.tensor.device == device if states is not None else True
        for tensor in [is_terminating, conditioning, log_rewards]:
            assert tensor.device == device if tensor is not None else True

        self.states = (
            states
            if states is not None
            else cast(StateType, env.states_from_batch_shape((0,)))
        )
        assert len(self.states.batch_shape) == 1

        self.conditioning = conditioning
        assert (
            self.conditioning is None
            or self.conditioning.shape == self.states.tensor.shape
        )

        self.is_terminating = (
            is_terminating
            if is_terminating is not None
            else torch.zeros(
                (len(self.states),), dtype=torch.bool, device=self.states.device
            )
        )
        assert (
            self.is_terminating.shape == self.states.batch_shape
            and self.is_terminating.dtype == torch.bool
        )

        # self._log_rewards can be torch.Tensor of shape same as self.states or None.
        if log_rewards is not None:
            self._log_rewards = log_rewards
        else:  # if log_rewards is None, there are two cases
            if (
                self.states.nelement() == 0
            ):  # 1) if we are initializing with empty states
                self._log_rewards = torch.full(
                    size=(0,), fill_value=0, dtype=torch.float, device=device
                )
            else:  # 2) we don't have log_rewards and need to compute them on the fly
                self._log_rewards = None
        assert self._log_rewards is None or (
            self._log_rewards.shape == self.states.batch_shape
            and self._log_rewards.dtype == torch.float
        )

    @property
    def intermediary_states(self) -> StateType:
        """Return the intermediary states."""
        return cast(StateType, self.states[~self.is_terminating])

    @property
    def terminating_states(self) -> StateType:
        """Return the terminating states."""
        return cast(StateType, self.states[self.is_terminating])

    @property
    def intermediary_conditioning(self) -> torch.Tensor | None:
        """Return the intermediary conditioning."""
        if self.conditioning is None:
            return None
        return self.conditioning[~self.is_terminating]

    @property
    def terminating_conditioning(self) -> torch.Tensor | None:
        """Return the terminating conditioning."""
        if self.conditioning is None:
            return None
        return self.conditioning[self.is_terminating]

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        return (
            f"StatesWrapper(n_states={len(self.states)}, "
            f"n_terminating={self.is_terminating.sum().item()})"
        )

    @property
    def last_states(self) -> StateType:
        """Get the last states, i.e. terminating states"""
        return self.terminating_states

    @property
    def log_rewards(self) -> torch.Tensor | None:
        """
        Returns the log rewards for the States as a tensor of shape (len(self.states),),
        with a value of `-float('inf')`for intermediate states.

        If the `log_rewards` are not provided during initialization, they are computed on the fly.
        """
        if self._log_rewards is None:
            self._log_rewards = torch.full(
                size=(len(self.states),),
                fill_value=-float("inf"),
                dtype=torch.float,
                device=self.states.device,
            )
            try:
                self._log_rewards[self.is_terminating] = self.env.log_reward(
                    self.last_states
                )
            except NotImplementedError:
                self._log_rewards[self.is_terminating] = torch.log(
                    self.env.reward(self.last_states)
                )

        assert self._log_rewards.shape == self.states.batch_shape
        return self._log_rewards

    @property
    def terminating_log_rewards(self) -> torch.Tensor:
        """Return the log rewards for the terminating states."""
        log_rewards = self.log_rewards
        assert log_rewards is not None
        return log_rewards[self.is_terminating]

    def extend(self, other: StatesWrapper[StateType]) -> None:
        """Extend this container with another StatesWrapper container."""
        assert len(self.states.batch_shape) == len(other.states.batch_shape) == 1

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
    ) -> StatesWrapper[StateType]:
        """Returns a subset of the states along the batch dimension.

        Note:
            The intermediary_states and terminating_states can have different batch shapes.
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

        # We can construct a new StatesWrapper with the same StateType
        return StatesWrapper[StateType](
            env=self.env,
            states=states,
            conditioning=conditioning,
            is_terminating=is_terminating,
            log_rewards=log_rewards,
        )
