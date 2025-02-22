from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Sequence, TypeVar, cast

import torch

if TYPE_CHECKING:
    from gfn.env import Env
    from gfn.states import States

from gfn.containers.base import Container

StateType = TypeVar("StateType", bound="States")


class StatePairs(Container, Generic[StateType]):
    """Container for pairs of states with optional conditioning.

    This container holds two sets of states (intermediary and terminating) along with
    optional conditioning tensors for each. This is useful for algorithms that need
    to process different types of states separately, such as flow matching.

    Attributes:
        env: Environment instance
        intermediary_states: First set of states (e.g., non-terminal states)
        terminating_states: Second set of states (e.g., terminal states)
        intermediary_conditioning: Optional conditioning tensor for intermediary states
        terminating_conditioning: Optional conditioning tensor for terminating states
    """

    def __init__(
        self,
        env: Env,
        intermediary_states: StateType | None = None,
        terminating_states: StateType | None = None,
        intermediary_conditioning: torch.Tensor | None = None,
        terminating_conditioning: torch.Tensor | None = None,
    ):
        """Initialize a StatePairs container.

        Args:
            env: Environment instance
            intermediary_states: First set of states
            terminating_states: Second set of states
            intermediary_conditioning: Optional conditioning for intermediary states
            terminating_conditioning: Optional conditioning for terminating states
        """
        self.env = env
        self.intermediary_states = (
            intermediary_states
            if intermediary_states is not None
            else cast(StateType, env.states_from_batch_shape((0,)))
        )
        self.terminating_states = (
            terminating_states
            if terminating_states is not None
            else cast(StateType, env.states_from_batch_shape((0,)))
        )
        self.intermediary_conditioning = intermediary_conditioning
        self.terminating_conditioning = terminating_conditioning

    def __len__(self) -> int:
        return len(self.intermediary_states) + len(self.terminating_states)

    def __repr__(self) -> str:
        return (
            f"StatePairs(n_intermediary={len(self.intermediary_states)}, "
            f"n_terminating={len(self.terminating_states)})"
        )

    @property
    def log_rewards(self) -> torch.Tensor | None:
        """Return the log rewards of the terminating states."""
        # TODO: Eventually, there would be no need to store a log_reward in terminating_states.
        return self.terminating_states.log_rewards

    @property
    def last_states(self) -> StateType:
        """Return the terminating states."""
        return self.terminating_states

    def extend(self, other: StatePairs[StateType]) -> None:
        """Extend this container with another StatePairs container."""
        self.intermediary_states.extend(other.intermediary_states)
        self.terminating_states.extend(other.terminating_states)

        if (
            self.intermediary_conditioning is not None
            and other.intermediary_conditioning is not None
        ):
            self.intermediary_conditioning = torch.cat(
                (self.intermediary_conditioning, other.intermediary_conditioning), dim=0
            )
        if (
            self.terminating_conditioning is not None
            and other.terminating_conditioning is not None
        ):
            self.terminating_conditioning = torch.cat(
                (self.terminating_conditioning, other.terminating_conditioning), dim=0
            )

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> StatePairs[StateType]:
        """Returns a subset of the states."""
        if isinstance(index, int):
            index = [index]

        # Cast the indexed states to maintain their type
        intermediary_states = cast(StateType, self.intermediary_states[index])
        terminating_states = cast(StateType, self.terminating_states[index])

        intermediary_conditioning = (
            self.intermediary_conditioning[index]
            if self.intermediary_conditioning is not None
            else None
        )
        terminating_conditioning = (
            self.terminating_conditioning[index]
            if self.terminating_conditioning is not None
            else None
        )

        # We can construct a new StatePairs with the same StateType
        return StatePairs[StateType](
            env=self.env,
            intermediary_states=intermediary_states,
            terminating_states=terminating_states,
            intermediary_conditioning=intermediary_conditioning,
            terminating_conditioning=terminating_conditioning,
        )
