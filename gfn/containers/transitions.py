from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from torchtyping import TensorType

from .trajectories import Trajectories

if TYPE_CHECKING:
    from ..envs import Env
    from .states import States

# Typing  -- n_transitions is either int or Tuple[int]
LongTensor = TensorType["n_transitions", torch.long]
BoolTensor = TensorType["n_transitions", torch.bool]
FloatTensor = TensorType["n_transitions", torch.float]


class Transitions:
    "Container for transitions"

    def __init__(
        self,
        env: Env,
        states: States | None = None,
        actions: LongTensor | None = None,
        is_done: BoolTensor | None = None,
        next_states: States | None = None,
        is_backward: bool = False,
    ):
        self.env = env
        self.is_backward = is_backward
        self.states = states if states is not None else env.States(batch_shape=(0,))
        assert len(self.states.batch_shape) == 1
        self.actions = (
            actions
            if actions is not None
            else torch.full(size=(0,), fill_value=-1, dtype=torch.long)
        )
        self.is_done = (
            is_done
            if is_done is not None
            else torch.full(size=(0,), fill_value=False, dtype=torch.bool)
        )
        self.next_states = (
            next_states if next_states is not None else env.States(batch_shape=(0,))
        )
        assert (
            len(self.next_states.batch_shape) == 1
            and self.states.batch_shape == self.next_states.batch_shape
        )

    @classmethod
    def from_trajectories(cls, trajectories: Trajectories) -> Transitions:
        "Create a Transitions from a Trajectories"
        states = trajectories.states[:-1][trajectories.actions != -1]
        next_states = trajectories.states[1:][trajectories.actions != -1]
        actions = trajectories.actions[trajectories.actions != -1]
        is_done = (
            next_states.is_sink_state
            if not trajectories.is_backward
            else next_states.is_initial_state
        )
        return cls(
            env=trajectories.env,
            states=states,
            actions=actions,
            is_done=is_done,
            next_states=next_states,
            is_backward=trajectories.is_backward,
        )

    @property
    def n_transitions(self) -> int:
        return self.states.batch_shape[0]

    def __len__(self) -> int:
        return self.n_transitions

    def __repr__(self):
        states_tensor = self.states.states
        next_states_tensor = self.next_states.states

        states_repr = ",\t".join(
            [
                f"{str(state.numpy())} -> {str(next_state.numpy())}"
                for state, next_state in zip(states_tensor, next_states_tensor)
            ]
        )
        return (
            f"Transitions(n_transitions={self.n_transitions}, "
            f"transitions={states_repr}, actions={self.actions}, "
            f"is_done={self.is_done}, rewards={self.rewards})"
        )

    @property
    def last_states(self) -> States:
        "Get the last states, i.e. terminating states"
        return self.states[self.is_done]

    @property
    def rewards(self) -> FloatTensor | None:
        if self.is_backward:
            return None
        else:
            rewards = torch.full(
                (self.n_transitions,),
                fill_value=-1.0,
                dtype=torch.float,
                device=self.states.device,
            )
            rewards[self.is_done] = self.env.reward(self.last_states)
            return rewards

    def __getitem__(self, index: int | Sequence[int]) -> Transitions:
        if isinstance(index, int):
            index = [index]
        states = self.states[index]
        actions = self.actions[index]
        is_done = self.is_done[index]
        next_states = self.next_states[index]
        return Transitions(
            env=self.env,
            states=states,
            actions=actions,
            is_done=is_done,
            next_states=next_states,
            is_backward=self.is_backward,
        )

    def extend(self, other: Transitions) -> None:
        self.states.extend(other.states)
        self.actions = torch.cat((self.actions, other.actions), dim=0)
        self.is_done = torch.cat((self.is_done, other.is_done), dim=0)
        self.next_states.extend(other.next_states)

    def sample(self, n_transitions: int) -> Transitions:
        "Sample a random subset of transitions"
        perm = torch.randperm(self.n_transitions)
        return self[perm[:n_transitions]]
