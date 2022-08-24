from dataclasses import dataclass
from typing import Optional

import torch
from torchtyping import TensorType

from gfn.containers import States
from gfn.envs import Env

# Typing  -- n_transitions is either int or Tuple[int]
LongTensor = TensorType["n_transitions", torch.long]
BoolTensor = TensorType["n_transitions", torch.bool]
FloatTensor = TensorType["n_transitions", torch.float]


@dataclass
class Transitions:
    "Container for transitions"
    env: Env
    n_transitions: int
    states: States
    actions: LongTensor
    next_states: States
    is_done: BoolTensor  # true when the corresponding action is the exit action for forward transitions
    is_backward: bool = False

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
    def rewards(self) -> Optional[FloatTensor]:
        if self.is_backward:
            return None
        else:
            rewards = torch.full(
                (self.n_transitions,),
                fill_value=-1.0,
                dtype=torch.float,
                device=self.states.device,
            )
            rewards[self.is_done] = self.env.reward(self.states[self.is_done])
            return rewards
