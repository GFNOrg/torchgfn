from dataclasses import dataclass
import torch
from torchtyping import TensorType
from gfn.envs import Env, AbstractStatesBatch


# Typing
LongTensor = TensorType["n_transitions", torch.long]
BoolTensor = TensorType["n_transitions", torch.bool]
FloatTensor = TensorType["n_transitions", torch.float]


@dataclass
class Transitions:
    "Container for transitions"
    env: Env
    n_transitions: int
    states: AbstractStatesBatch
    actions: LongTensor
    next_states: AbstractStatesBatch
    is_done: BoolTensor  # true when the corresponding action is the exit action
    rewards: FloatTensor  # should be zero for is_done=False

    def __repr__(self):
        states = self.states.states
        next_states = self.next_states.states
        assert states.ndim == 2

        states_repr = ",\t".join(
            [
                f"{str(state.numpy())}-> {str(next_state.numpy())}"
                for state, next_state in zip(states, next_states)
            ]
        )
        return (
            f"Transitions(n_transitions={self.n_transitions}, "
            f"transitions={states_repr}, actions={self.actions}, "
            f"is_done={self.is_done}, rewards={self.rewards})"
        )
