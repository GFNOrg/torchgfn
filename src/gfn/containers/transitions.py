from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from torchtyping import TensorType

if TYPE_CHECKING:
    from gfn.envs import Env
    from gfn.containers.states import States

from gfn.containers.base import Container

# Typing  -- n_transitions is either int or Tuple[int]
LongTensor = TensorType["n_transitions", torch.long]
BoolTensor = TensorType["n_transitions", torch.bool]
FloatTensor = TensorType["n_transitions", torch.float]
PairFloatTensor = TensorType["n_transitions", 2, torch.float]


class Transitions(Container):
    def __init__(
        self,
        env: Env,
        states: States | None = None,
        actions: LongTensor | None = None,
        is_done: BoolTensor | None = None,
        next_states: States | None = None,
        is_backward: bool = False,
        log_rewards: FloatTensor | None = None,
        log_probs: FloatTensor | None = None,
    ):
        """Container for transitions.

        Args:
            env (Env): Environment
            states (States, optional): States object with uni-dimensional batch_shape, representing the parents of the transitions. Defaults to None.
            actions (LongTensor, optional): Actions chosen at the parents of each transitions. Defaults to None.
            is_done (BoolTensor, optional): Whether the action is the exit action. Defaults to None.
            next_states (States, optional): States object with uni-dimensional batch_shape, representing the children of the transitions. Defaults to None.
            is_backward (bool, optional): Whether the transitions are backward transitions (i.e. next_states is the parent of states). Defaults to False.
            log_rewards (FloatTensor1D, optional): The log-rewards of the transitions (using a default value like -1 for non-terminating transitions). Defaults to None.
            log_probs (FloatTensor1D, optional): The log-probabilities of the actions. Defaults to None.

        When states and next_states are not None, the Transitions is an empty container that can be populated on the go.
        """
        self.env = env
        self.is_backward = is_backward
        self.states = (
            states
            if states is not None
            else env.States.from_batch_shape(batch_shape=(0,))
        )
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
            next_states
            if next_states is not None
            else env.States.from_batch_shape(batch_shape=(0,))
        )
        assert (
            len(self.next_states.batch_shape) == 1
            and self.states.batch_shape == self.next_states.batch_shape
        )

        self._log_rewards = log_rewards

        self.log_probs = log_probs if log_probs is not None else torch.zeros(0)

    @property
    def n_transitions(self) -> int:
        return self.states.batch_shape[0]

    def __len__(self) -> int:
        return self.n_transitions

    def __repr__(self):
        states_tensor = self.states.states_tensor
        next_states_tensor = self.next_states.states_tensor

        states_repr = ",\t".join(
            [
                f"{str(state.numpy())} -> {str(next_state.numpy())}"
                for state, next_state in zip(states_tensor, next_states_tensor)
            ]
        )
        return (
            f"Transitions(n_transitions={self.n_transitions}, "
            f"transitions={states_repr}, actions={self.actions}, "
            f"is_done={self.is_done})"
        )

    @property
    def last_states(self) -> States:
        "Get the last states, i.e. terminating states"
        return self.states[self.is_done]

    @property
    def log_rewards(self) -> FloatTensor | None:
        if self._log_rewards is not None:
            return self._log_rewards
        if self.is_backward:
            return None
        else:
            log_rewards = torch.full(
                (self.n_transitions,),
                fill_value=-1.0,
                dtype=torch.float,
                device=self.states.device,
            )
            try:
                log_rewards[self.is_done] = self.env.log_reward(self.last_states)
            except NotImplementedError:
                log_rewards[self.is_done] = torch.log(self.env.reward(self.last_states))
            return log_rewards

    @property
    def all_log_rewards(self) -> PairFloatTensor:
        """This is applicable to environments where all states are terminating.
        This function evaluates the rewards for all transitions that do not end in the sink state.
        This is useful for the Modified Detailed Balance loss."""
        if self.is_backward:
            raise NotImplementedError("Not implemented for backward transitions")
        is_sink_state = self.next_states.is_sink_state
        log_rewards = torch.full(
            (self.n_transitions, 2),
            fill_value=-1.0,
            dtype=torch.float,
            device=self.states.device,
        )
        try:
            log_rewards[~is_sink_state, 0] = self.env.log_reward(
                self.states[~is_sink_state]
            )
            log_rewards[~is_sink_state, 1] = self.env.log_reward(
                self.next_states[~is_sink_state]
            )
        except NotImplementedError:
            log_rewards[~is_sink_state, 0] = torch.log(
                self.env.reward(self.states[~is_sink_state])
            )
            log_rewards[~is_sink_state, 1] = torch.log(
                self.env.reward(self.next_states[~is_sink_state])
            )
        return log_rewards

    def __getitem__(self, index: int | Sequence[int]) -> Transitions:
        "Access particular transitions of the batch."
        if isinstance(index, int):
            index = [index]
        states = self.states[index]
        actions = self.actions[index]
        is_done = self.is_done[index]
        next_states = self.next_states[index]
        log_rewards = (
            self._log_rewards[index] if self._log_rewards is not None else None
        )
        log_probs = self.log_probs[index]
        return Transitions(
            env=self.env,
            states=states,
            actions=actions,
            is_done=is_done,
            next_states=next_states,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
        )

    def extend(self, other: Transitions) -> None:
        "Extend the Transitions object with another Transitions object."
        self.states.extend(other.states)
        self.actions = torch.cat((self.actions, other.actions), dim=0)
        self.is_done = torch.cat((self.is_done, other.is_done), dim=0)
        self.next_states.extend(other.next_states)
        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = torch.cat(
                (self._log_rewards, other._log_rewards), dim=0
            )
        else:
            self._log_rewards = None
        self.log_probs = torch.cat((self.log_probs, other.log_probs), dim=0)
