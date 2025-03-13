from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from gfn.actions import Actions
    from gfn.env import Env
    from gfn.states import States

from gfn.containers.base import Container
from gfn.utils.common import has_log_probs


class Transitions(Container):
    """Container for the transitions.

    Attributes:
        env: environment.
        is_backward: Whether the transitions are backward transitions (i.e.
            `next_states` is the parent of states).
        states: States object with uni-dimensional `batch_shape`, representing the
            parents of the transitions.
        actions: Actions chosen at the parents of each transitions.
        is_done: Whether the action is the exit action.
        next_states: States object with uni-dimensional `batch_shape`, representing
            the children of the transitions.
        log_probs: The log-probabilities of the actions.
    """

    def __init__(
        self,
        env: Env,
        states: States | None = None,
        conditioning: torch.Tensor | None = None,
        actions: Actions | None = None,
        is_done: torch.Tensor | None = None,
        next_states: States | None = None,
        is_backward: bool = False,
        log_rewards: torch.Tensor | None = None,
        log_probs: torch.Tensor | None = None,
    ):
        """Instantiates a container for transitions.

        When states and next_states are not None, the Transitions is an empty container
        that can be populated on the go.

        Args:
            env: Environment
            states: States object with uni-dimensional `batch_shape`, representing the
                parents of the transitions.
            conditioning: The conditioning of the transitions for conditional MDPs.
            actions: Actions chosen at the parents of each transitions.
            is_done: Tensor of shape (n_transitions,) indicating whether the action is the exit action.
            next_states: States object with uni-dimensional `batch_shape`, representing
                the children of the transitions.
            is_backward: Whether the transitions are backward transitions (i.e.
                `next_states` is the parent of states).
            log_rewards: Tensor of shape (n_transitions,) containing the log-rewards of the transitions (using a
                default value like `-float('inf')` for non-terminating transitions).
            log_probs: Tensor of shape (n_transitions,) containing the log-probabilities of the actions.

        Raises:
            AssertionError: If states and next_states do not have matching
                `batch_shapes`.
        """
        self.env = env
        self.conditioning = conditioning
        self.is_backward = is_backward

        # Assert that all tensors are in the same device as the environment.
        device = self.env.device
        for obj in [states, actions, next_states]:
            assert obj.tensor.device == device if obj is not None else True
        for tensor in [conditioning, is_done, log_rewards, log_probs]:
            assert tensor.device == device if tensor is not None else True

        self.states = states if states is not None else env.states_from_batch_shape((0,))
        assert len(self.states.batch_shape) == 1

        self.actions = (
            actions if actions is not None else env.actions_from_batch_shape((0,))
        )
        assert self.actions.batch_shape == self.states.batch_shape

        self.is_done = (
            is_done
            if is_done is not None
            else torch.full(size=(0,), fill_value=False, dtype=torch.bool, device=device)
        )
        assert (
            self.is_done.shape == (self.n_transitions,)
            and self.is_done.dtype == torch.bool
        )

        self.next_states = (
            next_states if next_states is not None else env.states_from_batch_shape((0,))
        )
        assert (
            len(self.next_states.batch_shape) == 1
            and self.states.batch_shape == self.next_states.batch_shape
        )
        self._log_rewards = (
            log_rewards
            if log_rewards is not None
            else torch.zeros(0, dtype=torch.float, device=device)
        )
        assert (
            self._log_rewards.shape == (self.n_transitions,)
            and self._log_rewards.dtype == torch.float
        )
        self.log_probs = (
            log_probs
            if log_probs is not None
            else torch.zeros(0, dtype=torch.float, device=device)
        )
        assert (
            self.log_probs.shape == (self.n_transitions,) or len(self.log_probs) == 0
        ) and self.log_probs.dtype == torch.float

    @property
    def n_transitions(self) -> int:
        return self.states.batch_shape[0]

    def __len__(self) -> int:
        return self.n_transitions

    def __repr__(self):
        states_tensor = self.states.tensor
        next_states_tensor = self.next_states.tensor

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
    def log_rewards(self) -> torch.Tensor | None:
        """Compute the tensor of shape (n_transitions,) containing the log rewards for the transitions."""
        if self._log_rewards is not None:
            return self._log_rewards
        if self.is_backward:
            return None
        else:
            log_rewards = torch.full(
                (self.n_transitions,),
                fill_value=-float("inf"),
                dtype=torch.float,
                device=self.states.device,
            )
            try:
                log_rewards[self.is_done] = self.env.log_reward(self.last_states)
            except NotImplementedError:
                log_rewards[self.is_done] = torch.log(self.env.reward(self.last_states))
            return log_rewards

    @property
    def all_log_rewards(self) -> torch.Tensor:
        """Calculate all log rewards for the transitions.

        This is applicable to environments where all states are terminating. This
        function evaluates the rewards for all transitions that do not end in the sink
        state. This is useful for the Modified Detailed Balance loss.

        Returns:
            log_rewards: Tensor of shape (n_transitions, 2) containing the log rewards
                for the transitions.

        Raises:
            NotImplementedError: when used for backward transitions.
        """
        if self.is_backward:
            raise NotImplementedError("Not implemented for backward transitions")
        is_sink_state = self.next_states.is_sink_state
        log_rewards = torch.full(
            (self.n_transitions, 2),
            fill_value=-float("inf"),
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

        assert (
            log_rewards.shape == (self.n_transitions, 2)
            and log_rewards.dtype == torch.float
        )
        return log_rewards

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> Transitions:
        """Access particular transitions of the batch."""
        if isinstance(index, int):
            index = [index]
        states = self.states[index]
        actions = self.actions[index]
        is_done = self.is_done[index]
        next_states = self.next_states[index]
        log_rewards = self._log_rewards[index] if self._log_rewards is not None else None

        # Only return logprobs if they exist.
        log_probs = self.log_probs[index] if has_log_probs(self) else None

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
        """Extend the Transitions object with another Transitions object."""
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.is_done = torch.cat((self.is_done, other.is_done), dim=0)
        self.next_states.extend(other.next_states)

        # Concatenate log_rewards of the trajectories.
        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = torch.cat((self._log_rewards, other._log_rewards), dim=0)
        # Will not be None if object is initialized as empty.
        else:
            self._log_rewards = None
        self.log_probs = torch.cat((self.log_probs, other.log_probs), dim=0)
