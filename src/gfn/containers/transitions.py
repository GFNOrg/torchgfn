from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from gfn.actions import Actions
    from gfn.env import Env
    from gfn.states import States

from gfn.containers.base import Container
from gfn.utils.common import ensure_same_device


class Transitions(Container):
    """Container for a batch of transitions.

    This class manages a collection of transitions (triplet of states, actions, and
    next states) and their corresponding properties.

    Attributes:
        env: The environment where the states and actions are defined.
        states: States with batch_shape (n_transitions,).
        conditioning: (Optional) Tensor of shape (n_transitions,) containing the
            conditioning for the transitions.
        actions: Actions with batch_shape (n_transitions,). The actions make the
            transitions from the `states` to the `next_states`.
        is_terminating: Boolean tensor of shape (n_transitions,) indicating whether the
            action is the exit action.
        next_states: States with batch_shape (n_transitions,).
        is_backward: Whether the transitions are backward transitions. When not
            is_backward, the `states` are the parents of the transitions and the
            `next_states` are the children. When is_backward, the `states` are the
            children of the transitions and the `next_states` are the parents.
        _log_rewards: (Optional) Tensor of shape (n_transitions,) containing the log
            rewards of the transitions.
        log_probs: (Optional) Tensor of shape (n_transitions,) containing the log
            probabilities of the actions.
        backward_log_probs: (Optional) Tensor of shape (n_transitions,) containing the
            backward log probabilities of the actions.
    """

    def __init__(
        self,
        env: Env,
        states: States | None = None,
        conditioning: torch.Tensor | None = None,
        actions: Actions | None = None,
        is_terminating: torch.Tensor | None = None,
        next_states: States | None = None,
        is_backward: bool = False,
        log_rewards: torch.Tensor | None = None,
        log_probs: torch.Tensor | None = None,
        backward_log_probs: torch.Tensor | None = None,
    ):
        """Initializes a Transitions instance.

        Args:
            env: The environment where the states and actions are defined.
            states: States with batch_shape (n_transitions,). If None, an empty States
                object is created.
            conditioning: Optional tensor of shape (n_transitions,) containing the
                conditioning for the transitions.
            actions: Actions with batch_shape (n_transitions,). If None, an empty Actions
                object is created.
            is_terminating: Boolean tensor of shape (n_transitions,) indicating whether
                the action is the exit action.
            next_states: States with batch_shape (n_transitions,). If None, an empty
                States object is created.
            is_backward: Whether the transitions are backward transitions.
            log_rewards: Optional tensor of shape (n_transitions,) containing the log
                rewards for the transitions. If None, computed on the fly when needed.
            log_probs: Optional tensor of shape (n_transitions,) containing the log
                probabilities of the actions.
            backward_log_probs: Optional tensor of shape (n_transitions,) containing the
                backward log probabilities of the actions.


        Note:
            When states and next_states are not None, the Transitions is initialized as
            an empty container that can be populated later with the `extend` method.
        """
        self.env = env
        self.is_backward = is_backward

        # Assert that all tensors are on the same device as the environment.
        device = self.env.device
        for obj in [states, actions, next_states]:
            ensure_same_device(obj.device, device) if obj is not None else True
        for tensor in [conditioning, is_terminating, log_rewards, log_probs]:
            ensure_same_device(tensor.device, device) if tensor is not None else True

        self.states = states if states is not None else env.states_from_batch_shape((0,))
        assert len(self.states.batch_shape) == 1
        batch_shape = self.states.batch_shape

        self.conditioning = conditioning
        assert self.conditioning is None or (
            self.conditioning.shape[: len(batch_shape)] == batch_shape
        )

        self.actions = (
            actions if actions is not None else env.actions_from_batch_shape((0,))
        )
        assert self.actions.batch_shape == batch_shape

        self.is_terminating = (
            is_terminating
            if is_terminating is not None
            else torch.full(size=(0,), fill_value=False, dtype=torch.bool, device=device)
        )
        assert (
            self.is_terminating.shape == (self.n_transitions,)
            and self.is_terminating.dtype == torch.bool
        )

        self.next_states = (
            next_states if next_states is not None else env.states_from_batch_shape((0,))
        )
        assert self.next_states.batch_shape == batch_shape

        self._log_rewards = log_rewards
        assert self._log_rewards is None or (
            self._log_rewards.shape == (self.n_transitions,)
            and self._log_rewards.is_floating_point()
        )

        self.log_probs = log_probs
        assert self.log_probs is None or (
            self.log_probs.shape == self.actions.batch_shape
            and self.log_probs.is_floating_point()
        )

        self.backward_log_probs = backward_log_probs
        assert self.backward_log_probs is None or (
            self.backward_log_probs.shape == self.actions.batch_shape
            and self.backward_log_probs.is_floating_point()
        )

    @property
    def device(self) -> torch.device:
        """The device on which the transitions are stored.

        Returns:
            The device object of the `self.states`.
        """
        return self.states.device

    @property
    def n_transitions(self) -> int:
        """The number of transitions in the container.

        Returns:
            The number of transitions.
        """
        return self.states.batch_shape[0]

    def __len__(self) -> int:
        """Returns the number of transitions in the container.

        Returns:
            The number of transitions.
        """
        return self.n_transitions

    def __repr__(self):
        """Returns a string representation of the Transitions container.

        Returns:
            A string summary of the transitions.
        """
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
        )

    @property
    def terminating_states(self) -> States:
        """The terminating states of the transitions.

        Returns:
            The terminating states.
        """
        return self.states[self.is_terminating]

    @property
    def log_rewards(self) -> torch.Tensor | None:
        """The log rewards for the transitions.

        Returns:
            Log rewards tensor of shape (n_transitions,). Non-terminating transitions
            have value -inf.

        Note:
            If not provided at initialization, log rewards are computed on demand for
            terminating transitions.
        """
        if self.is_backward:
            return None

        if self._log_rewards is None:
            self._log_rewards = torch.full(
                (self.n_transitions,),
                fill_value=-float("inf"),
                device=self.states.device,
            )
            self._log_rewards[self.is_terminating] = self.env.log_reward(
                self.terminating_states
            )

        assert self._log_rewards.shape == (self.n_transitions,)
        return self._log_rewards

    @property
    def all_log_rewards(self) -> torch.Tensor:
        """A helper method to compute the log rewards for all transitions

        This is applicable to environments where all states are terminating. This
        function evaluates the rewards for all transitions that do not end in the sink
        state. This is useful for the Modified Detailed Balance loss.

        Returns:
            Log rewards tensor of shape (n_transitions, 2) for the transitions.
        """
        # TODO: reuse self._log_rewards if it exists.
        if self.is_backward:
            raise NotImplementedError("Not implemented for backward transitions")
        is_sink_state = self.next_states.is_sink_state
        log_rewards = torch.full(
            (self.n_transitions, 2),
            fill_value=-float("inf"),
            device=self.states.device,
        )
        log_rewards[~is_sink_state, 0] = self.env.log_reward(self.states[~is_sink_state])
        log_rewards[~is_sink_state, 1] = self.env.log_reward(
            self.next_states[~is_sink_state]
        )

        assert (
            log_rewards.shape == (self.n_transitions, 2)
            and log_rewards.is_floating_point()
        )
        return log_rewards

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> Transitions:
        """Returns a subset of the transitions along the batch dimension.

        Args:
            index: Indices to select transitions.

        Returns:
            A new Transitions object with the selected transitions and associated data.
        """
        if isinstance(index, int):
            index = [index]

        states = self.states[index]
        conditioning = (
            self.conditioning[index] if self.conditioning is not None else None
        )
        actions = self.actions[index]
        is_terminating = self.is_terminating[index]
        next_states = self.next_states[index]
        log_rewards = self._log_rewards[index] if self._log_rewards is not None else None
        log_probs = self.log_probs[index] if self.log_probs is not None else None
        backward_log_probs = (
            self.backward_log_probs[index]
            if self.backward_log_probs is not None
            else None
        )
        return Transitions(
            env=self.env,
            states=states,
            conditioning=conditioning,
            actions=actions,
            is_terminating=is_terminating,
            next_states=next_states,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
            backward_log_probs=backward_log_probs,
        )

    def extend(self, other: Transitions) -> None:
        """Extends this Transitions object with another Transitions object.

        Args:
            Another Transitions object to append.
        """
        if self.conditioning is not None:
            # TODO: Support the case
            raise NotImplementedError(
                "`extend` is not implemented for conditional Transitions."
            )

        if len(other) == 0:
            return

        if len(self) == 0:
            if other._log_rewards is not None:
                self._log_rewards = torch.full(
                    size=(0,),
                    fill_value=-float("inf"),
                    device=self.device,
                )
            if other.log_probs is not None:
                self.log_probs = torch.full(
                    size=(0,),
                    fill_value=0.0,
                    device=self.device,
                )
            if other.backward_log_probs is not None:
                self.backward_log_probs = torch.full(
                    size=(0,),
                    fill_value=0.0,
                    device=self.device,
                )

        assert len(self.states.batch_shape) == len(other.states.batch_shape) == 1

        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.is_terminating = torch.cat(
            (self.is_terminating, other.is_terminating), dim=0
        )
        self.next_states.extend(other.next_states)

        # Concatenate log_rewards of the trajectories.
        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = torch.cat((self._log_rewards, other._log_rewards), dim=0)
        else:
            self._log_rewards = None

        # Concatenate log_probs of the trajectories.
        if self.log_probs is not None and other.log_probs is not None:
            self.log_probs = torch.cat((self.log_probs, other.log_probs), dim=0)
        else:
            self.log_probs = None

        # Concatenate backward_log_probs of the trajectories.
        if self.backward_log_probs is not None and other.backward_log_probs is not None:
            self.backward_log_probs = torch.cat(
                (self.backward_log_probs, other.backward_log_probs), dim=0
            )
        else:
            self.backward_log_probs = None
