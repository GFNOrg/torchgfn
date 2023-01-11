from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gfn.envs import Env
    from gfn.containers.states import States

import torch
from torchtyping import TensorType

from gfn.containers.base import Container
from gfn.containers.transitions import Transitions

# Typing  --- n_transitions is an int
Tensor2D = TensorType["max_length", "n_trajectories", torch.long]
FloatTensor2D = TensorType["max_length", "n_trajectories", torch.float]
Tensor2D2 = TensorType["n_trajectories", "shape"]
Tensor1D = TensorType["n_trajectories", torch.long]
FloatTensor1D = TensorType["n_trajectories", torch.float]


class Trajectories(Container):
    def __init__(
        self,
        env: Env,
        states: States | None = None,
        actions: Tensor2D | None = None,
        when_is_done: Tensor1D | None = None,
        is_backward: bool = False,
        log_rewards: FloatTensor1D | None = None,
        log_probs: FloatTensor2D | None = None,
    ) -> None:
        """Container for complete trajectories (starting in s_0 and ending in s_f).
        Trajectories are represented as a States object with bi-dimensional batch shape.
        The first dimension represents the time step, the second dimension represents the trajectory index.
        Because different trajectories may have different lengths, shorter trajectories are padded with
        the tensor representation of the terminal state (s_f or s_0 depending on the direction of the trajectory), and
        actions is appended with -1's.
        The actions are represented as a two dimensional tensor with the first dimension representing the time step
        and the second dimension representing the trajectory index.
        The when_is_done tensor represents the time step at which each trajectory ends.


        Args:
            env (Env): The environment in which the trajectories are defined.
            states (States, optional): The states of the trajectories. Defaults to None.
            actions (Tensor2D, optional): The actions of the trajectories. Defaults to None.
            when_is_done (Tensor1D, optional): The time step at which each trajectory ends. Defaults to None.
            is_backward (bool, optional): Whether the trajectories are backward or forward. Defaults to False.
            log_rewards (FloatTensor1D, optional): The log_rewards of the trajectories. Defaults to None.
            log_probs (FloatTensor2D, optional): The log probabilities of the trajectories' actions. Defaults to None.

        If states is None, then the states are initialized to an empty States object, that can be populated on the fly.
        If log_rewards is None, then `env.log_reward` is used to compute the rewards, at each call of self.log_rewards
        """
        self.env = env
        self.is_backward = is_backward
        self.states = (
            states
            if states is not None
            else env.States.from_batch_shape(batch_shape=(0, 0))
        )
        assert len(self.states.batch_shape) == 2
        self.actions = (
            actions
            if actions is not None
            else torch.full(size=(0, 0), fill_value=-1, dtype=torch.long)
        )
        self.when_is_done = (
            when_is_done
            if when_is_done is not None
            else torch.full(size=(0,), fill_value=-1, dtype=torch.long)
        )
        self._log_rewards = log_rewards
        self.log_probs = (
            log_probs
            if log_probs is not None
            else torch.full(size=(0, 0), fill_value=0, dtype=torch.float)
        )

    def __repr__(self) -> str:
        states = self.states.states_tensor.transpose(0, 1)
        assert states.ndim == 3
        trajectories_representation = ""
        for traj in states[:10]:
            one_traj_repr = []
            for step in traj:
                one_traj_repr.append(str(step.numpy()))
                if step.equal(self.env.s0 if self.is_backward else self.env.sf):
                    break
            trajectories_representation += "-> ".join(one_traj_repr) + "\n"
        return (
            f"Trajectories(n_trajectories={self.n_trajectories}, max_length={self.max_length}, First 10 trajectories:"
            + f"states=\n{trajectories_representation}, actions=\n{self.actions.transpose(0, 1)[:10].numpy()}, "
            + f"when_is_done={self.when_is_done[:10].numpy()})"
        )

    @property
    def n_trajectories(self) -> int:
        return self.states.batch_shape[1]

    def __len__(self) -> int:
        return self.n_trajectories

    @property
    def max_length(self) -> int:
        if len(self) == 0:
            return 0

        return self.actions.shape[0]

    @property
    def last_states(self) -> States:
        return self.states[self.when_is_done - 1, torch.arange(self.n_trajectories)]

    @property
    def log_rewards(self) -> FloatTensor1D | None:
        if self._log_rewards is not None:
            assert self._log_rewards.shape == (self.n_trajectories,)
            return self._log_rewards
        if self.is_backward:
            return None
        try:
            return self.env.log_reward(self.last_states)
        except NotImplementedError:
            return torch.log(self.env.reward(self.last_states))

    def __getitem__(self, index: int | Sequence[int]) -> Trajectories:
        "Returns a subset of the `n_trajectories` trajectories."
        if isinstance(index, int):
            index = [index]
        when_is_done = self.when_is_done[index]
        new_max_length = when_is_done.max().item() if len(when_is_done) > 0 else 0
        states = self.states[:, index]
        actions = self.actions[:, index]
        log_probs = self.log_probs[:, index]
        states = states[: 1 + new_max_length]
        actions = actions[:new_max_length]
        log_probs = log_probs[:new_max_length]
        log_rewards = (
            self._log_rewards[index] if self._log_rewards is not None else None
        )

        return Trajectories(
            env=self.env,
            states=states,
            actions=actions,
            when_is_done=when_is_done,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
        )

    def extend(self, other: Trajectories) -> None:
        """Extend the trajectories with another set of trajectories."""
        self.extend_actions(required_first_dim=max(self.max_length, other.max_length))
        other.extend_actions(required_first_dim=max(self.max_length, other.max_length))

        self.states.extend(other.states)
        self.actions = torch.cat((self.actions, other.actions), dim=1)
        self.when_is_done = torch.cat((self.when_is_done, other.when_is_done), dim=0)
        self.log_probs = torch.cat((self.log_probs, other.log_probs), dim=1)

        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = torch.cat(
                (self._log_rewards, other._log_rewards), dim=0
            )
        else:
            self._log_rewards = None

    def extend_actions(self, required_first_dim: int) -> None:
        """Extends the actions and log_probs along the first dimension by by adding -1s as necessary.
        This is useful for extending trajectories of different lengths."""
        if self.max_length >= required_first_dim:
            return
        self.actions = torch.cat(
            (
                self.actions,
                torch.full(
                    size=(
                        required_first_dim - self.actions.shape[0],
                        self.n_trajectories,
                    ),
                    fill_value=-1,
                    dtype=torch.long,
                ),
            ),
            dim=0,
        )
        self.log_probs = torch.cat(
            (
                self.log_probs,
                torch.full(
                    size=(
                        required_first_dim - self.log_probs.shape[0],
                        self.n_trajectories,
                    ),
                    fill_value=0,
                    dtype=torch.float,
                ),
            ),
            dim=0,
        )

    @staticmethod
    def revert_backward_trajectories(trajectories: Trajectories) -> Trajectories:

        assert trajectories.is_backward
        new_actions = torch.full_like(trajectories.actions, -1)
        new_actions = torch.cat(
            [new_actions, torch.full((1, len(trajectories)), -1)], dim=0
        )
        new_states = trajectories.env.sf.repeat(
            trajectories.when_is_done.max() + 1, len(trajectories), 1
        )
        new_when_is_done = trajectories.when_is_done + 1
        for i in range(len(trajectories)):
            new_actions[trajectories.when_is_done[i], i] = (
                trajectories.env.n_actions - 1
            )
            new_actions[: trajectories.when_is_done[i], i] = trajectories.actions[
                : trajectories.when_is_done[i], i
            ].flip(0)
            new_states[
                : trajectories.when_is_done[i] + 1, i
            ] = trajectories.states.states_tensor[
                : trajectories.when_is_done[i] + 1, i
            ].flip(
                0
            )
        new_states = trajectories.env.States(new_states)
        return Trajectories(
            env=trajectories.env,
            states=new_states,
            actions=new_actions,
            log_probs=trajectories.log_probs,
            when_is_done=new_when_is_done,
            is_backward=False,
        )

    def to_transitions(self) -> Transitions:
        """Returns a `Transitions` object from the trajectories"""
        states = self.states[:-1][self.actions != -1]
        next_states = self.states[1:][self.actions != -1]
        actions = self.actions[self.actions != -1]
        is_done = (
            next_states.is_sink_state
            if not self.is_backward
            else next_states.is_initial_state
        )
        if self._log_rewards is None:
            log_rewards = None
        else:
            log_rewards = torch.full_like(actions, fill_value=-1.0, dtype=torch.float)
            log_rewards[is_done] = torch.cat(
                [
                    self._log_rewards[self.when_is_done == i]
                    for i in range(self.when_is_done.max() + 1)
                ],
                dim=0,
            )
        log_probs = self.log_probs[self.actions != -1]
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

    def to_states(self) -> States:
        """Returns a `States` object from the trajectories, containing all states in the trajectories"""
        states = self.states.flatten()
        return states[~states.is_sink_state]

    def to_non_initial_intermediary_and_terminating_states(
        self,
    ) -> tuple[States, States]:
        """Returns a tuple of `States` objects from the trajectories, containing all non-initial intermediary and all terminating states in the trajectories

        Returns:
            Tuple[States, States]: - All the intermediary states in the trajectories that are not s0.
                                   - All the terminating states in the trajectories that are not s0.
        """
        states = self.states
        intermediary_states = states[~states.is_sink_state & ~states.is_initial_state]
        terminating_states = self.last_states
        terminating_states.log_rewards = self.log_rewards
        return intermediary_states, terminating_states
