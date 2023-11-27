from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gfn.actions import Actions
    from gfn.env import Env
    from gfn.states import States

import numpy as np
import torch
from torch import Tensor
from torchtyping import TensorType as TT

from gfn.containers.base import Container
from gfn.containers.transitions import Transitions


def is_tensor(t) -> bool:
    """Checks whether t is a torch.Tensor instance."""
    return isinstance(t, Tensor)


# TODO: remove env from this class?
class Trajectories(Container):
    """Container for complete trajectories (starting in $s_0$ and ending in $s_f$).

    Trajectories are represented as a States object with bi-dimensional batch shape.
    Actions are represented as an Actions object with bi-dimensional batch shape.
    The first dimension represents the time step, the second dimension represents
    the trajectory index. Because different trajectories may have different lengths,
    shorter trajectories are padded with the tensor representation of the terminal
    state ($s_f$ or $s_0$ depending on the direction of the trajectory), and
    actions is appended with dummy actions. The `when_is_done` tensor represents
    the time step at which each trajectory ends.

    Attributes:
        env: The environment in which the trajectories are defined.
        states: The states of the trajectories.
        actions: The actions of the trajectories.
        when_is_done: The time step at which each trajectory ends.
        is_backward: Whether the trajectories are backward or forward.
        log_rewards: The log_rewards of the trajectories.
        log_probs: The log probabilities of the trajectories' actions.

    """

    def __init__(
        self,
        env: Env,
        states: States | None = None,
        actions: Actions | None = None,
        when_is_done: TT["n_trajectories", torch.long] | None = None,
        is_backward: bool = False,
        log_rewards: TT["n_trajectories", torch.float] | None = None,
        log_probs: TT["max_length", "n_trajectories", torch.float] | None = None,
        estimator_outputs: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            env: The environment in which the trajectories are defined.
            states: The states of the trajectories.
            actions: The actions of the trajectories.
            when_is_done: The time step at which each trajectory ends.
            is_backward: Whether the trajectories are backward or forward.
            log_rewards: The log_rewards of the trajectories.
            log_probs: The log probabilities of the trajectories' actions.
            estimator_outputs: When forward sampling off-policy for an n-step
                trajectory, n forward passes will be made on some function approximator,
                which may need to be re-used (for example, for evaluating PF). To avoid
                duplicated effort, the outputs of the forward passes can be stored here.

        If states is None, then the states are initialized to an empty States object,
        that can be populated on the fly. If log_rewards is None, then `env.log_reward`
        is used to compute the rewards, at each call of self.log_rewards
        """
        self.env = env
        self.is_backward = is_backward
        self.states = (
            states.clone()  # TODO: Do we need this clone?
            if states is not None
            else env.States.from_batch_shape(batch_shape=(0, 0))
        )
        assert len(self.states.batch_shape) == 2
        self.actions = (
            actions
            if actions is not None
            else env.Actions.make_dummy_actions(batch_shape=(0, 0))
        )
        assert len(self.actions.batch_shape) == 2
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
        self.estimator_outputs = estimator_outputs

    def __repr__(self) -> str:
        states = self.states.tensor.transpose(0, 1)
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
            + f"states=\n{trajectories_representation}"
            # + f"actions=\n{self.actions.tensor.squeeze().transpose(0, 1)[:10].numpy()}, "
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

        return self.actions.batch_shape[0]

    @property
    def last_states(self) -> States:
        return self.states[self.when_is_done - 1, torch.arange(self.n_trajectories)]

    @property
    def log_rewards(self) -> TT["n_trajectories", torch.float] | None:
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
        """Returns a subset of the `n_trajectories` trajectories."""
        if isinstance(index, int):
            index = [index]
        when_is_done = self.when_is_done[index]
        new_max_length = when_is_done.max().item() if len(when_is_done) > 0 else 0
        states = self.states[:, index]
        actions = self.actions[:, index]
        states = states[: 1 + new_max_length]
        actions = actions[:new_max_length]
        if self.log_probs.shape != (0, 0):
            log_probs = self.log_probs[:, index]
            log_probs = log_probs[:new_max_length]
        else:
            log_probs = self.log_probs
        log_rewards = (
            self._log_rewards[index] if self._log_rewards is not None else None
        )

        if is_tensor(self.estimator_outputs):
            estimator_outputs = self.estimator_outputs[:, index]
            estimator_outputs = estimator_outputs[:new_max_length]
        else:
            estimator_outputs = None

        return Trajectories(
            env=self.env,
            states=states,
            actions=actions,
            when_is_done=when_is_done,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
            estimator_outputs=estimator_outputs,
        )

    @staticmethod
    def extend_log_probs(
        log_probs: TT["max_length", "n_trajectories", torch.float], new_max_length: int
    ) -> TT["max_max_length", "n_trajectories", torch.float]:
        """Extend the log_probs matrix by adding 0 until the required length is reached."""
        if log_probs.shape[0] >= new_max_length:
            return log_probs
        else:
            return torch.cat(
                (
                    log_probs,
                    torch.full(
                        size=(
                            new_max_length - log_probs.shape[0],
                            log_probs.shape[1],
                        ),
                        fill_value=0,
                        dtype=torch.float,
                        device=log_probs.device,
                    ),
                ),
                dim=0,
            )

    def extend(self, other: Trajectories) -> None:
        """Extend the trajectories with another set of trajectories.

        Extends along all attributes in turn (actions, states, when_is_done, log_probs,
        log_rewards).

        Args:
            other: an external set of Trajectories.
        """
        # TODO: The replay buffer is storing `dones` - this wastes a lot of space.
        self.actions.extend(other.actions)
        self.states.extend(other.states)
        self.when_is_done = torch.cat((self.when_is_done, other.when_is_done), dim=0)

        # For log_probs, we first need to make the first dimensions of self.log_probs and other.log_probs equal
        # (i.e. the number of steps in the trajectories), and then concatenate them
        new_max_length = max(self.log_probs.shape[0], other.log_probs.shape[0])
        self.log_probs = self.extend_log_probs(self.log_probs, new_max_length)
        other.log_probs = self.extend_log_probs(other.log_probs, new_max_length)

        self.log_probs = torch.cat((self.log_probs, other.log_probs), dim=1)

        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = torch.cat(
                (self._log_rewards, other._log_rewards),
                dim=0,
            )
        else:
            self._log_rewards = None

        # Either set, or append, estimator outputs if they exist in the submitted
        # trajectory.
        if self.estimator_outputs is None and is_tensor(other.estimator_outputs):
            self.estimator_outputs = other.estimator_outputs
        elif is_tensor(self.estimator_outputs) and is_tensor(other.estimator_outputs):
            batch_shape = self.actions.batch_shape
            n_bs = len(batch_shape)
            output_dtype = self.estimator_outputs.dtype

            if n_bs == 1:
                # Concatenate along the only batch dimension.
                self.estimator_outputs = torch.cat(
                    (self.estimator_outputs, other.estimator_outputs),
                    dim=0,
                )
            elif n_bs == 2:
                if self.estimator_outputs.shape[0] != other.estimator_outputs.shape[0]:
                    # First we need to pad the first dimension on either self or other.
                    self_shape = np.array(self.estimator_outputs.shape)
                    other_shape = np.array(other.estimator_outputs.shape)
                    required_first_dim = max(self_shape[0], other_shape[0])

                    # TODO: This should be a single reused function.
                    # The size of self needs to grow to match other along dim=0.
                    if self_shape[0] < other_shape[0]:
                        pad_dim = required_first_dim - self_shape[0]
                        pad_dim_full = (pad_dim,) + tuple(self_shape[1:])
                        output_padding = torch.full(
                            pad_dim_full,
                            fill_value=-float("inf"),
                            dtype=self.estimator_outputs.dtype,  # TODO: This isn't working! Hence the cast below...
                            device=self.estimator_outputs.device,
                        )
                        self.estimator_outputs = torch.cat(
                            (self.estimator_outputs, output_padding),
                            dim=0,
                        )

                    # The size of other needs to grow to match self along dim=0.
                    if other_shape[0] < self_shape[0]:
                        pad_dim = required_first_dim - other_shape[0]
                        pad_dim_full = (pad_dim,) + tuple(other_shape[1:])
                        output_padding = torch.full(
                            pad_dim_full,
                            fill_value=-float("inf"),
                            dtype=other.estimator_outputs.dtype,  # TODO: This isn't working! Hence the cast below...
                            device=other.estimator_outputs.device,
                        )
                        other.estimator_outputs = torch.cat(
                            (other.estimator_outputs, output_padding),
                            dim=0,
                        )

                # Concatenate the tensors along the second dimension.
                self.estimator_outputs = torch.cat(
                    (self.estimator_outputs, other.estimator_outputs),
                    dim=1,
                ).to(
                    dtype=output_dtype
                )  # Cast to prevent single precision becoming double precision... weird.

            # Sanity check. TODO: Remove?
            assert self.estimator_outputs.shape[:n_bs] == batch_shape

    def to_transitions(self) -> Transitions:
        """Returns a `Transitions` object from the trajectories."""
        states = self.states[:-1][~self.actions.is_dummy]
        next_states = self.states[1:][~self.actions.is_dummy]
        actions = self.actions[~self.actions.is_dummy]
        is_done = (
            next_states.is_sink_state
            if not self.is_backward
            else next_states.is_initial_state
        )
        if self._log_rewards is None:
            log_rewards = None
        else:
            log_rewards = torch.full(
                actions.batch_shape,
                fill_value=-float("inf"),
                dtype=torch.float,
                device=actions.device,
            )
            log_rewards[is_done] = torch.cat(
                [
                    self._log_rewards[self.when_is_done == i]
                    for i in range(self.when_is_done.max() + 1)
                ],
                dim=0,
            )
        log_probs = self.log_probs[~self.actions.is_dummy]
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
        """Returns all intermediate and terminating `States` from the trajectories.

        This is useful for the flow matching loss, that requires its inputs to be distinguished.

        Returns: a tuple containing all the intermediary states in the trajectories
            that are not s0, and all the terminating states in the trajectories that
            are not s0.
        """
        states = self.states
        intermediary_states = states[~states.is_sink_state & ~states.is_initial_state]
        terminating_states = self.last_states
        terminating_states.log_rewards = self.log_rewards
        return intermediary_states, terminating_states
