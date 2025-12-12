from __future__ import annotations

from typing import Sequence

import torch

from gfn.actions import Actions
from gfn.containers.base import Container
from gfn.containers.states_container import StatesContainer
from gfn.containers.transitions import Transitions
from gfn.env import Env
from gfn.states import DiscreteStates, GraphStates, States
from gfn.utils.common import ensure_same_device, is_int_dtype


# TODO: remove env from this class?
class Trajectories(Container):
    """Container for complete trajectories (starting in $s_0$ and ending in $s_f$).

    Trajectories are represented as a States object with bi-dimensional batch shape.
    Actions are represented as an Actions object with bi-dimensional batch shape.
    The first dimension represents the time step, the second dimension represents
    the trajectory index. Because different trajectories may have different lengths,
    shorter trajectories are padded with the tensor representation of the terminal
    state ($s_f$ or $s_0$ depending on the direction of the trajectory), and
    actions is appended with dummy actions. The `terminating_idx` tensor represents
    the time step at which each trajectory ends.

    Attributes:
        env: The environment where the states and actions are defined.
        states: States with batch_shape (max_length+1, n_trajectories).
        actions: Actions with batch_shape (max_length, n_trajectories).
        terminating_idx: Tensor of shape (n_trajectories,) indicating the time step
            at which each trajectory ends.
        is_backward: Whether the trajectories are backward or forward. When not
            is_backward, the `states` are ordered from initial to terminal states.
            When is_backward, the `states` are ordered from terminal to initial states.
        _log_rewards: (Optional) Tensor of shape (n_trajectories,) containing the
            log rewards of the trajectories.
        log_probs: (Optional) Tensor of shape (max_length, n_trajectories) indicating
            the log probabilities of the trajectories' actions.
        estimator_outputs: (Optional) Tensor of shape (max_length, n_trajectories, ...)
            containing outputs of a function approximator for each step.
    """

    def __init__(
        self,
        env: Env,
        states: States | None = None,
        actions: Actions | None = None,
        terminating_idx: torch.Tensor | None = None,
        is_backward: bool = False,
        log_rewards: torch.Tensor | None = None,
        log_probs: torch.Tensor | None = None,
        estimator_outputs: torch.Tensor | None = None,
    ) -> None:
        """Initializes a Trajectories instance.

        Args:
            env: The environment where the states and actions are defined.
            states: States with batch_shape (max_length+1, n_trajectories). If None,
                an empty States object is created.
            actions: Actions with batch_shape (max_length, n_trajectories). If None,
                an empty Actions object is created.
            terminating_idx: Tensor of shape (n_trajectories,) indicating the time step
                at which each trajectory ends.
            is_backward: Whether the trajectories are backward or forward.
            log_rewards: Optional tensor of shape (n_trajectories,) containing the
                log rewards of the trajectories. If None, computed on the fly when needed.
            log_probs: Optional tensor of shape (max_length, n_trajectories) indicating
                the log probabilities of the trajectories' actions.
            estimator_outputs: Optional tensor of shape (max_length, n_trajectories, ...)
                containing outputs of a function approximator for each step.
        Note:
            When states and actions are not None, the Trajectories is initialized as
            an empty container that can be populated later with the `extend` method.
        """
        self.env = env
        self.is_backward = is_backward

        # Assert that all tensors are on the same device as the environment.
        device = self.env.device
        if isinstance(device, str):
            device = torch.device(device)

        for obj in [states, actions]:
            if obj is not None:
                ensure_same_device(obj.device, device)

        for tensor in [
            terminating_idx,
            log_rewards,
            log_probs,
            estimator_outputs,
        ]:
            if tensor is not None:
                ensure_same_device(tensor.device, device)

        self.states = (
            states if states is not None else env.states_from_batch_shape((0, 0))
        )
        assert len(self.states.batch_shape) == 2

        self.actions = (
            actions if actions is not None else env.actions_from_batch_shape((0, 0))
        )
        assert (self.actions.batch_shape == self.states.batch_shape == (0, 0)) or (
            self.actions.batch_shape
            == (self.states.batch_shape[0] - 1, self.states.batch_shape[1])
        )

        self.terminating_idx = (
            terminating_idx
            if terminating_idx is not None
            else torch.full(size=(0,), fill_value=-1, device=device)
        )
        assert self.terminating_idx.shape == (self.n_trajectories,) and is_int_dtype(
            self.terminating_idx
        )

        self._log_rewards = log_rewards
        assert self._log_rewards is None or (
            self._log_rewards.shape == (self.n_trajectories,)
            and self._log_rewards.is_floating_point()
        )

        self.log_probs = log_probs
        assert self.log_probs is None or (
            self.log_probs.shape == self.actions.batch_shape
            and self.log_probs.is_floating_point()
        )

        self.estimator_outputs = estimator_outputs
        assert self.estimator_outputs is None or (
            self.estimator_outputs.shape[: len(self.states.batch_shape)]
            == self.actions.batch_shape
            and self.estimator_outputs.is_floating_point()
        )

    def __repr__(self) -> str:
        """Returns a string representation of the Trajectories container.

        Returns:
            A string summary of the trajectories.
        """
        trajectories_representation = ""
        n_traj_to_print = min(10, self.n_trajectories)

        if isinstance(self.states, GraphStates):
            for i in range(n_traj_to_print):
                trajectories_representation += str(self.states[..., i]) + "\n"
        else:
            states = self.states.tensor.transpose(0, 1)
            assert states.ndim == 3
            assert isinstance(self.env.s0, torch.Tensor)
            assert isinstance(self.env.sf, torch.Tensor)
            for traj in states[:n_traj_to_print]:
                one_traj_repr = []
                for step in traj:
                    one_traj_repr.append(str(step.cpu().numpy()))
                    if self.is_backward and step.equal(self.env.s0):
                        break
                    elif not self.is_backward and step.equal(self.env.sf):
                        break
                trajectories_representation += "-> ".join(one_traj_repr) + "\n"
        return (
            "Trajectories("
            + f"n_trajectories={self.n_trajectories}, max_length={self.max_length}\n"
            + f"First {n_traj_to_print} trajectories:\n"
            + f"states=\n{trajectories_representation}"
            + f"terminating_idx={self.terminating_idx[:10].cpu().numpy()})"
        )

    @property
    def device(self) -> torch.device:
        """The device on which the trajectories are stored.

        Returns:
            The device object of the `self.states`.
        """
        return self.states.device

    @property
    def n_trajectories(self) -> int:
        """The number of trajectories in the container.

        Returns:
            The number of trajectories.
        """
        return self.states.batch_shape[1]

    def __len__(self) -> int:
        """Returns the number of trajectories in the container.

        Returns:
            The number of trajectories.
        """
        return self.n_trajectories

    @property
    def max_length(self) -> int:
        """The maximum length of the trajectories in the container.

        Returns:
            The maximum trajectory length.
        """
        if len(self) == 0:
            return 0

        return self.actions.batch_shape[0]

    @property
    def terminating_states(self) -> States:
        """The terminating states of the trajectories.

        Returns:
            The terminating states.
        """
        return self.states[self.terminating_idx - 1, torch.arange(self.n_trajectories)]

    @property
    def log_rewards(self) -> torch.Tensor | None:
        """The log rewards for the trajectories.

        Returns:
            Log rewards tensor of shape (n_trajectories,).

        Note:
            If not provided at initialization, log rewards are computed on demand for
            terminating states.
        """
        if self.is_backward:  # TODO: Why can't backward trajectories have log_rewards?
            return None

        if self._log_rewards is None:
            self._log_rewards = self.env.log_reward(self.terminating_states)

        assert self._log_rewards.shape == (self.n_trajectories,)
        return self._log_rewards

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> Trajectories:
        """Returns a subset of the trajectories along the batch dimension.

        Args:
            index: Indices to select trajectories.

        Returns:
            A new Trajectories object with the selected trajectories and associated data.
        """
        if isinstance(index, int):
            index = [index]
        terminating_idx = self.terminating_idx[index]
        new_max_length = terminating_idx.max().item() if len(terminating_idx) > 0 else 0
        states = self.states[:, index]
        actions = self.actions[:, index]
        states = states[: 1 + new_max_length]
        actions = actions[:new_max_length]
        if self.log_probs is not None:
            log_probs = self.log_probs[:, index]
            log_probs = log_probs[:new_max_length]
        else:
            log_probs = None
        log_rewards = self._log_rewards[index] if self._log_rewards is not None else None
        if self.estimator_outputs is not None:
            # TODO: Is there a safer way to index self.estimator_outputs for
            #       for n-dimensional estimator outputs?
            #
            # First we index along the first dimension of the estimator outputs.
            # This can be thought of as the instance dimension, and is
            # compatible with all supported indexing approaches (dim=1).
            # All dims > 1 are not explicitly indexed unless the dimensionality
            # of `index` matches all dimensions of `estimator_outputs` aside
            # from the first (trajectory) dimension.
            estimator_outputs = self.estimator_outputs[:, index]
            # Next we index along the trajectory length (dim=0)
            estimator_outputs = estimator_outputs[:new_max_length]
        else:
            estimator_outputs = None

        return Trajectories(
            env=self.env,
            states=states,
            actions=actions,
            terminating_idx=terminating_idx,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
            estimator_outputs=estimator_outputs,
        )

    def extend(self, other: Trajectories) -> None:
        """Extends this Trajectories object with another Trajectories object.

        Extends along all attributes in turn (actions, states, terminating_idx, log_probs,
        log_rewards).

        Args:
            other: Another Trajectories to append.
        """
        if len(other) == 0:
            return

        if len(self) == 0:
            if other._log_rewards is not None:
                self._log_rewards = torch.full(
                    (0,), fill_value=-float("inf"), device=self.device
                )
            if other.log_probs is not None:
                self.log_probs = torch.full(
                    size=(0, 0), fill_value=0.0, device=self.device
                )
            if other.estimator_outputs is not None:
                self.estimator_outputs = torch.full(
                    size=(0, 0, *other.estimator_outputs.shape[2:]),
                    fill_value=0.0,
                    dtype=other.estimator_outputs.dtype,
                    device=self.device,
                )

        # TODO: The replay buffer is storing `dones` - this wastes a lot of space.
        self.actions.extend(other.actions)
        self.states.extend(other.states)  # n_trajectories comes from this.
        self.terminating_idx = torch.cat(
            (self.terminating_idx, other.terminating_idx), dim=0
        )

        # Concatenate log_rewards of the trajectories.
        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = torch.cat((self._log_rewards, other._log_rewards), dim=0)
            assert len(self._log_rewards) == self.actions.batch_shape[-1]
        else:
            self._log_rewards = None

        # For log_probs, use `pad_dim0_if_needed` to ensure both tensors have the same
        # number of steps (first dimension), padding with 0.0 if necessary, before
        # concatenation.
        if self.log_probs is not None and other.log_probs is not None:
            self.log_probs, other.log_probs = pad_dim0_if_needed(
                self.log_probs, other.log_probs, 0.0
            )
            self.log_probs = torch.cat((self.log_probs, other.log_probs), dim=1)
            assert self.log_probs.shape == self.actions.batch_shape
        else:
            self.log_probs = None

        # Do the same for estimator_outputs, but padding with -float("inf") instead of 0.0
        if self.estimator_outputs is not None and other.estimator_outputs is not None:
            self.estimator_outputs, other.estimator_outputs = pad_dim0_if_needed(
                self.estimator_outputs, other.estimator_outputs
            )
            self.estimator_outputs = torch.cat(
                (self.estimator_outputs, other.estimator_outputs), dim=1
            )
            assert (
                self.estimator_outputs.shape[: len(self.actions.batch_shape)]
                == self.actions.batch_shape
            )
        else:
            self.estimator_outputs = None

    def to_transitions(self) -> Transitions:
        """Returns a Transitions object from the current Trajectories.

        Returns:
            A Transitions object with the same states, actions, and log_rewards as the
            current Trajectories.
        """
        valid_action_mask = ~self.actions.is_dummy

        states = self.states[:-1][valid_action_mask]
        next_states = self.states[1:][valid_action_mask]
        actions = self.actions[valid_action_mask]
        is_terminating = (
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
                device=actions.device,
            )
            # TODO: Can we vectorize this?
            log_rewards[is_terminating] = torch.cat(
                [
                    self._log_rewards[self.terminating_idx == i]
                    for i in range(self.terminating_idx.max() + 1)
                ],
                dim=0,
            )
        log_probs = (
            self.log_probs[~self.actions.is_dummy]
            if self.log_probs is not None
            else None
        )

        return Transitions(
            env=self.env,
            states=states,
            actions=actions,
            is_terminating=is_terminating,
            next_states=next_states,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
            # FIXME: Add estimator_outputs.
        )

    def to_states_container(self) -> StatesContainer:
        """Returns a StatesContainer object from the current Trajectories.

        Returns:
            A StatesContainer object with the same states, actions, and log_rewards as the
            current Trajectories.
        """
        if not isinstance(self.states, DiscreteStates):
            raise TypeError("to_states_container only works with DiscreteStates")

        is_terminating = torch.zeros(
            self.states.batch_shape, dtype=torch.bool, device=self.states.device
        )
        is_terminating[self.terminating_idx - 1, torch.arange(len(self))] = True

        states = self.states
        is_valid = ~states.is_sink_state & (
            ~states.is_initial_state | (states.is_initial_state & is_terminating)
        )
        states = states[is_valid]
        is_terminating = is_terminating[is_valid]

        if self.log_rewards is None:
            log_rewards = None
        else:
            log_rewards = torch.full(
                self.states.batch_shape, fill_value=-float("inf"), device=states.device
            )
            log_rewards[self.terminating_idx - 1, torch.arange(len(self))] = (
                self.log_rewards
            )
            log_rewards = log_rewards[is_valid]

        return StatesContainer[DiscreteStates](
            env=self.env,
            states=states,
            is_terminating=is_terminating,
            log_rewards=log_rewards,
            # FIXME: Add log_probs and estimator_outputs.
        )

    def reverse_backward_trajectories(self) -> Trajectories:
        """Returns a reversed version of the backward trajectories."""
        assert self.is_backward, "Trajectories must be backward."
        # env.sf should never be None unless something went wrong during class
        # instantiation.
        if self.env.sf is None:
            raise AttributeError(
                "Something went wrong during the instantiation of environment {}".format(
                    self.env
                )
            )

        # Compute sequence lengths and maximum length
        seq_lengths = self.terminating_idx  # shape (n_trajectories,)
        max_len = int(seq_lengths.max().item())

        # Get actions and states
        actions = self.actions  # shape (max_len, n_trajectories *action_dim)
        states = self.states  # shape (max_len + 1, n_trajectories, *state_dim)

        # Initialize new actions and states
        new_actions = self.env.Actions.make_dummy_actions(
            (max_len + 1, len(self)), device=actions.device
        )
        # shape (max_len + 1, n_trajectories, *action_dim)
        new_states = self.env.States.make_sink_states(
            (max_len + 2, len(self)), device=states.device
        )
        # shape (max_len + 2, n_trajectories, *state_dim)

        # Create helper indices and masks
        idx = (
            torch.arange(max_len, device=seq_lengths.device)
            .unsqueeze(1)
            .expand(-1, len(self))
        )

        rev_idx = seq_lengths.unsqueeze(0) - 1 - idx  # shape (max_len, n_trajectories)
        mask = rev_idx >= 0  # shape (max_len, n_trajectories)

        # -------------------------------------------------------------
        # Replace the previous transpose-based reversal logic with a
        # version that operates directly in (time, trajectory, *) space.
        # -------------------------------------------------------------

        # 1. Reverse actions ---------------------------------------------------
        # Gather linear indices where the mask is valid
        time_idx, traj_idx = torch.nonzero(mask, as_tuple=True)  # 1-D tensors
        src_time_idx = rev_idx[mask]  # Corresponding source time indices
        # Assign reversed actions
        new_actions[time_idx, traj_idx] = actions[src_time_idx, traj_idx]
        # Insert EXIT action right after the last real action of every trajectory
        new_actions[seq_lengths, torch.arange(len(self), device=seq_lengths.device)] = (
            self.env.Actions.make_exit_actions((1,), device=actions.device)
        )

        # 2. Reverse states ----------------------------------------------------
        # The last state of the backward trajectories must be s0.
        assert torch.all(states[-1].is_initial_state), "Last state must be s0"

        # First state of the forward trajectories is s0 for every trajectory
        new_states[0] = self.env.States.make_initial_states(
            (len(self),), device=states.device
        )  # Broadcast over the trajectory dimension

        # We do not want to copy the last state (s0) from the backward trajectory.
        states_excl_last = states[:-1]  # shape (max_len, n_trajectories, *state_dim)
        new_states_data = new_states[1:-1]  # shape (max_len, n_trajectories, *state_dim)
        new_states_data[time_idx, traj_idx] = states_excl_last[src_time_idx, traj_idx]

        # ---------------------------------------------------------------------
        # new_actions / new_states already have the correct shapes
        #   new_actions: (max_len + 1, n_trajectories, *action_dim)
        #   new_states:  (max_len + 2, n_trajectories, *state_dim)
        # ---------------------------------------------------------------------

        reversed_trajectories = Trajectories(
            env=self.env,
            states=new_states,
            actions=new_actions,
            terminating_idx=self.terminating_idx + 1,
            is_backward=False,
            log_rewards=self.log_rewards,
            log_probs=None,  # We can't simply pass the trajectories.log_probs
            # Since `log_probs` is assumed to be the forward log probabilities.
            # FIXME: To resolve this, we can save log_pfs and log_pbs in the
            # trajectories object.
            estimator_outputs=None,  # Same as `log_probs`.
        )

        return reversed_trajectories


def pad_dim0_if_needed(
    a: torch.Tensor, b: torch.Tensor, value: float = -float("inf")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads tensor a or b to match the first dimension of the other.

    Args:
        a: First tensor.
        b: Second tensor.
        value: Value to use for padding.

    Returns:
        Tuple of tensors with the same first dimension.
    """
    if a.shape[0] == b.shape[0]:
        return a, b

    max_dim0 = max(a.shape[0], b.shape[0])

    tensor_to_pad = a if a.shape[0] < b.shape[0] else b

    pad_dim = max_dim0 - min(a.shape[0], b.shape[0])
    pad_dim_full = (pad_dim,) + tuple(tensor_to_pad.shape[1:])
    output_padding = torch.full(
        pad_dim_full,
        fill_value=value,
        dtype=tensor_to_pad.dtype,
        device=tensor_to_pad.device,
    )

    tensor_to_pad = torch.cat((tensor_to_pad, output_padding), dim=0)

    return (
        tensor_to_pad if a.shape[0] < b.shape[0] else a,
        tensor_to_pad if a.shape[0] > b.shape[0] else b,
    )
