from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch

from gfn.actions import Actions
from gfn.containers.base import Container
from gfn.containers.states_container import StatesContainer
from gfn.containers.transitions import Transitions
from gfn.env import Env
from gfn.states import DiscreteStates, GraphStates, States
from gfn.utils.common import ensure_same_device


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
        env: The environment in which the trajectories are defined.
        states: The states of the trajectories.
        actions: The actions of the trajectories.
        terminating_idx: Tensor of shape (n_trajectories,) indicating the time step at which each trajectory ends.
        is_backward: Whether the trajectories are backward or forward.
        log_rewards: Tensor of shape (n_trajectories,) containing the log rewards of the trajectories.
        log_probs: Tensor of shape (max_length, n_trajectories) indicating the log probabilities of the
            trajectories' actions.
    """

    def __init__(
        self,
        env: Env,
        states: States | None = None,
        conditioning: torch.Tensor | None = None,
        actions: Actions | None = None,
        terminating_idx: torch.Tensor | None = None,
        is_backward: bool = False,
        log_rewards: torch.Tensor | None = None,
        log_probs: torch.Tensor | None = None,
        estimator_outputs: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            env: The environment in which the trajectories are defined.
            states: The states of the trajectories.
            conditioning: The conditioning of the trajectories for conditional MDPs.
            actions: The actions of the trajectories.
            terminating_idx: Tensor of shape (n_trajectories,) indicating the time step at which each trajectory ends.
            is_backward: Whether the trajectories are backward or forward.
            log_rewards: Tensor of shape (n_trajectories,) containing the log rewards of the trajectories.
            log_probs: Tensor of shape (max_length, n_trajectories) indicating the log probabilities of
                the trajectories' actions.
            estimator_outputs: Tensor of shape (batch_shape, output_dim).
                When forward sampling off-policy for an n-step trajectory,
                n forward passes will be made on some function approximator,
                which may need to be re-used (for example, for evaluating PF). To avoid
                duplicated effort, the outputs of the forward passes can be stored here.

        If states is None, then the states are initialized to an empty States object,
        that can be populated on the fly. If log_rewards is None, then `env.log_reward`
        is used to compute the rewards, at each call of self.log_rewards
        """
        self.env = env
        self.conditioning = conditioning
        self.is_backward = is_backward

        # Assert that all tensors are on the same device as the environment.
        device = self.env.device
        if isinstance(device, str):
            device = torch.device(device)

        for obj in [states, actions]:
            if obj is not None:
                if isinstance(obj.tensor, GeometricBatch):
                    ensure_same_device(obj.tensor.x.device, device)
                elif isinstance(obj.tensor, TensorDict):
                    ensure_same_device(obj.tensor["x"].device, device)
                else:
                    ensure_same_device(obj.tensor.device, device)
        for tensor in [
            conditioning,
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

        assert self.conditioning is None or (
            self.conditioning.shape[: len(self.states.batch_shape)]
            == self.states.batch_shape
        )

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
            else torch.full(size=(0,), fill_value=-1, dtype=torch.long, device=device)
        )
        assert (
            self.terminating_idx.shape == (self.n_trajectories,)
            and self.terminating_idx.dtype == torch.long
        )

        self._log_rewards = log_rewards
        assert self._log_rewards is None or (
            self._log_rewards.shape == (self.n_trajectories,)
            and self._log_rewards.dtype == torch.float
        )

        self.log_probs = log_probs
        assert self.log_probs is None or (
            self.log_probs.shape == self.actions.batch_shape
            and self.log_probs.dtype == torch.float
        )

        self.estimator_outputs = estimator_outputs
        assert self.estimator_outputs is None or (
            self.estimator_outputs.shape[: len(self.states.batch_shape)]
            == self.actions.batch_shape
            and self.estimator_outputs.dtype == torch.float
        )

    def __repr__(self) -> str:
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
                    one_traj_repr.append(str(step.cpu().numpy()))  # step.__repr__()
                    if self.is_backward and step.equal(self.env.s0):
                        break
                    elif not self.is_backward and step.equal(self.env.sf):
                        break
                trajectories_representation += "-> ".join(one_traj_repr) + "\n"
        return (
            f"Trajectories(n_trajectories={self.n_trajectories}, max_length={self.max_length}\n"
            + f"First {n_traj_to_print} trajectories:\n"
            + f"states=\n{trajectories_representation}"
            # + f"actions=\n{self.actions.tensor.squeeze().transpose(0, 1)[:10].numpy()}, "
            + f"terminating_idx={self.terminating_idx[:10].cpu().numpy()})"
        )

    @property
    def device(self) -> torch.device:
        return self.states.device

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
    def terminating_states(self) -> States:
        """Return the terminating states."""
        return self.states[self.terminating_idx - 1, torch.arange(self.n_trajectories)]

    @property
    def log_rewards(self) -> torch.Tensor | None:
        """Returns the log rewards for the Trajectories as a tensor of shape (n_trajectories,).

        If the `log_rewards` are not provided during initialization, they are computed on the fly.
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
        """Returns a subset of the `n_trajectories` trajectories."""
        if isinstance(index, int):
            index = [index]
        terminating_idx = self.terminating_idx[index]
        new_max_length = terminating_idx.max().item() if len(terminating_idx) > 0 else 0
        states = self.states[:, index]
        conditioning = (
            self.conditioning[:, index] if self.conditioning is not None else None
        )
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
            conditioning=conditioning,
            actions=actions,
            terminating_idx=terminating_idx,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
            estimator_outputs=estimator_outputs,
        )

    def extend(self, other: Trajectories) -> None:
        """Extend the trajectories with another set of trajectories.

        Extends along all attributes in turn (actions, states, terminating_idx, log_probs,
        log_rewards).

        Args:
            other: an external set of Trajectories.
        """
        if self.conditioning is not None:
            # TODO: Support the case
            raise NotImplementedError(
                "`extend` is not implemented for conditional Trajectories."
            )

        if len(other) == 0:
            return

        if len(self) == 0:
            if other._log_rewards is not None:
                self._log_rewards = torch.full(
                    (0,), fill_value=-float("inf"), dtype=torch.float, device=self.device
                )
            if other.log_probs is not None:
                self.log_probs = torch.full(
                    size=(0, 0), fill_value=0, dtype=torch.float, device=self.device
                )
            if other.estimator_outputs is not None:
                self.estimator_outputs = torch.full(
                    size=(0, 0, *other.estimator_outputs.shape[2:]),
                    fill_value=0,
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

        # Do the same for estimator_outputs, but padding with -float("inf") instead of 0.0.
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

    def to_transitions(self) -> Transitions:
        """Returns a `Transitions` object from the trajectories."""
        if self.conditioning is not None:
            expand_dims = (self.max_length,) + tuple(self.conditioning.shape)
            conditioning = self.conditioning.unsqueeze(0).expand(expand_dims)[
                ~self.actions.is_dummy
            ]
        else:
            conditioning = None

        states = self.states[:-1][~self.actions.is_dummy]
        next_states = self.states[1:][~self.actions.is_dummy]
        actions = self.actions[~self.actions.is_dummy]
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
                dtype=torch.float,
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

        # Initialize log_probs None if not available
        if self.has_log_probs:
            log_probs = self.log_probs[~self.actions.is_dummy]  # type: ignore
        else:
            log_probs = None

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
        )

    def to_states_container(self) -> StatesContainer:
        """Returns a `StatesContainer` object from the trajectories."""
        if not isinstance(self.states, DiscreteStates):
            raise TypeError("to_states_container only works with DiscreteStates")

        is_terminating = torch.zeros(
            self.states.batch_shape, dtype=torch.bool, device=self.states.device
        )
        is_terminating[self.terminating_idx - 1, torch.arange(len(self))] = True

        states = self.states.flatten()
        is_terminating = is_terminating.flatten()

        is_valid = ~states.is_sink_state & (
            ~states.is_initial_state | (states.is_initial_state & is_terminating)
        )
        states = states[is_valid]
        is_terminating = is_terminating[is_valid]

        conditioning = None
        if self.conditioning is not None:
            conditioning = self.conditioning.repeat(
                self.max_length + 1, *((1,) * (len(self.conditioning.shape) - 1))
            )[is_valid]

        if self.log_rewards is None:
            log_rewards = None
        else:
            log_rewards = torch.full(
                (len(states),),
                fill_value=-float("inf"),
                dtype=torch.float,
                device=states.device,
            )
            # Get the original indices (before flattening and filtering).
            orig_batch_indices = torch.arange(
                self.states.batch_shape[0], device=states.device
            ).repeat_interleave(self.states.batch_shape[1])
            orig_traj_indices = torch.arange(
                self.states.batch_shape[1], device=states.device
            ).repeat(self.states.batch_shape[0])

            # Retain only the valid indices.
            valid_batch_indices = orig_batch_indices[is_valid]
            valid_traj_indices = orig_traj_indices[is_valid]

            # Assign rewards to valid terminating states.
            terminating_mask = is_terminating & (
                valid_batch_indices == (self.terminating_idx[valid_traj_indices] - 1)
            )
            log_rewards[terminating_mask] = self.log_rewards[
                valid_traj_indices[terminating_mask]
            ]

        return StatesContainer[DiscreteStates](
            env=self.env,
            states=states,
            conditioning=conditioning,
            is_terminating=is_terminating,
            log_rewards=log_rewards,
        )

    def reverse_backward_trajectories(self, debug: bool = False) -> Trajectories:
        """Return a reversed version of the backward trajectories."""
        assert self.is_backward, "Trajectories must be backward."

        # TODO: Implement reverse backward trajectories for GraphStates.
        if isinstance(self.env.States, GraphStates):
            raise NotImplementedError(
                "Reverse backward trajectories are not implemented for GraphStates."
            )

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
        actions = self.actions.tensor  # shape (max_len, n_trajectories *action_dim)
        states = self.states.tensor  # shape (max_len + 1, n_trajectories, *state_dim)

        # Initialize new actions and states
        new_actions = self.env.dummy_action.repeat(max_len + 1, len(self), 1).to(actions)
        # shape (max_len + 1, n_trajectories, *action_dim)
        new_states = self.env.sf.repeat(max_len + 2, len(self), 1).to(states)
        # shape (max_len + 2, n_trajectories, *state_dim)

        # Create helper indices and masks
        idx = torch.arange(max_len).unsqueeze(1).expand(-1, len(self)).to(seq_lengths)
        rev_idx = seq_lengths - 1 - idx  # shape (max_len, n_trajectories)
        mask = rev_idx >= 0  # shape (max_len, n_trajectories)
        rev_idx[:, 1:] += seq_lengths.cumsum(0)[:-1]

        # Transpose for easier indexing
        actions = actions.transpose(0, 1)
        # shape (n_trajectories, max_len, *action_dim)
        new_actions = new_actions.transpose(0, 1)
        # shape (n_trajectories, max_len + 1, *action_dim)
        states = states.transpose(0, 1)
        # shape (n_trajectories, max_len + 1, *state_dim)
        new_states = new_states.transpose(0, 1)
        # shape (n_trajectories, max_len + 2, *state_dim)
        rev_idx = rev_idx.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # Assign reversed actions to new_actions
        new_actions[:, :-1][mask] = actions[mask][rev_idx[mask]]
        new_actions[torch.arange(len(self)), seq_lengths] = self.env.exit_action

        # Assign reversed states to new_states
        assert isinstance(states[:, -1], torch.Tensor)
        assert isinstance(
            self.env.s0, torch.Tensor
        ), "reverse_backward_trajectories not supported for Graph trajectories"
        assert torch.all(states[:, -1] == self.env.s0), "Last state must be s0"
        new_states[:, 0] = self.env.s0
        new_states[:, 1:-1][mask] = states[:, :-1][mask][rev_idx[mask]]

        # Transpose back
        new_actions = new_actions.transpose(
            0, 1
        )  # shape (max_len + 1, n_trajectories, *action_dim)
        new_states = new_states.transpose(
            0, 1
        )  # shape (max_len + 2, n_trajectories, *state_dim)

        reversed_trajectories = Trajectories(
            env=self.env,
            states=self.env.states_from_tensor(new_states),
            conditioning=self.conditioning,
            actions=self.env.actions_from_tensor(new_actions),
            terminating_idx=self.terminating_idx + 1,
            is_backward=False,
            log_rewards=self.log_rewards,
            log_probs=None,  # We can't simply pass the trajectories.log_probs
            # Since `log_probs` is assumed to be the forward log probabilities.
            # FIXME: To resolve this, we can save log_pfs and log_pbs in the trajectories object.
            estimator_outputs=None,  # Same as `log_probs`.
        )

        # ------------------------------ DEBUG ------------------------------
        # If `debug` is True (expected only when testing), compare the
        # vectorized approach's results (above) to the for-loop results (below).
        if debug:
            _new_actions = self.env.dummy_action.repeat(max_len + 1, len(self), 1).to(
                actions
            )  # shape (max_len + 1, n_trajectories, *action_dim)
            _new_states = self.env.sf.repeat(max_len + 2, len(self), 1).to(
                states
            )  # shape (max_len + 2, n_trajectories, *state_dim)

            for i in range(len(self)):
                _new_actions[self.terminating_idx[i], i] = self.env.exit_action
                _new_actions[: self.terminating_idx[i], i] = self.actions.tensor[
                    : self.terminating_idx[i], i
                ].flip(0)

                _new_states[: self.terminating_idx[i] + 1, i] = self.states.tensor[
                    : self.terminating_idx[i] + 1, i
                ].flip(0)

            assert torch.all(new_actions == _new_actions)
            assert torch.all(new_states == _new_states)

        return reversed_trajectories


def pad_dim0_if_needed(
    a: torch.Tensor, b: torch.Tensor, value: float = -float("inf")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads tensor a or b to match the first dimension of the other."""
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
