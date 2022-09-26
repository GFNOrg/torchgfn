from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ..envs import Env
    from .states import States

import torch
from torchtyping import TensorType

# Typing  --- n_transitions is an int
Tensor2D = TensorType["max_length", "n_trajectories", torch.long]
Tensor2D2 = TensorType["n_trajectories", "shape"]
Tensor1D = TensorType["n_trajectories", torch.long]
FloatTensor1D = TensorType["n_trajectories", torch.float]


class Trajectories:
    """Class for keeping track of multiple COMPLETE trajectories, or backward trajectories."""

    def __init__(
        self,
        env: Env,
        states: States | None = None,
        actions: Tensor2D | None = None,
        when_is_done: Tensor1D | None = None,
        is_backward: bool = False,
        log_pfs: FloatTensor1D | None = None,  # log_probs of the trajectories
        log_pbs: FloatTensor1D | None = None,  # log_probs of the backward trajectories
    ) -> None:
        self.env = env
        self.is_backward = is_backward
        self.states = states if states is not None else env.States(batch_shape=(0, 0))
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
        self.log_pfs = log_pfs
        self.log_pbs = log_pbs

    def __repr__(self) -> str:
        states = self.states.states.transpose(0, 1)
        assert states.ndim == 3
        trajectories_representation = ""
        for traj in states:
            one_traj_repr = []
            for step in traj:
                one_traj_repr.append(str(step.numpy()))
                if step.equal(self.env.s_0 if self.is_backward else self.env.s_f):
                    break
            trajectories_representation += "-> ".join(one_traj_repr) + "\n"
        return (
            f"Trajectories(n_trajectories={self.n_trajectories}, max_length={self.max_length},"
            + f"states=\n{trajectories_representation}, actions=\n{self.actions.transpose(0, 1).numpy()}, "
            + f"when_is_done={self.when_is_done}, rewards={self.rewards})"
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
    def rewards(self) -> FloatTensor1D | None:
        if self.is_backward:
            return None
        return self.env.reward(self.last_states)

    def __getitem__(self, index: int | Sequence[int]) -> Trajectories:
        "Returns a subset of the `n_trajectories` trajectories."
        if isinstance(index, int):
            index = [index]
        when_is_done = self.when_is_done[index]
        new_max_length = when_is_done.max().item() if len(when_is_done) > 0 else 0
        states = self.states[:, index]
        actions = self.actions[:, index]
        states = states[: 1 + new_max_length]
        actions = actions[:new_max_length]
        log_pfs = self.log_pfs[index] if self.log_pfs is not None else None
        log_pbs = self.log_pbs[index] if self.log_pbs is not None else None
        return Trajectories(
            env=self.env,
            states=states,
            actions=actions,
            when_is_done=when_is_done,
            is_backward=self.is_backward,
            log_pfs=log_pfs,
            log_pbs=log_pbs,
        )

    def extend(self, other: Trajectories) -> None:
        """Extend the trajectories with another set of trajectories."""
        self.extend_actions(required_first_dim=max(self.max_length, other.max_length))
        other.extend_actions(required_first_dim=max(self.max_length, other.max_length))

        self.states.extend(other.states)
        self.actions = torch.cat((self.actions, other.actions), dim=1)
        self.when_is_done = torch.cat((self.when_is_done, other.when_is_done), dim=0)

    def extend_actions(self, required_first_dim: int) -> None:
        """Extends the actions along the first dimension by by adding -1s as necessary.
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

    def sample(self, n_trajectories: int) -> Trajectories:
        """Sample a random subset of trajectories."""
        perm = torch.randperm(self.n_trajectories)
        indices = perm[:n_trajectories]

        return self[indices]

    @staticmethod
    def revert_backward_trajectories(trajectories: Trajectories) -> Trajectories:
        """Revert the backward trajectories to forward trajectories."""
        assert trajectories.is_backward
        new_actions = torch.full_like(trajectories.actions, -1)
        new_actions = torch.cat(
            [new_actions, torch.full((1, len(trajectories)), -1)], dim=0
        )
        new_states = trajectories.env.s_f.repeat(  # type: ignore
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
            ] = trajectories.states.states[: trajectories.when_is_done[i] + 1, i].flip(
                0
            )
        new_states = trajectories.env.States(new_states)
        return Trajectories(
            env=trajectories.env,
            states=new_states,
            actions=new_actions,
            when_is_done=new_when_is_done,
            is_backward=False,
        )

    def save(self, directory: str) -> None:
        if self.is_backward:
            raise ValueError("Cannot save backward trajectories.")
        self.states.save(os.path.join(directory, "trajectories_states.pt"))
        torch.save(self.actions, os.path.join(directory, "trajectories_actions.pt"))
        torch.save(
            self.when_is_done, os.path.join(directory, "trajectories_when_is_done.pt")
        )

    def load(self, directory: str) -> None:
        self.states.load(os.path.join(directory, "trajectories_states.pt"))
        self.actions = torch.load(os.path.join(directory, "trajectories_actions.pt"))
        self.when_is_done = torch.load(
            os.path.join(directory, "trajectories_when_is_done.pt")
        )
