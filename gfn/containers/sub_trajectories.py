# TODO: merge with trajectories.py

from __future__ import annotations

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


class SubTrajectories:
    "Container for multiple sub-trajectories"

    def __init__(
        self,
        env: Env,
        states: States | None = None,
        actions: Tensor2D | None = None,
        when_is_done: Tensor1D | None = None,
    ) -> None:
        self.env = env
        self.states = states if states is not None else env.States(batch_shape=(0, 0))
        assert len(self.states.batch_shape) == 2
        self.actions = (
            actions
            if actions is not None
            else torch.full(
                size=(0, 0), fill_value=-1, dtype=torch.long, device=self.states.device
            )
        )
        self.when_is_done = (
            when_is_done
            if when_is_done is not None
            else torch.full(
                size=(0,), fill_value=-1, dtype=torch.long, device=self.states.device
            )
        )

    def __repr__(self) -> str:
        states = self.states.states.transpose(0, 1)
        assert states.ndim == 3
        trajectories_representation = ""
        for traj in states:
            one_traj_repr = []
            for step in traj:
                one_traj_repr.append(str(step.numpy()))
                if step.equal(self.env.s_f):
                    break
            trajectories_representation += "-> ".join(one_traj_repr) + "\n"
        return (
            f"Trajectories(n_trajectories={self.n_trajectories}, max_length={self.max_length},"
            f"states=\n{trajectories_representation}, actions=\n{self.actions.transpose(0, 1).numpy()}, "
            f"when_is_done={self.when_is_done})"
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
        return torch.any(self.actions != -1, dim=1).sum().item()

    def __getitem__(self, index: int | Sequence[int]) -> SubTrajectories:
        "Returns a subset of the `n_trajectories` trajectories."
        if isinstance(index, int):
            index = [index]
        when_is_done = self.when_is_done[index]
        states = self.states[:, index]
        actions = self.actions[:, index]
        new_max_length = (
            torch.any(actions != -1, dim=1).sum().item() if len(when_is_done) > 0 else 0
        )
        states = states[: 1 + new_max_length]
        actions = actions[:new_max_length]
        return SubTrajectories(
            env=self.env,
            states=states,
            actions=actions,
            when_is_done=when_is_done,
        )
