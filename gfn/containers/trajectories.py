from dataclasses import dataclass
from typing import Optional

import torch
from torchtyping import TensorType

from ..envs import Env
from .states import States

# Typing  --- n_transitions is an int
Tensor2D = TensorType["max_length", "n_trajectories", torch.long]
Tensor2D2 = TensorType["n_trajectories", "shape"]
Tensor1D = TensorType["n_trajectories", torch.long]
FloatTensor1D = TensorType["n_trajectories", torch.float]


@dataclass
class Trajectories:
    """Class for keeping track of multiple COMPLETE trajectories, or backward trajectories."""

    env: Env
    n_trajectories: int
    states: States
    actions: Tensor2D
    # The following field mentions how many actions were taken in each trajectory.
    when_is_done: Tensor1D
    is_backward: bool = False

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
        # states_repr = "\n".join(
        #     ["-> ".join([str(step.numpy()) for step in traj]) for traj in states]
        # )
        return (
            f"Trajectories(n_trajectories={self.n_trajectories}, "
            f"states=\n{trajectories_representation}, actions=\n{self.actions.transpose(0, 1).numpy()}, "
            f"when_is_done={self.when_is_done}, rewards={self.rewards})"
        )

    @property
    def last_states(self) -> States:
        mask = torch.nn.functional.one_hot(
            self.when_is_done - 1, num_classes=self.when_is_done.max().item() + 1
        ).T.bool()
        return self.states[mask]

    @property
    def rewards(self) -> Optional[FloatTensor1D]:
        if self.is_backward:
            return None
        return self.env.reward(self.last_states)

    def sample(self, n_trajectories: int) -> "Trajectories":
        """Sample a random subset of trajectories."""
        perm = torch.randperm(self.n_trajectories)
        indices = perm[:n_trajectories]

        states_raw = self.states.states[:, indices, ...]
        states = self.env.States(states=states_raw)
        actions = self.actions[:, indices]
        when_is_done = self.when_is_done[indices]
        return Trajectories(
            env=self.env,
            n_trajectories=n_trajectories,
            states=states,
            actions=actions,
            when_is_done=when_is_done,
            is_backward=self.is_backward,
        )
