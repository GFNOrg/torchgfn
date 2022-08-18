from dataclasses import dataclass
from typing import Optional

import torch
from torchtyping import TensorType

from gfn.containers import States
from gfn.envs import Env

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
    rewards: Optional[FloatTensor1D]
    last_states: States
    is_backwards: bool = False

    def __repr__(self) -> str:
        states = self.states.states
        assert states.ndim == 3
        states = states.transpose(0, 1)
        trajectories_representation = ""
        for traj in states:
            one_traj_repr = []
            for step in traj:
                one_traj_repr.append(str(step.numpy()))
                if step.equal(self.env.s_0 if self.is_backwards else self.env.s_f):
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

    def get_last_states_raw(self) -> Tensor2D2:
        return self.states.states[-1]

    def purge(self, raw_state) -> None:
        # TODO: obsolete - remove or update
        """Remove all trajectories that ended in the given state."""
        ndim = self.states.shape[-1]
        mask = (self.get_last_states_raw() == raw_state).sum(1) == ndim
        self.n_trajectories -= mask.sum().item()
        self.states.states = self.states.states[:, ~mask]
        self.states.update_masks()
        self.actions = self.actions[:, ~mask]
        self.when_is_done = self.when_is_done[~mask]
        if self.rewards is not None:
            self.rewards = self.rewards[~mask]

    def sample(self, n_trajectories: int) -> "Trajectories":
        """Sample a random subset of trajectories."""
        perm = torch.randperm(self.n_trajectories)
        indices = perm[:n_trajectories]

        states_raw = self.states.states[:, indices, ...]
        states = self.env.States(states=states_raw)
        actions = self.actions[:, indices]
        when_is_done = self.when_is_done[indices]
        rewards = self.rewards[indices] if self.rewards is not None else None
        last_states_raw = self.last_states.states[indices, ...]
        last_states = self.env.States(states=last_states_raw)
        return Trajectories(
            env=self.env,
            n_trajectories=n_trajectories,
            states=states,
            actions=actions,
            when_is_done=when_is_done,
            rewards=rewards,
            last_states=last_states,
            is_backwards=self.is_backwards,
        )
