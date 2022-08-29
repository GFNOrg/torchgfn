from typing import Optional, Sequence, Union

import torch
from torchtyping import TensorType

from ..envs import Env
from .states import States

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
        states: Optional[States] = None,
        actions: Optional[Tensor2D] = None,
        when_is_done: Optional[Tensor1D] = None,
        is_backward: bool = False,
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
            f"states=\n{trajectories_representation}, actions=\n{self.actions.transpose(0, 1).numpy()}, "
            f"when_is_done={self.when_is_done}, rewards={self.rewards})"
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
        return self.when_is_done.max().item()

    @property
    def last_states(self) -> States:
        return self.states[self.when_is_done - 1, torch.arange(self.n_trajectories)]

    @property
    def rewards(self) -> Optional[FloatTensor1D]:
        if self.is_backward:
            return None
        return self.env.reward(self.last_states)

    def __getitem__(self, index: Union[int, Sequence[int]]) -> "Trajectories":
        "Returns a subset of the `n_trajectories` trajectories."
        if isinstance(index, int):
            index = [index]
        when_is_done = self.when_is_done[index]
        new_max_length = when_is_done.max().item() if len(when_is_done) > 0 else 0
        states = self.states[:, index]
        actions = self.actions[:, index]
        states = states[: 1 + new_max_length]
        actions = actions[:new_max_length]
        return Trajectories(
            env=self.env,
            states=states,
            actions=actions,
            when_is_done=when_is_done,
            is_backward=self.is_backward,
        )

    # def __setitem__(
    #     self, index: Union[int, Sequence[int]], value: "Trajectories"
    # ) -> None:
    #     if isinstance(index, int):
    #         index = [index]
    #     self.states[:, index] = value.states
    #     self.actions[:, index] = value.actions
    #     self.when_is_done[index] = value.when_is_done

    def extend(self, other: "Trajectories") -> None:
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
                    size=(required_first_dim - self.max_length, self.n_trajectories),
                    fill_value=-1,
                    dtype=torch.long,
                ),
            ),
            dim=0,
        )

    def sample(self, n_trajectories: int) -> "Trajectories":
        """Sample a random subset of trajectories."""
        perm = torch.randperm(self.n_trajectories)
        indices = perm[:n_trajectories]

        return self[indices]
