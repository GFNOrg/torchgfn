from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from gfn.containers.states import States
from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions
from gfn.losses.base import (
    EdgeDecomposableLoss,
    Loss,
    StateDecomposableLoss,
    TrajectoryDecomposableLoss,
)

if TYPE_CHECKING:
    from gfn.envs import Env


class ReplayBuffer:
    def __init__(
        self,
        env: Env,
        loss_fn: Loss | None = None,
        objects_type: Literal["transitions", "trajectories", "states"] | None = None,
        capacity: int = 1000,
    ):
        self.env = env
        self.capacity = capacity
        self.terminating_states = None
        if objects_type == "trajectories" or isinstance(
            loss_fn, TrajectoryDecomposableLoss
        ):
            self.training_objects = Trajectories(env)
            self.objects_type = "trajectories"
        elif objects_type == "transitions" or isinstance(loss_fn, EdgeDecomposableLoss):
            self.training_objects = Transitions(env)
            self.objects_type = "transitions"
        elif objects_type == "states" or isinstance(loss_fn, StateDecomposableLoss):
            self.training_objects = env.States.from_batch_shape((0,))
            self.terminating_states = env.States.from_batch_shape((0,))
            self.objects_type = "states"
        else:
            raise ValueError(
                f"Unknown objects_type: {objects_type} and loss_fn: {loss_fn}"
            )

        self._is_full = False
        self._index = 0

    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {self.objects_type})"

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def add(self, training_objects: Transitions | Trajectories | tuple[States]):
        terminating_states = None
        if isinstance(training_objects, tuple):
            assert self.objects_type == "states" and self.terminating_states is not None
            training_objects, terminating_states = training_objects

        to_add = len(training_objects)

        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity

        self.training_objects.extend(training_objects)
        self.training_objects = self.training_objects[-self.capacity :]

        if self.terminating_states is not None:
            assert terminating_states is not None
            self.terminating_states.extend(terminating_states)
            self.terminating_states = self.terminating_states[-self.capacity :]

    def sample(self, n_trajectories: int) -> Transitions | Trajectories | tuple[States]:
        if self.terminating_states is not None:
            return (
                self.training_objects.sample(n_trajectories),
                self.terminating_states.sample(n_trajectories),
            )
        return self.training_objects.sample(n_trajectories)

    def save(self, directory: str):
        self.training_objects.save(os.path.join(directory, "training_objects"))
        if self.terminating_states is not None:
            self.terminating_states.save(os.path.join(directory, "terminating_states"))

    def load(self, directory: str):
        self.training_objects.load(os.path.join(directory, "training_objects"))
        self._index = len(self.training_objects)
        if self.terminating_states is not None:
            self.terminating_states.load(os.path.join(directory, "terminating_states"))
