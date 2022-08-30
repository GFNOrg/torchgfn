from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .trajectories import Trajectories
from .transitions import Transitions

if TYPE_CHECKING:
    from ..envs import Env


class ReplayBuffer:
    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
        objects: Literal["transitions", "trajectories"] = "trajectories",
    ):
        self.env = env
        self.capacity = capacity
        self.type = objects
        if objects == "transitions":
            self.training_objects = Transitions(env)
        else:
            self.training_objects = Trajectories(env)

        self._is_full = False
        self._index = 0

    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {self.type})"

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def add(self, training_objects: Transitions | Trajectories):
        to_add = len(training_objects)

        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity

        self.training_objects.extend(training_objects)
        self.training_objects = self.training_objects[-self.capacity :]

    def sample(self, n_objects: int) -> Transitions | Trajectories:
        return self.training_objects.sample(n_objects)
