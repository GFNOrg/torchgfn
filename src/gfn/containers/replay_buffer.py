from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions

if TYPE_CHECKING:
    from gfn.envs import Env

# TODO: fix the memory leak


class ReplayBuffer:
    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
        objects: Literal["transitions", "trajectories"] = "trajectories",
    ):
        self.env = env
        self.capacity = capacity
        self.objects_type = objects
        if objects == "transitions":
            self.training_objects = Transitions(env)
        else:
            self.training_objects = Trajectories(env)

        self._is_full = False
        self._index = 0

    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {self.objects_type})"

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

    def save(self, directory: str):
        self.training_objects.save(directory)

    def load(self, directory: str):
        self.training_objects.load(directory)
        self._index = len(self.training_objects)
