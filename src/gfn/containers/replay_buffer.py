from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions

if TYPE_CHECKING:
    from gfn.env import Env
    from gfn.states import States


class ReplayBuffer:
    """A replay buffer of trajectories or transitions.

    Attributes:
        env: the Environment instance.
        loss_fn: the Loss instance
        capacity: the size of the buffer.
        training_objects: the buffer of objects used for training.
        terminating_states: a States class representation of $s_f$.
        objects_type: the type of buffer (transitions, trajectories, or states).
    """

    def __init__(
        self,
        env: Env,
        objects_type: Literal["transitions", "trajectories", "states"],
        capacity: int = 1000,
    ):
        """Instantiates a replay buffer.
        Args:
            env: the Environment instance.
            loss_fn: the Loss instance.
            capacity: the size of the buffer.
            objects_type: the type of buffer (transitions, trajectories, or states).
        """
        self.env = env
        self.capacity = capacity
        self.terminating_states = None
        if objects_type == "trajectories":
            self.training_objects = Trajectories(env)
            self.objects_type = "trajectories"
        elif objects_type == "transitions":
            self.training_objects = Transitions(env)
            self.objects_type = "transitions"
        elif objects_type == "states":
            self.training_objects = env.States.from_batch_shape((0,))
            self.terminating_states = env.States.from_batch_shape((0,))
            self.objects_type = "states"
        else:
            raise ValueError(f"Unknown objects_type: {objects_type}")

        self._is_full = False
        self._index = 0

    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {self.objects_type})"

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def add(self, training_objects: Transitions | Trajectories | tuple[States]):
        """Adds a training object to the buffer."""
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
        """Samples `n_trajectories` training objects from the buffer."""
        if self.terminating_states is not None:
            return (
                self.training_objects.sample(n_trajectories),
                self.terminating_states.sample(n_trajectories),
            )
        return self.training_objects.sample(n_trajectories)

    def save(self, directory: str):
        """Saves the buffer to disk."""
        self.training_objects.save(os.path.join(directory, "training_objects"))
        if self.terminating_states is not None:
            self.terminating_states.save(os.path.join(directory, "terminating_states"))

    def load(self, directory: str):
        """Loads the buffer from disk."""
        self.training_objects.load(os.path.join(directory, "training_objects"))
        self._index = len(self.training_objects)
        if self.terminating_states is not None:
            self.terminating_states.load(os.path.join(directory, "terminating_states"))
