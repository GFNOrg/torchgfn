from __future__ import annotations

import os
from typing import cast, TypeVar, Generic

import torch

from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions
from gfn.containers.state_pairs import StatePairs
from gfn.states import DiscreteStates

from gfn.env import Env

ContainerType = TypeVar(
    "ContainerType", Trajectories, Transitions, StatePairs[DiscreteStates]
)


class ReplayBuffer(Generic[ContainerType]):
    """A replay buffer for GFlowNet training.

    The buffer stores training objects (trajectories, transitions, or state pairs)
    and provides functionality to add new objects and sample from the buffer.
    When the buffer is full, new objects replace old ones in a FIFO manner.
    """

    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
    ):
        """Instantiates a replay buffer.

        Args:
            env: the Environment instance.
            capacity: the size of the buffer.
        """
        self.env = env
        self.capacity = capacity
        self._is_full = False
        self.training_objects: ContainerType | None = None

    def __repr__(self):
        if self.training_objects is None:
            type_str = "empty"
        else:
            type_str = self.training_objects.__class__.__name__.lower()
        return (
            f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {type_str})"
        )

    def __len__(self):
        return 0 if self.training_objects is None else len(self.training_objects)

    def initialize(self, training_objects: ContainerType) -> None:
        """Initializes the buffer with a training object."""
        if self.training_objects is None:
            # Initialize with the same type as first added objects
            if isinstance(training_objects, Trajectories):
                self.training_objects = cast(ContainerType, Trajectories(self.env))
            elif isinstance(training_objects, Transitions):
                self.training_objects = cast(ContainerType, Transitions(self.env))
            elif isinstance(training_objects, StatePairs):
                self.training_objects = cast(ContainerType, StatePairs(self.env))
            else:
                raise ValueError(f"Unsupported type: {type(training_objects)}")

    def add(self, training_objects: ContainerType) -> None:
        """Adds a training object to the buffer."""
        self.initialize(training_objects)
        assert self.training_objects is not None

        to_add = len(training_objects)
        self._is_full |= len(self) + to_add >= self.capacity

        self.training_objects.extend(training_objects)
        self.training_objects = self.training_objects[-self.capacity :]

    def sample(self, n_trajectories: int) -> ContainerType:
        """Samples `n_trajectories` training objects from the buffer."""
        if self.training_objects is None:
            raise ValueError("Buffer is empty")
        return cast(ContainerType, self.training_objects.sample(n_trajectories))

    def save(self, directory: str):
        """Saves the buffer to disk."""
        if self.training_objects is not None:
            self.training_objects.save(os.path.join(directory, "training_objects"))

    def load(self, directory: str):
        """Loads the buffer from disk."""
        if self.training_objects is not None:
            self.training_objects.load(os.path.join(directory, "training_objects"))


class PrioritizedReplayBuffer(ReplayBuffer[ContainerType]):
    """A replay buffer of trajectories or transitions.

    Attributes:
        env: the Environment instance.
        capacity: the size of the buffer.
        training_objects: the buffer of objects used for training.
        cutoff_distance: threshold used to determine if new last_states are different
            enough from those already contained in the buffer.
        p_norm_distance: p-norm distance value to pass to torch.cdist, for the
            determination of novel states.
    """

    def __init__(
        self,
        env: "Env",
        capacity: int = 1000,
        cutoff_distance: float = 0.0,
        p_norm_distance: float = 1.0,
    ):
        """Instantiates a prioritized replay buffer.
        Args:
            env: the Environment instance.
            capacity: the size of the buffer.
            cutoff_distance: threshold used to determine if new last_states are
                different enough from those already contained in the buffer. If the
                cutoff is negative, all diversity caclulations are skipped (since all
                norms are >= 0).
            p_norm_distance: p-norm distance value to pass to torch.cdist, for the
                determination of novel states.
        """
        super().__init__(env, capacity)
        self.cutoff_distance = cutoff_distance
        self.p_norm_distance = p_norm_distance

    def _add_objs(
        self,
        training_objects: ContainerType,
    ):
        """Adds a training object to the buffer."""
        # Adds the objects to the buffer.
        self.initialize(training_objects)
        assert self.training_objects is not None
        self.training_objects.extend(training_objects)

        # Sort elements by logreward, capping the size at the defined capacity.
        assert self.training_objects.log_rewards is not None
        ix = torch.argsort(self.training_objects.log_rewards)
        self.training_objects = cast(ContainerType, self.training_objects[ix])
        self.training_objects = cast(
            ContainerType, self.training_objects[-self.capacity :]
        )

    def add(
        self,
        training_objects: ContainerType,
    ):
        """Adds a training object to the buffer."""
        to_add = len(training_objects)
        self._is_full |= len(self) + to_add >= self.capacity

        # The buffer isn't full yet.
        if len(self) < self.capacity:
            self._add_objs(training_objects)

        # Our buffer is full and we will prioritize diverse, high reward additions.
        else:
            log_rewards = training_objects.log_rewards

            if log_rewards is None:
                raise ValueError("log_rewards must be defined for prioritized replay.")

            # Sort the incoming elements by their log rewards.
            ix = torch.argsort(log_rewards, descending=True)
            training_objects = training_objects[ix]

            # Filter all batch logrewards lower than the smallest logreward in buffer.
            assert (
                self.training_objects is not None
                and self.training_objects.log_rewards is not None
            )
            min_reward_in_buffer = self.training_objects.log_rewards.min()
            idx_bigger_rewards = training_objects.log_rewards >= min_reward_in_buffer
            training_objects = training_objects[idx_bigger_rewards]

            # TODO: Concatenate input with final state for conditional GFN.
            # if self.is_conditional:
            #     batch = torch.cat(
            #         [dict_curr_batch["input"], dict_curr_batch["final_state"]],
            #         dim=-1,
            #     )
            #     buffer = torch.cat(
            #         [self.storage["input"], self.storage["final_state"]],
            #         dim=-1,
            #     )

            # If all trajectories were filtered, stop there.
            if not len(training_objects):
                return

            if self.cutoff_distance >= 0:
                # Filter the batch for diverse final_states with high reward.
                batch = training_objects.last_states.tensor.float()
                batch_dim = training_objects.last_states.batch_shape[0]
                batch_batch_dist = torch.cdist(
                    batch.view(batch_dim, -1).unsqueeze(0),
                    batch.view(batch_dim, -1).unsqueeze(0),
                    p=self.p_norm_distance,
                ).squeeze(0)

                # Finds the min distance at each row, and removes rows below the cutoff.
                r, w = torch.triu_indices(*batch_batch_dist.shape)  # Remove upper diag.
                batch_batch_dist[r, w] = torch.finfo(batch_batch_dist.dtype).max
                batch_batch_dist = batch_batch_dist.min(-1)[0]
                idx_batch_batch = batch_batch_dist > self.cutoff_distance
                training_objects = training_objects[idx_batch_batch]

                # Compute all pairwise distances between the remaining batch & buffer.
                batch = training_objects.last_states.tensor.float()
                buffer = self.training_objects.last_states.tensor.float()
                batch_dim = training_objects.last_states.batch_shape[0]
                tmp = self.training_objects.last_states
                buffer_dim = tmp.batch_shape[0]
                batch_buffer_dist = (
                    torch.cdist(
                        batch.view(batch_dim, -1).unsqueeze(0),
                        buffer.view(buffer_dim, -1).unsqueeze(0),
                        p=self.p_norm_distance,
                    )
                    .squeeze(0)
                    .min(-1)[0]  # Min calculated over rows - the batch elements.
                )

                # Filter the batch for diverse final_states w.r.t the buffer.
                idx_batch_buffer = batch_buffer_dist > self.cutoff_distance
                training_objects = training_objects[idx_batch_buffer]

            # If any training object remain after filtering, add them.
            if len(training_objects):
                self._add_objs(training_objects)
