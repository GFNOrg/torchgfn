from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import torch

from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions

if TYPE_CHECKING:
    from gfn.env import Env
    from gfn.states import States


class ReplayBuffer:
    """A replay buffer of trajectories, transitions, or states.

    Attributes:
        env: the Environment instance.
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
        self.objects_type = objects_type
        if objects_type == "trajectories":
            self.training_objects = Trajectories(env)
        elif objects_type == "transitions":
            self.training_objects = Transitions(env)
        elif objects_type == "states":
            self.training_objects = env.states_from_batch_shape((0,))
            self.terminating_states = env.states_from_batch_shape((0,))
            self.terminating_states.log_rewards = torch.zeros((0,), device=env.device)
        else:
            raise ValueError(f"Unknown objects_type: {objects_type}")

        self._is_full = False

    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {self.objects_type})"

    def __len__(self):
        return len(self.training_objects)

    def add(self, training_objects: Transitions | Trajectories | tuple[States]):
        """Adds a training object to the buffer."""
        terminating_states = None
        if isinstance(training_objects, tuple):
            assert self.objects_type == "states" and self.terminating_states is not None
            training_objects, terminating_states = training_objects  # pyright: ignore

        to_add = len(training_objects)

        self._is_full |= len(self) + to_add >= self.capacity

        self.training_objects.extend(training_objects)  # pyright: ignore
        self.training_objects = self.training_objects[
            -self.capacity :
        ]  # pyright: ignore

        if self.terminating_states is not None:
            assert terminating_states is not None
            self.terminating_states.extend(terminating_states)  # pyright: ignore
            self.terminating_states = self.terminating_states[-self.capacity :]

    def sample(self, n_trajectories: int) -> Transitions | Trajectories | tuple[States]:
        """Samples `n_trajectories` training objects from the buffer."""
        if self.terminating_states is not None:
            return (
                self.training_objects.sample(n_trajectories),  # pyright: ignore
                self.terminating_states.sample(n_trajectories),  # pyright: ignore
            )
        return self.training_objects.sample(n_trajectories)  # pyright: ignore

    def save(self, directory: str):
        """Saves the buffer to disk."""
        if self.objects_type == "states":
            raise ValueError("States cannot be saved")
        self.training_objects.save(  # pyright: ignore
            os.path.join(directory, "training_objects")
        )
        if self.terminating_states is not None:
            self.terminating_states.save(  # pyright: ignore
                os.path.join(directory, "terminating_states")
            )

    def load(self, directory: str):
        """Loads the buffer from disk."""
        self.training_objects.load(  # pyright: ignore
            os.path.join(directory, "training_objects")
        )
        if self.terminating_states is not None:
            self.terminating_states.load(  # pyright: ignore
                os.path.join(directory, "terminating_states")
            )


class PrioritizedReplayBuffer(ReplayBuffer):
    """A replay buffer of trajectories or transitions.

    Attributes:
        env: the Environment instance.
        capacity: the size of the buffer.
        training_objects: the buffer of objects used for training.
        terminating_states: a States class representation of $s_f$.
        objects_type: the type of buffer (transitions, trajectories, or states).
        cutoff_distance: threshold used to determine if new last_states are different
            enough from those already contained in the buffer.
        p_norm_distance: p-norm distance value to pass to torch.cdist, for the
            determination of novel states.
    """

    def __init__(
        self,
        env: Env,
        objects_type: Literal["transitions", "trajectories", "states"],
        capacity: int = 1000,
        cutoff_distance: float = 0.0,
        p_norm_distance: float = 1.0,
    ):
        """Instantiates a prioritized replay buffer.
        Args:
            env: the Environment instance.
            loss_fn: the Loss instance.
            capacity: the size of the buffer.
            objects_type: the type of buffer (transitions, trajectories, or states).
            cutoff_distance: threshold used to determine if new last_states are
                different enough from those already contained in the buffer. If the
                cutoff is negative, all diversity caclulations are skipped (since all
                norms are >= 0).
            p_norm_distance: p-norm distance value to pass to torch.cdist, for the
                determination of novel states.
        """
        super().__init__(env, objects_type, capacity)
        self.cutoff_distance = cutoff_distance
        self.p_norm_distance = p_norm_distance

    def _add_objs(
        self,
        training_objects: Transitions | Trajectories | tuple[States],
        terminating_states: States | None = None,
    ):
        """Adds a training object to the buffer."""
        # Adds the objects to the buffer.
        self.training_objects.extend(training_objects)  # pyright: ignore

        # Sort elements by logreward, capping the size at the defined capacity.
        ix = torch.argsort(self.training_objects.log_rewards)  # pyright: ignore
        self.training_objects = self.training_objects[ix]  # pyright: ignore
        self.training_objects = self.training_objects[
            -self.capacity :
        ]  # pyright: ignore

        # Add the terminating states to the buffer.
        if self.terminating_states is not None:
            assert terminating_states is not None
            self.terminating_states.extend(terminating_states)

            # Sort terminating states by logreward as well.
            self.terminating_states = self.terminating_states[ix]
            self.terminating_states = self.terminating_states[-self.capacity :]

    def add(self, training_objects: Transitions | Trajectories | tuple[States]):
        """Adds a training object to the buffer."""
        terminating_states = None
        if isinstance(training_objects, tuple):
            assert self.objects_type == "states" and self.terminating_states is not None
            training_objects, terminating_states = training_objects  # pyright: ignore

        to_add = len(training_objects)
        self._is_full |= len(self) + to_add >= self.capacity

        # The buffer isn't full yet.
        if len(self.training_objects) < self.capacity:
            self._add_objs(training_objects, terminating_states)

        # Our buffer is full and we will prioritize diverse, high reward additions.
        else:
            if (
                self.training_objects.log_rewards is None
                or training_objects.log_rewards is None  # pyright: ignore
            ):
                raise ValueError("log_rewards must be defined for prioritized replay.")

            # Sort the incoming elements by their logrewards.
            ix = torch.argsort(
                training_objects.log_rewards, descending=True  # pyright: ignore
            )  # pyright: ignore
            training_objects = training_objects[ix]  # pyright: ignore

            # Filter all batch logrewards lower than the smallest logreward in buffer.
            min_reward_in_buffer = self.training_objects.log_rewards.min()  # type: ignore  # FIXME
            idx_bigger_rewards = (
                training_objects.log_rewards >= min_reward_in_buffer  # pyright: ignore
            )  # pyright: ignore
            training_objects = training_objects[idx_bigger_rewards]  # pyright: ignore

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
                batch = training_objects.last_states.tensor.float()  # pyright: ignore
                batch_dim = training_objects.last_states.batch_shape[  # pyright: ignore
                    0
                ]  # pyright: ignore
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
                training_objects = training_objects[idx_batch_batch]  # pyright: ignore

                # Compute all pairwise distances between the remaining batch & buffer.
                batch = training_objects.last_states.tensor.float()  # pyright: ignore
                buffer = (
                    self.training_objects.last_states.tensor.float()  # pyright: ignore
                )  # pyright: ignore
                batch_dim = training_objects.last_states.batch_shape[  # pyright: ignore
                    0
                ]  # pyright: ignore
                tmp = self.training_objects.last_states  # pyright: ignore
                buffer_dim = tmp.batch_shape[0]  # pyright: ignore
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
                training_objects = training_objects[idx_batch_buffer]  # pyright: ignore

            # If any training object remain after filtering, add them.
            if len(training_objects):
                self._add_objs(training_objects)
