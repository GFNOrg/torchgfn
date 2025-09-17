from __future__ import annotations

import os
from typing import Protocol, Union, cast, runtime_checkable

import torch

from gfn.containers.states_container import StatesContainer
from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions
from gfn.env import Env


@runtime_checkable
class Container(Protocol):
    def __getitem__(self, idx): ...  # noqa: E704

    def extend(self, other): ...  # noqa: E704

    def __len__(self) -> int: ...  # noqa: E704

    @property
    def log_rewards(self) -> torch.Tensor | None: ...  # noqa: E704

    @property
    def terminating_states(self): ...  # noqa: E704


ContainerUnion = Union[Trajectories, Transitions, StatesContainer]


class ReplayBuffer:
    """A replay buffer for storing containers.

    Attributes:
        env: The environment associated with the containers.
        capacity: The maximum number of items the buffer can hold.
        training_objects: The buffer contents (Trajectories, Transitions,
            or StatesContainer). This is dynamically set based on the type of the
            first added object.
        prioritized_capacity: Whether to use prioritized capacity
            (keep highest-reward items).
        prioritized_sampling: Whether to sample items with probability proportional
            to their reward.
    """

    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
        prioritized_capacity: bool = False,
        prioritized_sampling: bool = False,
    ):
        """Initializes a ReplayBuffer instance.

        Args:
            env: The environment associated with the containers.
            capacity: The maximum number of items the buffer can hold.
            prioritized_capacity: If True, keep only the highest-reward items when full.
            prioritized_sampling: If True, sample items with probability proportional
                to their reward.
        """
        self.env = env
        self.capacity = capacity
        self._is_full = False
        self.training_objects: ContainerUnion | None = None
        self.prioritized_capacity = prioritized_capacity
        self.prioritized_sampling = prioritized_sampling

    @property
    def device(self) -> torch.device:
        """The device on which the buffer's data is stored.

        Returns:
            The device object of the buffer's contents.
        """
        assert self.training_objects is not None, "Buffer is empty, it has no device!"
        return self.training_objects.device

    def add(self, training_objects: ContainerUnion) -> None:
        """Adds a training object to the buffer.

        The type of the training objects is dynamically set based on the type of the
        first added object.

        Args:
            training_objects: The Trajectories, Transitions, or StatesContainer object
                to add.
        """
        if not isinstance(training_objects, ContainerUnion):
            raise TypeError("Must be a container type")

        self._add_objs(training_objects)

    def __repr__(self):
        """Returns a string representation of the ReplayBuffer.

        Returns:
            A string summary of the buffer.
        """
        if self.training_objects is None:
            type_str = "empty"
        else:
            type_str = self.training_objects.__class__.__name__.lower()
        return (
            f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {type_str})"
        )

    def __len__(self):
        """Returns the number of items in the buffer.

        Returns:
            The number of items in the buffer.
        """
        return 0 if self.training_objects is None else len(self.training_objects)

    def initialize(self, training_objects: ContainerUnion) -> None:
        """Initializes the buffer with the type of the first added object.

        Args:
            training_objects: The initial Trajectories, Transitions, or StatesContainer
                object to set the buffer type.
        """
        if isinstance(training_objects, Trajectories):
            self.training_objects = cast(ContainerUnion, Trajectories(self.env))
        elif isinstance(training_objects, Transitions):
            self.training_objects = cast(ContainerUnion, Transitions(self.env))
        elif isinstance(training_objects, StatesContainer):
            self.training_objects = cast(ContainerUnion, StatesContainer(self.env))
        else:
            raise ValueError(f"Unsupported type: {type(training_objects)}")

    def _add_objs(self, training_objects: ContainerUnion):
        """Adds a training object to the buffer, handling the capacity.

        Args:
            training_objects: The Trajectories, Transitions, or StatesContainer object
                to add.
        """
        if self.training_objects is None:
            self.initialize(training_objects)
        assert self.training_objects is not None
        assert isinstance(training_objects, type(self.training_objects))  # type: ignore

        # Clear fields that must be recomputed for Trajectories and Transitions.
        if isinstance(training_objects, (Trajectories, Transitions)):
            training_objects.log_probs = None
        if isinstance(training_objects, Trajectories):
            training_objects.estimator_outputs = None

        # Adds the objects to the buffer.
        self.training_objects.extend(training_objects)  # type: ignore

        # Sort elements by log reward, capping the size at the defined capacity.
        if self.prioritized_capacity:
            if (
                self.training_objects.log_rewards is None
                or training_objects.log_rewards is None
            ):
                raise ValueError("log_rewards must be defined for prioritized replay.")

            # Ascending sort.
            ix = torch.argsort(self.training_objects.log_rewards)
            self.training_objects = cast(ContainerUnion, self.training_objects[ix])

        assert self.training_objects is not None
        self.training_objects = cast(
            ContainerUnion, self.training_objects[-self.capacity :]
        )

    def sample(self, n_samples: int) -> ContainerUnion:
        """Samples training objects from the buffer.

        Args:
            n_samples: The number of items to sample.

        Returns:
            A sampled Trajectories, Transitions, or StatesContainer.
        """
        if self.training_objects is None:
            raise ValueError("Buffer is empty")

        # If the buffer is flagged as prioritised, draw samples proportionally to the
        # (exponentiated) log-rewards; otherwise, fall back to uniform sampling.
        if self.prioritized_sampling:
            log_rewards = self.training_objects.log_rewards

            if log_rewards is None:
                raise ValueError("log_rewards must be defined for prioritized sampling.")

            # Convert to a proper probability mass function.  Using the softmax of
            # the log-rewards ensures numerical stability even for widely varying
            # magnitudes.
            probs = torch.softmax(log_rewards, dim=0)

            # Decide whether to sample with replacement – this is required when the
            # request is larger than the buffer size.
            replacement = n_samples > len(self)

            indices = torch.multinomial(probs, n_samples, replacement=replacement)
            return self.training_objects[indices]

        # Uniform sampling (replacement-free) for the non-prioritised case.
        return cast(ContainerUnion, self.training_objects.sample(n_samples))

    def save(self, directory: str):
        """Saves the buffer to disk.

        Args:
            directory: The directory path where the buffer will be saved.
        """
        if self.training_objects is not None:
            self.training_objects.save(os.path.join(directory, "training_objects"))

    def load(self, directory: str):
        """Loads the buffer from disk.

        Args:
            directory: The directory path from which to load the buffer.
        """
        if self.training_objects is not None:
            self.training_objects.load(os.path.join(directory, "training_objects"))


class NormBasedDiversePrioritizedReplayBuffer(ReplayBuffer):
    """A replay buffer with diversity-based prioritization.

    Attributes:
        env: The environment associated with the containers.
        capacity: The maximum number of items the buffer can hold.
        training_objects: The buffer contents (Trajectories, Transitions,
            or StatesContainer). This is dynamically set based on the type of the
            first added object.
        prioritized_capacity: Whether to use prioritized capacity
            (keep highest-reward items). This is set to True by default.
        prioritized_sampling: Whether to sample items with probability proportional
            to their reward.
        cutoff_distance: Threshold used to determine whether a new terminating state
            is different enough from those already in the buffer.
        p_norm_distance: p-norm value for distance calculation (used in torch.cdist).
    """

    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
        cutoff_distance: float = 0.0,
        p_norm_distance: float = 1.0,
    ):
        """Initializes a NormBasedDiversePrioritizedReplayBuffer instance.

        Args:
            env: The environment associated with the containers.
            capacity: The maximum number of items the buffer can hold.
            cutoff_distance: Threshold used to determine whether a new terminating
                state is different enough from those already in the buffer.
            p_norm_distance: p-norm value for distance calculation (used in torch.cdist).
        """
        super().__init__(env, capacity, prioritized_capacity=True)
        self.cutoff_distance = cutoff_distance
        self.p_norm_distance = p_norm_distance

    def add(self, training_objects: ContainerUnion):
        """Adds a training object to the buffer with diversity-based prioritization.

        Args:
            training_objects: The Trajectories, Transitions, or StatesContainer object
                to add.
        """
        if not isinstance(training_objects, ContainerUnion):
            raise TypeError("Must be a container type")

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

            # Sort the incoming elements by their logrewards.
            ix = torch.argsort(log_rewards, descending=True)
            training_objects = cast(ContainerUnion, training_objects[ix])  # type: ignore

            # Filter all batch logrewards lower than the smallest logreward in buffer.
            assert (
                self.training_objects is not None
                and self.training_objects.log_rewards is not None
                and training_objects.log_rewards is not None
            )
            min_reward_in_buffer = self.training_objects.log_rewards.min()
            idx_bigger_rewards = training_objects.log_rewards >= min_reward_in_buffer
            training_objects = training_objects[idx_bigger_rewards]

            # TODO: Concatenate input with final state for conditional GFN.
            if training_objects.conditioning:
                raise NotImplementedError(
                    "{instance.__class__.__name__} does not yet support conditional GFNs."
                )
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
                batch = training_objects.terminating_states.tensor.to(
                    torch.get_default_dtype()
                )
                batch_dim = training_objects.terminating_states.batch_shape[0]
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
                batch = training_objects.terminating_states.tensor.to(
                    torch.get_default_dtype()
                )
                buffer = self.training_objects.terminating_states.tensor.to(
                    torch.get_default_dtype()
                )
                batch_dim = training_objects.terminating_states.batch_shape[0]
                tmp = self.training_objects.terminating_states
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
                training_objects = cast(
                    ContainerUnion, training_objects[idx_batch_buffer]
                )

            # If any training object remain after filtering, add them.
            if len(training_objects):
                self._add_objs(training_objects)


class TerminatingStateBuffer(ReplayBuffer):
    """A replay buffer for storing terminating states.

    Attributes:
        env: The environment associated with the containers.
        capacity: The maximum number of items the buffer can hold.
        training_objects: The buffer contents (StatesContainer).
    """

    def __init__(self, env: Env, capacity: int = 1000, **kwargs):
        super().__init__(env, capacity, **kwargs)
        self.training_objects = StatesContainer(env)

    def add(self, training_objects: ContainerUnion):
        # Extract the terminating states from the training objects.
        if not isinstance(training_objects, ContainerUnion):
            raise TypeError("Must be a StatesContainer")

        terminating_states = training_objects.terminating_states
        conditioning = training_objects.conditioning
        log_rewards = training_objects.log_rewards

        terminating_states_container = StatesContainer(
            env=self.env,
            states=terminating_states,
            conditioning=conditioning,
            is_terminating=torch.ones(
                len(terminating_states), dtype=torch.bool, device=self.env.device
            ),
            log_rewards=log_rewards,
        )

        self._add_objs(terminating_states_container)
