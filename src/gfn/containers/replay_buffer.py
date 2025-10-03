from __future__ import annotations

import os
from typing import Protocol, Union, cast, runtime_checkable

import torch
import torch.distributed as dist

from gfn.containers.message import Message, MessageType
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
ValidContainerTypes = (Trajectories, Transitions, StatesContainer)


class ReplayBuffer:
    """A replay buffer for storing containers.

    Attributes:
        env: The environment associated with the containers.
        capacity: The maximum number of items the buffer can hold.
        training_container: The buffer contents (Trajectories, Transitions,
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
        remote_manager_rank: int | None = None,
        remote_buffer_freq: int = 1,
    ):
        """Initializes a ReplayBuffer instance.

        Args:
            env: The environment associated with the containers.
            capacity: The maximum number of items the buffer can hold.
            prioritized_capacity: If True, keep only the highest-reward items when full.
            prioritized_sampling: If True, sample items with probability proportional
                to their reward.
            remote_manager_rank: Rank of the assigned remote replay buffer manager, or
                None if no remote manager is assigned.
            remote_buffer_freq: Frequency (in number of add() calls) at which to contact
                the remote buffer manager.
        """
        self.env = env
        self.capacity = capacity
        self._is_full = False
        self.training_container: ContainerUnion | None = None
        self.prioritized_capacity = prioritized_capacity
        self.prioritized_sampling = prioritized_sampling

        # Remote buffer fields
        self.remote_manager_rank = remote_manager_rank
        self.remote_buffer_freq = remote_buffer_freq
        self._add_counter = 0
        if self.remote_manager_rank is not None:
            backend = dist.get_backend()
            if backend != "gloo":
                raise RuntimeError(
                    f"Replay Buffer Manager is only supported with the 'gloo' backend, "
                    f"but the current backend is '{backend}'."
                )

    @property
    def device(self) -> torch.device:
        """The device on which the buffer's data is stored.

        Returns:
            The device object of the buffer's contents.
        """
        assert self.training_container is not None, "Buffer is empty, it has no device!"
        return self.training_container.device

    def add(self, training_container: ContainerUnion) -> None:
        """Adds a training container to the buffer.

        The type of the training container is dynamically set based on the type of the
        first added container.

        Args:
            training_container: The Trajectories, Transitions, or StatesContainer
                object to add.
        """
        if not isinstance(training_container, ValidContainerTypes):
            raise TypeError("Must be a container type")

        self._add_objs(training_container)

        # Handle remote buffer communication
        if self.remote_manager_rank is not None:
            self._add_counter += 1
            if self._add_counter % self.remote_buffer_freq == 0:
                self._send_objs(training_container)

    def _send_objs(self, training_container: ContainerUnion) -> float:
        """Sends a training container to the remote manager."""
        msg = Message(MessageType.DATA, training_container)
        msg_tensor = msg.serialize()

        # First send the length so the receiver knows how many bytes
        length_tensor = torch.IntTensor([len(msg_tensor)])
        dist.send(length_tensor, dst=self.remote_manager_rank)

        # Now send the actual content
        dist.send(msg_tensor, dst=self.remote_manager_rank)

        # Receive a dummy score back
        score = torch.zeros(1, dtype=torch.float32)
        dist.recv(score, src=self.remote_manager_rank)

        return score.item()

    def __repr__(self) -> str:
        """Returns a string representation of the ReplayBuffer.

        Returns:
            A string summary of the buffer.
        """
        if self.training_container is None:
            type_str = "empty"
        else:
            type_str = self.training_container.__class__.__name__.lower()
        return (
            f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {type_str})"
        )

    def __len__(self) -> int:
        """Returns the number of items in the buffer.

        Returns:
            The number of items in the buffer.
        """
        return 0 if self.training_container is None else len(self.training_container)

    def initialize(self, training_container: ContainerUnion) -> None:
        """Initializes the buffer with the type of the first added object.

        Args:
            training_container: The initial Trajectories, Transitions, or StatesContainer
                object to set the buffer type.
        """
        if isinstance(training_container, Trajectories):
            self.training_container = cast(ContainerUnion, Trajectories(self.env))
        elif isinstance(training_container, Transitions):
            self.training_container = cast(ContainerUnion, Transitions(self.env))
        elif isinstance(training_container, StatesContainer):
            self.training_container = cast(ContainerUnion, StatesContainer(self.env))
        else:
            raise ValueError(f"Unsupported type: {type(training_container)}")

    def _add_objs(self, training_container: ContainerUnion):
        """Adds a training object to the buffer, handling the capacity.

        Args:
            training_container: The Trajectories, Transitions, or StatesContainer object
                to add.
        """
        if self.training_container is None:
            self.initialize(training_container)
        assert self.training_container is not None
        assert isinstance(training_container, type(self.training_container))  # type: ignore

        # Adds the objects to the buffer.
        self.training_container.extend(training_container)  # type: ignore

        # Sort elements by log reward, capping the size at the defined capacity.
        if self.prioritized_capacity:
            if (
                self.training_container.log_rewards is None
                or training_container.log_rewards is None
            ):
                raise ValueError("log_rewards must be defined for prioritized replay.")

            # Ascending sort.
            ix = torch.argsort(self.training_container.log_rewards)
            self.training_container = cast(ContainerUnion, self.training_container[ix])

        assert self.training_container is not None
        self.training_container = cast(
            ContainerUnion, self.training_container[-self.capacity :]
        )

    def sample(self, n_samples: int) -> ContainerUnion:
        """Samples training objects from the buffer.

        Args:
            n_samples: The number of items to sample.

        Returns:
            A sampled Trajectories, Transitions, or StatesContainer.
        """
        if self.training_container is None:
            raise ValueError("Buffer is empty")

        # If the buffer is flagged as prioritised, draw samples proportionally to the
        # (exponentiated) log-rewards; otherwise, fall back to uniform sampling.
        if self.prioritized_sampling:
            log_rewards = self.training_container.log_rewards

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
            return self.training_container[indices]

        # Uniform sampling (replacement-free) for the non-prioritised case.
        return cast(ContainerUnion, self.training_container.sample(n_samples))

    def save(self, directory: str):
        """Saves the buffer to disk.

        Args:
            directory: The directory path where the buffer will be saved.
        """
        if self.training_container is not None:
            self.training_container.save(os.path.join(directory, "training_container"))

    def load(self, directory: str):
        """Loads the buffer from disk.

        Args:
            directory: The directory path from which to load the buffer.
        """
        if self.training_container is not None:
            self.training_container.load(os.path.join(directory, "training_container"))


class NormBasedDiversePrioritizedReplayBuffer(ReplayBuffer):
    """A replay buffer with diversity-based prioritization.

    Attributes:
        env: The environment associated with the containers.
        capacity: The maximum number of items the buffer can hold.
        training_container: The buffer contents (Trajectories, Transitions,
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

    def add(self, training_container: ContainerUnion):
        """Adds a training object to the buffer with diversity-based prioritization.

        Args:
            training_container: The Trajectories, Transitions, or StatesContainer object
                to add.
        """
        if not isinstance(training_container, ValidContainerTypes):
            raise TypeError("Must be a container type")

        to_add = len(training_container)
        self._is_full |= len(self) + to_add >= self.capacity

        # The buffer isn't full yet.
        if len(self) < self.capacity:
            self._add_objs(training_container)

        # Our buffer is full and we will prioritize diverse, high reward additions.
        else:
            log_rewards = training_container.log_rewards

            if log_rewards is None:
                raise ValueError("log_rewards must be defined for prioritized replay.")

            # Sort the incoming elements by their logrewards.
            ix = torch.argsort(log_rewards, descending=True)
            training_container = cast(ContainerUnion, training_container[ix])  # type: ignore

            # Filter all batch logrewards lower than the smallest logreward in buffer.
            assert (
                self.training_container is not None
                and self.training_container.log_rewards is not None
                and training_container.log_rewards is not None
            )
            min_reward_in_buffer = self.training_container.log_rewards.min()
            idx_bigger_rewards = training_container.log_rewards >= min_reward_in_buffer
            training_container = training_container[idx_bigger_rewards]

            # TODO: Concatenate input with final state for conditional GFN.
            if training_container.conditioning:
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
            if not len(training_container):
                return

            if self.cutoff_distance >= 0:
                # Filter the batch for diverse final_states with high reward.
                batch = training_container.terminating_states.tensor.to(
                    torch.get_default_dtype()
                )
                batch_dim = training_container.terminating_states.batch_shape[0]
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
                training_container = training_container[idx_batch_batch]

                # Compute all pairwise distances between the remaining batch & buffer.
                batch = training_container.terminating_states.tensor.to(
                    torch.get_default_dtype()
                )
                buffer = self.training_container.terminating_states.tensor.to(
                    torch.get_default_dtype()
                )
                batch_dim = training_container.terminating_states.batch_shape[0]
                tmp = self.training_container.terminating_states
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
                training_container = cast(
                    ContainerUnion, training_container[idx_batch_buffer]
                )

            # If any training object remain after filtering, add them.
            if len(training_container):
                self._add_objs(training_container)
