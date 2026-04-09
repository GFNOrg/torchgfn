from __future__ import annotations

from typing import Protocol, Union, cast, runtime_checkable

import torch

from gfn.containers.message import Message, MessageType
from gfn.containers.states_container import StatesContainer
from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions
from gfn.env import Env
from gfn.utils.common import Timer
from gfn.utils.distributed import recv, send


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
    """A replay buffer for storing training containers.

    Supports local-only operation and distributed remote buffer communication.

    Features:
        - **Local buffering**: Stores Trajectories, Transitions, or
          StatesContainers up to a fixed capacity.
        - **Prioritized capacity**: Optionally keeps only the highest-reward
          items when the buffer is full.
        - **Prioritized sampling**: Optionally samples with probability
          proportional to reward (softmax over log-rewards).
        - **Remote buffer communication**: When ``remote_manager_rank`` is set,
          periodically sends batched containers to a remote
          ``ReplayBufferManager`` and receives score dictionaries back.
        - **Communication backends**: The ``communication_backend`` parameter
          selects between ``"torch"`` (PyTorch distributed / Gloo) and
          ``"mpi"`` (MPI4PY, ~8-12 GB/s vs ~100 MB/s with Gloo).
        - **Async scoring**: When ``async_score`` is enabled, trajectory sends
          are fire-and-forget; scores are collected lazily on the next
          ``add()`` call (1-iteration stale), decoupling training throughput
          from buffer scoring latency.
        - **Timing instrumentation**: When ``timing`` is enabled, serialization,
          send, and receive durations are recorded for profiling.

    Attributes:
        env: The environment associated with the containers.
        capacity: The maximum number of items the buffer can hold.
        training_container: The buffer contents (Trajectories, Transitions,
            or StatesContainer). Dynamically set based on the type of the
            first added object.
    """

    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
        prioritized_capacity: bool = False,
        prioritized_sampling: bool = False,
        remote_manager_rank: int | None = None,
        remote_buffer_freq: int = 1,
        communication_backend: str = "mpi",
        timing: bool = False,
        async_score: bool = False,
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
            communication_backend: Communication backend for remote buffer operations.
                ``"mpi"`` uses MPI4PY (higher bandwidth), ``"torch"`` uses PyTorch
                distributed (Gloo/MPI).
            timing: If True, record durations for serialize/send/recv operations
                in ``timing_data`` for profiling.
            async_score: If True, trajectory sends are fire-and-forget; scores
                are collected lazily on the next add() call (1-iteration stale).
                Decouples training throughput from buffer scoring latency.
        """
        self.env = env
        self.capacity = capacity
        self._is_full = False
        self.training_container: ContainerUnion | None = None
        self.prioritized_capacity = prioritized_capacity
        self.prioritized_sampling = prioritized_sampling
        self.pending_container: ContainerUnion | None = None
        self.communication_backend = communication_backend
        self.timing = timing
        self.timing_data: dict[str, list[float]] = {}

        # Remote buffer fields
        self.remote_manager_rank = remote_manager_rank
        self.remote_buffer_freq = remote_buffer_freq
        self._add_counter = 0
        self.async_score = async_score
        self._pending_score: bool = False  # True when a score recv is outstanding.

    @property
    def device(self) -> torch.device:
        """The device on which the buffer's data is stored.

        Returns:
            The device object of the buffer's contents.
        """
        assert self.training_container is not None, "Buffer is empty, it has no device!"
        return self.training_container.device

    def add(self, training_container: ContainerUnion) -> dict[str, float] | None:
        """Adds a training container to the buffer.

        The type of the training container is dynamically set based on the type of the
        first added container.

        When ``async_score`` is enabled, scores are collected lazily: the first
        call returns None (no pending score yet), and subsequent calls return the
        score from the *previous* submission.  This decouples training throughput
        from buffer scoring latency.

        Args:
            training_container: The Trajectories, Transitions, or StatesContainer
                object to add.
        """
        assert isinstance(training_container, ContainerUnion), "Must be a container type"
        self._local_add(training_container)

        # Handle remote buffer communication.
        if self.remote_manager_rank is not None:
            self._add_counter += 1

            if self.pending_container is None:
                self.pending_container = self.initialize(training_container)
            assert self.pending_container is not None
            assert isinstance(training_container, type(self.pending_container))  # type: ignore

            self.pending_container.extend(training_container)  # type: ignore

            if isinstance(self.pending_container, (Trajectories, Transitions)):
                self.pending_container.log_probs = None
            if isinstance(self.pending_container, Trajectories):
                self.pending_container.estimator_outputs = None
            if self._add_counter % self.remote_buffer_freq == 0:
                if self.async_score:
                    # Collect stale score from previous send (if any), then
                    # fire-and-forget the new batch.
                    stale_score = self._collect_pending_score()
                    with Timer(self.timing_data, "send_objs", enabled=self.timing):
                        self._send_objs_async(self.pending_container)
                    self.pending_container = None
                    return stale_score
                else:
                    with Timer(self.timing_data, "send_objs", enabled=self.timing):
                        score = self._send_objs(self.pending_container)
                    self.pending_container = None
                    return score

    def _send_objs(self, training_container: ContainerUnion) -> dict[str, float]:
        """Sends a training container to the remote manager (synchronous)."""
        self._send_data(training_container)
        return self._recv_score()

    def _send_objs_async(self, training_container: ContainerUnion) -> None:
        """Sends a training container without waiting for the score response.

        The score will be collected on the next call to ``_collect_pending_score``.
        """
        self._send_data(training_container)
        self._pending_score = True

    def _collect_pending_score(self) -> dict[str, float] | None:
        """Collect a pending score response from a previous async send.

        Returns None if no score is pending (e.g., first iteration).
        """
        if not self._pending_score:
            return None
        score = self._recv_score()
        self._pending_score = False
        return score

    def drain_pending_score(self, timeout_sec: float = 30.0) -> dict[str, float] | None:
        """Drain any outstanding async score before shutdown.

        Should be called before sending the EXIT signal when async_score is
        enabled, to avoid leaving the buffer manager with an undelivered
        response.

        Uses a timeout to avoid hanging indefinitely if the buffer manager
        has crashed.  Returns None on timeout (score is lost).
        """
        if not self._pending_score:
            return None

        import logging
        from threading import Thread

        result: list = []

        def _recv_worker():
            try:
                result.append(self._recv_score())
            except Exception:
                pass  # Buffer manager may have crashed.

        t = Thread(target=_recv_worker, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)

        self._pending_score = False  # Clear regardless of success.
        if result:
            return result[0]

        logging.getLogger(__name__).warning(
            "drain_pending_score timed out after %.0fs — buffer manager "
            "may have crashed. Proceeding with shutdown.",
            timeout_sec,
        )
        return None

    def _send_data(self, training_container: ContainerUnion) -> None:
        """Send a training container to the remote manager."""
        msg = Message(MessageType.DATA, training_container)
        with Timer(self.timing_data, "serialize_objs", enabled=self.timing):
            msg_tensor = msg.serialize()
        with Timer(self.timing_data, "send_data", enabled=self.timing):
            send(
                msg_tensor,
                dst_rank=self.remote_manager_rank,
                backend=self.communication_backend,
            )

    def _recv_score(self) -> dict[str, float]:
        """Receive a score dictionary from the remote manager."""
        with Timer(self.timing_data, "recv_score", enabled=self.timing):
            _src_rank, score_tensor = recv(
                src_rank=self.remote_manager_rank,
                backend=self.communication_backend,
            )
        with Timer(self.timing_data, "deserialize_score", enabled=self.timing):
            return Message.deserialize(score_tensor).message_data

    def __repr__(self) -> str:
        """Returns a string representation of the ReplayBuffer.

        Returns:
            A string summary of the buffer.
        """
        if self.training_container is None:
            type_str = "empty"
        else:
            type_str = self.training_container.__class__.__name__
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
            return cast(ContainerUnion, Trajectories(self.env))  # type: ignore
        elif isinstance(training_container, Transitions):
            return cast(ContainerUnion, Transitions(self.env))  # type: ignore
        elif isinstance(training_container, StatesContainer):
            return cast(ContainerUnion, StatesContainer(self.env))  # type: ignore
        else:
            raise ValueError(f"Unsupported type: {type(training_container)}")

    def _local_add(self, training_container: ContainerUnion):
        """Adds a training object to the local buffer, handling capacity.

        Subclasses override this to customize local insertion logic (e.g.,
        diversity filtering). The base class ``add()`` calls this method,
        then handles remote buffer communication separately.

        Args:
            training_container: The Trajectories, Transitions, or StatesContainer object
                to add.
        """
        if self.training_container is None:
            self.training_container = self.initialize(training_container)
        assert self.training_container is not None
        assert isinstance(training_container, type(self.training_container))  # type: ignore

        self.training_container.extend(training_container)  # type: ignore

        # Clear fields that must be recomputed for Trajectories and Transitions.
        # Otherwise, we might accidentally use the old values.
        if isinstance(self.training_container, (Trajectories, Transitions)):
            self.training_container.log_probs = None
        if isinstance(self.training_container, Trajectories):
            self.training_container.estimator_outputs = None

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

    def save(self, path: str):
        """Saves the buffer to a single ``.pt`` file.

        Args:
            path: File path (e.g. ``"replay_buffer.pt"``).
        """
        if self.training_container is not None:
            self.training_container.save(path)

    def load(self, path: str):
        """Loads buffer contents from a ``.pt`` file saved by :meth:`save`.

        Args:
            path: File path to the saved buffer.
        """
        if self.training_container is not None:
            self.training_container = type(self.training_container).load(self.env, path)

    def timing_log(self) -> str:
        """Returns a formatted string of the timing information for the replay buffer."""
        log_str = "Replay Buffer Timing Information:\n"
        for key, times in self.timing_data.items():
            log_str += f"{key}: {sum(times):.4f} s\n"
        return log_str


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
        remote_manager_rank: int | None = None,
        remote_buffer_freq: int = 1,
        communication_backend: str = "mpi",
        timing: bool = False,
        async_score: bool = False,
    ):
        """Initializes a NormBasedDiversePrioritizedReplayBuffer instance.

        Args:
            env: The environment associated with the containers.
            capacity: The maximum number of items the buffer can hold.
            cutoff_distance: Threshold used to determine whether a new terminating
                state is different enough from those already in the buffer.
            p_norm_distance: p-norm value for distance calculation (used in torch.cdist).
            remote_manager_rank: Rank of the assigned remote replay buffer manager, or
                None if no remote manager is assigned.
            remote_buffer_freq: Frequency (in number of add() calls) at which to contact
                the remote buffer manager.
            communication_backend: Communication backend (``"mpi"`` or ``"torch"``).
            timing: If True, record operation durations for profiling.
            async_score: If True, trajectory sends are fire-and-forget; scores
                are collected lazily on the next add() call.
        """
        super().__init__(
            env,
            capacity,
            prioritized_capacity=True,
            remote_manager_rank=remote_manager_rank,
            remote_buffer_freq=remote_buffer_freq,
            communication_backend=communication_backend,
            timing=timing,
            async_score=async_score,
        )
        self.cutoff_distance = cutoff_distance
        self.p_norm_distance = p_norm_distance

    @staticmethod
    def _diversity_repr(container: ContainerUnion) -> torch.Tensor:
        """Returns the tensor used for pairwise distance in diversity filtering.

        For conditional GFNs, concatenates conditions with the state tensor so
        that identical states under different conditions are treated as distinct.
        """
        states = container.terminating_states
        repr_tensor = states.tensor.to(torch.get_default_dtype())
        if states.conditions is not None:
            repr_tensor = torch.cat(
                [repr_tensor, states.conditions.to(repr_tensor.dtype)], dim=-1
            )
        return repr_tensor

    def _local_add(self, training_container: ContainerUnion):
        """Adds with diversity-based prioritization to the local buffer.

        Overrides the base class hook so that ``add()`` (which handles remote
        communication) delegates local insertion here.
        """
        to_add = len(training_container)
        self._is_full |= len(self) + to_add >= self.capacity

        # The buffer isn't full yet.
        if len(self) < self.capacity:
            super()._local_add(training_container)
            return

        # Our buffer is full and we will prioritize diverse, high reward additions.
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

        # If all trajectories were filtered, stop there.
        if not len(training_container):
            return

        if self.cutoff_distance >= 0:
            # Filter the batch for diverse final_states with high reward.
            batch = self._diversity_repr(training_container)
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
            batch = self._diversity_repr(training_container)
            buffer = self._diversity_repr(self.training_container)
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

        # If any training objects remain after filtering, add them.
        if len(training_container):
            super()._local_add(training_container)


class TerminatingStateBuffer(ReplayBuffer):
    """A replay buffer for storing terminating states.

    Attributes:
        env: The environment associated with the containers.
        capacity: The maximum number of items the buffer can hold.
        training_container: The buffer contents (StatesContainer).
    """

    def __init__(self, env: Env, capacity: int = 1000, timing: bool = False, **kwargs):
        super().__init__(env, capacity, timing=timing, **kwargs)
        self.training_container = StatesContainer(env)

    def _local_add(self, training_container: ContainerUnion):
        """Extracts terminating states and adds them to the local buffer.

        Overrides the base class hook so that ``add()`` (which handles remote
        communication) delegates local insertion here.
        """
        terminating_states = training_container.terminating_states
        log_rewards = training_container.log_rewards

        terminating_states_container = StatesContainer(
            env=self.env,
            states=terminating_states,
            is_terminating=torch.ones(
                len(terminating_states), dtype=torch.bool, device=self.env.device
            ),
            log_rewards=log_rewards,
        )

        super()._local_add(terminating_states_container)
