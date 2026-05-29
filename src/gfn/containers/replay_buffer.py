from __future__ import annotations

from typing import Protocol, Union, cast, runtime_checkable

import torch

from gfn.containers.message import Message, MessageType
from gfn.containers.states_container import StatesContainer
from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions
from gfn.env import Env
from gfn.utils.common import Timer
from gfn.utils.distributed import AsyncSendHandle, isend, recv, send


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
        async_comm: bool = False,
        lazy_sort: bool = False,
        baseline_filtering: bool = False,
        scoring_only: bool = False,
        baseline_refresh_after: int = 10,
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
            async_comm: If True, uses ``isend`` for non-blocking data
                submission (deferred score collection on next ``add()``).
                The agent blocks only to collect the previous score and
                to wait for the prior ``isend`` to complete.
            lazy_sort: If True, defer concatenation and sorting until the
                buffer reaches 2x capacity. Useful for buffer manager ranks
                that only accumulate data and don't sample every iteration.
            baseline_filtering: If True, only send containers whose
                ``log_rewards`` exceed the baseline reported by the remote
                manager. Requires ``remote_manager_rank`` and a
                ``Trajectories`` or ``StatesContainer`` payload;
                ``Transitions`` is rejected.
            scoring_only: If True, convert payloads to a lightweight
                ``StatesContainer`` (terminating states + log-rewards) before
                sending. Local storage is unaffected.
            baseline_refresh_after: When ``baseline_filtering`` is enabled,
                bypass the filter on the next send after this many
                consecutive fully-filtered batches so the baseline can
                refresh.
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
        self.async_comm = async_comm
        self._pending_score: bool = False  # True when a score recv is outstanding.
        self._send_handle: AsyncSendHandle | None = (
            None  # Outstanding non-blocking send.
        )

        # Lazy-sort bookkeeping (only active when lazy_sort=True):
        # incoming batches are accumulated in a list and only flushed
        # (concatenated + sorted + truncated) when the total pending +
        # committed length reaches 2 * capacity.
        self.lazy_sort = lazy_sort
        self._pending_batches: list[ContainerUnion] = []
        self._pending_len: int = 0

        # Baseline filtering: skip sending containers worse than the
        # baseline reported by the remote manager. Updated from score
        # responses; bypasses itself after `baseline_refresh_after`
        # consecutive fully-filtered batches so the baseline can refresh.
        if baseline_filtering and remote_manager_rank is None:
            raise ValueError(
                "baseline_filtering=True requires remote_manager_rank to be set."
            )
        self.baseline_filtering = baseline_filtering
        self.baseline_refresh_after = baseline_refresh_after
        self._baseline_log_reward: float = float("-inf")
        self._consecutive_filtered_empty: int = 0
        self._baseline_total: int = 0
        self._baseline_kept: int = 0
        self._baseline_skipped_sends: int = 0

        # Scoring-only: send only terminating states + log_rewards.
        self.scoring_only = scoring_only

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

        When ``baseline_filtering`` is enabled, only trajectories with log-reward
        above the remote buffer's baseline are sent.  If all trajectories in the
        pending batch are below the baseline, the send is skipped entirely.

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
                if self.async_comm:

                    with Timer(
                        self.timing_data, "wait_previous_send", enabled=self.timing
                    ):
                        self._wait_previous_send()

                    with Timer(
                        self.timing_data, "wait_pending_score", enabled=self.timing
                    ):
                        stale_score = self._collect_pending_score()
                    self._update_baseline(stale_score)
                    self._filter_and_send(
                        self.pending_container, self._isend_and_defer_score
                    )
                    self.pending_container = None
                    return stale_score
                elif self.async_score:
                    # Collect stale score from previous send (if any), then
                    # fire-and-forget the new batch.
                    stale_score = self._collect_pending_score()
                    self._update_baseline(stale_score)
                    self._filter_and_send(self.pending_container, self._send_objs_async)
                    self.pending_container = None
                    return stale_score
                else:
                    score = self._filter_and_send(
                        self.pending_container, self._send_objs
                    )
                    if score is not None:
                        self._update_baseline(score)
                    self.pending_container = None
                    return score

    def _filter_and_send(self, container, send_fn):
        """Filter by baseline, prepare for remote, and send.

        Returns whatever ``send_fn`` returns (a score dict for sync sends,
        None for async sends), or None if baseline filtering drops everything.
        """
        filtered = self._filter_by_baseline(container)
        if filtered is None:
            return None
        with Timer(self.timing_data, "prepare_for_remote", enabled=self.timing):
            to_send = self._prepare_for_remote(filtered)
        with Timer(self.timing_data, "send_objs", enabled=self.timing):
            return send_fn(to_send)

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

    def _update_baseline(self, score_dict: dict[str, float] | None) -> None:
        """Extract and store the baseline log-reward from a score response.

        Called after receiving a score dict from the buffer manager.
        Only updates if baseline_filtering is enabled and the score dict
        contains a ``baseline_log_reward`` key.
        """
        if not self.baseline_filtering or score_dict is None:
            return
        if "baseline_log_reward" in score_dict:
            self._baseline_log_reward = score_dict["baseline_log_reward"]

    def _filter_by_baseline(
        self,
        container: ContainerUnion,
    ) -> ContainerUnion | None:
        """Filter a container to keep only items with log_reward >= baseline.

        Returns the (possibly subset) container, or None if every item is
        below the baseline.  After ``baseline_refresh_after`` consecutive
        fully-filtered batches, the next batch bypasses the filter so the
        worker can receive a fresh baseline.  ``Transitions`` is not
        supported (its log_rewards is per-transition with ``-inf`` for
        non-terminating rows, so per-row filtering would break DB/SubTB).
        """
        if not self.baseline_filtering:
            return container
        assert isinstance(container, (Trajectories, StatesContainer)), (
            "baseline_filtering supports Trajectories or StatesContainer only; "
            f"got {type(container).__name__}"
        )
        if self._baseline_log_reward == float("-inf"):
            return container  # No baseline yet — send everything.

        log_rewards = container.log_rewards
        if log_rewards is None:
            return container  # e.g. backward trajectories — nothing to filter on.

        n_total = len(container)
        assert log_rewards.shape == (
            n_total,
        ), f"log_rewards shape {tuple(log_rewards.shape)} must equal ({n_total},)"
        mask = log_rewards >= self._baseline_log_reward
        n_kept = int(mask.sum().item())
        self._baseline_total += n_total
        self._baseline_kept += n_kept

        if mask.all():
            self._consecutive_filtered_empty = 0
            return container
        if not mask.any():
            self._baseline_skipped_sends += 1
            self._consecutive_filtered_empty += 1
            if self._consecutive_filtered_empty >= self.baseline_refresh_after:
                self._consecutive_filtered_empty = 0
                return container  # Force a send to refresh the baseline.
            return None

        self._consecutive_filtered_empty = 0
        indices = torch.where(mask)[0]
        return container[indices]

    def _prepare_for_remote(self, container: ContainerUnion) -> ContainerUnion:
        """Convert a container to a lightweight form for remote scoring.

        When ``scoring_only`` is True, extracts terminating states and
        log-rewards into a ``StatesContainer``.  ``Transitions`` is
        rejected because its ``log_rewards`` shape does not match
        ``terminating_states`` (it is per-transition, not per-trajectory).
        When ``scoring_only`` is False, returns the container unchanged.
        """
        if not self.scoring_only:
            return container
        if isinstance(container, StatesContainer):
            return container
        assert isinstance(container, Trajectories), (
            "scoring_only supports Trajectories or StatesContainer only; "
            f"got {type(container).__name__}"
        )

        terminating_states = container.terminating_states
        log_rewards = container.log_rewards
        n = len(terminating_states)
        assert log_rewards is not None and log_rewards.shape == (n,), (
            "Trajectories must have log_rewards of shape (batch_size,) for "
            "scoring_only; backward trajectories are not supported."
        )
        return StatesContainer(
            env=self.env,
            states=terminating_states,
            is_terminating=torch.ones(
                n, dtype=torch.bool, device=terminating_states.device
            ),
            log_rewards=log_rewards,
        )

    def _isend_and_defer_score(self, training_container: ContainerUnion) -> None:
        """Non-blocking send (isend), deferred score: fire-and-forget data, collect score on next add().

        The send handle is kept alive in ``_send_handle`` until the next
        call to ``_wait_previous_send``.
        """
        assert self.remote_manager_rank is not None
        msg = Message(MessageType.DATA, training_container)
        with Timer(self.timing_data, "serialize_objs", enabled=self.timing):
            msg_tensor = msg.serialize()
        if self.timing:
            self.timing_data.setdefault("send_bytes", []).append(
                float(msg_tensor.numel() * msg_tensor.element_size())
            )
        with Timer(self.timing_data, "isend_data", enabled=self.timing):
            self._send_handle = isend(
                msg_tensor,
                dst_rank=self.remote_manager_rank,
                backend=self.communication_backend,
            )
        self._pending_score = True

    def _wait_previous_send(self) -> None:
        """Block until the previous non-blocking send has completed.

        This is typically near-instantaneous because MPI internally buffers
        the data, but guarantees the send buffer can be safely reused.
        """
        if self._send_handle is not None:
            with Timer(self.timing_data, "wait_send", enabled=self.timing):
                self._send_handle.wait()
            self._send_handle = None

    def drain_pending_score(self, timeout_sec: float = 30.0) -> dict[str, float] | None:
        """Drain any outstanding async score before shutdown.

        Should be called before sending the EXIT signal when ``async_score``
        or ``async_comm`` is enabled, to avoid leaving the buffer manager
        with an undelivered response.

        For ``async_comm`` mode this also waits for the outstanding
        non-blocking send to complete.

        Uses a timeout to avoid hanging indefinitely if the buffer manager
        has crashed.  Returns None on timeout (score is lost).
        """
        # Ensure any outstanding isend is flushed first.
        self._wait_previous_send()

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
        assert self.remote_manager_rank is not None
        msg = Message(MessageType.DATA, training_container)
        with Timer(self.timing_data, "serialize_objs", enabled=self.timing):
            msg_tensor = msg.serialize()
        if self.timing:
            self.timing_data.setdefault("send_bytes", []).append(
                float(msg_tensor.numel() * msg_tensor.element_size())
            )
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
            The number of items in the buffer (including pending batches).
        """
        committed = (
            0 if self.training_container is None else len(self.training_container)
        )
        return committed + self._pending_len

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

    def _flush_pending(self) -> None:
        """Concatenate all pending batches into ``training_container``.

        Called lazily when the accumulated size reaches 2 * capacity, or
        eagerly by callers that need a consistent view.

        Merges all pending batches into a single combined batch first, then
        extends ``training_container`` once to avoid extra copy cost.
        """
        if not self._pending_batches:
            return

        with Timer(self.timing_data, "local_add/extend", enabled=self.timing):
            # Merge pending batches into one combined batch (cheap: starts
            # from 0 items), then extend training_container once.
            assert self.training_container is not None
            combined = self.initialize(self._pending_batches[0])
            for batch in self._pending_batches:
                combined.extend(batch)  # type: ignore
            self.training_container.extend(combined)  # type: ignore  # single extend
            self._pending_batches.clear()
            self._pending_len = 0

        # Clear fields that must be recomputed for Trajectories and Transitions.
        if isinstance(self.training_container, (Trajectories, Transitions)):
            self.training_container.log_probs = None
        if isinstance(self.training_container, Trajectories):
            self.training_container.estimator_outputs = None

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

        if self.lazy_sort:
            # Accumulate the incoming batch without concatenating yet.
            self._pending_batches.append(training_container)
            self._pending_len += len(training_container)

            total_len = len(self.training_container) + self._pending_len

            # Flush, sort, and truncate only when we hit 2x capacity.
            if total_len >= 2 * self.capacity:
                self._flush_pending()
                self._sort_and_truncate(training_container)
        else:
            # Eager path: extend, sort, and truncate on every add.
            with Timer(self.timing_data, "local_add/extend", enabled=self.timing):
                self.training_container.extend(training_container)  # type: ignore

            # Clear fields that must be recomputed.
            if isinstance(self.training_container, (Trajectories, Transitions)):
                self.training_container.log_probs = None
            if isinstance(self.training_container, Trajectories):
                self.training_container.estimator_outputs = None

            self._sort_and_truncate(training_container)

    def _sort_and_truncate(self, training_container: ContainerUnion) -> None:
        """Sort by log-reward (if prioritized) and truncate to capacity."""
        assert self.training_container is not None

        if self.prioritized_capacity:
            if (
                self.training_container.log_rewards is None
                or training_container.log_rewards is None
            ):
                raise ValueError("log_rewards must be defined for prioritized replay.")

            with Timer(self.timing_data, "local_add/sort", enabled=self.timing):
                ix = torch.argsort(self.training_container.log_rewards)
                self.training_container = cast(
                    ContainerUnion, self.training_container[ix]
                )

        with Timer(self.timing_data, "local_add/truncate", enabled=self.timing):
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
        if self.training_container is None or len(self.training_container) == 0:
            raise ValueError("Buffer is empty")

        # Sample from the committed container only — pending batches are not
        # flushed here so that lazy sorting can accumulate up to 2x capacity.

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
            replacement = n_samples > len(self.training_container)

            indices = torch.multinomial(probs, n_samples, replacement=replacement)
            return self.training_container[indices]

        # Uniform sampling (replacement-free) for the non-prioritised case.
        return cast(ContainerUnion, self.training_container.sample(n_samples))

    def save(self, path: str):
        """Saves the buffer to a single ``.pt`` file.

        Args:
            path: File path (e.g. ``"replay_buffer.pt"``).
        """
        self._flush_pending()
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
            total = sum(times)
            count = len(times)
            mean = total / count if count > 0 else 0.0
            if key == "send_bytes":
                log_str += (
                    f"  {key}: total={total / 1e6:.1f} MB, "
                    f"count={count}, avg={mean / 1e6:.2f} MB\n"
                )
            else:
                log_str += (
                    f"  {key}: total={total:.4f}s, " f"count={count}, mean={mean:.4f}s\n"
                )
        # Effective bandwidth (send_bytes / send_data time).
        send_bytes = self.timing_data.get("send_bytes", [])
        send_times = self.timing_data.get("send_data", [])
        if send_bytes and send_times:
            total_bytes = sum(send_bytes)
            total_time = sum(send_times)
            if total_time > 0:
                bw = (total_bytes / 1e6) / total_time
                log_str += f"  effective_send_bandwidth: {bw:.1f} MB/s\n"
        # Baseline filtering stats.
        if self._baseline_total > 0:
            pct_filtered = 100.0 * (1.0 - self._baseline_kept / self._baseline_total)
            log_str += (
                f"  baseline_filtering: {self._baseline_kept}/{self._baseline_total} "
                f"kept ({pct_filtered:.1f}% filtered), "
                f"{self._baseline_skipped_sends} sends skipped entirely\n"
            )

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
        async_comm: bool = False,
        lazy_sort: bool = False,
        baseline_filtering: bool = False,
        scoring_only: bool = False,
        baseline_refresh_after: int = 10,
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
            async_comm: If True, fully non-blocking send and recv.
            lazy_sort: If True, defer concatenation and sorting until 2x capacity.
            baseline_filtering: See :class:`ReplayBuffer`.
            scoring_only: See :class:`ReplayBuffer`.
            baseline_refresh_after: See :class:`ReplayBuffer`.
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
            async_comm=async_comm,
            lazy_sort=lazy_sort,
            baseline_filtering=baseline_filtering,
            scoring_only=scoring_only,
            baseline_refresh_after=baseline_refresh_after,
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

        # The buffer isn't full yet — delegate to base class (lazy sorting ok).
        if len(self) < self.capacity:
            super()._local_add(training_container)
            return

        # Flush any pending lazy batches so diversity filtering sees all data.
        # This defeats lazy sorting once the buffer is full, but diversity
        # filtering requires a consistent view of the committed container.
        self._flush_pending()

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

    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
        communication_backend: str = "mpi",
        timing: bool = False,
        **kwargs,
    ):
        super().__init__(
            env,
            capacity,
            communication_backend=communication_backend,
            timing=timing,
            **kwargs,
        )
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
