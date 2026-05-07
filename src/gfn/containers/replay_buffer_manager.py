import math
from collections import defaultdict
from typing import Callable, Optional

from gfn.containers.message import Message, MessageType
from gfn.containers.replay_buffer import (
    NormBasedDiversePrioritizedReplayBuffer,
    ReplayBuffer,
)
from gfn.env import Env
from gfn.utils.common import Timer
from gfn.utils.distributed import AsyncSendHandle, isend, recv, send

# MPI tag constants for multiplexing independent message channels.
# Data messages (trajectory submissions + score responses) use tag 0.
# Metadata queries/responses use tag 1 so they never collide with
# pending async score responses on the same rank pair.
DATA_TAG = 0
METADATA_TAG = 1


class ReplayBufferManager:

    def __init__(
        self,
        env: Env,
        rank: int,
        num_training_ranks: int,
        scoring_function: Optional[Callable[..., dict[str, float]]] = None,
        diverse_replay_buffer: bool = False,
        capacity: int = 10000,
        remote_manager_rank: int | None = None,
        communication_backend: str = "mpi",
        timing: bool = False,
        store_locally: bool = True,
    ):
        self.store_locally = store_locally
        self.rank = rank
        self.is_running = True
        self.exit_counter = 0
        self.num_training_ranks = num_training_ranks
        self.scoring_function = scoring_function or self.default_scoring_function
        self.communication_backend = communication_backend
        self._pending_sends: list[AsyncSendHandle] = []
        self._timing_data: dict[str, list[float]] = defaultdict(list)
        self._comm_stats: dict[int, dict] = {}
        self.timing = timing

        self.diverse_replay_buffer = diverse_replay_buffer
        self.capacity = capacity
        self.remote_manager_rank = remote_manager_rank
        if self.diverse_replay_buffer:
            self.replay_buffer = NormBasedDiversePrioritizedReplayBuffer(
                env,
                capacity=self.capacity,
                communication_backend=self.communication_backend,
                timing=self.timing,
                lazy_sort=True,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                env,
                capacity=self.capacity,
                prioritized_capacity=True,  # Always prioritize high reward items.
                remote_manager_rank=self.remote_manager_rank,
                remote_buffer_freq=1,
                communication_backend=self.communication_backend,
                timing=self.timing,
                lazy_sort=True,
            )

    def default_scoring_function(self, obj, sender_rank: int = -1) -> dict[str, float]:
        """Default score function if none provided, placeholder."""
        return {"score": math.inf}

    def _inject_baseline_log_reward(self, score_dict: dict[str, float]) -> None:
        """Add ``baseline_log_reward`` to *score_dict* when the buffer is full.

        Reports the worst (min) log-reward currently held in the replay
        buffer so that agents with ``baseline_filtering=True`` can skip
        sending trajectories that would be immediately evicted.

        The key is only added once the buffer has reached capacity;
        before that every trajectory is accepted anyway.
        """
        if (
            self.replay_buffer is not None
            and self.replay_buffer.training_container is not None
            and len(self.replay_buffer) >= self.capacity
        ):
            buf_log_rewards = self.replay_buffer.training_container.log_rewards
            if buf_log_rewards is not None and buf_log_rewards.numel() > 0:
                score_dict["baseline_log_reward"] = float(buf_log_rewards.min().item())

    def _compute_metadata(self) -> dict:
        raise NotImplementedError(
            "_compute_metadata is not implemented for default replay buffer manager"
        )

    def run(self, async_send: bool = True):
        """Runs on remote buffer manager ranks. Waits for training data, computes reward, sends back scores.

        Args:
            async_send: If True (default), use non-blocking ``isend`` for
                responses.  If False, use blocking ``send`` for responses.
        """
        handler = self._handle_message_async if async_send else self._handle_message_sync
        while self.is_running:
            with Timer(self._timing_data, "recv", enabled=self.timing):
                sender_rank, msg, msg_data_len = self._recv_object()
            handler(sender_rank, msg, msg_data_len)
        self._print_timing_summary()

    def _prune_completed_sends(self) -> None:
        """Remove completed non-blocking sends from the pending list."""
        self._pending_sends = [h for h in self._pending_sends if not h.is_complete()]

    def _handle_message_async(self, sender_rank: int, msg, msg_data_len: int = 0):
        """Dispatch a message using non-blocking ``isend`` for responses."""
        # Prune completed sends to bound memory growth.
        self._prune_completed_sends()

        if msg.message_type == MessageType.DATA:
            if self.store_locally:
                with Timer(self._timing_data, "replay_add", enabled=self.timing):
                    self.replay_buffer.add(msg.message_data)
            with Timer(self._timing_data, "scoring", enabled=self.timing):
                score_dict = self.scoring_function(
                    msg.message_data, sender_rank=sender_rank
                )
            self._inject_baseline_log_reward(score_dict)
            with Timer(self._timing_data, "send", enabled=self.timing):
                message = Message(message_type=MessageType.DATA, message_data=score_dict)
                message_tensor = message.serialize()
                response_bytes = message_tensor.numel() * message_tensor.element_size()
                handle = isend(
                    message_tensor,
                    dst_rank=sender_rank,
                    backend=self.communication_backend,
                )
                self._pending_sends.append(handle)

            stats = self._comm_stats.setdefault(
                sender_rank,
                {
                    "n_requests": 0,
                    "bytes_recv": 0,
                    "bytes_sent": 0,
                },
            )
            stats["n_requests"] += 1
            stats["bytes_recv"] += msg_data_len
            stats["bytes_sent"] += response_bytes

        elif msg.message_type == MessageType.GET_METADATA:
            metadata = self._compute_metadata()
            msg = Message(message_type=MessageType.DATA, message_data=metadata)
            metadata_tensor = msg.serialize()
            handle = isend(
                metadata_tensor,
                dst_rank=sender_rank,
                backend=self.communication_backend,
                tag=METADATA_TAG,
            )
            self._pending_sends.append(handle)

        elif msg.message_type == MessageType.EXIT:
            self.exit_counter = self.exit_counter + 1
            if self.exit_counter == self.num_training_ranks:
                self.is_running = False
                print(
                    f"Manager - Replay buffer {self.rank} received exit signals from all training ranks. Exiting."
                )
        else:
            raise ValueError(
                f"Manager - Rank {self.rank} received unknown message type: {msg.message_type}"
            )

    def _handle_message_sync(self, sender_rank: int, msg, msg_data_len: int = 0):
        """Dispatch a message using blocking ``send`` for responses.

        Simpler than the async variant and uses zero CPU while the send
        is in flight, making it preferable when all ranks share a CPU.
        """
        if msg.message_type == MessageType.DATA:
            if self.store_locally:
                with Timer(self._timing_data, "replay_add", enabled=self.timing):
                    self.replay_buffer.add(msg.message_data)
            with Timer(self._timing_data, "scoring", enabled=self.timing):
                score_dict = self.scoring_function(
                    msg.message_data, sender_rank=sender_rank
                )
            self._inject_baseline_log_reward(score_dict)
            with Timer(self._timing_data, "send", enabled=self.timing):
                message = Message(message_type=MessageType.DATA, message_data=score_dict)
                message_tensor = message.serialize()
                response_bytes = message_tensor.numel() * message_tensor.element_size()
                send(
                    message_tensor,
                    dst_rank=sender_rank,
                    backend=self.communication_backend,
                )

            stats = self._comm_stats.setdefault(
                sender_rank,
                {
                    "n_requests": 0,
                    "bytes_recv": 0,
                    "bytes_sent": 0,
                },
            )
            stats["n_requests"] += 1
            stats["bytes_recv"] += msg_data_len
            stats["bytes_sent"] += response_bytes

        elif msg.message_type == MessageType.GET_METADATA:
            metadata = self._compute_metadata()
            msg = Message(message_type=MessageType.DATA, message_data=metadata)
            metadata_tensor = msg.serialize()
            send(
                metadata_tensor,
                dst_rank=sender_rank,
                backend=self.communication_backend,
                tag=METADATA_TAG,
            )

        elif msg.message_type == MessageType.EXIT:
            self.exit_counter = self.exit_counter + 1
            if self.exit_counter == self.num_training_ranks:
                self.is_running = False
                print(
                    f"Manager - Replay buffer {self.rank} received exit signals from all training ranks. Exiting."
                )
        else:
            raise ValueError(
                f"Manager - Rank {self.rank} received unknown message type: {msg.message_type}"
            )

    def _recv_object(self):
        sender_rank, byte_tensor = recv(backend=self.communication_backend)

        # Deserialize back into object.
        msg = Message.deserialize(byte_tensor)
        return sender_rank, msg, len(byte_tensor)

    @staticmethod
    def send_termination_signal(manager_rank: int, backend: str) -> None:
        """Sends a termination signal to the replay buffer manager."""
        msg = Message(message_type=MessageType.EXIT, message_data=None)
        msg_bytes = msg.serialize()
        send(msg_bytes, dst_rank=manager_rank, backend=backend)

    @staticmethod
    def get_metadata(manager_rank: int, backend: str) -> dict:
        """Sends a get metadata signal to the replay buffer manager.

        Uses ``METADATA_TAG`` so the response is never confused with
        pending data/score messages on the default tag.
        """
        msg = Message(message_type=MessageType.GET_METADATA, message_data=None)
        msg_bytes = msg.serialize()

        # The request goes on the default DATA_TAG (the buffer's recv loop
        # listens on tag=0 for all incoming messages).  The *response* comes
        # back on METADATA_TAG so it cannot collide with async score replies.
        send(msg_bytes, dst_rank=manager_rank, backend=backend)
        _src_rank, metadata_tensor = recv(
            manager_rank, backend=backend, tag=METADATA_TAG
        )
        metadata = Message.deserialize(metadata_tensor)
        return metadata.message_data

    def _print_timing_summary(self) -> None:
        """Print communication and timing stats at shutdown."""
        total_requests = sum(s["n_requests"] for s in self._comm_stats.values())
        total_recv = sum(s["bytes_recv"] for s in self._comm_stats.values())
        total_sent = sum(s["bytes_sent"] for s in self._comm_stats.values())

        print(
            f"\n{'=' * 60}\n"
            f"Buffer rank {self.rank} — communication summary\n"
            f"{'=' * 60}\n"
            f"  Total DATA requests : {total_requests}\n"
            f"  Total bytes recv    : {total_recv:,}\n"
            f"  Total bytes sent    : {total_sent:,}\n",
            flush=True,
        )

        if self._comm_stats:
            print("  Per-rank breakdown:", flush=True)
            for rank in sorted(self._comm_stats):
                s = self._comm_stats[rank]
                print(
                    f"    rank {rank:3d}: "
                    f"{s['n_requests']:6d} reqs, "
                    f"recv {s['bytes_recv']:>12,} B, "
                    f"sent {s['bytes_sent']:>12,} B",
                    flush=True,
                )

        if self.timing and self._timing_data:
            print(
                f"\n  Timing profile (seconds):\n"
                f"  {'phase':<15s} {'count':>8s} {'total':>10s} "
                f"{'mean':>10s} {'min':>10s} {'max':>10s}",
                flush=True,
            )
            for phase in ["recv", "replay_add", "scoring", "send"]:
                vals = self._timing_data.get(phase, [])
                if not vals:
                    continue
                print(
                    f"  {phase:<15s} {len(vals):8d} {sum(vals):10.4f} "
                    f"{sum(vals)/len(vals):10.6f} {min(vals):10.6f} "
                    f"{max(vals):10.6f}",
                    flush=True,
                )

        # Replay buffer internal timing breakdown.
        if self.replay_buffer is not None and self.replay_buffer.timing_data:
            print(
                f"\n  Replay buffer _local_add breakdown (seconds):\n"
                f"  {'phase':<25s} {'count':>8s} {'total':>10s} "
                f"{'mean':>10s} {'min':>10s} {'max':>10s}",
                flush=True,
            )
            for phase, vals in self.replay_buffer.timing_data.items():
                if not vals:
                    continue
                print(
                    f"  {phase:<25s} {len(vals):8d} {sum(vals):10.4f} "
                    f"{sum(vals)/len(vals):10.6f} {min(vals):10.6f} "
                    f"{max(vals):10.6f}",
                    flush=True,
                )

        print(f"{'=' * 60}\n", flush=True)
