import math
from typing import Callable, Optional

from gfn.containers.message import Message, MessageType
from gfn.containers.replay_buffer import (
    NormBasedDiversePrioritizedReplayBuffer,
    ReplayBuffer,
)
from gfn.env import Env
from gfn.utils.distributed import recv, send

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
    ):
        self.rank = rank
        self.is_running = True
        self.exit_counter = 0
        self.num_training_ranks = num_training_ranks
        self.scoring_function = scoring_function or self.default_scoring_function
        self.communication_backend = communication_backend

        self.diverse_replay_buffer = diverse_replay_buffer
        self.capacity = capacity
        self.remote_manager_rank = remote_manager_rank
        if self.diverse_replay_buffer:
            self.replay_buffer = NormBasedDiversePrioritizedReplayBuffer(
                env,
                capacity=self.capacity,
                communication_backend=self.communication_backend,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                env,
                capacity=self.capacity,
                prioritized_capacity=True,  # Always prioritize high reward items.
                remote_manager_rank=self.remote_manager_rank,
                remote_buffer_freq=1,
                communication_backend=self.communication_backend,
            )

    def default_scoring_function(self, obj, sender_rank: int = -1) -> dict[str, float]:
        """Default score function if none provided, placeholder."""
        return {"score": math.inf}

    def _compute_metadata(self) -> dict:
        raise NotImplementedError(
            "_compute_metadata is not implemented for default replay buffer manager"
        )

    def run(self):
        """Runs on remote buffer manager ranks. Waits for training data, computes reward, sends back."""

        while self.is_running:
            # Receive data
            sender_rank, msg, _msg_data_len = self._recv_object()

            # Received some data to add to the buffer.
            if msg.message_type == MessageType.DATA:
                self.replay_buffer.add(msg.message_data)
                score_dict = self.scoring_function(
                    msg.message_data, sender_rank=sender_rank
                )
                message = Message(message_type=MessageType.DATA, message_data=score_dict)
                message_tensor = message.serialize()
                send(
                    message_tensor,
                    dst_rank=sender_rank,
                    backend=self.communication_backend,
                )

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
        _src_rank, metadata_tensor = recv(manager_rank, backend=backend, tag=METADATA_TAG)
        metadata = Message.deserialize(metadata_tensor)
        return metadata.message_data
