import math
from typing import Callable, Optional

import torch
import torch.distributed as dist

from gfn.containers.message import Message, MessageType
from gfn.containers.replay_buffer import (
    ContainerUnion,
    NormBasedDiversePrioritizedReplayBuffer,
    ReplayBuffer,
)
from gfn.env import Env


class ReplayBufferManager:

    def __init__(
        self,
        env: Env,
        rank: int,
        num_training_ranks: int,
        scoring_function: Optional[Callable[[ContainerUnion], float]] = None,
        diverse_replay_buffer: bool = False,
        capacity: int = 10000,
        remote_manager_rank: int | None = None,
    ):
        self.rank = rank
        self.is_running = True
        self.exit_counter = 0
        self.num_training_ranks = num_training_ranks
        self.scoring_function = scoring_function or self.default_scoring_function
        backend = dist.get_backend()
        if backend != "gloo":
            raise RuntimeError(
                f"Replay Buffer Manager is only supported with the 'gloo' backend, "
                f"but the current backend is '{backend}'."
            )

        self.diverse_replay_buffer = diverse_replay_buffer
        self.capacity = capacity
        self.remote_manager_rank = remote_manager_rank
        if self.diverse_replay_buffer:
            self.replay_buffer = NormBasedDiversePrioritizedReplayBuffer(
                env, capacity=self.capacity
            )
        else:
            self.replay_buffer = ReplayBuffer(
                env,
                capacity=self.capacity,
                prioritized_capacity=False,
                remote_manager_rank=self.remote_manager_rank,
                remote_buffer_freq=1,
            )

    def default_scoring_function(self, obj) -> float:
        """Default score function if none provided, placeholder."""
        return math.inf

    def _compute_metadata(self) -> dict:
        raise NotImplementedError(
            "_compute_metadata is not implemented for default replay buffer manager"
        )

    def run(self):
        """Runs on remote buffer manager ranks. Waits for training data, computes reward, sends back."""

        while self.is_running:
            # Receive data
            sender_rank, msg, msg_data_len = self._recv_object()

            # Recieved some data to add to the buffer.
            if msg.message_type == MessageType.DATA:
                score = self.scoring_function(msg.message_data)
                score_tensor = torch.tensor([score], dtype=torch.float32)
                print(f"Manager - Rank {self.rank} score: {score}")
                print(f"Manager - Rank {self.rank} sending score to rank {sender_rank}")
                dist.send(score_tensor, dst=sender_rank)
                self.replay_buffer.add(msg.message_data)

            elif msg.message_type == MessageType.GET_METADATA:
                metadata = self._compute_metadata()
                msg = Message(message_type=MessageType.DATA, message_data=metadata)
                metadata_tensor = msg.serialize()
                length_metadata_tensor = torch.IntTensor([len(metadata_tensor)])
                dist.send(length_metadata_tensor, dst=sender_rank)
                dist.send(metadata_tensor, dst=sender_rank)

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
        # Receive the length.
        length_tensor = torch.IntTensor([0])
        sender_rank = dist.recv(length_tensor)
        length = length_tensor.item()

        # Receive the actual serialized data.
        byte_tensor = torch.ByteTensor(length)
        dist.recv(byte_tensor, src=sender_rank)

        # Deserialize back into object.
        # obj_bytes = bytes(byte_tensor.tolist()). # TODO -- Remove?
        msg = Message.deserialize(byte_tensor)
        return sender_rank, msg, length

    @staticmethod
    def send_termination_signal(manager_rank: int):
        """Sends a termination signal to the replay buffer manager."""
        rank = dist.get_rank()
        msg = Message(message_type=MessageType.EXIT, message_data=None)
        msg_bytes = msg.serialize()
        length_tensor = torch.IntTensor([len(msg_bytes)])
        dist.send(length_tensor, dst=manager_rank)
        dist.send(msg_bytes, dst=manager_rank)
        print(
            f"Rank {rank} sent termination signal to replay buffer manager {manager_rank}."
        )

    @staticmethod
    def get_metadata(manager_rank: int) -> dict:
        """Sends a get metadata signal to the replay buffer manager."""
        msg = Message(message_type=MessageType.GET_METADATA, message_data=None)
        msg_bytes = msg.serialize()

        length_tensor = torch.IntTensor([len(msg_bytes)])
        dist.send(length_tensor, dst=manager_rank)

        dist.send(msg_bytes, dst=manager_rank)
        length_metadata_tensor = torch.IntTensor([0])

        dist.recv(length_metadata_tensor, src=manager_rank)
        metadata_tensor = torch.ByteTensor(length_metadata_tensor.item())

        dist.recv(metadata_tensor, src=manager_rank)
        metadata = Message.deserialize(metadata_tensor)
        return metadata.message_data
