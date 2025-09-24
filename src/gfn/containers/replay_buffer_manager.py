from typing import Callable, Optional

import torch
import torch.distributed as dist

from .message import Message, MessageType


class ReplayBufferManager:

    def __init__(
        self,
        rank: int,
        num_training_ranks: int,
        scoring_function: Optional[Callable[[object], float]] = None,
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

    def default_scoring_function(self, obj) -> float:
        """Default reward function if none provided"""
        return float(len(str(obj)) * 0.1)

    def run(self):
        """Runs on remote buffer manager ranks. Waits for training data, computes dummy reward, sends back."""

        while self.is_running:
            # Receive data
            sender_rank, msg, msg_data_len = self._recv_object()

            if msg.message_type == MessageType.DATA:
                reward = self.scoring_function(msg.message_data)
                reward_tensor = torch.tensor([reward], dtype=torch.float32)
                dist.send(reward_tensor, dst=sender_rank)

            elif msg.message_type == MessageType.EXIT:
                self.exit_counter = self.exit_counter + 1
                if self.exit_counter == self.num_training_ranks:
                    self.is_running = False
                    print(
                        f"Replay buffer manager {self.rank} received exit signals from all training ranks. Exiting."
                    )
            else:
                raise ValueError(
                    f"Rank {self.rank} received unknown message type: {msg.message_type}"
                )

    def _recv_object(self):
        # Receive the length
        length_tensor = torch.IntTensor([0])
        sender_rank = dist.recv(length_tensor)
        length = length_tensor.item()

        # Receive the actual serialized data
        byte_tensor = torch.ByteTensor(length)
        dist.recv(byte_tensor, src=sender_rank)

        # Deserialize back into object
        # obj_bytes = bytes(byte_tensor.tolist())
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
