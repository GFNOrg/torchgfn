from typing import Callable, Optional

import torch
import torch.distributed as dist

from .message import Message, MessageType


class ReplayBufferManager:

    def __init__(
        self, rank: int, reward_function: Optional[Callable[[object], float]] = None
    ):
        self.rank = rank
        self.is_running = True
        self.reward_function = reward_function or self._default_reward

    def _default_reward(self, obj) -> float:
        """Default reward function if none provided"""
        return float(len(str(obj)) * 0.1)

    def run(self):
        """Runs on remote buffer manager ranks. Waits for training data, computes dummy reward, sends back."""

        while self.is_running:
            # Receive data
            sender_rank, msg, msg_data_len = self._recv_object()

            if msg.type == MessageType.DATA:
                reward = self.reward_function(msg.data)
                reward_tensor = torch.tensor([reward], dtype=torch.float32)
                dist.send(reward_tensor, dst=sender_rank)

            else:
                raise ValueError(
                    f"Rank {self.rank} received unknown message type: {msg.type}"
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
