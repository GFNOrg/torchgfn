from __future__ import annotations

from enum import Enum, auto
from typing import Any

import dill as pickle
import torch


class MessageType(Enum):
    DATA = auto()
    GET_METADATA = auto()
    EXIT = auto()
    MODE_REPORT = auto()  # Buffer -> Coordinator: new mode hashes + training rank
    GET_POPULATION_STATS = auto()  # Training rank -> Coordinator: request stats


class Message:
    def __init__(self, message_type: MessageType, message_data: Any = None):
        self.message_type = message_type
        self.message_data = message_data

    def serialize(self) -> torch.ByteTensor:
        """Convert message into a tensor of bytes."""
        obj_bytes = pickle.dumps(self)
        return torch.frombuffer(bytearray(obj_bytes), dtype=torch.uint8)  # type: ignore[return-value]

    @staticmethod
    def deserialize(byte_tensor: torch.ByteTensor) -> Message:
        """Reconstruct Message from a tensor of bytes."""
        obj_bytes = bytes(byte_tensor.numpy())
        return pickle.loads(obj_bytes)
