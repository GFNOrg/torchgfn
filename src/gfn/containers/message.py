from __future__ import annotations

from enum import Enum, auto
from typing import Any

import dill as pickle
import numpy as np
import torch


class MessageType(Enum):
    DATA = auto()
    GET_METADATA = auto()
    EXIT = auto()


class Message:
    def __init__(self, message_type: MessageType, message_data: Any = None):
        self.message_type = message_type
        self.message_data = message_data

    def serialize(self) -> torch.Tensor:
        """Convert message into a tensor of bytes."""
        obj_bytes = pickle.dumps(self)
        # bytes → numpy → torch tensor
        arr = np.frombuffer(obj_bytes, dtype=np.uint8)
        return torch.from_numpy(
            arr.copy()
        ).contiguous()  # copy so tensor owns its memory

    @staticmethod
    def deserialize(byte_tensor: torch.Tensor) -> Message:
        """Reconstruct Message from a tensor of bytes."""
        # torch tensor → numpy → bytes
        obj_bytes = byte_tensor.numpy().tobytes()
        return pickle.loads(obj_bytes)
