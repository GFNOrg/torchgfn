from enum import Enum, auto
from typing import Any

import dill as pickle
import torch


class MessageType(Enum):
    DATA = auto()
    EXIT = auto()


class Message:
    def __init__(self, type: MessageType, data: Any = None):
        self.type = type
        self.data = data

    def serialize(self) -> torch.ByteTensor:
        """Convert message into a tensor of bytes."""
        obj_bytes = pickle.dumps(self)
        return torch.ByteTensor(list(obj_bytes))

    @staticmethod
    def deserialize(byte_tensor: torch.ByteTensor) -> Message:
        """Reconstruct Message from a tensor of bytes."""
        obj_bytes = bytes(byte_tensor.tolist())
        return pickle.loads(obj_bytes)
