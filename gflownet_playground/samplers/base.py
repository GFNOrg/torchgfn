from abc import ABC, abstractmethod
from typing import List
import torch
from torchtyping import TensorType

from dataclasses import dataclass


@dataclass
class Trajectory:
    """Class for keeping track of a trajectory."""
    states: TensorType['k']
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand


class Sampler(ABC):
    "Base class for trajectory samplers, with an extra functions that only returns last states"
    ""
