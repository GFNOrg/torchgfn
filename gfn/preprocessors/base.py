from abc import ABC, abstractmethod
from typing import Tuple

from torchtyping import TensorType

from ..containers import States
from ..envs import Env

# Typing
OutputTensor = TensorType["batch_shape", "dim_in"]


class Preprocessor(ABC):
    """
    Base class for Preprocessors. The goal is to transform tensors representing raw states
    to tensors that can be used as input to neural networks.
    """

    name: str = "Preprocessor"

    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    @abstractmethod
    def output_dim(self) -> Tuple:
        pass

    @abstractmethod
    def preprocess(self, states: States) -> OutputTensor:
        pass

    def __call__(self, states: States) -> OutputTensor:
        return self.preprocess(states)

    def __repr__(self):
        return f"{self.name} of {self.env} with output_dim={self.output_dim}"


class IdentityPreprocessor(Preprocessor):
    "Simple preprocessor applicable to environments with uni-dimensional states."
    name = "IdentityPreprocessor"

    @property
    def output_dim(self):
        return self.env.ndim

    def preprocess(self, states):
        return states.states.float()


class EnumPreprocessor(Preprocessor):
    "Preprocessor applicable to environments with discrete states."
    name = "EnumPreprocessor"

    @property
    def output_dim(self):
        return 1

    def preprocess(self, states):
        return self.env.get_states_indices(states).long()
