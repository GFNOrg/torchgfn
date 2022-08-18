from abc import ABC, abstractmethod
from typing import Tuple

from torchtyping import TensorType

from gfn.containers import States
from gfn.envs import Env

# Typing
OutputTensor = TensorType["batch_shape", "dim_in", float]


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

    def __call__(self, states):
        return self.preprocess(states)

    def __repr__(self):
        return self.name


class IdentityPreprocessor(Preprocessor):
    "Simple preprocessor applicable to environments with uni-dimensional states."
    name = "IdentityPreprocessor"

    @property
    def output_dim(self):
        return self.env.ndim

    def preprocess(self, states):
        return states.states.float()
