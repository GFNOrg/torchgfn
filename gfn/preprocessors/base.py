from abc import ABC, abstractmethod
from torchtyping import TensorType
from gfn.envs.env import Env, AbstractStatesBatch
from typing import Tuple


# Typing
batch_shape = None
dim_in = None
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
    def preprocess(self, states: AbstractStatesBatch) -> OutputTensor:
        pass

    def __call__(self, states):
        return self.preprocess(states)

    def __repr__(self):
        return self.name


class IdentityPreprocessor(Preprocessor):
    "Simple preprocessor applicable to environments with unidimensional states."
    name = "IdentityPreprocessor"

    @property
    def output_dim(self):
        return self.env.ndim

    def preprocess(self, states):
        return states.states.float()
