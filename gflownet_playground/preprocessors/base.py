from abc import ABC, abstractmethod
from torchtyping import TensorType
from gflownet_playground.envs.env import Env
from typing import Tuple


class Preprocessor(ABC):
    """
    Base class for Preprocessors. The goal is to transform tensors representing raw states
    to tensors that can be used as input to neural networks.
    """

    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    @abstractmethod
    def output_dim(self) -> Tuple:
        pass

    @abstractmethod
    def preprocess(self, states: TensorType['k': ..., 'state_dim': ...]) -> TensorType['k', 'dim_in']:
        """
        :param states: Tensor of shape (k x state_dim)
        :outputs: Tensor of shape (k x dim_in) where dim_in is the input dimension of the neural network
        """
        pass


class IdentityPreprocessor(Preprocessor):
    "simple preprocessor applicable to environments with unidimensional states."
    @property
    def output_dim(self):
        return self.env.state_dim

    def preprocess(self, states):
        return states
