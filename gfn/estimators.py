from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch.nn as nn
import torch
from torchtyping import TensorType
from typing import Union, Literal
from gfn.preprocessors.base import Preprocessor
from gfn.envs.env import Env, AbstractStatesBatch


# Typing
batch_shape = None
input_dim = None
output_dim = None
InputTensor = TensorType['batch_shape', 'input_dim', float]
OutputTensor = TensorType['batch_shape', 'output_dim', float]
OutputTensor1D = TensorType['batch_shape', 1, float]


@dataclass(eq=True, unsafe_hash=True)
class GFNModule(ABC):
    "Abstract Base Class for all functions/approximators/estimators used"
    input_dim: Union[int, None]
    output_dim: int
    output_type: Literal['free'] = 'free'

    @abstractmethod
    def __call__(self, input: InputTensor) -> OutputTensor:
        pass


class LogEdgeFlowEstimator:
    def __init__(self, preprocessor: Preprocessor, env: Env, module: GFNModule):
        assert module.input_dim is None or module.input_dim == preprocessor.output_dim
        assert module.output_dim == env.n_actions - 1
        assert module.output_type == 'free'
        self.preprocessor = preprocessor
        self.module = module
        self.env = env

    def __call__(self, states: AbstractStatesBatch) -> OutputTensor:
        return self.module(self.preprocessor(states))


class LogStateFlowEstimator:
    def __init__(self, preprocessor: Preprocessor, module: GFNModule):
        assert module.input_dim is None or module.input_dim == preprocessor.output_dim
        assert module.output_dim == 1
        assert module.output_type == 'free'
        self.preprocessor = preprocessor
        self.module = module

    def __call__(self, states: AbstractStatesBatch) -> OutputTensor1D:
        return self.module(self.preprocessor(states))


class LogitPFEstimator:
    def __init__(self, preprocessor: Preprocessor, env: Env, module: GFNModule):
        assert module.input_dim is None or module.input_dim == preprocessor.output_dim
        assert module.output_dim == env.n_actions
        assert module.output_type == 'free'
        self.preprocessor = preprocessor
        self.module = module

    def __call__(self, states: AbstractStatesBatch) -> OutputTensor:
        return self.module(self.preprocessor(states))


class LogitPBEstimator:
    def __init__(self, preprocessor: Preprocessor, env: Env, module: GFNModule):
        assert module.input_dim is None or module.input_dim == preprocessor.output_dim
        assert module.output_dim == env.n_actions - 1
        assert module.output_type == 'free'
        self.preprocessor = preprocessor
        self.module = module

    def __call__(self, states: AbstractStatesBatch) -> OutputTensor:
        return self.module(self.preprocessor(states))


@dataclass
class LogZEstimator:
    logZ: TensorType[0]
