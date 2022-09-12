from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType

from gfn.envs import Env

# Typing

InputTensor = TensorType["batch_shape", "input_shape", float]
OutputTensor = TensorType["batch_shape", "output_dim", float]


class GFNModule(ABC):
    """Abstract Base Class for all functions/approximators/estimators used.
    Each module takes a preprocessed tensor as input, and outputs a tensor of logits,
    or log flows. The input dimension of the module (e.g.) Neural network, is deduced
    from the environment's preprocessor's output dimension"""

    def __init__(
        self,
        output_dim: int,
        input_shape: Optional[Tuple[int]] = None,
        **kwargs,
    ) -> None:
        self.output_dim = output_dim
        self.input_shape = input_shape
        del kwargs

    def named_parameters(self) -> dict:
        """Returns a dictionary of all (learnable) parameters of the module. Not needed
        for NeuralNet modules, given that those inherit this function from nn.Module"""
        return {}

    @abstractmethod
    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        pass


class NeuralNet(nn.Module, GFNModule):
    def __init__(
        self,
        input_shape: Tuple[int],
        output_dim: int,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh"]] = "relu",
        torso: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        super(nn.Module, self).__init__(
            output_dim=output_dim, input_shape=input_shape, **kwargs
        )

        if torso is None:
            assert (
                n_hidden_layers is not None and n_hidden_layers >= 0
            ), "n_hidden_layers must be >= 0"
            assert activation_fn is not None, "activation_fn must be provided"
            activation = nn.ReLU if activation_fn == "relu" else nn.Tanh
            if len(input_shape) > 1:
                raise NotImplementedError(
                    "Only 1D inputs are supported for new NeuralNet creation"
                )
            input_dim = input_shape[0]
            self.torso = [nn.Linear(input_dim, hidden_dim), activation()]
            for _ in range(n_hidden_layers - 1):
                self.torso.append(nn.Linear(hidden_dim, hidden_dim))
                self.torso.append(activation())
            self.torso = nn.Sequential(*self.torso)
            self.torso.hidden_dim = hidden_dim
        else:
            self.torso = torso
        self.last_layer = nn.Linear(self.torso.hidden_dim, output_dim)
        self.device = None

    def forward(self, preprocessed_states: InputTensor) -> OutputTensor:
        if self.device is None:
            self.device = preprocessed_states.device
            self.to(self.device)
        logits = self.torso(preprocessed_states)
        logits = self.last_layer(logits)
        return logits


class Tabular(GFNModule):
    def __init__(self, env: Env, output_dim: int, **kwargs) -> None:
        super().__init__(output_dim=output_dim, **kwargs)

        self.logits = torch.zeros(
            (env.n_states, output_dim),
            dtype=torch.float,
            device=env.device,
            requires_grad=True,
        )

    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        # Note that only the EnumPreprocessor is compatible with the Tabular module
        assert preprocessed_states.dtype == torch.long
        outputs = self.logits[preprocessed_states.squeeze(-1)]
        return outputs

    def named_parameters(self) -> dict:
        return {"logits": self.logits}


class ZeroGFNModule(GFNModule):
    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        out = torch.zeros(*preprocessed_states.shape[:-1], self.output_dim).to(
            preprocessed_states.device
        )
        return out


class Uniform(ZeroGFNModule):
    """Use this module for uniform policies for example"""

    pass
