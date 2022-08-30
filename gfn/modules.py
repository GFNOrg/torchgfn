from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torchtyping import TensorType

from gfn.envs import Env

# Typing
batch_shape = None
input_dim = None
output_dim = None
InputTensor = TensorType["batch_shape", "input_dim", float]
OutputTensor = TensorType["batch_shape", "output_dim", float]


@dataclass(eq=True, unsafe_hash=True)
class GFNModule(ABC):
    "Abstract Base Class for all functions/approximators/estimators used"
    input_dim: Optional[int]
    output_dim: int
    output_type: Literal["free", "positive"] = "free"

    @abstractmethod
    def __call__(self, input: InputTensor) -> OutputTensor:
        pass

    def named_parameters(self) -> Iterator:
        # Mimics torch.nn.Module.named_parameters()
        return iter([])


class NeuralNet(nn.Module, GFNModule):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh"]] = "relu",
        torso: Optional[nn.Module] = None,
        **kwargs
    ):
        del kwargs
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if torso is None:
            assert (
                n_hidden_layers is not None and n_hidden_layers >= 0
            ), "n_hidden_layers must be >= 0"
            assert activation_fn is not None, "activation_fn must be provided"
            activation = nn.ReLU if activation_fn == "relu" else nn.Tanh
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


class Tabular(nn.Module, GFNModule):
    def __init__(self, env: Env, output_dim: int, **kwargs) -> None:
        del kwargs
        super().__init__()

        self.env = env
        self.input_dim = None

        self.tensors = nn.ParameterList(
            [
                torch.zeros(output_dim, dtype=torch.float, device=env.device)
                for _ in range(env.n_states)
            ]
        )
        self.output_dim = output_dim

    def forward(self, preprocessed_states: InputTensor) -> OutputTensor:
        # Note that only the IdentityPreprocessor is compatible with the Tabular module, and only linear batches are possible
        assert preprocessed_states.ndim == 2
        states_indices = self.env.get_states_indices(
            self.env.States(preprocessed_states)
        )
        outputs = [self.tensors[index] for index in states_indices]
        if len(outputs) > 0:
            return torch.stack(outputs)
        else:
            return torch.tensor(
                [[] for _ in range(self.env.ndim)], device=preprocessed_states.device
            ).T


class Uniform(GFNModule):
    def __init__(self, output_dim: int, **kwargs):
        del kwargs
        """
        :param n_actions: the number of all possible actions
        """
        self.input_dim = None
        self.output_dim = output_dim

    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        logits = torch.zeros(*preprocessed_states.shape[:-1], self.output_dim).to(
            preprocessed_states.device
        )
        return logits


class ZeroGFNModule(GFNModule):
    def __init__(self):
        self.input_dim = None
        self.output_dim = 1

    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        out = torch.zeros(preprocessed_states.shape[0], self.output_dim).to(
            preprocessed_states.device
        )
        return out
