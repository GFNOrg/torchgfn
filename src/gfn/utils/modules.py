"""This file contains some examples of modules that can be used with GFN."""

from typing import Literal, Optional

import torch
import torch.nn as nn
from torchtyping import TensorType


class NeuralNet(nn.Module):
    """Implements a basic MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh"]] = "relu",
        torso: Optional[nn.Module] = None,
    ):
        """
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (Optional[int], optional): Number of units per hidden layer. Defaults to 256.
            n_hidden_layers (Optional[int], optional): Number of hidden layers. Defaults to 2.
            activation_fn (Optional[Literal[relu, tanh]], optional): Activation function. Defaults to "relu".
            torso (Optional[nn.Module], optional): If provided, this module will be used as the torso of the network (i.e. all layers except last layer). Defaults to None.
        """
        super().__init__()
        self._output_dim = output_dim

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

    def forward(
        self, preprocessed_states: TensorType["batch_shape", "input_dim", float]
    ) -> TensorType["batch_shape", "output_dim", float]:
        if self.device is None:
            self.device = preprocessed_states.device
            self.to(self.device)
        logits = self.torso(preprocessed_states)
        logits = self.last_layer(logits)
        return logits


class Tabular(nn.Module):
    """Implements a tabular function approximator. Only compatible with the EnumPreprocessor."""

    def __init__(self, n_states: int, output_dim: int) -> None:
        """
        Args:
            n_states (int): Number of states in the environment.
            output_dim (int): Output dimension.
        """

        self.table = torch.zeros(
            (n_states, output_dim),
            dtype=torch.float,
        )

        self.table = nn.parameter.Parameter(self.table)

        self.device = None

    def __call__(
        self, preprocessed_states: TensorType["batch_shape", "input_dim", float]
    ) -> TensorType["batch_shape", "output_dim", float]:
        if self.device is None:
            self.device = preprocessed_states.device
            self.table = self.table.to(self.device)
        assert preprocessed_states.dtype == torch.long
        outputs = self.table[preprocessed_states.squeeze(-1)]
        return outputs


class Uniform(nn.Module):
    def __init__(self, output_dim: int) -> None:
        """Implements a zero function approximator, i.e. a function that always outputs 0.

        Args:
            output_dim (int): Output dimension. This is typically n_actions if it implements
                a Uniform PF, or n_actions-1 if it implements a Uniform PB.
        """
        self.output_dim = output_dim

    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        out = torch.zeros(*preprocessed_states.shape[:-1], self.output_dim).to(
            preprocessed_states.device
        )
        return out
