"""This file contains some examples of modules that can be used with GFN."""

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType as TT


class NeuralNet(nn.Module):
    """Implements a basic MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh", "elu"]] = "relu",
        torso: Optional[nn.Module] = None,
    ):
        """Instantiates a MLP instance.

        Args:
            input_dim: input dimension.
            output_dim: output dimension.
            hidden_dim: Number of units per hidden layer.
            n_hidden_layers: Number of hidden layers.
            activation_fn: Activation function.
            torso: If provided, this module will be used as the torso of the network
                (i.e. all layers except last layer).
        """
        super().__init__()
        self._output_dim = output_dim

        if torso is None:
            assert (
                n_hidden_layers is not None and n_hidden_layers >= 0
            ), "n_hidden_layers must be >= 0"
            assert activation_fn is not None, "activation_fn must be provided"
            if activation_fn == "elu":
                activation = nn.ELU
            elif activation_fn == "relu":
                activation = nn.ReLU
            elif activation_fn == "tanh":
                activation = nn.Tanh
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
        self, preprocessed_states: TT["batch_shape", "input_dim", float]
    ) -> TT["batch_shape", "output_dim", float]:
        """Forward method for the neural network.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the MLP.
        Returns: out, a set of continuous variables.
        """
        if self.device is None:
            self.device = preprocessed_states.device
            self.to(self.device)
        out = self.torso(preprocessed_states)
        out = self.last_layer(out)
        return out


class Tabular(nn.Module):
    """Implements a tabular policy.

    This class is only compatible with the EnumPreprocessor.

    Attributes:
        table: a tensor with dimensions [n_states, output_dim].
        device: the device that holds this policy.
    """

    def __init__(self, n_states: int, output_dim: int) -> None:
        """
        Args:
            n_states (int): Number of states in the environment.
            output_dim (int): Output dimension.
        """

        super().__init__()

        self.table = torch.zeros(
            (n_states, output_dim),
            dtype=torch.float,
        )

        self.table = nn.parameter.Parameter(self.table)
        self.device = None

    def forward(
        self, preprocessed_states: TT["batch_shape", "input_dim", float]
    ) -> TT["batch_shape", "output_dim", float]:
        if self.device is None:
            self.device = preprocessed_states.device
            self.table = self.table.to(self.device)
        assert preprocessed_states.dtype == torch.long
        outputs = self.table[preprocessed_states.squeeze(-1)]
        return outputs


class DiscreteUniform(nn.Module):
    """Implements a uniform distribution over discrete actions.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator.

    Attributes:
        output_dim: The size of the output space.
    """

    def __init__(self, output_dim: int) -> None:
        """Initializes the uniform function approximiator.

        Args:
            output_dim (int): Output dimension. This is typically n_actions if it
                implements a Uniform PF, or n_actions-1 if it implements a Uniform PB.
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(
        self, preprocessed_states: TT["batch_shape", "input_dim", float]
    ) -> TT["batch_shape", "output_dim", float]:
        out = torch.zeros(*preprocessed_states.shape[:-1], self.output_dim).to(
            preprocessed_states.device
        )
        return out
