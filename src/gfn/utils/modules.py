"""This file contains some examples of modules that can be used with GFN."""

from typing import Literal, Optional

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer


class MLP(nn.Module):
    """Implements a basic MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh", "elu"]] = "relu",
        trunk: Optional[nn.Module] = None,
        add_layer_norm: bool = False,
    ):
        """Instantiates a MLP instance.

        Args:
            input_dim: input dimension.
            output_dim: output dimension.
            hidden_dim: Number of units per hidden layer.
            n_hidden_layers: Number of hidden layers.
            activation_fn: Activation function.
            trunk: If provided, this module will be used as the trunk of the network
                (i.e. all layers except last layer).
            add_layer_norm: If True, add a LayerNorm after each linear layer.
                (incompatible with `trunk` argument)
        """
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

        if trunk is None:
            assert hidden_dim is not None, "hidden_dim must be provided"
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
            if add_layer_norm:
                arch = [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    activation(),
                ]
            else:
                arch = [nn.Linear(input_dim, hidden_dim), activation()]
            for _ in range(n_hidden_layers - 1):
                arch.append(nn.Linear(hidden_dim, hidden_dim))
                if add_layer_norm:
                    arch.append(nn.LayerNorm(hidden_dim))
                arch.append(activation())
            self.trunk = nn.Sequential(*arch)
            self.trunk.hidden_dim = torch.tensor(hidden_dim)
            self._hidden_dim = hidden_dim
        else:
            self.trunk = trunk
            assert hasattr(trunk, "hidden_dim") and isinstance(
                trunk.hidden_dim, torch.Tensor
            ), "trunk must have a hidden_dim attribute"
            self._hidden_dim = int(trunk.hidden_dim.item())
        self.last_layer = nn.Linear(self._hidden_dim, output_dim)

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        """Forward method for the neural network.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the MLP. The shape of the tensor should be (*batch_shape, input_dim).
        Returns: a tensor of shape (*batch_shape, output_dim).
        """
        out = self.trunk(preprocessed_states)
        out = self.last_layer(out)
        return out

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @input_dim.setter
    def input_dim(self, value: int):
        self._input_dim = value

    @output_dim.setter
    def output_dim(self, value: int):
        self._output_dim = value


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

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        """Forward method for the tabular policy.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the tabular policy. The shape of the tensor should be (*batch_shape, 1).
        Returns: a tensor of shape (*batch_shape, output_dim).
        """
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

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        """Forward method for the uniform distribution.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the uniform distribution. The shape of the tensor should be (*batch_shape, input_dim).

        Returns: a tensor of shape (*batch_shape, output_dim).
        """
        out = torch.zeros(*preprocessed_states.shape[:-1], self.output_dim).to(
            preprocessed_states.device
        )
        return out


class LinearTransformer(nn.Module):
    """The Linear Transformer module.

    Implements Transformers are RNNs: Fast Autoregressive Transformers with Linear
        Attention. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François
        Fleuret, ICML 2020.

    Expresses self-attention as a linear dot-product of kernel feature maps and makes
    use the associativity property of matrix products to reduce the complexity of the
    attention computation from O(n^2) to O(n).

    Implementation from https://github.com/lucidrains/linear-attention-transformer.

    Args:
        dim: The dimension of the input.
        depth: The depth of the transformer.
        max_seq_len: The maximum sequence length.
        n_heads: The number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        max_seq_len: int,
        n_heads: int = 8,
        causal: bool = False,
    ):
        super().__init__()
        assert isinstance(dim, int) and dim > 0, "dim must be a positive integer"
        assert isinstance(depth, int) and depth > 0, "depth must be a positive integer"
        assert (
            isinstance(max_seq_len, int) and max_seq_len > 0
        ), "max_seq_len must be a positive integer"

        self.module = LinearAttentionTransformer(
            dim,
            depth,
            max_seq_len,
            heads=n_heads,
            causal=causal,
            dim_head=None,
            bucket_size=64,
            ff_chunks=1,
            ff_glu=False,
            ff_dropout=0.0,
            attn_layer_dropout=0.0,
            attn_dropout=0.0,
            reversible=False,
            blindspot_size=1,
            n_local_attn_heads=0,
            local_attn_window_size=128,
            receives_context=False,
            attend_axially=False,
            pkm_layers=tuple(),
            pkm_num_keys=128,
            linformer_settings=None,
            context_linformer_settings=None,
            shift_tokens=False,
        )
        # TODO: Should we have a final linear layer as part of this module?
        # The output dimension is the same as the embedding dimension.
        self.output_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
