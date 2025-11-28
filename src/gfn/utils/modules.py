"""This file contains some examples of modules that can be used with GFN."""

import math
from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from tensordict import TensorDict
from torch import Tensor
from torch_geometric.nn import DirGNNConv, GCNConv, GINConv

from gfn.actions import GraphActions, GraphActionType
from gfn.utils.common import is_int_dtype
from gfn.utils.graphs import GeometricBatch, get_edge_indices

ACTIVATION_FNS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


class MLP(nn.Module):
    """Implements a basic MLP with optional noisy layers for exploration.

    When `trunk` is provided, the MLP will be a wrapper around the trunk.

    See `Noisy Networks for Exploration (Fortunato et al., ICLR 2018)` for more details
    on the noisy layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: Optional[int] = 2,
        n_noisy_layers: int = 0,
        activation_fn: Literal["relu", "leaky_relu", "tanh", "elu"] = "relu",
        trunk: Optional[nn.Module] = None,
        add_layer_norm: bool = False,
        std_init: float = 0.1,
    ):
        """Initializes a new MLP.

        Args:
            input_dim: The dimension of the input.
            output_dim: The dimension of the output.
            hidden_dim: The dimension of the hidden layers.
            n_hidden_layers: The number of hidden layers.
            n_noisy_layers: The number of layers which are noisy, including the
                input and output layers.
            activation_fn: The activation function to use.
            trunk: A custom trunk to use. If None, a new trunk will be created.
            add_layer_norm: Whether to add layer normalization to the hidden layers.
            noise_std: The inital value of the noise standard deviation for noisy layers.
        """
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

        # Validate hidden layer inputs
        assert n_noisy_layers >= 0, "n_noisy_layers must be non-negative (>= 0)."
        assert std_init > 0, "std_init must be positive (> 0)."

        # If a trunk is provided, the MLP will be a wrapper around the trunk.
        if trunk is not None:
            assert (
                n_noisy_layers <= 1
            ), "when trunk is provided, n_noisy_layers must be 0 or 1."
            self.trunk = trunk
            assert hasattr(trunk, "hidden_dim") and isinstance(
                trunk.hidden_dim, torch.Tensor
            ), "trunk must have a hidden_dim attribute"
            self._hidden_dim = int(trunk.hidden_dim.item())
        # If no trunk is provided, we build the MLP from scratch. Noisy layers are added
        # to the end of the network.
        else:
            assert n_hidden_layers is not None, "n_hidden_layers must be provided"
            assert n_hidden_layers >= 0, "n_hidden_layers must be non-negative (>= 0)."
            assert (
                n_noisy_layers <= n_hidden_layers + 1
            ), "n_noisy_layers must be <= n_hidden_layers + the output layer."
            assert hidden_dim is not None, "hidden_dim must be provided"
            assert (
                activation_fn in ACTIVATION_FNS
            ), "activation_fn must be one of " + str(ACTIVATION_FNS.keys())

            activation = ACTIVATION_FNS[activation_fn]

            # Initialize the input layer (never noisy).
            if add_layer_norm:
                arch = [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    activation(),
                ]
            else:
                arch = [
                    nn.Linear(input_dim, hidden_dim),
                    activation(),
                ]

            # Add the hidden layers. Put all noisy layers near the output.
            n_noisy_hidden_layers = max(0, n_noisy_layers - 1)
            n_noiseless_hidden_layers = n_hidden_layers - n_noisy_hidden_layers
            hidden_layer_types = [nn.Linear] * n_noiseless_hidden_layers + [
                NoisyLinear
            ] * n_noisy_hidden_layers

            for layer_type in hidden_layer_types:
                if isinstance(layer_type, NoisyLinear):
                    arch.append(layer_type(hidden_dim, hidden_dim, std_init=std_init))
                else:
                    arch.append(layer_type(hidden_dim, hidden_dim))

                if add_layer_norm:
                    arch.append(nn.LayerNorm(hidden_dim))

                arch.append(activation())

            self.trunk = nn.Sequential(*arch)
            self.trunk.hidden_dim = torch.tensor(hidden_dim)
            self._hidden_dim = hidden_dim

        # Initialize the output layer.
        if n_noisy_layers == 0:
            self.last_layer = nn.Linear(self._hidden_dim, output_dim)
        else:
            self.last_layer = NoisyLinear(
                self._hidden_dim, output_dim, std_init=std_init  # type: ignore
            )

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        """Forward method for the neural network.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the MLP. The shape of the tensor should be (*batch_shape, input_dim).
        Returns: a tensor of shape (*batch_shape, output_dim).
        """
        if not preprocessed_states.is_floating_point():
            preprocessed_states = preprocessed_states.to(torch.get_default_dtype())

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
        """Initializes a new Tabular module.

        Args:
            n_states: The number of states.
            output_dim: The dimension of the output.
        """
        super().__init__()
        self.table = nn.parameter.Parameter(torch.zeros((n_states, output_dim)))
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
        assert is_int_dtype(preprocessed_states)
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
        """Initializes a new DiscreteUniform module.

        Args:
            output_dim: The dimension of the output.
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
        causal: Whether to use causal attention.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        max_seq_len: int,
        n_heads: int = 8,
        causal: bool = False,
    ):
        """Initializes a new LinearTransformer module.

        Args:
            dim: The dimension of the input.
            depth: The depth of the transformer.
            max_seq_len: The maximum sequence length.
            n_heads: The number of attention heads.
            causal: Whether to use causal attention.
        """
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


class GraphActionGNN(nn.Module):
    """Implements a GNN for graph action prediction."""

    def __init__(
        self,
        num_node_classes: int,
        num_edge_classes: int,
        directed: bool,
        embedding_dim: int = 128,
        num_conv_layers: int = 1,
        is_backward: bool = False,
    ) -> None:
        """Initializes a new GraphActionGNN module.

        Args:
            n_nodes: The number of nodes in the graph.
            directed: Whether the graph is directed.
            num_edge_classes: The number of edge classes.
            num_conv_layers: The number of convolutional layers.
            embedding_dim: The dimension of the node embeddings.
            is_backward: Whether the GNN is used for a backward policy.
        """
        super().__init__()
        assert num_node_classes > 0, "num_node_classes must be greater than 0"
        assert embedding_dim > 0, "embedding_dim must be greater than 0"
        assert num_conv_layers > 0, "num_conv_layers must be greater than 0"
        assert isinstance(num_node_classes, int), "n_nodes must be an integer"
        assert isinstance(embedding_dim, int), "embedding_dim must be an integer"
        assert isinstance(num_conv_layers, int), "num_conv_layers must be an integer"
        assert isinstance(directed, bool), "directed must be a boolean"
        assert isinstance(is_backward, bool), "is_backward must be a boolean"
        self.num_node_classes = num_node_classes
        self.hidden_dim = self.embedding_dim = embedding_dim
        self.is_backward = is_backward
        self.is_directed = directed
        self.num_conv_layers = num_conv_layers
        self.num_edge_classes = num_edge_classes

        # Node embedding layer.
        self.embedding = nn.Embedding(num_node_classes, self.embedding_dim)
        self.conv_blks = nn.ModuleList()

        if directed:
            for i in range(num_conv_layers):
                self.conv_blks.extend(
                    [
                        DirGNNConv(
                            GCNConv(
                                self.embedding_dim if i == 0 else self.hidden_dim,
                                self.hidden_dim,
                            ),
                            alpha=0.5,
                            root_weight=True,
                        ),
                        # Process in/out components separately
                        nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(
                                        self.hidden_dim // 2,
                                        self.hidden_dim // 2,
                                    ),
                                    nn.ReLU(),
                                    nn.Linear(
                                        self.hidden_dim // 2,
                                        self.hidden_dim // 2,
                                    ),
                                )
                                for _ in range(2)  # 1 for in & 1 for out-features.
                            ]
                        ),
                    ]
                )
        else:  # Undirected case.
            for i in range(num_conv_layers):
                self.conv_blks.extend(
                    [
                        GINConv(
                            MLP(
                                input_dim=(
                                    self.embedding_dim if i == 0 else self.hidden_dim
                                ),
                                output_dim=self.hidden_dim,
                                hidden_dim=self.hidden_dim,
                                n_hidden_layers=1,
                                add_layer_norm=True,
                            ),
                        ),
                        nn.Sequential(
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                        ),
                    ]
                )

        self.norm = nn.LayerNorm(self.hidden_dim)

        # Heads operating on per-graph pooled features
        self.action_type_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=3,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )
        self.node_class_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=self.num_node_classes,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )
        self.node_index_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=1,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )
        self.edge_class_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=self.num_edge_classes,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

    @property
    def input_dim(self):
        return 1  # placeholder TODO: remove this

    @staticmethod
    def _group_mean(tensor: torch.Tensor, batch_ptr: torch.Tensor) -> torch.Tensor:
        # Safe mean over ragged graphs using ptr; returns zeros for empty graphs
        if tensor.numel() == 0:
            B = batch_ptr.numel() - 1
            return torch.zeros(B, 0, device=batch_ptr.device)
        cumsum = torch.zeros(
            (len(tensor) + 1, tensor.size(-1)), dtype=tensor.dtype, device=tensor.device
        )
        cumsum[1:] = torch.cumsum(tensor, dim=0)
        size = batch_ptr[1:] - batch_ptr[:-1]
        denom = torch.clamp(size, min=1).to(tensor.dtype)
        sums = cumsum[batch_ptr[1:]] - cumsum[batch_ptr[:-1]]
        means = sums / denom[:, None]
        means[size == 0] = 0
        return means

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        node_features = states_tensor.x
        B = len(states_tensor)
        lengths = states_tensor.ptr[1:] - states_tensor.ptr[:-1]
        max_nodes = int(lengths.max().item())
        device = node_features.device

        # Embed node classes
        if node_features.numel() > 0:
            x = self.embedding(node_features.squeeze(-1))
        # Handle the case where the graph has no nodes. We use zeros as
        # features, so we can continue the forward pass.
        else:
            x = torch.zeros(0, self.hidden_dim, device=device)

        # Message passing with residual connections and layer norms
        for i in range(0, len(self.conv_blks), 2):
            x_new = self.conv_blks[i](x, states_tensor.edge_index)  # GIN/GCN conv.
            if self.is_directed:
                assert isinstance(self.conv_blks[i + 1], nn.ModuleList)
                x_in, x_out = torch.chunk(x_new, 2, dim=-1)

                # Process each component separately through its own MLP.
                mlp_in, mlp_out = self.conv_blks[i + 1]
                x_in = mlp_in(x_in)
                x_out = mlp_out(x_out)
                x_new = torch.cat([x_in, x_out], dim=-1)
            else:
                x_new = self.conv_blks[i + 1](x_new)  # Linear -> ReLU -> Linear.

            x = x_new + x if i > 0 else x_new  # Residual connection.
            x = self.norm(x)  # Layernorm.

            # Pool to graph features
        graph_emb = (
            self._group_mean(x, states_tensor.ptr)
            if x.numel() > 0
            else torch.zeros(B, self.hidden_dim, device=device)
        )

        # Action type and class logits
        action_type = self.action_type_mlp(graph_emb)
        node_class_logits = self.node_class_mlp(graph_emb)
        edge_class_logits = self.edge_class_mlp(graph_emb)

        # Node index logits
        if self.is_backward:
            node_index_logits = self.node_index_mlp(x).squeeze(-1)
            node_index_logits = torch.split(node_index_logits, lengths.tolist(), dim=0)
            node_index_logits = torch.nn.utils.rnn.pad_sequence(
                list(node_index_logits), batch_first=True
            )
        else:
            node_index_logits = torch.zeros(B, max_nodes + 1, device=device)

        # Edge-index logits via pairwise dot products
        # Pad to max_nodes across batch for gathering candidate edges
        seqs = torch.split(x, lengths.tolist())
        padded = torch.nn.utils.rnn.pad_sequence(
            list(seqs), batch_first=True
        )  # (B, max_nodes, hidden_dim)

        feature_dim = (self.hidden_dim // 2) if self.is_directed else self.hidden_dim
        if self.is_directed:
            source_features = padded[..., :feature_dim]
            target_features = padded[..., feature_dim:]
            scores = torch.einsum("bnf,bmf->bnm", source_features, target_features)
        else:
            scores = torch.einsum("bnf,bmf->bnm", padded, padded)
        scores = scores / math.sqrt(max(1, feature_dim))

        ei0, ei1 = get_edge_indices(max_nodes, self.is_directed, device)
        edge_index_logits = scores[:, ei0, ei1]

        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.NODE_CLASS_KEY: node_class_logits,
                GraphActions.NODE_INDEX_KEY: node_index_logits,
                GraphActions.EDGE_CLASS_KEY: edge_class_logits,
                GraphActions.EDGE_INDEX_KEY: edge_index_logits,
            },
            batch_size=B,
        )


class GraphEdgeActionMLP(nn.Module):
    """Network that processes flattened adjacency matrices to predict graph actions.

    Unlike the GNN-based GraphActionGNN, this module uses standard MLPs to process
    the entire adjacency matrix as a flattened vector. This approach:

    1. Can directly process global graph structure without message passing.
    2. May be more effective for small graphs where global patterns are important.
    3. Does not require complex graph neural network operations.

    The module architecture consists of:
    - An MLP to process the flattened adjacency matrix into an embedding.
    - An edge MLP that predicts logits for each possible edge action.
    - An exit MLP that predicts a logit for the exit action.

    Args:
        n_nodes: Number of nodes in the graph.
        directed: Whether the graph is directed or undirected.
        n_hidden_layers: Number of hidden layers in the MLP for the edge actions.
        n_hidden_layers_exit: Number of hidden layers in the MLP for the exit action.
        embedding_dim: Dimension of internal embeddings.
        is_backward: Whether this is a backward policy.
    """

    def __init__(
        self,
        n_nodes: int,
        directed: bool,
        num_node_classes: int,
        num_edge_classes: int,
        n_hidden_layers: int = 2,
        n_hidden_layers_exit: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        """Initializes a new GraphEdgeActionMLP module.

        Args:
            n_nodes: The number of nodes in the graph.
            directed: Whether the graph is directed.
            num_node_classes: The number of node classes.
            num_edge_classes: The number of edge classes.
            n_hidden_layers: The number of hidden layers in the main MLP.
            n_hidden_layers_exit: The number of hidden layers in the exit MLP.
            embedding_dim: The dimension of the embeddings.
            is_backward: Whether the MLP is used for a backward policy.
        """
        super().__init__()
        assert n_nodes > 0, "n_nodes must be greater than 0"
        assert embedding_dim > 0, "embedding_dim must be greater than 0"
        assert n_hidden_layers > 0, "n_hidden_layers must be greater than 0"
        assert n_hidden_layers_exit > 0, "n_hidden_layers_exit must be greater than 0"
        assert isinstance(n_nodes, int), "n_nodes must be an integer"
        assert isinstance(embedding_dim, int), "embedding_dim must be an integer"
        assert isinstance(n_hidden_layers, int), "n_hidden_layers must be an integer"
        assert isinstance(
            n_hidden_layers_exit, int
        ), "n_hidden_layers_exit must be an integer"
        assert isinstance(directed, bool), "directed must be a boolean"
        assert isinstance(is_backward, bool), "is_backward must be a boolean"
        self._input_dim = n_nodes**2
        self.n_nodes = n_nodes
        self.is_directed = directed
        self.is_backward = is_backward
        self.hidden_dim = embedding_dim
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes

        # MLP for processing the flattened adjacency matrix
        self.mlp = MLP(
            input_dim=n_nodes**2,  # Flattened adjacency matrix
            output_dim=embedding_dim,
            hidden_dim=embedding_dim,
            n_hidden_layers=n_hidden_layers,
            add_layer_norm=True,
        )

        # Exit action MLP
        self.exit_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=1,
            hidden_dim=embedding_dim,
            n_hidden_layers=n_hidden_layers_exit,
            add_layer_norm=True,
        )

        # Edge prediction MLP
        # Output dimension.
        edges_dim = self.n_nodes**2 - self.n_nodes
        if not self.is_directed:
            edges_dim = edges_dim // 2  # No double-counting.

        if not self.is_backward:
            out_dim = edges_dim + 1  # +1 for exit action.
        else:
            out_dim = edges_dim

        self._output_dim = int(out_dim)
        self._edges_dim = int(edges_dim)

        self.node_class_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=self.num_node_classes,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )
        self.features_embedding = nn.Embedding(self.num_node_classes, embedding_dim)
        self.node_index_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=1,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )
        self.edge_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=self.edges_dim,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

        self.edge_class_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=self.num_edge_classes,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def edges_dim(self) -> int:
        return self._edges_dim

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        """Forward pass to compute action logits from graph states.

        Process:
        1. Convert the graph representation to adjacency matrices
        2. Process the flattened adjacency matrices through the main MLP
        3. Predict logits for edge actions and exit action

        Args:
            states_tensor: A GeometricBatch containing graph state information

        Returns:
            A tensor of logits for all possible actions
        """
        device = states_tensor.x.device
        # Convert the graph to adjacency matrix.
        adj_matrices = torch.zeros(
            (len(states_tensor), self.n_nodes, self.n_nodes),
            device=device,
        )

        # Fill the adjacency matrices from edge indices
        if states_tensor.edge_index.numel() > 0:
            for i in range(len(states_tensor)):
                eis = states_tensor[i].edge_index
                adj_matrices[i, eis[0], eis[1]] = 1

        # Flatten the adjacency matrices for the MLP
        adj_matrices_flat = adj_matrices.view(len(states_tensor), -1)

        # Process with MLP
        embedding = self.mlp(adj_matrices_flat)

        # Generate edge and exit actions
        edge_actions = self.edge_mlp(embedding)

        action_type = torch.ones(len(states_tensor), 3, device=device) * float("-inf")
        edge_class_logits = torch.zeros(
            len(states_tensor), self.num_edge_classes, device=device
        )
        lengths = states_tensor.ptr[1:] - states_tensor.ptr[:-1]
        if self.is_backward:
            action_type[..., GraphActionType.ADD_EDGE] = 0.0
            node_features = self.features_embedding(states_tensor.x)
            node_index_logits = self.node_index_mlp(node_features).squeeze(-1)
            node_index_logits = torch.split(node_index_logits, lengths.tolist(), dim=0)
            node_index_logits = torch.nn.utils.rnn.pad_sequence(
                list(node_index_logits), batch_first=True
            )
        else:
            exit_action = self.exit_mlp(embedding).squeeze(-1)
            action_type[..., GraphActionType.ADD_EDGE] = 0.0
            action_type[..., GraphActionType.EXIT] = exit_action
            edge_class_logits = self.edge_class_mlp(embedding)
            node_class_logits = self.node_class_mlp(embedding)
            node_index_logits = torch.zeros(
                len(states_tensor), self.n_nodes + 1, device=device
            )

        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.NODE_CLASS_KEY: node_class_logits,
                GraphActions.NODE_INDEX_KEY: node_index_logits,
                GraphActions.EDGE_CLASS_KEY: edge_class_logits,
                GraphActions.EDGE_INDEX_KEY: edge_actions,
            },
            batch_size=len(states_tensor),
        )


class GraphScalarMLP(nn.Module):
    """Graph encoder that maps adjacency structure to n scalar output."""

    def __init__(
        self,
        n_nodes: int,
        directed: bool,
        embedding_dim: int = 128,
        n_hidden_layers: int = 2,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        assert n_nodes > 0, "n_nodes must be positive"
        assert embedding_dim > 0, "embedding_dim must be positive"
        assert n_hidden_layers >= 0, "n_hidden_layers must be non-negative"
        self.n_nodes = n_nodes
        self.is_directed = directed
        self.input_dim = n_nodes**2

        self.backbone = MLP(
            input_dim=n_nodes**2,
            output_dim=embedding_dim,
            hidden_dim=embedding_dim,
            n_hidden_layers=n_hidden_layers,
            add_layer_norm=True,
        )
        self.head = MLP(
            input_dim=embedding_dim,
            output_dim=n_outputs,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

    def forward(self, states_tensor: GeometricBatch) -> torch.Tensor:
        """Encode graphs into a scalar per element of the batch."""
        batch_size = len(states_tensor)
        device = states_tensor.x.device
        adj = torch.zeros((batch_size, self.n_nodes, self.n_nodes), device=device)

        if states_tensor.edge_index.numel() > 0:
            for i in range(batch_size):
                edges = states_tensor[i].edge_index
                adj[i, edges[0], edges[1]] = 1
                if not self.is_directed:
                    adj[i, edges[1], edges[0]] = 1

        embedding = self.backbone(adj.view(batch_size, -1))
        return self.head(embedding)


class GraphActionUniform(nn.Module):
    """Implements a uniform distribution over discrete actions given a graph state.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator.

    Attributes:
        output_dim: The size of the output space.
    """

    def __init__(
        self,
        edges_dim: int,
        num_edge_classes: int,
        num_node_classes: int,
    ) -> None:
        """Initializes a new GraphActionUniform module.

        Args:
            edges_dim: The dimension of edge_index in GraphActions.
            num_edge_classes: The number of edge classes.
            num_node_classes: The number of node classes.
        """
        """Initializes the uniform function approximiator.

        Args:
            edges_dim (int): The dimension of edge_index in GraphActions.
            num_edge_classes (int): Number of edge classes.
            num_node_classes (int): Number of node classes.
        """
        super().__init__()
        self.input_dim = 1  # has no effect
        self.edges_dim = edges_dim
        self.num_edge_classes = num_edge_classes
        self.num_node_classes = num_node_classes

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        """Forward method for the uniform distribution.

        Args:
            states_tensor: a batch of states appropriately preprocessed for
                ingestion by the uniform distribution.

        Returns:
            A TensorDict containing logits for each action component, with all values
              set to 1 to represent a uniform distribution:
            - GraphActions.ACTION_TYPE_KEY: Tensor of shape [*batch_shape, 3] for the 3
              possible action types
            - GraphActions.EDGE_CLASS_KEY: Tensor of shape [*batch_shape,
              num_edge_classes] for edge class logits
            - GraphActions.NODE_CLASS_KEY: Tensor of shape [*batch_shape,
              num_node_classes] for node class logits
            - GraphActions.EDGE_INDEX_KEY: Tensor of shape [*batch_shape, edges_dim]
              for edge index logits
        """
        device = states_tensor.x.device
        max_nodes = int(torch.max(states_tensor.ptr[1:] - states_tensor.ptr[:-1]))
        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: torch.ones(
                    len(states_tensor), 3, device=device
                ),
                GraphActions.EDGE_CLASS_KEY: torch.ones(
                    len(states_tensor), self.num_edge_classes, device=device
                ),
                GraphActions.NODE_CLASS_KEY: torch.ones(
                    len(states_tensor), self.num_node_classes, device=device
                ),
                GraphActions.NODE_INDEX_KEY: torch.ones(
                    len(states_tensor), max_nodes, device=device
                ),
                GraphActions.EDGE_INDEX_KEY: torch.ones(
                    len(states_tensor), self.edges_dim, device=device
                ),
            },
            batch_size=len(states_tensor),
        )


class NoisyLinear(nn.Linear):
    """Noisy Linear Layer.

    Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added to the weights.
    This induced stochasticity can be used in RL networks for the agent's policy to aid
    efficient exploration. The parameters of the noise are learned with gradient descent
    along with any other remaining network weights. Factorized Gaussian noise is the
    type of noise usually employed.

    Taken from torchrl v0.9.2.

    Args:
        in_features (int): input features dimension
        out_features (int): out features dimension
        bias (bool, optional): if ``True``, a bias term will be added to the matrix multiplication: Ax + b.
            Defaults to ``True``
        device (DEVICE_TYPING, optional): device of the layer.
            Defaults to ``"cpu"``
        dtype (torch.dtype, optional): dtype of the parameters.
            Defaults to ``None`` (default pytorch dtype)
        std_init (scalar, optional): initial value of the Gaussian standard deviation before optimization.
            Defaults to ``0.1``

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std_init: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))  # type: ignore
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)  # type: ignore

    def _scale_noise(self, size: int | torch.Size) -> torch.Tensor:
        if isinstance(size, int):
            size = (size,)  # type: ignore
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @property
    def weight(self) -> torch.Tensor:
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon  # type: ignore
        else:
            return self.weight_mu

    @property
    def bias(self) -> torch.Tensor | None:
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon  # type: ignore
            else:
                return self.bias_mu
        else:
            return None


class AutoregressiveDiscreteSequenceModel(ABC, nn.Module):

    @abstractmethod
    def init_carry(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Initialize the carry for the sequence model.

        Args:
            batch_size (int): Batch size.
            device (torch.device): Device to allocate carry tensors on.

        Returns:
            dict[str, torch.Tensor]: Initialized carry.
        """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        carry: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the logits for the next tokens in the sequence.

        Args:
            x (torch.Tensor): (B, T) tensor of input token indices where ``T`` is the
                number of newly supplied timesteps (``T`` may be 1 for incremental
                decoding).
            carry (dict[str, torch.Tensor]): Carry from previous steps for recurrent
                processing (e.g., hidden states).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Logits for the next token
                at each supplied timestep with shape (B, T, vocab) and updated carry.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of the vocabulary (excluding BOS token)."""


class RecurrentDiscreteSequenceModel(AutoregressiveDiscreteSequenceModel):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        rnn_kind = rnn_type.lower()
        if rnn_kind not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'.")

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in the range [0, 1].")

        self._vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_kind

        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 for BOS token
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.lstm: nn.LSTM | None
        self.gru: nn.GRU | None
        if rnn_kind == "lstm":
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
            self.gru = None
        else:
            self.gru = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
            self.lstm = None
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def init_carry(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        carry: dict[str, torch.Tensor] = {
            "hidden": torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            ),
        }
        if self.rnn_type == "lstm":
            carry["cell"] = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )
        return carry

    def forward(
        self,
        x: torch.Tensor,
        carry: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 2:
            raise ValueError("Expected input tensor with shape (batch, timesteps).")

        batch, timesteps = x.size()
        device = x.device

        if "hidden" not in carry:
            raise KeyError("Carry must provide a 'hidden' state tensor.")

        hidden = carry["hidden"]
        if hidden.size(1) != batch:
            raise ValueError(
                "Hidden state batch dimension does not match the provided tokens."
            )
        if hidden.device != device:
            raise ValueError(
                "Hidden state tensor must live on the same device as input tokens."
            )

        embedded = self.embedding(x)

        if self.rnn_type == "lstm":
            lstm = self.lstm
            if lstm is None:
                raise RuntimeError("LSTM module was not initialized.")
            if "cell" not in carry:
                raise KeyError("LSTM carry must provide a 'cell' state tensor.")
            cell = carry["cell"]
            if cell.size(1) != batch:
                raise ValueError(
                    "Cell state batch dimension does not match the provided tokens."
                )
            if cell.device != device:
                raise ValueError(
                    "Cell state tensor must live on the same device as input tokens."
                )
            outputs, (hidden_next, cell_next) = lstm(embedded, (hidden, cell))
            updated_carry: dict[str, torch.Tensor] = {
                "hidden": hidden_next,
                "cell": cell_next,
            }
        else:
            gru = self.gru
            if gru is None:
                raise RuntimeError("GRU module was not initialized.")
            outputs, hidden_next = gru(embedded, hidden)
            updated_carry = {
                "hidden": hidden_next,
            }

        logits = self.output_projection(outputs)
        return logits, updated_carry

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class _AutoregressiveTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.linear1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        key_carry: torch.Tensor,
        value_carry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, timesteps, _ = hidden.size()

        normed_hidden = self.norm1(hidden)

        q = self.q_proj(normed_hidden)
        k = self.k_proj(normed_hidden)
        v = self.v_proj(normed_hidden)

        q = q.view(batch, timesteps, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, timesteps, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, timesteps, self.num_heads, self.head_dim).transpose(1, 2)

        carry_length = key_carry.size(2)
        updated_key_carry = torch.cat((key_carry, k), dim=2)
        updated_value_carry = torch.cat((value_carry, v), dim=2)

        attn_scores = torch.matmul(q, updated_key_carry.transpose(-2, -1)) / math.sqrt(
            float(self.head_dim)
        )

        if timesteps > 1 or carry_length > 0:
            total_kv_length = carry_length + timesteps
            kv_positions = torch.arange(
                total_kv_length, device=hidden.device, dtype=torch.long
            )
            query_positions = torch.arange(
                timesteps, device=hidden.device, dtype=torch.long
            ).unsqueeze(1)
            causal_mask = kv_positions.unsqueeze(0) <= (query_positions + carry_length)
            attn_scores = attn_scores.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, updated_value_carry)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch, timesteps, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)

        residual = hidden
        hidden = residual + self.residual_dropout(attn_output)

        ff_input = self.norm2(hidden)
        ff_hidden = self.linear1(ff_input)
        ff_hidden = self.ff_dropout(F.gelu(ff_hidden))
        ff_hidden = self.linear2(ff_hidden)

        hidden = hidden + self.residual_dropout(ff_hidden)
        return hidden, updated_key_carry, updated_value_carry


class TransformerDiscreteSequenceModel(AutoregressiveDiscreteSequenceModel):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_layers: int,
        max_position_embeddings: int,
        dropout: float = 0.0,
        positional_embedding: Literal["learned", "sinusoidal"] = "learned",
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must lie in [0, 1].")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if positional_embedding not in {"learned", "sinusoidal"}:
            raise ValueError("positional_embedding must be 'learned' or 'sinusoidal'.")

        self._vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = embedding_dim // num_heads
        self._positional_embedding_type = positional_embedding

        self.token_embedding = nn.Embedding(
            vocab_size + 1, embedding_dim
        )  # +1 for BOS token
        if self._positional_embedding_type == "learned":
            self.position_embedding = nn.Embedding(
                max_position_embeddings, embedding_dim
            )
        else:
            self.position_embedding = SinusoidalPositionalEmbedding(
                embedding_dim=embedding_dim,
                max_length=max_position_embeddings,
            )
        self.embedding_dropout = nn.Dropout(dropout)

        blocks: list[_AutoregressiveTransformerBlock] = []
        for _ in range(num_layers):
            blocks.append(
                _AutoregressiveTransformerBlock(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
            )

        self.layers = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.key_names = [f"key_{idx}" for idx in range(num_layers)]
        self.value_names = [f"value_{idx}" for idx in range(num_layers)]

    def init_carry(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        weight = self.token_embedding.weight
        carry: dict[str, torch.Tensor] = {
            "position": torch.zeros(batch_size, dtype=torch.long, device=device),
        }
        empty_key = weight.new_empty(batch_size, self.num_heads, 0, self.head_dim).to(
            device
        )
        empty_value = weight.new_empty(batch_size, self.num_heads, 0, self.head_dim).to(
            device
        )
        for key_name, value_name in zip(self.key_names, self.value_names):
            carry[key_name] = empty_key.clone()
            carry[value_name] = empty_value.clone()

        return carry

    def forward(
        self,
        x: torch.Tensor,
        carry: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 2:
            raise ValueError("Expected input tensor with shape (batch, timesteps).")

        batch, timesteps = x.size()
        device = x.device
        if "position" not in carry:
            raise KeyError("Carry must include a 'position' tensor.")

        positions = carry["position"]
        if positions.size(0) != batch:
            raise ValueError(
                "Position carry batch dimension does not match the provided tokens."
            )
        if positions.device != device:
            raise ValueError(
                "Position tensor must live on the same device as input tokens."
            )
        if torch.any(positions >= self.max_position_embeddings):
            raise ValueError(
                "Position index exceeds configured positional embedding range."
            )

        position_offsets = torch.arange(timesteps, device=device, dtype=positions.dtype)
        position_indices = positions.unsqueeze(1) + position_offsets
        if torch.any(position_indices >= self.max_position_embeddings):
            raise ValueError(
                "Position index exceeds configured positional embedding range."
            )

        hidden = self.token_embedding(x) + self.position_embedding(position_indices)
        hidden = self.embedding_dropout(hidden)

        updated_carry: dict[str, torch.Tensor] = {}

        for idx, layer in enumerate(self.layers):
            key_name = self.key_names[idx]
            value_name = self.value_names[idx]
            if key_name not in carry or value_name not in carry:
                raise KeyError(
                    "Transformer carry is missing key/value tensors for layer" f" {idx}."
                )
            key_carry = carry[key_name]
            value_carry = carry[value_name]
            if key_carry.size(0) != batch or key_carry.size(1) != self.num_heads:
                raise ValueError(
                    "Key carry shape is incompatible with the provided tokens."
                )
            if value_carry.size(0) != batch or value_carry.size(1) != self.num_heads:
                raise ValueError(
                    "Value carry shape is incompatible with the provided tokens."
                )
            if (key_carry.size(-1) != self.head_dim) or (
                value_carry.size(-1) != self.head_dim
            ):
                raise ValueError("Key/value carry head dimension mismatch detected.")
            if key_carry.device != device or value_carry.device != device:
                raise ValueError("Key/value carry tensors must share the input device.")
            hidden, updated_key_carry, updated_value_carry = layer(
                hidden, key_carry, value_carry
            )
            updated_carry[key_name] = updated_key_carry
            updated_carry[value_name] = updated_value_carry

        hidden = self.final_norm(hidden)
        logits = self.output_projection(hidden)

        updated_carry["position"] = positions + timesteps
        return logits, updated_carry

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


def sinusoidal_position_encoding(
    length: int,
    embedding_dim: int,
    base: float = 10000.0,
) -> Tensor:
    """Create 1D sinusoidal positional embeddings.

    Args:
        length: Number of positions to encode. Must be non-negative.
        embedding_dim: Dimensionality of each embedding. Must be positive.
        base: Exponential base used to compute the angular frequencies.

    Returns:
        A ``(length, embedding_dim)`` tensor of sinusoidal encodings.

    Raises:
        ValueError: If ``length`` is negative, ``embedding_dim`` is not positive,
            or ``base`` is not positive.
    """

    assert length >= 0, "length must be non-negative."
    assert embedding_dim > 0, "embedding_dim must be positive."
    assert base > 0, "base must be positive."

    if length == 0:
        return torch.empty(0, embedding_dim)

    positions = torch.arange(length).unsqueeze(1)
    div_input = torch.arange(0, embedding_dim, 2)
    div_term = torch.exp(div_input * (-math.log(base) / embedding_dim))
    embeddings = torch.zeros(length, embedding_dim)
    angles = positions * div_term
    embeddings[:, 0::2] = torch.sin(angles)

    if embedding_dim % 2 == 0:
        embeddings[:, 1::2] = torch.cos(angles)
    else:
        embeddings[:, 1::2] = torch.cos(angles)[:, : embedding_dim // 2]

    return embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for transformer-style models.

    The module caches a precomputed table of embeddings and extends it on demand.
    Forward accepts either a sequence length or explicit position indices.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_length: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert max_length >= 0, "max_length must be non-negative."
        assert embedding_dim > 0, "embedding_dim must be positive."
        assert base > 0, "base must be positive."

        self.embedding_dim = int(embedding_dim)
        self.base = float(base)

        pe = sinusoidal_position_encoding(max_length, self.embedding_dim, base=self.base)
        self._pe: Tensor
        self.register_buffer("_pe", pe)

    @property
    def pe(self) -> Tensor:
        """Return the cached positional embedding table."""
        return self._pe

    def forward(
        self,
        positions: Optional[Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tensor:
        """Look up positional embeddings.

        Args:
            positions: Optional tensor of position indices. Can have any shape,
                and the returned embeddings will append ``embedding_dim`` to that
                shape. Defaults to ``None``.
            seq_len: Optional sequence length. When provided, returns the first
                ``seq_len`` embeddings from the table.

        Returns:
            Tensor of positional embeddings on the same device/dtype as the
            cached table.

        Raises:
            ValueError: If both or neither of ``positions`` and ``seq_len`` are
                provided, or if indices exceed the cached range.
        """

        if (positions is None) == (seq_len is None):
            raise ValueError("Provide exactly one of positions or seq_len.")

        if positions is not None:
            flat_positions = positions.reshape(-1)
            gathered = self._pe.index_select(0, flat_positions)
            return gathered.view(
                positions.shape[0], positions.shape[1], self.embedding_dim
            )
        else:
            return self._pe[:seq_len]


class DiffusionPISTimeEncoding(nn.Module):
    """Time Encoding Module for DiffusionPISGradNet.

    See DiffusionPISGradNet for more details.
    """

    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
        )
        self.register_buffer(
            "pe", torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            t: torch.Tensor
        """
        t_sin = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).sin()  # type: ignore
        t_cos = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).cos()  # type: ignore
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class DiffusionPISStateEncoding(nn.Module):
    """State Encoding Module for DiffusionPISGradNet.

    See DiffusionPISGradNet for more details.
    """

    def __init__(self, x_dim: int, s_emb_dim: int) -> None:
        super().__init__()

        self.s_model = nn.Linear(x_dim, s_emb_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.s_model(s)


class DiffusionPISJointPolicy(nn.Module):
    """Joint Policy Module for DiffusionPISGradNet.

    See DiffusionPISGradNet for more details.
    """

    def __init__(
        self,
        s_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.GELU(),  # Because this model accepts embeddings (linear projections).
            nn.Linear(s_emb_dim, hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(1e-8)  # type: ignore
            self.model[-1].bias.data.fill_(0.0)  # type: ignore

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.model(s_emb + t_emb)


class DiffusionPISGradNetForward(nn.Module):  # TODO: support Learnable Backward policy
    """PISGradNet for diffusion sampling.

    This architecture was first introduced in Path Integral Sampler (PIS) (https://arxiv.org/abs/2111.15141)
    and adapted for GFlowNet-based training by Sendera et al. (https://arxiv.org/abs/2508.03044).

    Attributes:
        s_dim: The dimension of the states.
        harmonics_dim: The dimension of the Fourier features.
        t_emb_dim: The dimension of the time embedding.
        s_emb_dim: The dimension of the state embedding.
        hidden_dim: The dimension of the hidden layers.
        joint_layers: The number of layers in the joint policy.
        zero_init: Whether to initialize the weights and biases of the final layer to zero.
        out_dim: The dimension of the output.
        t_model: The time encoding module.
        s_model: The state encoding module.
        joint_model: The joint policy module.
    """

    def __init__(
        self,
        s_dim: int,  # dimension of states (== target.dim)
        harmonics_dim: int = 64,
        t_emb_dim: int = 64,
        s_emb_dim: int = 64,
        hidden_dim: int = 64,
        joint_layers: int = 2,
        zero_init: bool = False,
        # predict_flow: bool,  # TODO: support predict flow for db or subtb
        # share_embeddings: bool = False,
        # flow_harmonics_dim: int = 64,
        # flow_t_emb_dim: int = 64,
        # flow_s_emb_dim: int = 64,
        # flow_hidden_dim: int = 64,
        # flow_layers: int = 2,
        # lp: bool,  # TODO: support Langevin parameterization
        # lp_layers: int = 3,
        # lp_scaling_per_dimension: bool = True,
        # clipping: bool = False,  # TODO: support clipping
        # out_clip: float = 1e4,
        # lp_clip: float = 1e2,
        # learn_variance: bool = True,  # TODO: support learnable variance
        # log_var_range: float = 4.0,
    ):
        """Initialize the PISGradNetForward.

        Args:
            s_dim: The dimension of the states.
            harmonics_dim: The dimension of the Fourier features.
            t_emb_dim: The dimension of the time embedding.
            s_emb_dim: The dimension of the state embedding.
            hidden_dim: The dimension of the hidden layers.
            joint_layers: The number of layers in the joint policy.
            zero_init: Whether to initialize the weights and biases of the final layer to zero.
        """
        super().__init__()
        self.s_dim = s_dim
        self.input_dim = s_dim + 1  # + 1 for time, for the default IdentityPreprocessor
        self.harmonics_dim = harmonics_dim
        self.t_emb_dim = t_emb_dim
        self.s_emb_dim = s_emb_dim
        self.hidden_dim = hidden_dim
        self.joint_layers = joint_layers
        self.zero_init = zero_init
        self.out_dim = s_dim  # 2 * out_dim if learn_variance is True

        assert (
            self.s_emb_dim == self.t_emb_dim
        ), "Dimensionality of state embedding and time embedding should be the same!"

        self.t_model = DiffusionPISTimeEncoding(
            self.harmonics_dim, self.t_emb_dim, self.hidden_dim
        )
        self.s_model = DiffusionPISStateEncoding(self.s_dim, self.s_emb_dim)
        self.joint_model = DiffusionPISJointPolicy(
            self.s_emb_dim,
            self.hidden_dim,
            self.out_dim,
            self.joint_layers,
            self.zero_init,
        )

    def forward(
        self,
        preprocessed_states: torch.Tensor,
        # grad_logr_fn: Callable,  # TODO: grad_logr_fn for lp
    ) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            preprocessed_states: The preprocessed states (shape: (*batch_shape, s_dim + 1))

        Returns:
            The output of the module (shape: (*batch_shape, s_dim)).
        """
        s = preprocessed_states[..., :-1]
        t = preprocessed_states[..., -1]
        s_emb = self.s_model(s)
        t_emb = self.t_model(t)
        out = self.joint_model(s_emb, t_emb)

        # TODO: learn variance, lp, clipping, ...
        if torch.isnan(out).any():
            print("+ out has {} nans".format(torch.isnan(out).sum()))
            out = torch.nan_to_num(out)

        return out


class DiffusionFixedBackwardModule(nn.Module):
    """Fixed Backward Module for DiffusionPISGradNet.

    Attributes:
        input_dim: The dimension of the input.
    """

    def __init__(self, s_dim: int):
        """Initialize the FixedBackwardModule.

        Args:
            s_dim: The dimension of the states.
        """
        super().__init__()
        self.input_dim = s_dim + 1  # + 1 for time, for the default IdentityPreprocessor

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            preprocessed_states: The preprocessed states (shape: (*batch_shape, s_dim + 1))

        Returns:
            The output of the module (shape: (*batch_shape, s_dim)).
        """
        return torch.zeros_like(preprocessed_states[..., :-1])
