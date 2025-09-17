"""This file contains some examples of modules that can be used with GFN."""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
from tensordict import TensorDict
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

        # Edge-index logits via pairwise dot products
        # Pad to max_nodes across batch for gathering candidate edges
        lengths = (states_tensor.ptr[1:] - states_tensor.ptr[:-1]).to(torch.long)
        max_nodes = int(lengths.max().item())

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
        if self.is_backward:
            action_type[..., GraphActionType.ADD_EDGE] = 0.0
        else:
            exit_action = self.exit_mlp(embedding).squeeze(-1)
            action_type[..., GraphActionType.ADD_EDGE] = 0.0
            action_type[..., GraphActionType.EXIT] = exit_action
            edge_class_logits = self.edge_class_mlp(embedding)

        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.NODE_CLASS_KEY: torch.zeros(
                    len(states_tensor), 1, device=device
                ),
                GraphActions.EDGE_CLASS_KEY: edge_class_logits,
                GraphActions.EDGE_INDEX_KEY: edge_actions,
            },
            batch_size=len(states_tensor),
        )


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
            A TensorDict containing logits for each action component, with all values set to 1 to represent a uniform distribution:
            - GraphActions.ACTION_TYPE_KEY: Tensor of shape [*batch_shape, 3] for the 3 possible action types
            - GraphActions.EDGE_CLASS_KEY: Tensor of shape [*batch_shape, num_edge_classes] for edge class logits
            - GraphActions.NODE_CLASS_KEY: Tensor of shape [*batch_shape, num_node_classes] for node class logits
            - GraphActions.EDGE_INDEX_KEY: Tensor of shape [*batch_shape, edges_dim] for edge index logits
        """
        device = states_tensor.x.device
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
                GraphActions.EDGE_INDEX_KEY: torch.ones(
                    len(states_tensor), self.edges_dim, device=device
                ),
            },
            batch_size=len(states_tensor),
        )


class NoisyLinear(nn.Linear):
    """Noisy Linear Layer.

    Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
    be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
    with gradient descent along with any other remaining network weights. Factorized Gaussian
    noise is the type of noise usually employed.

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
