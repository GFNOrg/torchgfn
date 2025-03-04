"""Train a GFlowNet to generate ring graphs.

This example demonstrates training a GFlowNet to generate graphs that are rings - where each vertex
has exactly two neighbors and the edges form a single cycle containing all vertices. The environment
supports both directed and undirected ring generation.

Key components:
- RingGraphBuilding: Environment for building ring graphs
- RingPolicyModule: GNN-based policy network for predicting actions
- directed_reward/undirected_reward: Reward functions for validating ring structures
"""

import math
import time
from typing import Optional

from matplotlib import patches
import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, GCNConv, DirGNNConv

from gfn.actions import Actions, GraphActions, GraphActionType
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym import GraphBuilding
from gfn.modules import DiscretePolicyEstimator
from gfn.preprocessors import Preprocessor
from gfn.states import GraphStates
from gfn.utils.modules import MLP
from gfn.containers import ReplayBuffer


REW_VAL = 100.0
EPS_VAL = 1e-6


def directed_reward(states: GraphStates) -> torch.Tensor:
    """Compute reward for directed ring graphs.

    This function evaluates if a graph forms a valid directed ring (directed cycle).
    A valid directed ring must satisfy these conditions:
    1. Each node must have exactly one outgoing edge (row sum = 1 in adjacency matrix)
    2. Each node must have exactly one incoming edge (column sum = 1 in adjacency matrix)
    3. Following the edges must form a single cycle that includes all nodes

    The reward is binary:
    - REW_VAL (100.0) for valid directed rings
    - EPS_VAL (1e-6) for invalid structures

    Args:
        states: A batch of graph states to evaluate

    Returns:
        A tensor of rewards with the same batch shape as states
    """
    if states.tensor.edge_index.numel() == 0:
        return torch.full(states.batch_shape, EPS_VAL)

    out = torch.full((len(states),), EPS_VAL)  # Default reward.

    for i in range(len(states)):
        graph = states[i]
        adj_matrix = torch.zeros(graph.tensor.num_nodes, graph.tensor.num_nodes)
        adj_matrix[graph.tensor.edge_index[0], graph.tensor.edge_index[1]] = 1

        # Check if each node has exactly one outgoing edge (row sum = 1)
        if not torch.all(adj_matrix.sum(dim=1) == 1):
            continue

        # Check that each node has exactly one incoming edge (column sum = 1)
        if not torch.all(adj_matrix.sum(dim=0) == 1):
            continue

        # Starting from node 0, follow edges and see if we visit all nodes
        # and return to the start
        visited = []
        current = 0  # Start from node 0

        while current not in visited:
            visited.append(current)

            # Get the outgoing neighbor
            current = torch.where(adj_matrix[current] == 1)[0].item()

            # If we've visited all nodes and returned to 0, it's a valid ring
            if len(visited) == graph.tensor.num_nodes and current == 0:
                out[i] = REW_VAL
                break

    return out.view(*states.batch_shape)


def undirected_reward(states: GraphStates) -> torch.Tensor:
    """Compute reward for undirected ring graphs.

    This function evaluates if a graph forms a valid undirected ring (cycle).
    A valid undirected ring must satisfy these conditions:
    1. Each node must have exactly two neighbors (degree = 2)
    2. The graph must form a single connected cycle including all nodes

    The reward is binary:
    - REW_VAL (100.0) for valid undirected rings
    - EPS_VAL (1e-6) for invalid structures

    The algorithm:
    1. Checks that all nodes have degree 2
    2. Performs a traversal starting from node 0, following edges
    3. Checks if the traversal visits all nodes and returns to start

    Args:
        states: A batch of graph states to evaluate

    Returns:
        A tensor of rewards with the same batch shape as states
    """
    if states.tensor.edge_index.numel() == 0:
        return torch.full(states.batch_shape, EPS_VAL)

    out = torch.full((len(states),), EPS_VAL)  # Default reward.

    for i in range(len(states)):
        graph = states[i]
        if graph.tensor.num_nodes == 0:
            continue
        adj_matrix = torch.zeros(graph.tensor.num_nodes, graph.tensor.num_nodes)
        adj_matrix[graph.tensor.edge_index[0], graph.tensor.edge_index[1]] = 1
        adj_matrix[graph.tensor.edge_index[1], graph.tensor.edge_index[0]] = 1

        # In an undirected ring, every vertex should have degree 2.
        if not torch.all(adj_matrix.sum(dim=1) == 2):
            continue

        # Traverse the cycle starting from vertex 0.
        start_vertex = 0
        visited = [start_vertex]
        neighbors = torch.where(adj_matrix[start_vertex] == 1)[0]
        if neighbors.numel() == 0:
            continue
        # Arbitrarily choose one neighbor to begin the traversal.
        current = neighbors[0].item()
        prev = start_vertex

        while True:
            if current == start_vertex:
                break
            visited.append(current)
            current_neighbors = torch.where(adj_matrix[current] == 1)[0]
            # Exclude the neighbor we just came from.
            current_neighbors_list = [n.item() for n in current_neighbors]
            possible = [n for n in current_neighbors_list if n != prev]
            if len(possible) != 1:
                break
            next_node = possible[0]
            prev, current = current, next_node

        if current == start_vertex and len(visited) == graph.tensor.num_nodes:
            out[i] = REW_VAL

    return out.view(*states.batch_shape)


class RingPolicyModule(nn.Module):
    """Simple module which outputs a fixed logits for the actions, depending on the number of edges.

    Args:
        n_nodes: The number of nodes in the graph.
    """

    def __init__(
        self,
        n_nodes: int,
        directed: bool,
        num_conv_layers: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        super().__init__()
        self.hidden_dim = self.embedding_dim = embedding_dim
        self.is_directed = directed
        self.is_backward = is_backward
        self.n_nodes = n_nodes
        self.num_conv_layers = num_conv_layers

        # Node embedding layer.
        self.embedding = nn.Embedding(n_nodes, self.embedding_dim)
        self.conv_blks = nn.ModuleList()
        self.exit_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=1,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

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
                                        self.hidden_dim // 2, self.hidden_dim // 2
                                    ),
                                    nn.ReLU(),
                                    nn.Linear(
                                        self.hidden_dim // 2, self.hidden_dim // 2
                                    ),
                                )
                                for _ in range(
                                    2
                                )  # One for in-features, one for out-features.
                            ]
                        ),
                    ]
                )
        else:  # Undirected case.
            for _ in range(num_conv_layers):
                self.conv_blks.extend(
                    [
                        GINConv(
                            MLP(
                                input_dim=self.embedding_dim,
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

    def _group_mean(
        self, tensor: torch.Tensor, batch_ptr: torch.Tensor
    ) -> torch.Tensor:
        cumsum = torch.zeros(
            (len(tensor) + 1, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        cumsum[1:] = torch.cumsum(tensor, dim=0)

        # Subtract the end val from each batch idx fom the start val of each batch idx.
        size = batch_ptr[1:] - batch_ptr[:-1]
        return (cumsum[batch_ptr[1:]] - cumsum[batch_ptr[:-1]]) / size[:, None]

    def forward(self, states_tensor: Batch) -> torch.Tensor:
        node_features, batch_ptr = (
            states_tensor.x,
            states_tensor.ptr,
        )
        batch_size = int(torch.prod(states_tensor.batch_shape))

        # Multiple action type convolutions with residual connections.
        x = self.embedding(node_features.squeeze().int())
        for i in range(0, len(self.conv_blks), 2):
            x_new = self.conv_blks[i](x, states_tensor.edge_index)  # GIN/GCN conv.
            if self.is_directed:
                x_in, x_out = torch.chunk(x_new, 2, dim=-1)
                # Process each component separately through its own MLP
                x_in = self.conv_blks[i + 1][0](x_in)  # First MLP in ModuleList.
                x_out = self.conv_blks[i + 1][1](x_out)  # Second MLP in ModuleList.
                x_new = torch.cat([x_in, x_out], dim=-1)
            else:
                x_new = self.conv_blks[i + 1](x_new)  # Linear -> ReLU -> Linear.

            x = x_new + x if i > 0 else x_new  # Residual connection.
            x = self.norm(x)  # Layernorm.

        # This MLP computes the exit action.
        node_feature_means = self._group_mean(x, batch_ptr)
        exit_action = self.exit_mlp(node_feature_means)

        x = x.reshape(*states_tensor.batch_shape, self.n_nodes, self.hidden_dim)

        # Undirected.
        if self.is_directed:
            feature_dim = self.hidden_dim // 2
            source_features = x[..., :feature_dim]
            target_features = x[..., feature_dim:]

            # Dot product between source and target features (asymmetric).
            edgewise_dot_prod = torch.einsum(
                "bnf,bmf->bnm", source_features, target_features
            )
            edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(
                torch.tensor(feature_dim)
            )

            i_up, j_up = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)
            i_lo, j_lo = torch.tril_indices(self.n_nodes, self.n_nodes, offset=-1)

            # Combine them.
            i0 = torch.cat([i_up, i_lo])
            i1 = torch.cat([j_up, j_lo])
            out_size = self.n_nodes**2 - self.n_nodes

        else:
            # Dot product between all node features (symmetric).
            edgewise_dot_prod = torch.einsum("bnf,bmf->bnm", x, x)
            edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(
                torch.tensor(self.hidden_dim)
            )
            i0, i1 = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)
            out_size = (self.n_nodes**2 - self.n_nodes) // 2

        # Grab the needed elems from the adjacency matrix and reshape.
        edge_actions = edgewise_dot_prod[
            torch.arange(batch_size)[:, None, None], i0, i1
        ]
        edge_actions = edge_actions.reshape(*states_tensor["batch_shape"], out_size)

        if self.is_backward:
            return edge_actions
        else:
            return torch.cat([edge_actions, exit_action], dim=-1)


class RingGraphBuilding(GraphBuilding):
    """Environment for building ring graphs with discrete action space.

    This environment is specialized for creating ring graphs where each node has
    exactly two neighbors and the edges form a single cycle. The environment supports
    both directed and undirected graphs.

    In each state, the policy can:
    1. Add an edge between existing nodes
    2. Use the exit action to terminate graph building

    The action space is discrete, with size:
    - For directed graphs: n_nodes^2 - n_nodes + 1 (all possible directed edges + exit)
    - For undirected graphs: (n_nodes^2 - n_nodes)/2 + 1 (upper triangle + exit)

    Args:
        n_nodes: The number of nodes in the graph.
        state_evaluator: A function that evaluates a state and returns a reward.
        directed: Whether the graph should be directed.
    """

    def __init__(self, n_nodes: int, state_evaluator: callable, directed: bool):
        self.n_nodes = n_nodes
        if directed:
            # all off-diagonal edges + exit.
            self.n_actions = (n_nodes**2 - n_nodes) + 1
        else:
            # bottom triangle + exit.
            self.n_actions = ((n_nodes**2 - n_nodes) // 2) + 1
        super().__init__(feature_dim=n_nodes, state_evaluator=state_evaluator)
        self.is_discrete = True  # actions here are discrete, needed for FlowMatching
        self.is_directed = directed

    def make_actions_class(self) -> type[Actions]:
        env = self

        class RingActions(Actions):
            """Actions for building ring graphs.

            Actions are represented as discrete indices where:
            - 0 to n_actions-2: Adding an edge between specific nodes
            - n_actions-1: EXIT action to terminate the trajectory
            - n_actions: DUMMY action (used for padding)
            """

            action_shape = (1,)
            dummy_action = torch.tensor([env.n_actions])
            exit_action = torch.tensor([env.n_actions - 1])

        return RingActions

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class RingStates(GraphStates):
            """Represents the state of a ring graph building process.

            This class extends GraphStates to specifically handle ring graph states.
            Each state represents a graph with a fixed number of nodes where edges
            are being added incrementally to form a ring structure.

            The state representation consists of:
            - node_feature: Node IDs for each node in the graph (shape: [n_nodes, 1])
            - edge_feature: Features for each edge (shape: [n_edges, 1])
            - edge_index: Indices representing the source and target nodes for each edge (shape: [n_edges, 2])

            Special states:
            - s0: Initial state with n_nodes and no edges
            - sf: Terminal state (used as a placeholder)

            The class provides masks for both forward and backward actions to determine
            which actions are valid from the current state.
            """
            s0 = Data(
                x=torch.arange(env.n_nodes)[:, None],
                edge_attr=torch.ones((0, 1)),
                edge_index=torch.ones((2, 0), dtype=torch.long),
            )
            sf = Data(
                x=-torch.ones(env.n_nodes)[:, None],
                edge_attr=torch.zeros((0, 1)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )

            def __init__(self, tensor: Batch):
                self.tensor = tensor
                self.node_features_dim = tensor.x.shape[-1]
                self.edge_features_dim = tensor.edge_attr.shape[-1]
                self._log_rewards: Optional[float] = None

                self.n_nodes = env.n_nodes
                self.n_actions = env.n_actions

            @property
            def forward_masks(self):
                """Compute masks for valid forward actions from the current state.

                A forward action is valid if:
                1. The edge doesn't already exist in the graph
                2. The edge connects two distinct nodes

                For directed graphs, all possible src->dst edges are considered.
                For undirected graphs, only the upper triangular portion of the adjacency matrix is used.

                The last action is always the EXIT action, which is always valid.

                Returns:
                    Tensor: Boolean mask of shape [batch_size, n_actions] where True indicates valid actions
                """
                # Allow all actions.
                forward_masks = torch.ones(len(self), self.n_actions, dtype=torch.bool)

                if env.is_directed:
                    i_up, j_up = torch.triu_indices(
                        self.n_nodes, self.n_nodes, offset=1
                    )  # Upper triangle.
                    i_lo, j_lo = torch.tril_indices(
                        self.n_nodes, self.n_nodes, offset=-1
                    )  # Lower triangle.

                    # Combine them
                    ei0 = torch.cat([i_up, i_lo])
                    ei1 = torch.cat([j_up, j_lo])
                else:
                    ei0, ei1 = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)

                # Remove existing edges.
                for i in range(len(self)):
                    existing_edges = self[i].tensor.edge_index
                    assert torch.all(existing_edges >= 0)  # TODO: convert to test.

                    if existing_edges.numel() == 0:
                        edge_idx = torch.zeros(0, dtype=torch.bool)
                    else:
                        edge_idx = torch.logical_and(
                            existing_edges[0][..., None] == ei0[None],
                            existing_edges[1][..., None] == ei1[None],
                        )

                        # Collapse across the edge dimension.
                        if len(edge_idx.shape) == 2:
                            edge_idx = edge_idx.sum(0).bool()

                        # Adds an unmasked exit action.
                        edge_idx = torch.cat((edge_idx, torch.BoolTensor([False])))
                        forward_masks[i, edge_idx] = (
                            False  # Disallow the addition of this edge.
                        )

                return forward_masks.view(*self.batch_shape, self.n_actions)

            @forward_masks.setter
            def forward_masks(self, value: torch.Tensor):
                pass  # fwd masks is computed on the fly

            @property
            def backward_masks(self):
                """Compute masks for valid backward actions from the current state.

                A backward action is valid if:
                1. The edge exists in the current graph (i.e., can be removed)

                For directed graphs, all existing edges are considered for removal.
                For undirected graphs, only the upper triangular edges are considered.

                The EXIT action is not included in backward masks.

                Returns:
                    Tensor: Boolean mask of shape [batch_size, n_actions-1] where True indicates valid actions
                """
                # Disallow all actions.
                backward_masks = torch.zeros(
                    len(self), self.n_actions - 1, dtype=torch.bool
                )

                for i in range(len(self)):
                    existing_edges = self[i].tensor.edge_index

                    if env.is_directed:
                        i_up, j_up = torch.triu_indices(
                            self.n_nodes, self.n_nodes, offset=1
                        )  # Upper triangle.
                        i_lo, j_lo = torch.tril_indices(
                            self.n_nodes, self.n_nodes, offset=-1
                        )  # Lower triangle.

                        # Combine them
                        ei0 = torch.cat([i_up, i_lo])
                        ei1 = torch.cat([j_up, j_lo])
                    else:
                        ei0, ei1 = torch.triu_indices(
                            self.n_nodes, self.n_nodes, offset=1
                        )

                    if len(existing_edges) == 0:
                        edge_idx = torch.zeros(0, dtype=torch.bool)
                    else:
                        edge_idx = torch.logical_and(
                            existing_edges[0][..., None] == ei0[None],
                            existing_edges[1][..., None] == ei1[None],
                        )
                        # Collapse across the edge dimension.
                        if len(edge_idx.shape) == 2:
                            edge_idx = edge_idx.sum(0).bool()

                        backward_masks[i, edge_idx] = (
                            True  # Allow the removal of this edge.
                        )

                return backward_masks.view(*self.batch_shape, self.n_actions - 1)

            @backward_masks.setter
            def backward_masks(self, value: torch.Tensor):
                pass  # bwd masks is computed on the fly

        return RingStates

    def _step(self, states: GraphStates, actions: Actions) -> GraphStates:
        """Take a step in the environment by applying actions to states.

        Args:
            states: Current states batch
            actions: Actions to apply

        Returns:
            New states after applying the actions
        """
        actions = self.convert_actions(states, actions)
        new_states = super()._step(states, actions)
        assert isinstance(new_states, GraphStates)
        return new_states

    def _backward_step(self, states: GraphStates, actions: Actions) -> GraphStates:
        """Take a backward step in the environment.

        Args:
            states: Current states batch
            actions: Actions to apply in reverse

        Returns:
            New states after applying the backward actions
        """
        actions = self.convert_actions(states, actions)
        new_states = super()._backward_step(states, actions)
        assert isinstance(new_states, GraphStates)
        return new_states

    def convert_actions(self, states: GraphStates, actions: Actions) -> GraphActions:
        """Converts the action from discrete space to graph action space.

        This method maps discrete action indices to specific graph operations:
        - GraphActionType.ADD_EDGE: Add an edge between specific nodes
        - GraphActionType.EXIT: Terminate trajectory
        - GraphActionType.DUMMY: No-op action (for padding)

        Args:
            states: Current states batch
            actions: Discrete actions to convert

        Returns:
            Equivalent actions in the GraphActions format
        """
        # TODO: factor out into utility function.
        action_tensor = actions.tensor.squeeze(-1).clone()
        action_type = torch.where(
            action_tensor == self.n_actions - 1,
            GraphActionType.EXIT,
            GraphActionType.ADD_EDGE,
        )
        action_type[action_tensor == self.n_actions] = GraphActionType.DUMMY

        # Convert action indices to source-target node pairs
        # TODO: factor out into utility function.
        if self.is_directed:
            i_up, j_up = torch.triu_indices(
                self.n_nodes, self.n_nodes, offset=1
            )  # Upper triangle.
            i_lo, j_lo = torch.tril_indices(
                self.n_nodes, self.n_nodes, offset=-1
            )  # Lower triangle.

            # Combine them
            ei0 = torch.cat([i_up, i_lo])
            ei1 = torch.cat([j_up, j_lo])

        else:
            ei0, ei1 = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)

        # Adds -1 "edge" representing exit, -2 "edge" representing dummy.
        ei0 = torch.cat((ei0, torch.IntTensor([-1, -2])), dim=0)
        ei1 = torch.cat((ei1, torch.IntTensor([-1, -2])), dim=0)

        # Indexes either the second last element (exit) or la
        # action_tensor[action_tensor >= (self.n_actions - 1)] = 0
        ei0, ei1 = ei0[action_tensor], ei1[action_tensor]

        out = GraphActions(
            TensorDict(
                {
                    "action_type": action_type,
                    "features": torch.ones(action_tensor.shape + (1,)),
                    "edge_index": torch.stack([ei0, ei1], dim=-1),
                },
                batch_size=action_tensor.shape,
            )
        )
        return out


class GraphPreprocessor(Preprocessor):
    """Preprocessor for graph states to extract the tensor representation.

    This simple preprocessor extracts the torch_geometric Batch from GraphStates to make
    it compatible with the policy networks. It doesn't perform any complex
    transformations, just ensuring the tensors are accessible in the right format.

    Args:
        feature_dim: The dimension of features in the graph (default: 1)
    """

    def __init__(self, feature_dim: int = 1):
        super().__init__(output_dim=feature_dim)

    def preprocess(self, states: GraphStates) -> Batch:
        return states.tensor

    def __call__(self, states: GraphStates) -> Batch:
        return self.preprocess(states)


def render_states(states: GraphStates, state_evaluator: callable, directed: bool):
    """Visualize a batch of graph states as ring structures.

    This function creates a matplotlib visualization of graph states, rendering them
    as circular layouts with nodes positioned evenly around a circle. For directed
    graphs, edges are shown as arrows; for undirected graphs, edges are shown as lines.

    The visualization includes:
    - Circular positioning of nodes
    - Drawing edges between connected nodes
    - Displaying the reward value for each graph

    Args:
        states: A batch of graphs to visualize
        state_evaluator: Function to compute rewards for each graph
        directed: Whether to render directed or undirected edges
    """
    rewards = state_evaluator(states)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    for i in range(8):
        current_ax = ax[i // 4, i % 4]
        state = states[i]
        n_circles = state.tensor.num_nodes
        radius = 5
        xs, ys = [], []
        for j in range(n_circles):
            angle = 2 * math.pi * j / n_circles
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            xs.append(x)
            ys.append(y)
            current_ax.add_patch(
                patches.Circle((x, y), 0.5, facecolor="none", edgecolor="black")
            )

        edge_index = states[i].tensor.edge_index

        for edge in edge_index.T:
            start_x, start_y = xs[edge[0]], ys[edge[0]]
            end_x, end_y = xs[edge[1]], ys[edge[1]]
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx**2 + dy**2)
            dx, dy = dx / length, dy / length

            circle_radius = 0.5
            head_thickness = 0.2

            start_x += dx * circle_radius
            start_y += dy * circle_radius
            if directed:
                end_x -= dx * circle_radius
                end_y -= dy * circle_radius
                current_ax.arrow(
                    start_x,
                    start_y,
                    end_x - start_x,
                    end_y - start_y,
                    head_width=head_thickness,
                    head_length=head_thickness,
                    fc="black",
                    ec="black",
                )

            else:
                end_x -= dx * (circle_radius + head_thickness)
                end_y -= dy * (circle_radius + head_thickness)
                current_ax.plot([start_x, end_x], [start_y, end_y], color="black")

        current_ax.set_title(f"State {i}, $r={rewards[i]:.2f}$")
        current_ax.set_xlim(-(radius + 1), radius + 1)
        current_ax.set_ylim(-(radius + 1), radius + 1)
        current_ax.set_aspect("equal")
        current_ax.set_xticks([])
        current_ax.set_yticks([])

    plt.show()


class AdjacencyPolicyModule(nn.Module):
    """Policy network that processes flattened adjacency matrices to predict graph actions.

    Unlike the GNN-based RingPolicyModule, this module uses standard MLPs to process
    the entire adjacency matrix as a flattened vector. This approach:

    1. Can directly process global graph structure without message passing
    2. May be more effective for small graphs where global patterns are important
    3. Does not require complex graph neural network operations

    The module architecture consists of:
    - An MLP to process the flattened adjacency matrix into an embedding
    - An edge MLP that predicts logits for each possible edge action
    - An exit MLP that predicts a logit for the exit action

    Args:
        n_nodes: Number of nodes in the graph
        directed: Whether the graph is directed or undirected
        embedding_dim: Dimension of internal embeddings (default: 128)
        is_backward: Whether this is a backward policy (default: False)
    """

    def __init__(
        self,
        n_nodes: int,
        directed: bool,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.is_directed = directed
        self.is_backward = is_backward
        self.hidden_dim = embedding_dim

        # MLP for processing the flattened adjacency matrix
        self.mlp = MLP(
            input_dim=n_nodes * n_nodes,  # Flattened adjacency matrix
            output_dim=embedding_dim,
            hidden_dim=embedding_dim,
            n_hidden_layers=2,
            add_layer_norm=True,
        )

        # Exit action MLP
        self.exit_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=1,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

        # Edge prediction MLP
        if directed:
            # For directed graphs: all off-diagonal elements
            out_size = n_nodes**2 - n_nodes
        else:
            # For undirected graphs: upper triangle without diagonal
            out_size = (n_nodes**2 - n_nodes) // 2

        self.edge_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=out_size,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

    def forward(self, states_tensor: Batch) -> torch.Tensor:
        """Forward pass to compute action logits from graph states.

        Process:
        1. Convert the graph representation to adjacency matrices
        2. Process the flattened adjacency matrices through the main MLP
        3. Predict logits for edge actions and exit action

        Args:
            states_tensor: A torch_geometric Batch containing graph state information

        Returns:
            A tensor of logits for all possible actions
        """
        # Convert the graph to adjacency matrix
        batch_size = int(states_tensor.batch_size)
        adj_matrices = torch.zeros(
            (batch_size, self.n_nodes, self.n_nodes),
            device=states_tensor.x.device,
        )

        # Fill the adjacency matrices from edge indices
        if states_tensor.edge_index.numel() > 0:
            for i in range(batch_size):
                eis = states_tensor[i].edge_index
                adj_matrices[i, eis[0], eis[1]] = 1

        # Flatten the adjacency matrices for the MLP
        adj_matrices_flat = adj_matrices.view(batch_size, -1)

        # Process with MLP
        embedding = self.mlp(adj_matrices_flat)

        # Generate edge and exit actions
        edge_actions = self.edge_mlp(embedding)
        exit_action = self.exit_mlp(embedding)

        if self.is_backward:
            return edge_actions
        else:
            return torch.cat([edge_actions, exit_action], dim=-1)


if __name__ == "__main__":
    """
    Main execution for training a GFlowNet to generate ring graphs.

    This script demonstrates the complete workflow of training a GFlowNet
    to generate valid ring structures in both directed and undirected settings.

    Configurable parameters:
    - N_NODES: Number of nodes in the graph (default: 5)
    - N_ITERATIONS: Number of training iterations (default: 1000)
    - LR: Learning rate for optimizer (default: 0.001)
    - BATCH_SIZE: Batch size for training (default: 128)
    - DIRECTED: Whether to generate directed rings (default: True)
    - USE_BUFFER: Whether to use a replay buffer (default: False)
    - USE_GNN: Whether to use GNN-based policy (True) or MLP-based policy (False)

    The script performs the following steps:
    1. Initialize the environment and policy networks
    2. Train the GFlowNet using trajectory balance
    3. Visualize sample generated graphs
    """
    N_NODES = 4
    N_ITERATIONS = 200
    LR = 0.001
    BATCH_SIZE = 1024
    DIRECTED = True
    USE_BUFFER = False
    USE_GNN = True  # Set to False to use MLP with adjacency matrices instead of GNN

    state_evaluator = undirected_reward if not DIRECTED else directed_reward
    torch.random.manual_seed(7)
    env = RingGraphBuilding(
        n_nodes=N_NODES, state_evaluator=state_evaluator, directed=DIRECTED
    )

    # Choose model type based on USE_GNN flag
    if USE_GNN:
        module_pf = RingPolicyModule(env.n_nodes, DIRECTED)
        module_pb = RingPolicyModule(env.n_nodes, DIRECTED, is_backward=True)
    else:
        module_pf = AdjacencyPolicyModule(env.n_nodes, DIRECTED)
        module_pb = AdjacencyPolicyModule(env.n_nodes, DIRECTED, is_backward=True)

    pf = DiscretePolicyEstimator(
        module=module_pf, n_actions=env.n_actions, preprocessor=GraphPreprocessor()
    )
    pb = DiscretePolicyEstimator(
        module=module_pb,
        n_actions=env.n_actions,
        preprocessor=GraphPreprocessor(),
        is_backward=True,
    )
    gflownet = TBGFlowNet(pf, pb)
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    replay_buffer = ReplayBuffer(
        env,
        objects_type="trajectories",
        capacity=BATCH_SIZE,
        prioritized=True,
    )

    losses = []

    t1 = time.time()
    for iteration in range(N_ITERATIONS):
        trajectories = gflownet.sample_trajectories(
            env,
            n=BATCH_SIZE,
            save_logprobs=True,  # pyright: ignore
            epsilon=0.2 * (1 - iteration / N_ITERATIONS),
        )
        training_samples = gflownet.to_training_samples(trajectories)

        # Collect rewards for reporting.
        if isinstance(training_samples, tuple):
            last_states = training_samples[1]
        else:
            last_states = training_samples.last_states
        assert isinstance(last_states, GraphStates)
        rewards = state_evaluator(last_states)

        if USE_BUFFER:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                if iteration > 20:
                    training_samples = training_samples[: BATCH_SIZE // 2]
                    buffer_samples = replay_buffer.sample(
                        n_trajectories=BATCH_SIZE // 2
                    )
                    training_samples.extend(buffer_samples)

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples)  # pyright: ignore
        pct_rings = torch.mean(rewards > 0.1, dtype=torch.float) * 100
        print(
            "Iteration {} - Loss: {:.02f}, rings: {:.0f}%".format(
                iteration, loss.item(), pct_rings
            )
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    t2 = time.time()
    print("Time:", t2 - t1)

    # This comes from the gflownet, not the buffer.
    last_states = trajectories.last_states[:8]
    assert isinstance(last_states, GraphStates)
    render_states(last_states, state_evaluator, DIRECTED)
