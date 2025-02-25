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
    """Compute the reward of a graph.

    Specifically, the reward is 1 if the graph is a ring, 1e-6 otherwise.

    Args:
        states: A batch of graphs.

    Returns:
        A tensor of rewards.
    """
    if states.tensor["edge_index"].shape[0] == 0:
        return torch.full(states.batch_shape, EPS_VAL)

    out = torch.full((len(states),), EPS_VAL)  # Default reward.

    for i in range(len(states)):
        start, end = states.tensor["batch_ptr"][i], states.tensor["batch_ptr"][i + 1]
        nodes_index_range = states.tensor["node_index"][start:end]
        edge_index_mask = torch.all(
            states.tensor["edge_index"] >= nodes_index_range[0], dim=-1
        ) & torch.all(states.tensor["edge_index"] <= nodes_index_range[-1], dim=-1)
        masked_edge_index = (
            states.tensor["edge_index"][edge_index_mask] - nodes_index_range[0]
        )

        n_nodes = nodes_index_range.shape[0]
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        adj_matrix[masked_edge_index[:, 0], masked_edge_index[:, 1]] = 1

        if not torch.all(adj_matrix.sum(dim=1) == 1):
            continue

        visited, current = [], 0
        while current not in visited:
            visited.append(current)

            def set_diff(tensor1, tensor2):
                mask = ~torch.isin(tensor1, tensor2)
                return tensor1[mask]

            # Find an unvisited neighbor
            neighbors = torch.where(adj_matrix[current] == 1)[0]
            valid_neighbours = set_diff(neighbors, torch.tensor(visited))

            # Visit the first valid neighbor.
            if len(valid_neighbours) == 1:
                current = valid_neighbours[0]
            elif len(valid_neighbours) == 0:
                break
            else:
                break  # TODO: This actually should never happen, should be caught on line 45.

        # Check if we visited all vertices and the last vertex connects back to start.
        if len(visited) == n_nodes and adj_matrix[current][0] == 1:
            out[i] = REW_VAL

    return out.view(*states.batch_shape)


def undirected_reward(states: GraphStates) -> torch.Tensor:
    """Compute the reward of a graph.

    Specifically, the reward is 1 if the graph is an undirected ring, 1e-6 otherwise.

    Args:
        states: A batch of graphs.

    Returns:
        A tensor of rewards.
    """
    if states.tensor["edge_index"].shape[0] == 0:
        return torch.full(states.batch_shape, EPS_VAL)

    out = torch.full((len(states),), EPS_VAL)  # Default reward.

    for i in range(len(states)):
        start, end = states.tensor["batch_ptr"][i], states.tensor["batch_ptr"][i + 1]
        nodes_index_range = states.tensor["node_index"][start:end]
        edge_index_mask = torch.all(
            states.tensor["edge_index"] >= nodes_index_range[0], dim=-1
        ) & torch.all(states.tensor["edge_index"] <= nodes_index_range[-1], dim=-1)
        masked_edge_index = (
            states.tensor["edge_index"][edge_index_mask] - nodes_index_range[0]
        )

        n_nodes = nodes_index_range.shape[0]
        if n_nodes == 0:
            continue

        # Construct a symmetric adjacency matrix for the undirected graph.
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        if masked_edge_index.shape[0] > 0:
            adj_matrix[masked_edge_index[:, 0], masked_edge_index[:, 1]] = 1
            adj_matrix[masked_edge_index[:, 1], masked_edge_index[:, 0]] = 1

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

        if current == start_vertex and len(visited) == n_nodes:
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
                        nn.ModuleList([
                            nn.Sequential(
                                nn.Linear(self.hidden_dim//2, self.hidden_dim//2),
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim//2, self.hidden_dim//2)
                            ) for _ in range(2)  # One for in-features, one for out-features.
                        ])
                    ])
        else:  # Undirected case.
            for _ in range(num_conv_layers):
                self.conv_blks.extend(
                    [
                        GINConv(
                            MLP(
                                input_dim=self.embedding_dim if i == 0 else self.hidden_dim,
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

    def forward(self, states_tensor: TensorDict) -> torch.Tensor:
        node_features, batch_ptr = (
            states_tensor["node_feature"],
            states_tensor["batch_ptr"],
        )
        batch_size = int(torch.prod(states_tensor["batch_shape"]))

        edge_index = torch.where(
            states_tensor["edge_index"][..., None] == states_tensor["node_index"]
        )[2].reshape(
            states_tensor["edge_index"].shape
        )  # (M, 2)

        # Multiple action type convolutions with residual connections.
        x = self.embedding(node_features.squeeze().int())
        for i in range(0, len(self.conv_blks), 2):
            x_new = self.conv_blks[i](x, edge_index.T)  # GIN/GCN conv.
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

        x = x.reshape(*states_tensor["batch_shape"], self.n_nodes, self.hidden_dim)

        # Undirected.
        if self.is_directed:
            feature_dim = self.hidden_dim // 2
            source_features = x[..., :feature_dim]
            target_features = x[..., feature_dim:]

            # Dot product between source and target features (asymmetric).
            edgewise_dot_prod = torch.einsum("bnf,bmf->bnm", source_features, target_features)
            edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(torch.tensor(feature_dim))

            i_up, j_up = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)
            i_lo, j_lo = torch.tril_indices(self.n_nodes, self.n_nodes, offset=-1)

            # Combine them.
            i0 = torch.cat([i_up, i_lo])
            i1 = torch.cat([j_up, j_lo])
            out_size = self.n_nodes**2 - self.n_nodes

        else:
            # Dot product between all node features (symmetric).
            edgewise_dot_prod = torch.einsum("bnf,bmf->bnm", x, x)
            edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(torch.tensor(self.hidden_dim))
            i0, i1 = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)
            out_size = (self.n_nodes**2 - self.n_nodes) // 2

        # Grab the needed elems from the adjacency matrix and reshape.
        edge_actions = x[torch.arange(batch_size)[:, None, None], i0, i1]
        edge_actions = edge_actions.reshape(*states_tensor["batch_shape"], out_size)

        if self.is_backward:
            return edge_actions
        else:
            return torch.cat([edge_actions, exit_action], dim=-1)


class RingGraphBuilding(GraphBuilding):
    """Override the GraphBuilding class to create have discrete actions.

    Specifically, at initialization, we have n nodes.
    The policy can only add edges between existing nodes or use the exit action.
    The action space is thus discrete and of size n^2 + 1, where the last action is the exit action,
    and the first n^2 actions are the possible edges.

    Args:
        n_nodes: The number of nodes in the graph.
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
            action_shape = (1,)
            dummy_action = torch.tensor([env.n_actions])
            exit_action = torch.tensor([env.n_actions - 1])

        return RingActions

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class RingStates(GraphStates):
            s0 = TensorDict(
                {
                    "node_feature": torch.arange(env.n_nodes)[:, None],
                    "edge_feature": torch.ones((0, 1)),
                    "edge_index": torch.ones((0, 2), dtype=torch.long),
                },
                batch_size=(),
            )
            sf = TensorDict(
                {
                    "node_feature": -torch.ones(env.n_nodes)[:, None],
                    "edge_feature": torch.zeros((0, 1)),
                    "edge_index": torch.zeros((0, 2), dtype=torch.long),
                },
                batch_size=(),
            )

            def __init__(self, tensor: TensorDict):
                self.tensor = tensor
                self.node_features_dim = tensor["node_feature"].shape[-1]
                self.edge_features_dim = tensor["edge_feature"].shape[-1]
                self._log_rewards: Optional[float] = None

                self.n_nodes = env.n_nodes
                self.n_actions = env.n_actions

            @property
            def forward_masks(self):
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
                    existing_edges = (
                        self[i].tensor["edge_index"]
                        - self.tensor["node_index"][self.tensor["batch_ptr"][i]]
                    )
                    assert torch.all(existing_edges >= 0)  # TODO: convert to test.

                    if len(existing_edges) == 0:
                        edge_idx = torch.zeros(0, dtype=torch.bool)
                    else:
                        edge_idx = torch.logical_and(
                            existing_edges[:, 0] == ei0.unsqueeze(-1),
                            existing_edges[:, 1] == ei1.unsqueeze(-1),
                        )

                        # Collapse across the edge dimension.
                        if len(edge_idx.shape) == 2:
                            edge_idx = edge_idx.sum(1).bool()

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
                # Disallow all actions.
                backward_masks = torch.zeros(
                    len(self), self.n_actions - 1, dtype=torch.bool
                )

                for i in range(len(self)):
                    existing_edges = (
                        self[i].tensor["edge_index"]
                        - self.tensor["node_index"][self.tensor["batch_ptr"][i]]
                    )

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
                            existing_edges[:, 0] == ei0.unsqueeze(-1),
                            existing_edges[:, 1] == ei1.unsqueeze(-1),
                        )
                        # Collapse across the edge dimension.
                        if len(edge_idx.shape) == 2:
                            edge_idx = edge_idx.sum(1).bool()

                        backward_masks[i, edge_idx] = (
                            True  # Allow the removal of this edge.
                        )

                return backward_masks.view(*self.batch_shape, self.n_actions - 1)

            @backward_masks.setter
            def backward_masks(self, value: torch.Tensor):
                pass  # bwd masks is computed on the fly

        return RingStates

    def _step(self, states: GraphStates, actions: Actions) -> GraphStates:
        actions = self.convert_actions(states, actions)
        new_states = super()._step(states, actions)
        assert isinstance(new_states, GraphStates)
        return new_states

    def _backward_step(self, states: GraphStates, actions: Actions) -> GraphStates:
        actions = self.convert_actions(states, actions)
        new_states = super()._backward_step(states, actions)
        assert isinstance(new_states, GraphStates)
        return new_states

    def convert_actions(self, states: GraphStates, actions: Actions) -> GraphActions:
        """Converts the action from discrete space to graph action space."""
        action_tensor = actions.tensor.squeeze(-1).clone()
        action_type = torch.where(
            action_tensor == self.n_actions - 1,
            GraphActionType.EXIT,
            GraphActionType.ADD_EDGE,
        )
        action_type[action_tensor == self.n_actions] = GraphActionType.DUMMY

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

        offset = states.tensor["node_index"][states.tensor["batch_ptr"][:-1]]
        out = GraphActions(
            TensorDict(
                {
                    "action_type": action_type,
                    "features": torch.ones(action_tensor.shape + (1,)),
                    "edge_index": torch.stack([ei0, ei1], dim=-1) + offset[:, None],
                },
                batch_size=action_tensor.shape,
            )
        )
        return out


class GraphPreprocessor(Preprocessor):
    """Extract the tensor from the states."""

    def __init__(self, feature_dim: int = 1):
        super().__init__(output_dim=feature_dim)

    def preprocess(self, states: GraphStates) -> TensorDict:
        return states.tensor

    def __call__(self, states: GraphStates) -> torch.Tensor:
        return self.preprocess(states)


def render_states(states: GraphStates, state_evaluator: callable, directed: bool):
    """Render the states as a matplotlib plot.

    Args:
        states: A batch of graphs.
    """
    rewards = state_evaluator(states)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    for i in range(8):
        current_ax = ax[i // 4, i % 4]
        state = states[i]
        n_circles = state.tensor["node_feature"].shape[0]
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

        edge_index = states[i].tensor["edge_index"]
        edge_index = torch.where(
            edge_index[..., None] == states[i].tensor["node_index"]
        )[2].reshape(edge_index.shape)

        for edge in edge_index:
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


if __name__ == "__main__":
    N_NODES = 4
    N_ITERATIONS = 1000
    LR = 0.001
    BATCH_SIZE = 128
    DIRECTED = True
    USE_BUFFER = False

    state_evaluator = undirected_reward if not DIRECTED else directed_reward
    torch.random.manual_seed(7)
    env = RingGraphBuilding(
        n_nodes=N_NODES, state_evaluator=state_evaluator, directed=DIRECTED
    )
    module_pf = RingPolicyModule(env.n_nodes, DIRECTED, num_conv_layers=2)
    module_pb = RingPolicyModule(env.n_nodes, DIRECTED, is_backward=True, num_conv_layers=2)
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
            env, n=BATCH_SIZE, save_logprobs=True,  # pyright: ignore
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
                    training_samples = training_samples[:BATCH_SIZE // 2]
                    buffer_samples = replay_buffer.sample(n_trajectories=BATCH_SIZE // 2)
                    training_samples.extend(buffer_samples)

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples)  # pyright: ignore
        pct_rings = torch.mean(rewards > 0.1, dtype=torch.float) * 100
        print("Iteration {} - Loss: {:.02f}, rings: {:.0f}%".format(
            iteration, loss.item(), pct_rings)
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    t2 = time.time()
    print("Time:", t2 - t1)

    # This comes from the gflownet, not the buffer.
    last_states = trajectories.last_states[:8]
    assert isinstance(last_states, GraphStates)
    render_states(last_states, state_evaluator, DIRECTED)
