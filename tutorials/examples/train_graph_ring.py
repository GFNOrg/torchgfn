"""Train a GFlowNet to generate ring graphs.

This example demonstrates training a GFlowNet to generate graphs that are rings - where
each vertex has exactly two neighbors and the edges form a single cycle containing all
vertices. The environment supports both directed and undirected ring generation.

This problem has a number of modes that grows factorially with the number of nodes.
This makes learning from scratch very difficult on even relatively small graphs.

For usage see train_graph_ring.py -h

Key components:
- `RingGraphBuilding`: Environment for building ring graphs
- `RingPolicyModule`: GNN-based policy network for predicting actions
- `directed_reward`/`undirected_reward`: Reward functions for validating ring structures.
"""

import math
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from matplotlib import patches

from gfn.actions import GraphActions
from gfn.containers import ReplayBuffer
from gfn.estimators import DiscreteGraphPolicyEstimator
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym.graph_building import GraphBuildingOnEdges
from gfn.states import GraphStates
from gfn.utils.common import set_seed
from gfn.utils.modules import GraphActionGNN, GraphEdgeActionMLP


class RingReward(object):
    """
    This function evaluates if a graph forms a valid ring (directed or
        undirected cycle).

    Args:
        directed: Whether the graph is directed.
        reward_val: The reward for valid directed rings.
        eps_val: The reward for invalid structures.

    Returns:
        A tensor of rewards with the same batch shape as states
    """

    def __init__(
        self,
        directed: bool,
        reward_val: float = 100.0,
        eps_val: float = 1e-6,
        device: torch.device = torch.device("cpu"),
    ):
        self.directed = directed
        self.reward_val = reward_val
        self.eps_val = eps_val
        self.device = device

    def __call__(self, states: GraphStates) -> torch.Tensor:
        if self.directed:
            return self.directed_reward(states)
        else:
            return self.undirected_reward(states)

    def directed_reward(self, states: GraphStates) -> torch.Tensor:
        """Compute reward for directed ring graphs.

        This function evaluates if a graph forms a valid directed ring (cycle).
        A valid directed ring must satisfy these conditions:
        1. Each node must have exactly one outgoing edge (row sum = 1 in
            adjacency matrix).
        2. Each node must have exactly one incoming edge (column sum = 1 in
            adjacency matrix).
        3. Following the edges must form a single cycle that includes all nodes.

        Args:
            states: A batch of graph states to evaluate.

        Returns:
            A tensor of rewards with the same batch shape as states.
        """
        out = torch.full(
            (len(states),), self.eps_val, device=self.device
        )  # Default reward.

        for i in range(len(states)):
            graph = states[i]
            graph_tensor = graph.tensor
            adj_matrix = torch.zeros(graph_tensor.x.size(0), graph_tensor.x.size(0))
            adj_matrix[graph_tensor.edge_index[0], graph_tensor.edge_index[1]] = 1

            # Check if each node has exactly one outgoing edge (row sum = 1)
            if not torch.all(adj_matrix.sum(dim=1) == 1):
                continue

            # Check that each node has exactly one incoming edge (column sum = 1)
            if not torch.all(adj_matrix.sum(dim=0) == 1):
                continue

            # Starting from node 0, follow edges and see if we visit all nodes
            # and return to the start
            visited, current = [], 0  # Start from node 0.

            while current not in visited:
                visited.append(current)

                # Get the outgoing neighbor
                current = torch.where(adj_matrix[int(current)] == 1)[0].item()

                # If we've visited all nodes and returned to 0, it's a valid ring
                if len(visited) == graph_tensor.x.size(0) and current == 0:
                    out[i] = self.reward_val
                    break

        return out.view(*states.batch_shape)

    def undirected_reward(self, states: GraphStates) -> torch.Tensor:
        """Compute reward for undirected ring graphs.

        This function evaluates if a graph forms a valid undirected ring (cycle).
        A valid undirected ring must satisfy these conditions:
        1. Each node must have exactly two neighbors (degree = 2)
        2. The graph must form a single connected cycle including all nodes.

        The algorithm:
        1. Checks that all nodes have degree 2
        2. Performs a traversal starting from node 0, following edges
        3. Checks if the traversal visits all nodes and returns to start

        Args:
            states: A batch of graph states to evaluate

        Returns:
            A tensor of rewards with the same batch shape as states
        """
        out = torch.full(
            (len(states),), self.eps_val, device=self.device
        )  # Default reward.

        for i in range(len(states)):
            graph = states[i]
            if graph.tensor.x.size(0) == 0:
                continue
            adj_matrix = torch.zeros(graph.tensor.x.size(0), graph.tensor.x.size(0))
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
                visited.append(int(current))
                current_neighbors = torch.where(adj_matrix[int(current)] == 1)[0]
                # Exclude the neighbor we just came from.
                current_neighbors_list = [n.item() for n in current_neighbors]
                possible = [n for n in current_neighbors_list if n != prev]
                if len(possible) != 1:
                    break
                next_node = possible[0]
                prev, current = current, next_node

            if current == start_vertex and len(visited) == graph.tensor.x.size(0):
                out[i] = self.reward_val

        return out.view(*states.batch_shape)


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
        n_circles = state.tensor.x.size(0)
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


def init_gflownet(
    num_nodes: int,
    directed: bool,
    use_gnn: bool,
    embedding_dim: int,
    num_conv_layers: int,
    num_edge_classes: int,
    device: torch.device,
) -> TBGFlowNet:
    # Choose model type based on USE_GNN flag
    if use_gnn:
        module_pf = GraphActionGNN(
            num_node_classes=num_nodes,
            directed=directed,
            num_conv_layers=num_conv_layers,
            num_edge_classes=num_edge_classes,
            embedding_dim=embedding_dim,
        )
        module_pb = GraphActionGNN(
            num_node_classes=num_nodes,
            directed=directed,
            is_backward=True,
            num_conv_layers=num_conv_layers,
            num_edge_classes=num_edge_classes,
            embedding_dim=embedding_dim,
        )
    else:
        module_pf = GraphEdgeActionMLP(
            num_nodes,
            directed,
            num_node_classes=num_nodes,
            num_edge_classes=num_edge_classes,
            embedding_dim=embedding_dim,
        )
        module_pb = GraphEdgeActionMLP(
            num_nodes,
            directed,
            is_backward=True,
            num_node_classes=num_nodes,
            num_edge_classes=num_edge_classes,
            embedding_dim=embedding_dim,
        )

    pf = DiscreteGraphPolicyEstimator(
        module=module_pf,
    )
    pb = DiscreteGraphPolicyEstimator(
        module=module_pb,
        is_backward=True,
    )
    gflownet = TBGFlowNet(pf, pb).to(device)
    return gflownet


def init_env(n_nodes: int, directed: bool, device: torch.device) -> GraphBuildingOnEdges:
    state_evaluator = RingReward(
        directed=directed,
        reward_val=100.0,
        eps_val=1e-6,
        device=device,
    )

    return GraphBuildingOnEdges(
        n_nodes=n_nodes,
        state_evaluator=state_evaluator,
        directed=directed,
        device=device,
    )


def main(args: Namespace):
    """
    Main execution for training a GFlowNet to generate ring graphs.

    This script demonstrates the complete workflow of training a GFlowNet
    to generate valid ring structures in both directed and undirected settings.

    For usage see train_graph_ring.py -h

    The script performs the following steps:
        1. Initialize the environment and policy networks.
        2. Train the GFlowNet using trajectory balance.
        3. Visualize sample generated graphs.
    """
    device = torch.device(args.device)
    set_seed(args.seed)
    env = init_env(args.n_nodes, args.directed, device)

    gflownet = init_gflownet(
        args.n_nodes,
        args.directed,
        args.use_gnn,
        args.embedding_dim,
        args.num_conv_layers,
        env.num_edge_classes,
        device,
    )

    # Create the optimizer.
    non_logz_params = [
        v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
    ]
    if "logZ" in dict(gflownet.named_parameters()):
        logz_params = [dict(gflownet.named_parameters())["logZ"]]
    else:
        logz_params = []

    params = [
        {"params": non_logz_params, "lr": args.lr},
        # Log Z gets dedicated learning rate (typically higher).
        {"params": logz_params, "lr": args.lr_Z},
    ]
    optimizer = torch.optim.Adam(params)

    replay_buffer = ReplayBuffer(
        env,
        capacity=args.batch_size,
        prioritized_capacity=True,
        prioritized_sampling=True,
    )

    losses = []

    t1 = time.time()
    epsilon_dict = defaultdict(float)
    for iteration in range(args.n_iterations):
        epsilon_dict[GraphActions.ACTION_TYPE_KEY] = args.action_type_epsilon
        epsilon_dict[GraphActions.EDGE_INDEX_KEY] = args.edge_index_epsilon

        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            epsilon=epsilon_dict,
        )
        training_samples = gflownet.to_training_samples(trajectories)

        # Collect rewards for reporting.
        terminating_states = training_samples.terminating_states
        assert isinstance(terminating_states, GraphStates)
        rewards = env.reward(terminating_states)

        if args.use_buffer:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                if iteration > 10:
                    training_samples = training_samples[: args.batch_size // 2]
                    buffer_samples = replay_buffer.sample(n_samples=args.batch_size // 2)
                    training_samples.extend(buffer_samples)  # type: ignore

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
        pct_rings = torch.mean(rewards > 0.1, dtype=torch.get_default_dtype()) * 100
        print(
            "Iteration {} - Loss: {:.02f}, rings: {:.0f}%".format(
                iteration, loss.item(), pct_rings
            )
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print("+ Total Training Time: {} minutes".format((time.time() - t1) / 60))

    # This comes from the gflownet, not the buffer.
    if args.plot:
        samples_to_render = trajectories.terminating_states[:8]
        assert isinstance(samples_to_render, GraphStates)
        render_states(samples_to_render, env.reward, args.directed)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a GFlowNet to generate ring graphs")

    # Problem size (number of modes is factorial in n_nodes for directed rings).
    parser.add_argument(
        "--n_nodes", type=int, default=4, help="Number of nodes in the graph"
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        default=True,
        help="Whether to generate directed rings",
    )

    # GFlowNet estimator parameters.
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        default=True,
        help="Use GNN-based policy (True) or MLP-based policy (False)",
    )
    parser.add_argument(
        "--num_conv_layers", type=int, default=1, help="Number of convolutional layers"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )

    # Training parameters.
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=200, help="Number of training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument("--lr_Z", type=float, default=0.1, help="Learning rate for logZ")
    parser.add_argument(
        "--use_buffer",
        action="store_true",
        default=True,
        help="Whether to use replay buffer",
    )

    # Exploration parameters.
    parser.add_argument(
        "--action_type_epsilon",
        type=float,
        default=0.0,
        help="Epsilon for action type exploration",
    )
    parser.add_argument(
        "--edge_index_epsilon",
        type=float,
        default=0.0,
        help="Epsilon for edge index exploration",
    )

    # Misc parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (cpu or cuda)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Whether to plot generated graphs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for random number generator.",
    )

    args = parser.parse_args()
    main(args)
