"""Generate expert data for ring graphs.

This module provides functionality to generate all possible ring graphs for a given number of nodes.
It supports both directed and undirected rings, using the same environment as train_graph_ring.py.
"""

from argparse import ArgumentParser, Namespace
from collections import defaultdict
import itertools
import time
from typing import Union, Literal, List, Tuple, Dict, Any
import numpy as np
import torch
from tensordict import TensorDict
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.modules import DiscreteGraphPolicyEstimator
from gfn.samplers import Sampler
from gfn.states import GraphStates
from gfn.gym.graph_building import GraphBuildingOnEdges
from gfn.actions import GraphActions, GraphActionType
from gfn.utils.graphs import from_edge_indices
from gfn.utils.modules import GraphEdgeActionGNN
from train_graph_ring import RingReward, init_env, init_gflownet, render_states

def generate_all_rings(
    n_nodes: int, 
    device: Union[str, torch.device, Literal["cpu", "cuda"]] = "cpu",
    cut_off: int = 10000,
) -> GraphStates:
    """Generate all possible ring graphs for a given number of nodes using GraphActions.
    
    Args:
        n_nodes: Number of nodes in the graph
        device: Device to use for tensor operations ("cpu" or "cuda")
        
    Returns:
        List of tuples (final_state, actions) where:
            - final_state is the GraphState representing a valid ring
            - actions is the list of GraphActions that build the ring
    """
    device = torch.device(device)
    state_evaluator = RingReward(directed=True, device=device)
    env = GraphBuildingOnEdges(
        n_nodes=n_nodes,
        state_evaluator=state_evaluator,
        directed=True,
        device=device
    )
    
    valid_rings = []
    for perm in itertools.permutations(range(1, n_nodes)):
        # Start with empty state
        state = env.reset(batch_shape=())
        
        # Create the sequence of nodes to connect
        nodes_sequence = [0] + list(perm) + [0]  # Add 0 at end to close the ring
        
        # Add edges between consecutive nodes
        for i in range(len(nodes_sequence) - 1):
            src, dst = nodes_sequence[i], nodes_sequence[i + 1]
            edge_idx = from_edge_indices(src, dst, n_nodes, is_directed=True)
            
            action_dict = TensorDict({
                GraphActions.ACTION_TYPE_KEY: torch.tensor([GraphActionType.ADD_EDGE], device=device),
                GraphActions.NODE_CLASS_KEY: torch.zeros(1, dtype=torch.long, device=device),
                GraphActions.EDGE_CLASS_KEY: torch.zeros(1, dtype=torch.long, device=device),
                GraphActions.EDGE_INDEX_KEY: torch.tensor([edge_idx], dtype=torch.long, device=device)
            }, batch_size=[1])
            action = env.Actions.from_tensor_dict(action_dict)
            state = env.step(state, action)

        valid_rings.append(state)
        if len(valid_rings) >= cut_off:
            break

    ring_examples = env.States.stack(valid_rings)
    ring_examples.data = ring_examples.data.flatten()
    return ring_examples


def main(args: Namespace):
    """
    Main execution for training a GFlowNet to generate ring graphs.

    This script demonstrates the complete workflow of training a GFlowNet
    to generate valid ring structures in both directed and undirected settings.

    Configurable parameters:
    - n_nodes: Number of nodes in the graph (default: 5)
    - n_iterations: Number of training iterations (default: 1000)
    - lr: Learning rate for optimizer (default: 0.001)
    - batch_size: Batch size for training (default: 128)
    - directed: Whether to generate directed rings (default: True)
    - use_buffer: Whether to use a replay buffer (default: False)
    - use_gnn: Whether to use GNN-based policy (True) or MLP-based policy (False)

    The script performs the following steps:
    1. Initialize the environment and policy networks
    2. Train the GFlowNet using trajectory balance
    3. Visualize sample generated graphs
    """
    device = torch.device(args.device)
    torch.random.manual_seed(7)

    env = init_env(args.n_nodes, True, device)
    gflownet = init_gflownet(
        num_nodes=args.n_nodes,
        directed=True,
        use_gnn=True,
        num_conv_layers=args.num_conv_layers,
        num_edge_classes=env.num_edge_classes,
        device=device,
    )
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)

    replay_buffer = ReplayBuffer(
        env,
        capacity=args.replay_buffer_max_size,
        prioritized=True,
    )

    # --- PREFILL REPLAY BUFFER ---
    ring_examples = generate_all_rings(args.n_nodes, device, args.replay_buffer_max_size)
    backward_sampler = Sampler(gflownet.pb)
    sample_trajectories = backward_sampler.sample_trajectories(
        env,
        n=args.replay_buffer_max_size,
        states=ring_examples,
        save_logprobs=True,
    )
    sample_trajectories = sample_trajectories.reverse_backward_trajectories()
    replay_buffer.add(sample_trajectories)

    losses = []

    t1 = time.time()
    epsilon_dict = defaultdict(float)
    for iteration in range(args.n_iterations):
        epsilon_dict[GraphActions.ACTION_TYPE_KEY] = 0.0

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

        with torch.no_grad():
            replay_buffer.add(training_samples)
            training_samples = training_samples[: args.batch_size // 2]
            buffer_samples = replay_buffer.sample(
                n_trajectories=args.batch_size // 2
            )
            training_samples.extend(buffer_samples)  # type: ignore

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
        pct_rings = torch.mean(rewards > 0.1, dtype=torch.float) * 100
        print(
            "Iteration {} - Loss: {:.02f}, rings: {:.0f}%".format(
                iteration, loss.item(), pct_rings
            )
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    t2 = time.time()
    print("Time:", t2 - t1)

    if args.plot:
        samples_to_render = trajectories.terminating_states[:8]
        assert isinstance(samples_to_render, GraphStates)
        render_states(samples_to_render, env.reward, True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a GFlowNet to generate ring graphs")

    # Model parameters
    parser.add_argument(
        "--n_nodes", type=int, default=4, help="Number of nodes in the graph"
    )

    parser.add_argument(
        "--num_conv_layers", type=int, default=1, help="Number of convolutional layers"
    )

    # Training parameters
    parser.add_argument(
        "--n_iterations", type=int, default=200, help="Number of training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )

    parser.add_argument(
        "--replay_buffer_max_size", type=int, default=1024 * 16, help="Max size of the replay buffer"
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

    args = parser.parse_args()
    main(args)