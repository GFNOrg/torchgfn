"""Generate expert data for ring graphs.

This module provides functionality to generate all possible ring graphs for a given number of nodes.
It supports both directed and undirected rings, using the same environment as train_graph_ring.py.
"""

import itertools
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Literal, Union

import torch
from tensordict import TensorDict
from train_graph_ring import RingReward, init_env, init_gflownet, render_states

from gfn.actions import GraphActions, GraphActionType
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.containers.trajectories import Trajectories
from gfn.gym.graph_building import GraphBuildingOnEdges
from gfn.samplers import Sampler
from gfn.states import GraphStates
from gfn.utils.graphs import from_edge_indices


def generate_all_rings(
    n_nodes: int,
    device: Union[str, torch.device, Literal["cpu", "cuda"]] = "cpu",
    max_rings: int = 10000,
) -> GraphStates:
    """Generate all possible ring graphs for a given number of nodes using GraphActions.

    Args:
        n_nodes: Number of nodes in the graph
        device: Device to use for tensor operations ("cpu" or "cuda")
        max_rings: Maximum number of rings to generate.

    Returns:
        List of tuples (final_state, actions) where:
            - final_state is the GraphState representing a valid ring
            - actions is the list of GraphActions that build the ring
    """
    device = torch.device(device)
    state_evaluator = RingReward(directed=True, device=device)
    env = GraphBuildingOnEdges(
        n_nodes=n_nodes, state_evaluator=state_evaluator, directed=True, device=device
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

            action_dict = TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.tensor(
                        [GraphActionType.ADD_EDGE], device=device
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.zeros(
                        1, dtype=torch.long, device=device
                    ),
                    GraphActions.EDGE_CLASS_KEY: torch.zeros(
                        1, dtype=torch.long, device=device
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.tensor(
                        [edge_idx], dtype=torch.long, device=device
                    ),
                },
                batch_size=[1],
            )
            action = env.Actions.from_tensor_dict(action_dict)
            state = env.step(state, action)

        valid_rings.append(state)
        if len(valid_rings) >= max_rings:
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
        embedding_dim=args.embedding_dim,
        num_conv_layers=args.num_conv_layers,
        num_edge_classes=env.num_edge_classes,
        device=device,
    )

    # 3. Create the optimizer
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
        capacity=args.replay_buffer_max_size,
        prioritized=True,
    )

    # If the user has requested to use expert data to prefill the replay buffer,
    # generate all possible ring graphs, determine their backward trajectory,
    # reverse those to create forward trajectories, and add those to the replay
    # buffer.
    if args.use_expert_data:
        # Generate all possible ring graphs.
        ring_examples = generate_all_rings(
            args.n_nodes, device, args.replay_buffer_max_size
        )
        backward_sampler = Sampler(gflownet.pb)
        sample_trajectories = backward_sampler.sample_trajectories(
            env,
            n=args.replay_buffer_max_size,
            states=ring_examples,
            save_logprobs=True,
        )
        sample_trajectories = sample_trajectories.reverse_backward_trajectories()
        replay_buffer.add(sample_trajectories)

        final_states = sample_trajectories.terminating_states
        assert isinstance(final_states, GraphStates)
        rewards = env.reward(final_states)
        print("Mean reward of expert data:", rewards.mean().item())

    # Train the GFlowNet to generate ring graphs, drawing on examples from the
    # replay buffer.
    losses = []

    t1 = time.time()
    epsilon_dict = defaultdict(float)
    for iteration in range(args.n_iterations):
        epsilon_dict[GraphActions.ACTION_TYPE_KEY] = 0.1

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

        # Add the training samples to the replay buffer. If the user requested the use
        # of expert data, gflownet samples are only added after the first
        # n_iterations_with_expert_data iterations. The gflownet loss is calculated on
        # the a 50% mixture of the gflownet samples and the replay buffer samples.
        with torch.no_grad():
            if (
                iteration < args.n_iterations_with_only_expert_data
                and args.use_expert_data
            ):
                training_samples = replay_buffer.sample(n_trajectories=args.batch_size)
            else:
                # A mix of 50% the replay buffer and 50% the gflownet samples.
                replay_buffer.add(training_samples)
                training_samples = training_samples[: args.batch_size // 2]
                replay_buffer_samples = replay_buffer.sample(
                    n_trajectories=args.batch_size // 2
                )
                assert isinstance(replay_buffer_samples, Trajectories)
                training_samples.extend(replay_buffer_samples)

        if iteration > 101 and iteration % 25 == 0:
            print(
                "sum of rewards in buffer: {}",
                sum(env.reward(replay_buffer.training_objects.terminating_states)) / 100,
            )

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
        pct_rings = torch.mean(rewards > 0.1, dtype=torch.float) * 100
        print(
            "Iteration {} - Loss: {:.02f}, rings: {:.0f}%, logZ: {:.06f}".format(
                iteration, loss.item(), pct_rings, gflownet.logZ.item()
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
        "--n_nodes", type=int, default=8, help="Number of nodes in the graph"
    )

    parser.add_argument(
        "--num_conv_layers", type=int, default=1, help="Number of convolutional layers"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=256, help="Embedding dimension"
    )

    # Training parameters
    parser.add_argument(
        "--n_iterations", type=int, default=500, help="Number of training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument("--lr_Z", type=float, default=0.1, help="Learning rate for logZ")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )

    parser.add_argument(
        "--replay_buffer_max_size",
        type=int,
        default=1024 * 16,
        help="Max size of the replay buffer",
    )

    # Misc parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Whether to plot generated graphs",
    )

    # Expert data parameters.
    parser.add_argument(
        "--use_expert_data",
        action="store_true",
        default=False,
        help="Whether to use expert data to prefill the replay buffer.",
    )
    parser.add_argument(
        "--n_iterations_with_only_expert_data",
        type=int,
        default=100,
        help="Number of iterations where no gflownet samples are added to the replay buffer.",
    )

    args = parser.parse_args()
    main(args)
