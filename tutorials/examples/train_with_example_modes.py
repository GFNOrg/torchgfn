"""Train a GFlowNet to generate ring graphs using example modes.

This script demonstrates how to use example modes to warm-start gflownet exploration.
We show this in the context of generating ring graphs, where the number of modes
quickly grows with the number of nodes in the graph, making learning from scratch
very difficult on even relatively small graphs. Here, we show how to use example
modes to warm-start gflownet exploration allows us to learn a gflownet that can
sample all modes.

For usage see train_with_example_modes.py -h

The script performs the following steps:
    1. Initialize the environment and policy networks.
    2. If using expert data, generates all possible ring graphs, and pre-fills the
        replay buffer with 1/2 of their forward trajectories (found by computing the
        backward trajectories from the final states, then reversing them).
    3. Train the GFlowNet using trajectory balance, with each batch containing a
        mix of 50% replay buffer and 50% gflownet samples.
    4. At the end of training we evaluate the GFlowNet's ability to recover all
        modes.
    5. Optionally, we plot samples of generated graphs.

This tutorial uses the same environment as train_graph_ring.py.
"""

import copy
import itertools
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Literal, Union

import numpy as np
import torch
from tensordict import TensorDict
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
from train_graph_ring import RingReward, init_env, init_gflownet, render_states

from gfn.actions import GraphActions, GraphActionType
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.containers.trajectories import Trajectories
from gfn.gym.graph_building import GraphBuildingOnEdges
from gfn.samplers import Sampler
from gfn.states import GraphStates
from gfn.utils.common import set_seed
from gfn.utils.graphs import from_edge_indices, hash_graph
from gfn.utils.training import lr_grad_ratio


def per_step_decay(num_steps: int, total_drop: float) -> float:
    """
    Compute the per-step decay multiplier y (γ) so that after ``num_steps``
    scheduler steps the learning rate has been multiplied by ``total_drop``.

    lr_final = lr_init * y**num_steps  -->  y = total_drop**(1/num_steps)

    Args
    ----
    num_steps   : total number of scheduler.step() calls you will make
    total_drop  : desired overall multiplier (e.g. 0.1 for a 10× drop)

    Returns
    -------
    y : float  # per-step multiplier
    """
    if not (0.0 < total_drop <= 1.0):
        raise ValueError("total_drop must be in (0, 1].")
    return total_drop ** (1.0 / num_steps)


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


def count_recovered_modes(final_states: GraphStates, mode_hashes: set[str]) -> int:
    """Count the number of unique modes in the final states that are in the mode_hashes."""
    _hashes = copy.deepcopy(mode_hashes)

    found = 0
    for example in final_states:
        example_hash = hash_graph(example.tensor[0], directed=True)
        if example_hash in _hashes:
            found += 1
            _hashes.remove(example_hash)
    return found


def main(args: Namespace):
    """
    Main execution for training a GFlowNet to generate ring graphs.

    For usage see train_with_example_modes.py -h

    The function performs the following steps:
        1. Initialize the environment and policy networks.
        2. If using expert data, generates all possible ring graphs, and pre-fills the
           replay buffer with 1/2 of their forward trajectories (found by computing the
           backward trajectories from the final states, then reversing them).
        3. Train the GFlowNet using trajectory balance, with each batch containing a
           mix of 50% replay buffer and 50% gflownet samples.
        4. At the end of training we evaluate the GFlowNet's ability to recover all
           modes.
        5. Optionally, we plot samples of generated graphs.
    """
    device = torch.device(args.device)
    set_seed(args.seed if args.seed is not None else 1234)

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

    # Create the learning rate scheduler.
    if args.use_lr_scheduler:
        gamma = per_step_decay(args.n_iterations, 0.1)  # 10x drop over n_iterations.
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                lambda s: gamma**s,  # non-logZ decay
                lambda s: gamma**s,  # logZ decay
            ],
        )
    else:
        scheduler = None

    # Generate all possible ring graphs (and compute their hashes).
    ring_modes = generate_all_rings(args.n_nodes, device, args.replay_buffer_max_size)
    n_modes_total = len(ring_modes)
    mode_hashes = {hash_graph(m.tensor[0], directed=True) for m in ring_modes}
    print(f"+ Number of modes: {n_modes_total}")

    # Create the replay buffer.
    replay_buffer = ReplayBuffer(
        env,
        capacity=args.replay_buffer_max_size,
        prioritized_capacity=True,
        prioritized_sampling=True,
    )

    # If the user has requested to use expert data to prefill the replay buffer
    # determine the backward trajectories of the ring_examples, reverse them to
    # create forward trajectories, and add those to the replay buffer.
    if args.use_expert_data:

        n_expert_data = n_modes_total // 2  # Use half of the modes as expert data.
        idx = np.arange(n_modes_total)
        np.random.shuffle(idx)
        ring_expert_data = ring_modes[idx[:n_expert_data]]  # type: ignore
        assert gflownet.pb is not None
        backward_sampler = Sampler(gflownet.pb)

        trajectories = backward_sampler.sample_trajectories(
            env,
            states=ring_expert_data,
            save_logprobs=False,  # Not used.
        )
        trajectories = trajectories.reverse_backward_trajectories()
        training_samples = gflownet.to_training_samples(trajectories)
        replay_buffer.add(training_samples)

        # Report the proportion of modes in the replay buffer expert data.
        final_states = training_samples.terminating_states
        assert isinstance(final_states, GraphStates)
        rewards = env.reward(final_states)
        print("+ Mean reward of expert data:", rewards.mean().item())

        # Ensures that all replay buffer expert data are in ring_modes and unique.
        assert count_recovered_modes(final_states, mode_hashes) == len(final_states)

    # Train the GFlowNet to generate ring graphs, drawing on examples from the
    # replay buffer.
    losses = []

    time_start = time.time()

    # Exploration on action type & edge index.
    epsilon_dict = defaultdict(float)
    epsilon_dict[GraphActions.ACTION_TYPE_KEY] = args.action_type_epsilon
    epsilon_dict[GraphActions.EDGE_INDEX_KEY] = args.edge_index_epsilon

    pbar = trange(args.n_iterations)
    for iteration in pbar:
        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            epsilon=epsilon_dict,
        )
        training_samples = gflownet.to_training_samples(trajectories)

        # Collect rewards for reporting.
        terminating_states = training_samples.terminating_states
        assert isinstance(terminating_states, GraphStates)
        rewards = env.reward(terminating_states)
        pct_rings = torch.mean(rewards > 0.1, dtype=torch.get_default_dtype()) * 100

        # Add the training samples to the replay buffer. If the user requested the use
        # of expert data, gflownet samples are only added after the first
        # n_iterations_with_expert_data iterations. The gflownet loss is calculated on
        # the a 50% mixture of the gflownet samples and the replay buffer samples.
        with torch.no_grad():
            # Mix of 50% replay buffer and 50% gflownet samples.
            replay_buffer.add(training_samples)
            training_samples = training_samples[: args.batch_size // 2]
            replay_buffer_samples = replay_buffer.sample(n_samples=args.batch_size // 2)
            assert isinstance(replay_buffer_samples, Trajectories)
            training_samples.extend(replay_buffer_samples)

        if iteration % 100 == 0 or iteration == 0 or iteration == args.n_iterations - 1:
            n_modes_in_buffer = sum(
                env.reward(
                    replay_buffer.training_objects.terminating_states  # type: ignore
                )
                > 0.1
            )

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)  # type: ignore
        loss.backward()
        lr_g_ratios = lr_grad_ratio(optimizer)  # lr * grad_norm / param_norm.
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        final_states = training_samples.terminating_states
        assert isinstance(final_states, GraphStates)
        found = count_recovered_modes(final_states, mode_hashes)

        pbar.set_postfix(
            iter=iteration,
            loss=loss.item(),
            pct_gfn_sampled_rings=pct_rings,
            unique_modes_in_batch="{}/{}".format(found, n_modes_total),
            logZ=gflownet.logZ.item(),  # type: ignore
            n_modes_in_buffer=n_modes_in_buffer,
            grad_norm_ratios=", ".join("{:.{}g}".format(v, 3) for v in lr_g_ratios),
        )

    print("+ Total Training Time: {} minutes".format((time.time() - time_start) / 60))

    # Report mode coverage.
    sample_trajectories = gflownet.sample_trajectories(
        env,
        n=n_modes_total * 100,  # Oversample to ensure all modes are seen.
        save_logprobs=False,  # Not used.
    )
    final_states = sample_trajectories.terminating_states
    assert isinstance(final_states, GraphStates)
    found = count_recovered_modes(final_states, mode_hashes)
    print(f"+ Sampler discovered {found} / {n_modes_total} modes.")

    if args.plot:
        samples_to_render = trajectories.terminating_states[:8]
        assert isinstance(samples_to_render, GraphStates)
        render_states(samples_to_render, env.reward, True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a GFlowNet to generate ring graphs.")

    # Problem size (number of modes is factorial in n_nodes).
    parser.add_argument(
        "--n_nodes", type=int, default=7, help="Number of nodes in the graph"
    )
    parser.add_argument(
        "--use_expert_data",
        action="store_true",
        default=False,
        help="Whether to use expert data to prefill the replay buffer.",
    )

    # GFlowNet estimator parameters.
    parser.add_argument(
        "--num_conv_layers", type=int, default=2, help="Number of convolutional layers"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )

    # Training parameters.
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=500, help="Number of training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument("--lr_Z", type=float, default=0.1, help="Learning rate for logZ")
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        default=False,
        help="Whether to use a learning rate scheduler.",
    )
    parser.add_argument(
        "--replay_buffer_max_size",
        type=int,
        default=1024 * 16,
        help="Max size of the replay buffer",
    )

    # Exploration parameters.
    parser.add_argument(
        "--action_type_epsilon",
        type=float,
        default=0.0,
        help="Epsilon for exploration on action type.",
    )
    parser.add_argument(
        "--edge_index_epsilon",
        type=float,
        default=0.0,
        help="Epsilon for exploration on edge index.",
    )

    # Misc parameters.
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
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for random number generator.",
    )

    args = parser.parse_args()
    print("+ Training Arguments:", args)
    main(args)
