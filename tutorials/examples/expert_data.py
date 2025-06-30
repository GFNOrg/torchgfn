"""Generate expert data for ring graphs.

This module provides functionality to generate all possible ring graphs for a given number of nodes.
It supports both directed and undirected rings, using the same environment as train_graph_ring.py.
"""

import itertools
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Iterable, Literal, Union

import torch
from tensordict import TensorDict
from torch.optim.lr_scheduler import LambdaLR
from train_graph_ring import RingReward, init_env, init_gflownet, render_states

from gfn.actions import GraphActions, GraphActionType
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.containers.trajectories import Trajectories
from gfn.gym.graph_building import GraphBuildingOnEdges
from gfn.samplers import Sampler
from gfn.states import GraphStates
from gfn.utils.graphs import from_edge_indices


def grad_norm(params: Iterable[torch.nn.Parameter], p: float = 2) -> float:
    """
    Returns the p-norm of all gradients in ``params`` (ignores params with no grad).
    Example: grad_norm(model.parameters())               # total L2 norm
             grad_norm(model.parameters(), p=float('inf'))  # max-grad
    """
    grads = [p_.grad for p_ in params if p_.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm(p) for g in grads]), p).item()


def param_norm(params: Iterable[torch.nn.Parameter], p: float = 2) -> float:
    """
    Total p-norm of a collection of parameters.
    Example:
        model_pnorm = param_norm(model.parameters())        # L2 norm
        max_abs     = param_norm(model.parameters(), p=float('inf'))
    """
    with torch.no_grad():  # no grad tracking needed
        norms = [p_.data.norm(p) for p_ in params]
    return torch.norm(torch.stack(norms), p).item() if norms else 0.0


def lr_grad_ratio(optimizer: torch.optim.Optimizer) -> list[float]:
    """Return (lr·‖g‖₂)/‖θ‖₂ for each param group."""
    out = []
    for group in optimizer.param_groups:
        lr = group["lr"]
        g_norm = grad_norm(group["params"])
        p_norm = param_norm(group["params"])
        out.append((lr * g_norm) / p_norm if p_norm else 0.0)

    return out


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
    epsilon_dict[GraphActions.ACTION_TYPE_KEY] = 0.0  # Exploration on action type.

    for iteration in range(args.n_iterations):

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
        pct_rings = torch.mean(rewards > 0.1, dtype=torch.float) * 100

        # Add the training samples to the replay buffer. If the user requested the use
        # of expert data, gflownet samples are only added after the first
        # n_iterations_with_expert_data iterations. The gflownet loss is calculated on
        # the a 50% mixture of the gflownet samples and the replay buffer samples.
        # TODO: During iteration < args.n_iterations_with_only_expert_data, no
        # non-ring trajectories will be seen!
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
        loss.backward()

        # Print stats about the gradient and parameter norms.
        lr_grad_ratios = lr_grad_ratio(optimizer)
        # gnorm = grad_norm(gflownet.parameters())
        # pnorm = param_norm(gflownet.parameters())
        # lr = optimizer.param_groups[0]["lr"]  # Ignores logz param.
        # update_ratio = (lr * gnorm) / pnorm if pnorm else 0.0

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

        print(
            "Iter {}: loss: {:.02f}, rings: {:.0f}%, logZ: {:.06f}, grad_norms: {}".format(
                iteration,
                loss.item(),
                pct_rings,
                gflownet.logZ.item(),
                ", ".join("{:.{}g}".format(v, 4) for v in lr_grad_ratios),
            )
        )

    t2 = time.time()
    print("Total Training Time:", t2 - t1)

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
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )

    # Training parameters
    parser.add_argument(
        "--n_iterations", type=int, default=500, help="Number of training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        default=False,
        help="Whether to use a learning rate scheduler.",
    )
    parser.add_argument("--lr_Z", type=float, default=0.1, help="Learning rate for logZ")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
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
    print(args)
    main(args)
