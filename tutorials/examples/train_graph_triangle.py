"""Minimal training example for a graph-building environment with node addition.

This script demonstrates training a GFlowNet on a generic graph-building task where
the policy can ADD_NODE, ADD_EDGE, or EXIT. The reward here encourages forming
an undirected triangle (3 nodes, 3 edges).

Run:
  python tutorials/examples/train_graph_build_nodes.py --device cpu --plot
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Callable

import torch

from gfn.actions import GraphActions
from gfn.containers import ReplayBuffer
from gfn.estimators import DiscreteGraphPolicyEstimator
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym.graph_building import GraphBuilding
from gfn.states import GraphStates
from gfn.utils.common import set_seed
from gfn.utils.modules import GraphActionGNN


class TriangleReward:
    """Reward high if the graph is an undirected triangle, else a small epsilon.

    - Undirected triangle: exactly 3 nodes and edges {(0,1),(1,2),(0,2)} ignoring labels.
    """

    def __init__(
        self,
        reward_val: float = 100.0,
        eps_val: float = 1e-6,
        device: torch.device | str = "cpu",
    ):
        self.reward_val = reward_val
        self.eps_val = eps_val
        self.device = torch.device(device)

    def __call__(self, states: GraphStates) -> torch.Tensor:
        out = torch.full((len(states),), self.eps_val, device=self.device)
        for i in range(len(states)):
            g = states[i].tensor
            if g.x is None or g.edge_index is None:
                continue
            if g.x.size(0) != 3:
                continue
            # Build undirected adjacency
            adj = torch.zeros(3, 3, device=self.device)
            if g.edge_index.numel() > 0:
                adj[g.edge_index[0], g.edge_index[1]] = 1
                adj[g.edge_index[1], g.edge_index[0]] = 1
            # Triangle has 3 undirected edges and degree 2 for each node
            if (adj.sum() / 2 == 3) and torch.all(adj.sum(dim=1) == 2):
                out[i] = self.reward_val
        return out.view(*states.batch_shape)


def init_env(device: torch.device) -> GraphBuilding:
    state_evaluator = TriangleReward(device=device)
    # Start from empty graph; allow 5 node class and 1 edge class, undirected
    env = GraphBuilding(
        num_node_classes=5,
        num_edge_classes=1,
        state_evaluator=state_evaluator,
        is_directed=False,
        device=device,
    )
    return env


def init_gflownet(
    env: GraphBuilding, embedding_dim: int, num_conv_layers: int, device: torch.device
) -> TBGFlowNet:
    pf_module = GraphActionGNN(
        num_node_classes=env.num_node_classes,
        num_edge_classes=env.num_edge_classes,
        directed=env.is_directed,
        embedding_dim=embedding_dim,
        num_conv_layers=num_conv_layers,
        is_backward=False,
    )
    pb_module = GraphActionGNN(
        num_node_classes=env.num_node_classes,
        num_edge_classes=env.num_edge_classes,
        directed=env.is_directed,
        embedding_dim=embedding_dim,
        num_conv_layers=num_conv_layers,
        is_backward=True,
    )

    pf = DiscreteGraphPolicyEstimator(module=pf_module)
    pb = DiscreteGraphPolicyEstimator(module=pb_module, is_backward=True)
    gflownet = TBGFlowNet(pf, pb).to(device)
    return gflownet


def render_states(states: GraphStates, evaluator: Callable[[GraphStates], torch.Tensor]):
    import matplotlib.pyplot as plt
    from matplotlib import patches

    rewards = evaluator(states)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(min(8, len(states))):
        a = ax[i // 4, i % 4]
        g = states[i].tensor
        n = g.x.size(0)
        radius = 3
        xs, ys = [], []
        for j in range(n):
            ang = 2 * math.pi * j / max(1, n)
            x = radius * math.cos(ang)
            y = radius * math.sin(ang)
            xs.append(x)
            ys.append(y)
            a.add_patch(
                patches.Circle((x, y), 0.25, facecolor="none", edgecolor="black")
            )
        if g.edge_index.numel() > 0:
            for e in g.edge_index.T:
                sx, sy = xs[int(e[0])], ys[int(e[0])]
                ex, ey = xs[int(e[1])], ys[int(e[1])]
                a.plot([sx, ex], [sy, ey], color="black")
        a.set_title(f"r={rewards.view(-1)[i]:.2f}")
        a.set_xlim(-(radius + 1), radius + 1)
        a.set_ylim(-(radius + 1), radius + 1)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)
    env = init_env(device)
    gflownet = init_gflownet(env, args.embedding_dim, args.num_conv_layers, device)

    # Optimizer
    non_logz_params = [
        v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
    ]
    logz_params = (
        [dict(gflownet.named_parameters())["logZ"]]
        if "logZ" in dict(gflownet.named_parameters())
        else []
    )
    optimizer = torch.optim.Adam(
        [
            {"params": non_logz_params, "lr": args.lr},
            {"params": logz_params, "lr": args.lr_Z},
        ]
    )

    # Optional replay buffer
    replay_buffer = ReplayBuffer(
        env,
        capacity=args.batch_size,
        prioritized_capacity=True,
        prioritized_sampling=True,
    )

    epsilon = {
        GraphActions.ACTION_TYPE_KEY: args.epsilon_action_type,
        GraphActions.NODE_CLASS_KEY: args.epsilon_node_class,
        GraphActions.EDGE_CLASS_KEY: args.epsilon_edge_class,
        GraphActions.EDGE_INDEX_KEY: args.epsilon_edge_index,
    }

    t0 = time.time()
    for it in range(args.n_iterations):
        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            epsilon=epsilon,
        )
        training_samples = gflownet.to_training_samples(trajectories)
        gflownet_training_samples = training_samples

        # Buffer mix-in
        if args.use_buffer:
            with torch.no_grad():
                replay_buffer.add(training_samples)
            if it > 10:
                training_samples = training_samples[: args.batch_size // 2]
                buffer_samples = replay_buffer.sample(n_samples=args.batch_size // 2)
                training_samples.extend(buffer_samples)  # type: ignore

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
        # Quick metric: percent triangles
        final_states = gflownet_training_samples.terminating_states
        assert isinstance(final_states, GraphStates)
        rewards = env.reward(final_states)
        pct_tri = (rewards > 0.1).to(torch.get_default_dtype()).mean() * 100
        print(
            f"Iter {it:04d} | Loss {loss.item():.3f} | triangles {pct_tri.item():.1f}%"
        )
        loss.backward()
        optimizer.step()

    print(f"Training time: {(time.time() - t0)/60:.2f} min")

    if args.plot:
        samples_to_render = trajectories.terminating_states[:8]
        assert isinstance(samples_to_render, GraphStates)
        render_states(samples_to_render, env.reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GFlowNet with ADD_NODE and ADD_EDGE actions"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dim for policy heads"
    )
    parser.add_argument(
        "--num_conv_layers", type=int, default=1, help="Number of GNN layers"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--n_iterations", type=int, default=300, help="Training iterations"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lr_Z", type=float, default=5e-2, help="Learning rate for logZ"
    )
    parser.add_argument(
        "--use_buffer", action="store_true", default=True, help="Use replay buffer"
    )
    # Exploration epsilons per action component
    parser.add_argument("--epsilon_action_type", type=float, default=0.0)
    parser.add_argument("--epsilon_node_class", type=float, default=0.0)
    parser.add_argument("--epsilon_edge_class", type=float, default=0.0)
    parser.add_argument("--epsilon_edge_index", type=float, default=0.0)
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
