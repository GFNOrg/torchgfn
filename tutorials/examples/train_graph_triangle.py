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
import torch.nn as nn
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.nn import DirGNNConv, GCNConv, GINConv

from gfn.actions import GraphActions, GraphActionType
from gfn.containers import ReplayBuffer
from gfn.estimators import DiscreteGraphPolicyEstimator
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym.graph_building import GraphBuilding
from gfn.states import GraphStates
from gfn.utils.common import set_seed
from gfn.utils.graphs import get_edge_indices
from gfn.utils.modules import MLP


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


class GraphNodeEdgeAction(nn.Module):
    """GNN-based graph policy head for ADD_NODE, ADD_EDGE, and EXIT.

    This mirrors the structure of `GraphEdgeActionGNN` while handling dynamic
    graph sizes and producing logits for node/edge classes and add-node actions.
    """

    def __init__(
        self,
        num_node_classes: int,
        num_edge_classes: int,
        directed: bool,
        embedding_dim: int = 128,
        num_conv_layers: int = 1,
        is_backward: bool = False,
    ) -> None:
        super().__init__()
        assert num_node_classes > 0 and num_edge_classes > 0
        assert embedding_dim > 0 and num_conv_layers > 0
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.directed = directed
        self.hidden_dim = embedding_dim
        self.is_backward = is_backward
        self.input_dim = 1

        # Node class embedding lookup
        self.embedding = nn.Embedding(num_node_classes, embedding_dim)

        # Message passing stacks (similar to GraphEdgeActionGNN)
        self.conv_blks = nn.ModuleList()
        if directed:
            for i in range(num_conv_layers):
                self.conv_blks.extend(
                    [
                        DirGNNConv(
                            GCNConv(
                                embedding_dim if i == 0 else self.hidden_dim,
                                self.hidden_dim,
                            ),
                            alpha=0.5,
                            root_weight=True,
                        ),
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
                                for _ in range(2)
                            ]
                        ),
                    ]
                )
        else:
            for i in range(num_conv_layers):
                self.conv_blks.extend(
                    [
                        GINConv(
                            MLP(
                                input_dim=embedding_dim if i == 0 else self.hidden_dim,
                                output_dim=self.hidden_dim,
                                hidden_dim=self.hidden_dim,
                                n_hidden_layers=1,
                                add_layer_norm=True,
                            )
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
        self.exit_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=1,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )
        self.add_node_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=1,
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
        device = states_tensor.x.device
        B = len(states_tensor)

        # Embed node classes
        if states_tensor.x.numel() > 0:
            x = self.embedding(states_tensor.x.squeeze(-1))
        else:
            x = torch.zeros(0, self.hidden_dim, device=device)

        # Message passing with residual connections and layer norms
        for i in range(0, len(self.conv_blks), 2):
            if len(self.conv_blks) == 0:
                break
            x_new = self.conv_blks[i](x, states_tensor.edge_index)
            if self.directed:
                assert isinstance(self.conv_blks[i + 1], nn.ModuleList)
                x_in, x_out = torch.chunk(x_new, 2, dim=-1)
                mlp_in, mlp_out = self.conv_blks[i + 1]
                x_in = mlp_in(x_in)
                x_out = mlp_out(x_out)
                x_new = torch.cat([x_in, x_out], dim=-1)
            else:
                x_new = self.conv_blks[i + 1](x_new)
            x = x_new + x if i > 0 else x_new
            x = self.norm(x)

        # Pool to graph features
        graph_emb = (
            self._group_mean(x, states_tensor.ptr)
            if x.numel() > 0
            else torch.zeros(B, self.hidden_dim, device=device)
        )

        # Action type logits
        action_type = torch.ones(B, 3, device=device) * float("-inf")
        add_node_logit = self.add_node_mlp(graph_emb).squeeze(-1)
        action_type[..., GraphActionType.ADD_NODE] = add_node_logit
        action_type[..., GraphActionType.ADD_EDGE] = 0.0
        if not self.is_backward:
            exit_logit = self.exit_mlp(graph_emb).squeeze(-1)
            action_type[..., GraphActionType.EXIT] = exit_logit

        # Class logits
        node_class_logits = self.node_class_mlp(graph_emb)
        edge_class_logits = self.edge_class_mlp(graph_emb)

        # Edge-index logits via pairwise dot products
        # Pad to max_nodes across batch for gathering candidate edges
        max_nodes = 0
        for b in range(B):
            start, end = int(states_tensor.ptr[b].item()), int(
                states_tensor.ptr[b + 1].item()
            )
            max_nodes = max(max_nodes, end - start)
        if max_nodes == 0:
            edge_index_logits = torch.zeros(B, 0, device=device)
        else:
            padded = torch.zeros(B, max_nodes, self.hidden_dim, device=device)
            for b in range(B):
                start, end = int(states_tensor.ptr[b].item()), int(
                    states_tensor.ptr[b + 1].item()
                )
                if end > start:
                    padded[b, : (end - start)] = x[start:end]

            if self.directed:
                feature_dim = self.hidden_dim // 2
                source_features = padded[..., :feature_dim]
                target_features = padded[..., feature_dim:]
                scores = torch.einsum("bnf,bmf->bnm", source_features, target_features)
                scores = scores / math.sqrt(max(1, feature_dim))
            else:
                scores = torch.einsum("bnf,bmf->bnm", padded, padded)
                scores = scores / math.sqrt(max(1, self.hidden_dim))

            ei0, ei1 = get_edge_indices(max_nodes, self.directed, device)
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
    pf_module = GraphNodeEdgeAction(
        num_node_classes=env.num_node_classes,
        num_edge_classes=env.num_edge_classes,
        directed=env.is_directed,
        embedding_dim=embedding_dim,
        num_conv_layers=num_conv_layers,
        is_backward=False,
    )
    pb_module = GraphNodeEdgeAction(
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
        "--embedding_dim", type=int, default=64, help="Embedding dim for policy heads"
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
        "--lr_Z", type=float, default=1e-1, help="Learning rate for logZ"
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
