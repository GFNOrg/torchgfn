"""
The goal of this script is to reproduce the results of DAG-GFlowNet for
Bayesian structure learning (Deleu et al., 2022) using the GraphEnv.

Specifically, we consider a randomly generated (under the Erdős-Rényi model) linear-Gaussian
Bayesian network over `num_nodes` nodes. We generate 100 datapoints from it, and use them to
calculate the BGe score. The GFlowNet is learned to generate directed acyclic graphs (DAGs)
proportionally to their BGe score, using the modified DB loss.

Some expected results on the Erdős-Rényi model with 4 nodes and 4 edges:

(On-policy training)
python train_bayesian_structure.py \
    --no_buffer \
    --batch_size 32 \
    --sampling_batch_size 32 \
    --use_gnn \
    --max_epsilon 0.0 \
    --min_epsilon 0.0
>> Expected SHD: ~5.0
>> Expected edges: ~6.0
>> ROC-AUC: ~0.7

(Off-policy training with epsilon-noisy exploration)
python train_bayesian_structure.py \
    --no_buffer \
    --batch_size 32 \
    --sampling_batch_size 32 \
    --use_gnn \
    --max_epsilon 0.9 \
    --min_epsilon 0.0
>> Expected SHD: ~4.2
>> Expected edges: ~5.2
>> ROC-AUC: ~0.9
"""

from argparse import ArgumentParser, Namespace
from collections import defaultdict

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torch_geometric.data import Batch as GeometricBatch
from tqdm import trange

from gfn.actions import GraphActions, GraphActionType
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.containers.trajectories import Trajectories
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym.bayesian_structure import BayesianStructure
from gfn.gym.helpers.bayesian_structure.evaluation import (
    expected_edges,
    expected_shd,
    posterior_estimate,
    threshold_metrics,
)
from gfn.gym.helpers.bayesian_structure.factories import get_scorer
from gfn.modules import DiscreteGraphPolicyEstimator
from gfn.utils.common import set_seed
from gfn.utils.modules import GraphActionUniform, GraphEdgeActionGNN, GraphEdgeActionMLP


class DAGEdgeActionMLP(GraphEdgeActionMLP):

    def __init__(
        self,
        n_nodes: int,
        num_edge_classes: int,
        n_hidden_layers: int = 2,
        n_hidden_layers_exit: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        super().__init__(
            n_nodes=n_nodes,
            directed=True,
            num_edge_classes=num_edge_classes,
            n_hidden_layers=n_hidden_layers,
            n_hidden_layers_exit=n_hidden_layers_exit,
            embedding_dim=embedding_dim,
            is_backward=is_backward,
        )

    @property
    def edges_dim(self) -> int:
        return self.n_nodes**2


class DAGEdgeActionGNN(GraphEdgeActionGNN):
    """Simple module which outputs a fixed logits for the actions, depending on the number of nodes.

    Args:
        n_nodes: The number of nodes in the graph.
    """

    def __init__(
        self,
        n_nodes: int,
        num_edge_classes: int,
        num_conv_layers: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        super().__init__(
            n_nodes=n_nodes,
            directed=True,
            num_edge_classes=num_edge_classes,
            num_conv_layers=num_conv_layers,
            embedding_dim=embedding_dim,
            is_backward=is_backward,
        )

    @property
    def edges_dim(self) -> int:
        return self.n_nodes**2

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        node_features, batch_ptr = (states_tensor.x, states_tensor.ptr)

        # Multiple action type convolutions with residual connections.
        x = self.embedding(node_features.squeeze().int())
        for i in range(0, len(self.conv_blks), 2):
            x_new = self.conv_blks[i](x, states_tensor.edge_index)  # GIN/GCN conv.
            assert isinstance(self.conv_blks[i + 1], nn.ModuleList)
            x_in, x_out = torch.chunk(x_new, 2, dim=-1)

            # Process each component separately through its own MLP.
            mlp_in, mlp_out = self.conv_blks[i + 1]
            x_in = mlp_in(x_in)
            x_out = mlp_out(x_out)
            x_new = torch.cat([x_in, x_out], dim=-1)

            x = x_new + x if i > 0 else x_new  # Residual connection.
            x = self.norm(x)  # Layernorm.

        # This MLP computes the exit action.
        node_feature_means = self._group_mean(x, batch_ptr)
        if not self.is_backward:
            exit_action = self.exit_mlp(node_feature_means).squeeze(-1)

        x = x.reshape(*states_tensor.batch_shape, self.n_nodes, self.hidden_dim)

        feature_dim = self.hidden_dim // 2
        source_features = x[..., :feature_dim]
        target_features = x[..., feature_dim:]

        # Dot product between source and target features (asymmetric).
        edgewise_dot_prod = torch.einsum(
            "bnf,bmf->bnm", source_features, target_features
        )
        edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(torch.tensor(feature_dim))

        # Grab the needed elems from the adjacency matrix and reshape.
        edge_actions = edgewise_dot_prod.flatten(1, 2)
        assert edge_actions.shape == (*states_tensor.batch_shape, self.edges_dim)

        action_type = torch.zeros(*states_tensor.batch_shape, 3, device=x.device)
        if self.is_backward:
            action_type[..., GraphActionType.ADD_EDGE] = 1
        else:
            action_type[..., GraphActionType.ADD_EDGE] = 1 - exit_action
            action_type[..., GraphActionType.EXIT] = exit_action

        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.EDGE_CLASS_KEY: torch.zeros(
                    *states_tensor.batch_shape, self.num_edge_classes, device=x.device
                ),  # TODO: make it learnable.
                GraphActions.NODE_CLASS_KEY: torch.zeros(
                    *states_tensor.batch_shape, 1, device=x.device
                ),
                GraphActions.EDGE_INDEX_KEY: edge_actions,
            },
            batch_size=states_tensor.batch_shape,
        )


def main(args: Namespace):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Create the scorer
    scorer, _, gt_graph = get_scorer(
        args.graph_name,
        args.prior_name,
        args.num_nodes,
        args.num_edges,
        args.num_samples,
        args.node_names,
        rng=rng,
    )

    # Create the environment
    env = BayesianStructure(
        n_nodes=args.num_nodes,
        state_evaluator=scorer.state_evaluator,
        device=device,
    )

    if args.use_gnn:
        pf_module = DAGEdgeActionGNN(
            env.n_nodes,
            env.num_edge_classes,
            args.num_layers,
            args.embedding_dim,
        )
    else:
        pf_module = DAGEdgeActionMLP(
            env.n_nodes,
            env.num_edge_classes,
            args.num_layers,
            1,
            args.embedding_dim,
        )

    pb_module = GraphActionUniform(
        env.n_actions - 1,  # equivalent to env.n_nodes**2
        env.num_edge_classes,
        env.num_node_classes,
    )
    pf = DiscreteGraphPolicyEstimator(
        module=pf_module,
    )
    pb = DiscreteGraphPolicyEstimator(
        module=pb_module,
        is_backward=True,
    )
    gflownet = TBGFlowNet(pf, pb)
    gflownet = gflownet.to(device)

    # Log Z gets dedicated learning rate (typically higher).
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )
    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.n_iterations * 0.5), int(args.n_iterations * 0.8)],
        gamma=0.1,
    )

    # Replay buffer
    replay_buffer = None
    if args.use_buffer:
        replay_buffer = ReplayBuffer(
            env,
            capacity=args.buffer_capacity,
            prioritized=args.buffer_prioritize,
        )

    epsilon_dict = defaultdict(float)
    pbar = trange(args.n_iterations, dynamic_ncols=True)
    for it in pbar:
        # Schedule epsilon for the first half of training
        eps = args.min_epsilon + (
            (args.max_epsilon - args.min_epsilon)
            * max(0.0, 1.0 - it / (args.n_iterations / 2))
        )
        epsilon_dict[GraphActions.ACTION_TYPE_KEY] = eps
        epsilon_dict[GraphActions.EDGE_INDEX_KEY] = eps

        trajectories = gflownet.sample_trajectories(
            env,
            n=args.sampling_batch_size,
            save_logprobs=True if not args.use_buffer else False,
            epsilon=epsilon_dict,
        )
        _training_samples = gflownet.to_training_samples(trajectories)

        if args.use_buffer:
            assert replay_buffer is not None
            with torch.no_grad():
                replay_buffer.add(_training_samples)
            if it < args.prefill or len(replay_buffer) < args.batch_size:
                continue
            training_samples = replay_buffer.sample(n_trajectories=args.batch_size)
        else:
            training_samples = _training_samples
        assert isinstance(training_samples, Trajectories)  # TODO: Other containers

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        assert training_samples.log_rewards is not None
        pbar.set_postfix(
            {
                "Loss": loss.item(),
                "log_r_mean": training_samples.log_rewards.mean().item(),
            },
        )

    # Compute the metrics
    with torch.no_grad():
        posterior_samples = posterior_estimate(
            gflownet,
            env,
            num_samples=args.num_samples_posterior,
            batch_size=args.sampling_batch_size,
        )

    print(f"Expected SHD: {expected_shd(posterior_samples, gt_graph)}")
    print(f"Expected edges: {expected_edges(posterior_samples)}")
    thres_metrics = threshold_metrics(posterior_samples, gt_graph)
    for k, v in thres_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    parser = ArgumentParser(
        "Train a GFlowNet to generate a DAG for Bayesian structure learning."
    )
    # Environment parameters
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument(
        "--num_edges",
        type=int,
        default=4,
        help="Number of edges in the sampled erdos renyi graph",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples."
    )
    parser.add_argument("--graph_name", type=str, default="erdos_renyi_lingauss")
    parser.add_argument("--prior_name", type=str, default="uniform")
    parser.add_argument(
        "--node_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of node names.",
    )
    parser.add_argument(
        "--num_samples_posterior",
        type=int,
        default=1000,
        help="Number of samples for posterior approximation.",
    )

    # GFlowNet and policy parameters
    parser.add_argument("--use_gnn", action="store_true")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--max_epsilon", type=float, default=0.9)
    parser.add_argument("--min_epsilon", type=float, default=0.0)

    # Replay buffer parameters
    parser.add_argument("--no_buffer", dest="use_buffer", action="store_false")
    parser.add_argument("--buffer_capacity", type=int, default=1000)
    parser.add_argument("--buffer_prioritize", action="store_true")
    parser.add_argument("--prefill", type=int, default=30)
    parser.add_argument("--sampling_batch_size", type=int, default=32)

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_Z", type=float, default=1e-1)
    parser.add_argument("--n_iterations", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)

    # Misc parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    if not args.use_buffer:
        assert args.sampling_batch_size == args.batch_size

    main(args)
