"""
The goal of this script is to reproduce the results of DAG-GFlowNet for
Bayesian structure learning (Deleu et al., 2022) using the GraphEnv.

Specifically, we consider a randomly generated (under the Erdős-Rényi model) linear-Gaussian
Bayesian network over `num_nodes` nodes. We generate 100 datapoints from it, and use them to
calculate the BGe score. The GFlowNet is learned to generate directed acyclic graphs (DAGs)
proportionally to their BGe score, using the modified DB loss.

Some expected results on the Erdős-Rényi model with 5 nodes and 5 edges:

(On-policy training)
python train_bayesian_structure.py \
    --seed 0 \
    --no_buffer \
    --max_epsilon 0.0 \
    --min_epsilon 0.0
>> Expected SHD: 6.99
>> Expected edges: 10.00
>> ROC-AUC: 0.68
>> Jensen-Shannon divergence: 0.36

(Off-policy training with epsilon-noisy exploration, which is the default)
python train_bayesian_structure.py --seed 0
>> Expected SHD: 6.31
>> Expected edges: 8.80
>> ROC-AUC: 0.78
>> Jensen-Shannon divergence: 0.02
"""

from argparse import ArgumentParser, Namespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.nn import global_add_pool
from tqdm import trange

from gfn.actions import GraphActions, GraphActionType
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.containers.trajectories import Trajectories
from gfn.estimators import DiscreteGraphPolicyEstimator
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym.bayesian_structure import BayesianStructure
from gfn.gym.helpers.bayesian_structure.evaluation import (
    expected_edges,
    expected_shd,
    posterior_estimate,
    threshold_metrics,
)
from gfn.gym.helpers.bayesian_structure.factories import get_scorer
from gfn.gym.helpers.bayesian_structure.jsd import (
    get_full_posterior,
    get_gfn_exact_posterior,
    jensen_shannon_divergence,
    posterior_exact,
)
from gfn.utils.common import set_seed
from gfn.utils.modules import GraphActionGNN, GraphActionUniform, GraphEdgeActionMLP


class DAGEdgeActionMLP(GraphEdgeActionMLP):

    def __init__(
        self,
        n_nodes: int,
        num_node_classes: int,
        num_edge_classes: int,
        n_hidden_layers: int = 2,
        n_hidden_layers_exit: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        super().__init__(
            n_nodes=n_nodes,
            num_node_classes=num_node_classes,
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


class DAGEdgeActionGNN(GraphActionGNN):
    """Simple GNN-based edge action module

    Args:
        n_nodes: The number of nodes in the graph.
        num_edge_classes: The number of edge classes.
        num_conv_layers: The number of GNN layers.
        embedding_dim: The dimension of embeddings.
        is_backward: Whether the module is used for backward action prediction.
    """

    def __init__(
        self,
        n_nodes: int,
        num_node_classes: int,
        num_edge_classes: int,
        num_conv_layers: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        self.n_nodes = n_nodes
        super().__init__(
            num_node_classes=num_node_classes,
            directed=True,
            num_edge_classes=num_edge_classes,
            num_conv_layers=num_conv_layers,
            embedding_dim=embedding_dim,
            is_backward=is_backward,
        )

    @property
    def edges_dim(self) -> int:
        return self.num_node_classes**2

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        node_features, batch_ptr = (states_tensor.x, states_tensor.ptr)

        # Multiple action type convolutions with residual connections.
        x = self.embedding(node_features.squeeze().long())
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

        x_reshaped = x.reshape(len(states_tensor), self.n_nodes, self.hidden_dim)

        feature_dim = self.hidden_dim // 2
        source_features = x_reshaped[..., :feature_dim]
        target_features = x_reshaped[..., feature_dim:]

        # Dot product between source and target features (asymmetric).
        edgewise_dot_prod = torch.einsum(
            "bnf,bmf->bnm", source_features, target_features
        )
        edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(torch.tensor(feature_dim))

        # Grab the needed elems from the adjacency matrix and reshape.
        edge_actions = edgewise_dot_prod.flatten(1, 2)
        assert edge_actions.shape == (len(states_tensor), self.edges_dim)

        action_type = torch.ones(
            len(states_tensor), 3, device=x_reshaped.device
        ) * float("-inf")
        if self.is_backward:
            action_type[..., GraphActionType.ADD_EDGE] = 1
        else:
            # This MLP computes the exit action.
            node_feature_means = self._group_mean(x, batch_ptr)
            exit_action = self.action_type_mlp(node_feature_means)[
                ..., GraphActionType.EXIT
            ]
            action_type[..., GraphActionType.ADD_EDGE] = F.logsigmoid(-exit_action)
            action_type[..., GraphActionType.EXIT] = F.logsigmoid(exit_action)

        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.EDGE_CLASS_KEY: torch.zeros(
                    len(states_tensor), self.num_edge_classes, device=x.device
                ),  # TODO: make it learnable.
                GraphActions.NODE_CLASS_KEY: torch.zeros(
                    len(states_tensor), self.num_node_classes, device=x.device
                ),
                GraphActions.NODE_INDEX_KEY: torch.zeros(
                    len(states_tensor), self.n_nodes, device=x.device
                ),
                GraphActions.EDGE_INDEX_KEY: edge_actions,
            },
            batch_size=len(states_tensor),
        )


class DAGEdgeActionGNNv2(nn.Module):
    """
    GNN-based edge action module, adapted from the implementation of
    https://github.com/GFNOrg/GFN_vs_HVI/blob/master/dags/dag_gflownet/nets/gnn/gflownet.py

    Args:
        n_nodes: The number of nodes in the graph.
        num_edge_classes: The number of edge classes.
        num_conv_layers: The number of GNN layers.
        embedding_dim: The dimension of embeddings.
        num_heads: The number of attention heads.
        is_backward: Whether the module is used for backward action prediction.
    """

    def __init__(
        self,
        n_nodes: int,
        num_node_classes: int,
        num_edge_classes: int,
        num_conv_layers: int = 2,
        embedding_dim: int = 128,
        num_heads: int = 4,
        is_backward: bool = False,
    ):
        super().__init__()

        assert n_nodes > 0, "n_nodes must be greater than 0"
        assert embedding_dim > 0, "embedding_dim must be greater than 0"
        assert isinstance(n_nodes, int), "n_nodes must be an integer"
        assert isinstance(embedding_dim, int), "embedding_dim must be an integer"
        assert isinstance(is_backward, bool), "is_backward must be a boolean"
        self._input_dim = 1  # Each node input is a single integer before embedding.
        self._n_nodes = n_nodes
        self.num_node_classes = num_node_classes
        self.embedding_dim = embedding_dim
        self.is_backward = is_backward
        self.num_edge_classes = num_edge_classes

        self._output_dim = self.n_nodes**2
        if not self.is_backward:
            self._output_dim += 1  # +1 for exit action.

        self.node_embedding = nn.Embedding(num_node_classes, embedding_dim)
        self.edge_embedding = nn.Parameter(
            torch.randn(1, embedding_dim).clamp(min=-2.0, max=2.0)
        )

        self.graph_network = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "node_mlp": nn.Sequential(
                            nn.Linear(embedding_dim * 2 + embedding_dim, embedding_dim),
                            nn.ReLU(),
                            nn.Linear(embedding_dim, embedding_dim),
                        ),
                        "edge_mlp": nn.Sequential(
                            nn.Linear(embedding_dim * 2 + embedding_dim, embedding_dim),
                            nn.ReLU(),
                            nn.Linear(embedding_dim, embedding_dim),
                        ),
                        "global_mlp": nn.Sequential(
                            nn.Linear(embedding_dim * 2 + embedding_dim, embedding_dim),
                            nn.ReLU(),
                            nn.Linear(embedding_dim, embedding_dim),
                        ),
                    }
                )
                for _ in range(num_conv_layers)
            ]
        )

        self.projection = nn.Linear(embedding_dim, embedding_dim * 3)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

        self.senders_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.receivers_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.stop_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def edges_dim(self) -> int:
        return self.n_nodes**2

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        node_features, edge_index, batch_ptr = (
            states_tensor.x,
            states_tensor.edge_index,
            states_tensor.ptr,
        )
        # node_features: (n_graphs * n_nodes, 1)
        # edge_index: (2, \sum_{i=1}^{n_graphs} n_edges_i)
        # batch_ptr: (n_graphs + 1)
        n_graphs = batch_ptr.shape[0] - 1

        # batch_ptr to batch_idx
        batch_idx = torch.repeat_interleave(
            torch.arange(n_graphs, device=batch_ptr.device),
            batch_ptr[1:] - batch_ptr[:-1],
        )
        # batch_idx: (n_graphs * n_nodes)

        node_embs = self.node_embedding(node_features.squeeze())
        # node_embs: (n_graphs * n_nodes, embedding_dim)

        edge_embs = self.edge_embedding.repeat(edge_index.shape[1], 1)
        # edge_embs: (\sum_{i=1}^{n_graphs} n_edges_i, embedding_dim)

        globals = torch.zeros(n_graphs, self.embedding_dim, device=node_embs.device)
        # globals: (n_graphs, embedding_dim)

        for layer in self.graph_network:
            # Edge update
            edge_input = torch.cat(
                [node_embs[edge_index[0]], node_embs[edge_index[1]], edge_embs],
                dim=-1,
            )
            # Note the skip-connection
            edge_embs = edge_embs + layer["edge_mlp"](edge_input)  # pyright: ignore

            # Node update
            node_aggr = global_add_pool(
                edge_embs, edge_index[1], size=node_embs.shape[0]
            )
            node_input = torch.cat([node_embs, node_aggr, globals[batch_idx]], dim=-1)
            node_embs = node_embs + layer["node_mlp"](node_input)  # pyright: ignore
            # Global update
            edge_aggr_global = global_add_pool(
                global_add_pool(edge_embs, edge_index[1], size=node_embs.shape[0]),
                batch_idx,
                size=n_graphs,
            )
            node_aggr_global = global_add_pool(node_embs, batch_idx, size=n_graphs)

            global_input = torch.cat(
                [node_aggr_global, edge_aggr_global, globals], dim=-1
            )
            globals = globals + layer["global_mlp"](global_input)  # pyright: ignore

        # Reshape the node features, and project into keys, queries, and values
        node_embs = node_embs.reshape(n_graphs, self.n_nodes, self.embedding_dim)
        node_embs = self.projection(node_embs)
        queries, keys, values = torch.chunk(node_embs, 3, dim=-1)
        # queries: (n_graphs, n_nodes, embedding_dim)

        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)
        # (n_nodes, n_graphs, embedding_dim)

        attn_output, _ = self.attention(queries, keys, values)
        attn_output = attn_output.transpose(0, 1)
        # (n_nodes, n_graphs, embedding_dim)

        senders = self.senders_mlp(attn_output)
        receivers = self.receivers_mlp(attn_output)
        edge_actions = torch.bmm(senders, receivers.transpose(1, 2)).view(n_graphs, -1)

        temperature = nn.functional.softplus(self.temperature)
        edge_actions = edge_actions / temperature

        # Make TensorDict output
        action_type = torch.ones(len(states_tensor), 3, device=node_embs.device) * float(
            "-inf"
        )
        if self.is_backward:
            action_type[..., GraphActionType.ADD_EDGE] = 0.0  # log(1.0)
        else:
            stop_logits = self.stop_mlp(globals).squeeze(-1)
            action_type[..., GraphActionType.ADD_EDGE] = F.logsigmoid(-stop_logits)
            action_type[..., GraphActionType.EXIT] = F.logsigmoid(stop_logits)

        return TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: action_type,
                GraphActions.EDGE_CLASS_KEY: torch.zeros(
                    len(states_tensor),
                    self.num_edge_classes,
                    device=node_embs.device,
                ),  # TODO: make it learnable.
                GraphActions.NODE_CLASS_KEY: torch.zeros(
                    len(states_tensor), self.num_node_classes, device=node_embs.device
                ),
                GraphActions.NODE_INDEX_KEY: torch.zeros(
                    len(states_tensor), self.n_nodes, device=node_embs.device
                ),
                GraphActions.EDGE_INDEX_KEY: edge_actions,
            },
            batch_size=len(states_tensor),
        )


def main(args: Namespace):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Create the scorer
    scorer, data, gt_graph = get_scorer(
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

    if args.module == "mlp":
        pf_module = DAGEdgeActionMLP(
            n_nodes=env.n_nodes,
            num_node_classes=env.num_node_classes,
            num_edge_classes=env.num_edge_classes,
            n_hidden_layers=args.num_layers,
            n_hidden_layers_exit=1,
            embedding_dim=args.embedding_dim,
        )
    elif args.module == "gnn":
        pf_module = DAGEdgeActionGNN(
            n_nodes=env.n_nodes,
            num_node_classes=env.num_node_classes,
            num_edge_classes=env.num_edge_classes,
            num_conv_layers=args.num_layers,
            embedding_dim=args.embedding_dim,
        )
    elif args.module == "gnn_v2":
        pf_module = DAGEdgeActionGNNv2(
            n_nodes=env.n_nodes,
            num_node_classes=env.num_node_classes,
            num_edge_classes=env.num_edge_classes,
            num_conv_layers=args.num_layers,
            embedding_dim=args.embedding_dim,
            num_heads=4,
        )
    else:
        raise ValueError(f"Invalid module: {args.module}")

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
    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.n_iterations * args.n_steps_per_iteration * 0.5)],
        gamma=0.1,
    )

    optimizer_logZ = None
    if "logZ" in dict(gflownet.named_parameters()):
        params_logZ = [
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        ]
        optimizer_logZ = torch.optim.SGD(params_logZ, momentum=0.8)

    # Replay buffer
    replay_buffer = None
    if args.use_buffer:
        replay_buffer = ReplayBuffer(env, capacity=args.buffer_capacity)

    epsilon_dict = defaultdict(float)
    total_niter = args.n_iterations + (args.prefill if args.use_buffer else 0)
    pbar = trange(total_niter, dynamic_ncols=True)
    for it in pbar:
        # Schedule epsilon throughout training
        eps = args.min_epsilon + (
            (args.max_epsilon - args.min_epsilon)
            * max(0.0, 1.0 - it / (total_niter // 2))
        )
        epsilon_dict[GraphActions.ACTION_TYPE_KEY] = eps / (args.num_nodes**2)
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

        for _ in range(args.n_steps_per_iteration):
            if args.use_buffer:
                assert replay_buffer is not None
                training_samples = replay_buffer.sample(n_samples=args.batch_size)
            else:
                training_samples = _training_samples
            assert isinstance(training_samples, Trajectories)

            optimizer.zero_grad()
            if optimizer_logZ is not None:
                optimizer_logZ.zero_grad()
            loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if optimizer_logZ is not None:
                optimizer_logZ.step()

        assert training_samples.log_rewards is not None
        postfix = {
            "Loss": loss.item(),
            "log_r_mean": training_samples.log_rewards.mean().item(),
        }
        if isinstance(gflownet.logZ, nn.Parameter):
            postfix["logZ"] = gflownet.logZ.item()
        pbar.set_postfix(postfix)

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

    ### Jensen-Shannon divergence
    if env.n_nodes < 6:
        full_posterior = get_full_posterior(
            scorer, data, env, gt_graph.node_names, verbose=False
        )
        exact_posterior = get_gfn_exact_posterior(
            posterior_exact(
                env, gflownet.pf, gt_graph.node_names, batch_size=args.batch_size
            )
        )
        jsd = jensen_shannon_divergence(full_posterior, exact_posterior)
        print(f"Jensen-Shannon divergence: {jsd}")


if __name__ == "__main__":
    parser = ArgumentParser(
        "Train a GFlowNet to generate a DAG for Bayesian structure learning."
    )
    # Environment parameters
    parser.add_argument("--num_nodes", type=int, default=5)
    parser.add_argument(
        "--num_edges",
        type=int,
        default=5,
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
    parser.add_argument(
        "--module",
        type=str,
        default="gnn_v2",
        choices=["mlp", "gnn", "gnn_v2"],
    )
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--max_epsilon", type=float, default=1.0)
    parser.add_argument("--min_epsilon", type=float, default=0.1)

    # Replay buffer parameters
    parser.add_argument("--no_buffer", dest="use_buffer", action="store_false")
    parser.add_argument("--buffer_capacity", type=int, default=100000)
    parser.add_argument("--prefill", type=int, default=100)
    parser.add_argument("--sampling_batch_size", type=int, default=32)

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_Z", type=float, default=1e-1)
    parser.add_argument("--n_iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_steps_per_iteration", type=int, default=1)

    # Misc parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    if not args.use_buffer:
        assert args.sampling_batch_size == args.batch_size

    main(args)
