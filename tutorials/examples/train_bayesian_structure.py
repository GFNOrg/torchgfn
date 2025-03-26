"""
The goal of this script is to reproduce the results of DAG-GFlowNet for
Bayesian structure learning (Deleu et al., 2022) using the GraphEnv.

Specifically, we consider a randomly generated (under the Erdős-Rényi model) linear-Gaussian
Bayesian network over `num_nodes` nodes. We generate 100 datapoints from it, and use them to
calculate the BGe score. The GFlowNet is learned to generate directed acyclic graphs (DAGs)
proportionally to their BGe score, using the modified DB loss.

Key components:
- BayesianStructure: Environment for Bayesian structure learning
- LinearTransformerPolicyModule: Linear transformer policy module
- ModifiedDBGFlowNet: GFlowNet with modified detailed balance loss
"""

from typing import cast

import torch
from torch import nn
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.utils import to_dense_adj
from tqdm import trange

from gfn.actions import Actions
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym.bayesian_structure import BayesianStructure
from gfn.gym.helpers.bayesian_structure.factories import get_scorer
from gfn.modules import DiscretePolicyEstimator
from gfn.preprocessors import Preprocessor
from gfn.states import GraphStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP

DEFAULT_SEED = 4444


class AdjacencyPolicyModule(nn.Module):
    """Policy network that processes flattened adjacency matrices to predict graph actions.

    Unlike the GNN-based RingPolicyModule, this module uses standard MLPs to process
    the entire adjacency matrix as a flattened vector. This approach:

    1. Can directly process global graph structure without message passing
    2. May be more effective for small graphs where global patterns are important
    3. Does not require complex graph neural network operations

    The module architecture consists of:
    - An MLP to process the flattened adjacency matrix into an embedding
    - An edge MLP that predicts logits for each possible edge action
    - An exit MLP that predicts a logit for the exit action

    Args:
        num_nodes: Number of nodes in the graph
        directed: Whether the graph is directed or undirected
        embedding_dim: Dimension of internal embeddings (default: 128)
        is_backward: Whether this is a backward policy (default: False)
    """

    def __init__(self, num_nodes: int, embedding_dim=128, is_backward=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.is_backward = is_backward
        self.hidden_dim = embedding_dim

        # MLP for processing the flattened adjacency matrix
        self.mlp = MLP(
            input_dim=num_nodes * num_nodes,  # Flattened adjacency matrix
            output_dim=embedding_dim,
            hidden_dim=embedding_dim,
            n_hidden_layers=2,
            add_layer_norm=True,
        )

        # Exit action MLP
        self.exit_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=1,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

        self.edge_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=num_nodes**2,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

    def forward(self, states_tensor: GeometricBatch) -> torch.Tensor:
        """Forward pass to compute action logits from graph states.

        Process:
        1. Convert the graph representation to adjacency matrices
        2. Process the flattened adjacency matrices through the main MLP
        3. Predict logits for edge actions and exit action

        Args:
            states_tensor: A GeometricBatch containing graph state information

        Returns:
            A tensor of logits for all possible actions
        """
        # Convert the graph to adjacency matrix
        batch_size = int(states_tensor.batch_size)

        adj_matrix_list = []
        for i in range(batch_size):
            graph = states_tensor[i]
            adj_matrix_list.append(
                to_dense_adj(graph.edge_index, max_num_nodes=self.num_nodes)
            )
        adj_matrices = torch.cat(adj_matrix_list, dim=0)
        # (batch_size, num_nodes, num_nodes)

        # Flatten the adjacency matrices for the MLP
        adj_matrices_flat = adj_matrices.view(batch_size, -1)

        # Process with MLP
        embedding = self.mlp(adj_matrices_flat)

        # Generate edge and exit actions
        edge_actions = self.edge_mlp(embedding)
        exit_action = self.exit_mlp(embedding)

        if self.is_backward:
            return edge_actions
        else:
            return torch.cat([edge_actions, exit_action], dim=-1)


class GeometricDiscreteUniform(nn.Module):
    """Implements a uniform distribution over discrete actions given a graph state.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator.

    Attributes:
        output_dim: The size of the output space.
    """

    def __init__(self, output_dim: int) -> None:
        """Initializes the uniform function approximiator.

        Args:
            output_dim (int): Output dimension. This is typically n_actions if it
                implements a Uniform PF, or n_actions-1 if it implements a Uniform PB.
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(self, preprocessed_states: GeometricBatch) -> torch.Tensor:
        """Forward method for the uniform distribution.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the uniform distribution. The shape of the tensor should be (*batch_shape, input_dim).

        Returns: a tensor of shape (*batch_shape, output_dim).
        """
        out = torch.zeros(*preprocessed_states.batch_shape, self.output_dim).to(
            preprocessed_states.x.device
        )
        return out


class GraphPreprocessor(Preprocessor):
    """Preprocessor for graph states to extract the tensor representation.

    This simple preprocessor extracts the GeometricBatch from GraphStates to make
    it compatible with the policy networks. It doesn't perform any complex
    transformations, just ensuring the tensors are accessible in the right format.

    Args:
        feature_dim: The dimension of features in the graph (default: 1)
    """

    def __init__(self, feature_dim: int = 1):
        super().__init__(output_dim=feature_dim)

    def preprocess(self, states: GraphStates) -> GeometricBatch:
        return states.tensor

    def __call__(self, states: GraphStates) -> GeometricBatch:
        return self.preprocess(states)


def get_random_action(env: BayesianStructure, states: GraphStates) -> Actions:
    """Perform a random action in the environment."""
    assert isinstance(states, env.States)

    action_masks = cast(torch.Tensor, states.forward_masks)
    # shape: (batch_size, n_actions) where n_actions = num_nodes^2 + 1

    action_probs = torch.rand(action_masks.shape, device=action_masks.device)
    action_probs = action_probs * action_masks
    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
    action_tensor = torch.multinomial(action_probs, num_samples=1)

    return env.Actions(action_tensor)  # type: ignore


def main(args):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    rng = torch.Generator(device="cpu")  # The device should be cpu
    rng.manual_seed(seed)

    # Create the scorer
    scorer, _, _ = get_scorer(
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
        num_nodes=args.num_nodes,
        state_evaluator=scorer.state_evaluator,
        device=device,
    )

    pf_module = AdjacencyPolicyModule(env.num_nodes, args.embedding_dim)
    pb_module = GeometricDiscreteUniform(env.n_actions - 1)
    pf = DiscretePolicyEstimator(
        module=pf_module, n_actions=env.n_actions, preprocessor=GraphPreprocessor()
    )
    pb = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        preprocessor=GraphPreprocessor(),
        is_backward=True,
    )
    gflownet = TBGFlowNet(pf, pb)
    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)

    # Replay buffer
    # replay_buffer = ReplayBuffer(  # TODO
    #     env,
    #     capacity=args.buffer_capacity,
    #     prioritized=args.buffer_prioritize,
    # )

    pbar = trange(args.n_iterations, dynamic_ncols=True)
    for iteration in pbar:
        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            epsilon=args.min_epsilon
            + (
                (args.max_epsilon - args.min_epsilon)
                * min(0.0, 1.0 - iteration / (args.n_iterations / 2))
            ),  # Schedule for the first half of training
        )
        training_samples = gflownet.to_training_samples(trajectories)

        # if args.use_buffer:  # TODO
        #     with torch.no_grad():
        #         replay_buffer.add(training_samples)
        #         if iteration >= args.prefill:
        #             training_samples = training_samples[: args.batch_size // 2]
        #             buffer_samples = replay_buffer.sample(
        #                 n_trajectories=args.batch_size // 2
        #             )
        #             training_samples.extend(buffer_samples)  # type: ignore

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
        loss.backward()
        optimizer.step()

        assert training_samples.log_rewards is not None
        pbar.set_postfix(
            {
                "Loss": loss.item(),
                "log_r_mean": training_samples.log_rewards.mean().item(),
            },
        )

        # TODO: Add metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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

    # GFlowNet and policy parameters
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--max_epsilon", type=float, default=0.5)
    parser.add_argument("--min_epsilon", type=float, default=0.1)

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)

    # Misc parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    main(args)
