from copy import deepcopy
from typing import Callable, Literal, Tuple

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv

from gfn.actions import Actions
from gfn.env import GraphEnv, NonValidActionsError
from gfn.states import GraphStates


class GraphBuilding(GraphEnv):
    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        state_evaluator: Callable[[Batch], torch.Tensor] | None = None,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        s0 = Data(
            x=torch.zeros((num_nodes, node_feature_dim)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        ).to(device_str)
        exit_action = torch.tensor(
            [-float("inf"), -float("inf")], device=torch.device(device_str)
        )
        dummy_action = torch.tensor(
            [float("inf"), float("inf")], device=torch.device(device_str)
        )
        if state_evaluator is None:
            state_evaluator = GCNConvEvaluator(node_feature_dim)
        self.state_evaluator = state_evaluator

        super().__init__(
            s0=s0,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            action_shape=(2,),
            dummy_action=dummy_action,
            exit_action=exit_action,
            device_str=device_str,
        )

    def step(self, states: GraphStates, actions: Actions) -> GraphStates:
        """Step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to add.

        Returns the next graph the new GraphStates.
        """
        if not self.is_action_valid(states, actions):
            raise NonValidActionsError("Invalid action.")
        graphs: Batch = deepcopy(states.data)
        assert len(graphs) == len(actions)

        edge_index = torch.cat([graphs.edge_index, actions.tensor.T], dim=1)
        graphs.edge_index = edge_index
        return self.States(graphs)

    def backward_step(self, states: GraphStates, actions: Actions) -> GraphStates:
        """Backward step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to remove.

        Returns the previous graph as a new GraphStates.
        """
        if not self.is_action_valid(states, actions, backward=True):
            raise NonValidActionsError("Invalid action.")
        graphs: Batch = deepcopy(states.data)
        assert len(graphs) == len(actions)

        for i, act in enumerate(actions.tensor):
            edge_index = graphs[i].edge_index
            edge_index = edge_index[:, edge_index[1] != act]
            graphs[i].edge_index = edge_index

        return self.States(graphs)

    def is_action_valid(
        self, states: GraphStates, actions: Actions, backward: bool = False
    ) -> bool:
        current_edges = states.data.edge_index
        new_edges = actions.tensor

        if torch.any(new_edges[:, 0] == new_edges[:, 1]):
            return False
        if current_edges.shape[1] == 0:
            return not backward

        if backward:
            some_edges_not_exist = torch.any(
                torch.all(current_edges[:, None, :] != new_edges.T[:, :, None], dim=0)
            )
            return not some_edges_not_exist
        else:
            some_edges_exist = torch.any(
                torch.all(current_edges[:, None, :] == new_edges.T[:, :, None], dim=0)
            )
            return not some_edges_exist

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        per_node_rew = self.state_evaluator(final_states.data)
        node_batch_idx = final_states.data.batch
        return torch.bincount(node_batch_idx, weights=per_node_rew)

    @property
    def log_partition(self) -> float:
        "Returns the logarithm of the partition function."
        raise NotImplementedError

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        "Returns a one-dimensional tensor representing the true distribution."
        raise NotImplementedError

    def make_random_states_tensor(self, batch_shape: Tuple) -> GraphStates:
        """Generates random states tensor of shape (*batch_shape, num_nodes, node_feature_dim)."""
        return self.States.from_batch_shape(batch_shape)


class GCNConvEvaluator:
    def __init__(self, num_features):
        self.net = GCNConv(num_features, 1)

    def __call__(self, batch: Batch) -> torch.Tensor:
        return self.net(batch.x, batch.edge_index).squeeze(-1)
