from copy import deepcopy
from typing import Callable, Literal, Tuple

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv

from gfn.actions import GraphActions, GraphActionType
from gfn.env import GraphEnv, NonValidActionsError
from gfn.states import GraphStates


class GraphBuilding(GraphEnv):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        state_evaluator: Callable[[Batch], torch.Tensor] | None = None,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        s0 = Data().to(device_str)

        if state_evaluator is None:
            state_evaluator = GCNConvEvaluator(node_feature_dim)
        self.state_evaluator = state_evaluator

        super().__init__(
            s0=s0,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            device_str=device_str,
        )

    def step(self, states: GraphStates, actions: GraphActions) -> GraphStates:
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

        if actions.action_type == GraphActionType.ADD_NODE:
            if graphs.x is None:
                graphs.x = actions.features
            else:
                graphs.x = torch.cat([graphs.x, actions.features])

        if actions.action_type == GraphActionType.ADD_EDGE:
            assert actions.edge_index is not None
            if graphs.edge_attr is None:
                graphs.edge_attr = actions.features
                assert graphs.edge_index is None
                graphs.edge_index = actions.edge_index
            else:
                graphs.edge_attr = torch.cat([graphs.edge_attr, actions.features])
                graphs.edge_index = torch.cat([graphs.edge_index, actions.edge_index], dim=1)

        return self.States(graphs)


    def backward_step(self, states: GraphStates, actions: GraphActions) -> GraphStates:
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

        if actions.action_type == GraphActionType.ADD_NODE:
            assert graphs.x is not None
            is_equal = torch.any(
                torch.all(graphs.x[:, None] == actions.features, dim=-1),
                dim=-1
            )
            graphs.x = graphs.x[~is_equal]
        elif actions.action_type == GraphActionType.ADD_EDGE:
            assert actions.edge_index is not None
            is_equal = torch.all(graphs.edge_index[:, None] == actions.edge_index[:, :, None], dim=0)
            is_equal = torch.any(is_equal, dim=0)
            graphs.edge_attr = graphs.edge_attr[~is_equal]
            graphs.edge_index = graphs.edge_index[:, ~is_equal]

        return self.States(graphs)

    def is_action_valid(
        self, states: GraphStates, actions: GraphActions, backward: bool = False
    ) -> bool:
        if actions.action_type == GraphActionType.ADD_NODE:
            if actions.edge_index is not None:
                return False
            if states.data.x is None:
                return not backward
            
            equal_nodes_per_batch = torch.all(
                states.data.x == actions.features[:, None], dim=-1
            ).reshape(states.data.batch_size, -1)
            equal_nodes_per_batch = torch.sum(equal_nodes_per_batch, dim=-1)

            if backward:  # TODO: check if no edge are connected?
                return torch.all(equal_nodes_per_batch == 1)
            return torch.all(equal_nodes_per_batch == 0)
        
        if actions.action_type == GraphActionType.ADD_EDGE:
            assert actions.edge_index is not None
            if torch.any(actions.edge_index[0] == actions.edge_index[1]):
                return False
            if states.data.num_nodes is None or states.data.num_nodes == 0:
                return False
            if torch.any(actions.edge_index > states.data.num_nodes):
                return False
            
            batch_idx = actions.edge_index % actions.batch_shape[0] 
            if torch.any(batch_idx != torch.arange(actions.batch_shape[0])):
                return False
            if states.data.edge_attr is None:
                return True

            equal_edges_per_batch_attr = torch.all(
                states.data.edge_attr == actions.features[:, None], dim=-1
            ).reshape(states.data.batch_size, -1)
            equal_edges_per_batch_attr = torch.sum(equal_edges_per_batch_attr, dim=-1)
    
            equal_edges_per_batch_index = torch.all(
                states.data.edge_index[:, None] == actions.edge_index[:, :, None], dim=0
            ).reshape(states.data.batch_size, -1)
            equal_edges_per_batch_index = torch.sum(equal_edges_per_batch_index, dim=-1)
            if backward:
                return torch.all(equal_edges_per_batch_attr == 1) and torch.all(equal_edges_per_batch_index == 1)
            return torch.all(equal_edges_per_batch_attr == 0) and torch.all(equal_edges_per_batch_index == 0)


    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        return self.state_evaluator(final_states.data)

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
        out = self.net(batch.x, batch.edge_index)
        out = out.reshape(batch.batch_size, -1)
        return out.mean(-1)