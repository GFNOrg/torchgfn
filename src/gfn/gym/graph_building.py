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
                graphs.x = actions.features[:, None, :]
            else:
                graphs.x = torch.cat([graphs.x, actions.features[:, None, :]], dim=1)

        if actions.action_type == GraphActionType.ADD_EDGE:
            assert actions.edge_index is not None
            if graphs.edge_attr is None:
                graphs.edge_attr = actions.features[:, None, :]
                assert graphs.edge_index is None
                graphs.edge_index = actions.edge_index[:, :, None]
            else:
                graphs.edge_attr = torch.cat([graphs.edge_attr, actions.features[:, None, :]], dim=1)
                graphs.edge_index = torch.cat([graphs.edge_index, actions.edge_index[:, :, None]], dim=2)

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
            is_equal = torch.all(graphs.x == actions.features[:, None], dim=-1)
            assert torch.all(torch.sum(is_equal, dim=-1) == 1)
            graphs.x = graphs.x[~is_equal].reshape(states.data.batch_size, -1, self.node_feature_dim)

        elif actions.action_type == GraphActionType.ADD_EDGE:
            assert actions.edge_index is not None
            is_equal = torch.all(graphs.edge_index == actions.edge_index[:, :, None], dim=1)
            assert torch.all(torch.sum(is_equal, dim=-1) == 1)
            graphs.edge_attr = graphs.edge_attr[~is_equal].reshape(states.data.batch_size, -1, self.edge_feature_dim)
            edge_index = graphs.edge_index.permute(0, 2, 1)[~is_equal]
            graphs.edge_index = edge_index.reshape(states.data.batch_size, -1, 2).permute(0, 2, 1)

        return self.States(graphs)

    def is_action_valid(
        self, states: GraphStates, actions: GraphActions, backward: bool = False
    ) -> bool:
        if actions.action_type == GraphActionType.ADD_NODE:
            if actions.edge_index is not None:
                return False
            if states.data.x is None:
                return not backward
            
            equal_nodes_per_batch = torch.sum(
                torch.all(states.data.x == actions.features[:, None], dim=-1),
                dim=-1
            )

            if backward:  # TODO: check if no edge are connected?
                return torch.all(equal_nodes_per_batch == 1)
            return torch.all(equal_nodes_per_batch == 0)
        
        if actions.action_type == GraphActionType.ADD_EDGE:
            assert actions.edge_index is not None
            if torch.any(actions.edge_index[:, 0] == actions.edge_index[:, 1]):
                return False
            if states.data.edge_index is None:
                return not backward

            equal_edges_per_batch_attr = torch.sum(
                torch.all(states.data.edge_attr == actions.features[:, None], dim=-1),
                dim=-1
            )
            equal_edges_per_batch_index = torch.sum(
                torch.all(states.data.edge_index == actions.edge_index[:, :, None], dim=1),
                dim=-1
            )
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
        out = torch.empty(len(batch), device=batch.x.device)
        for i in range(len(batch)):  # looks like net doesn't work with batch
            out[i] = self.net(batch.x[i], batch.edge_index[i]).mean()
        
        return out
