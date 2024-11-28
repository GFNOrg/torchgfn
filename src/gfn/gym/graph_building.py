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
        feature_dim: int,
        state_evaluator: Callable[[Batch], torch.Tensor] | None = None,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        s0 = Data(
            x=torch.zeros((0, feature_dim), dtype=torch.float32),
            edge_attr=torch.zeros((0, feature_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        ).to(device_str)
        sf = Data(
            x=torch.ones((1, feature_dim), dtype=torch.float32) * float("inf"),
        ).to(device_str)

        if state_evaluator is None:
            state_evaluator = GCNConvEvaluator(feature_dim)
        self.state_evaluator = state_evaluator

        super().__init__(
            s0=s0,
            sf=sf,
            device_str=device_str,
        )

    def step(self, states: GraphStates, actions: GraphActions) -> Data:
        """Step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to add.

        Returns the next graph the new GraphStates.
        """
        if not self.is_action_valid(states, actions):
            raise NonValidActionsError("Invalid action.")
        graphs: Batch = deepcopy(states.data)

        action_type = actions.action_type[0]
        assert torch.all(actions.action_type == action_type)

        if action_type == GraphActionType.ADD_NODE:
            assert len(graphs) == len(actions)
            if graphs.x is None:
                graphs.x = actions.features
            else:
                graphs.x = torch.cat([graphs.x, actions.features])

        if action_type == GraphActionType.ADD_EDGE:
            assert len(graphs) == len(actions)
            assert actions.edge_index is not None
            if graphs.edge_attr is None:
                graphs.edge_attr = actions.features
                assert graphs.edge_index is None
                graphs.edge_index = actions.edge_index
            else:
                graphs.edge_attr = torch.cat([graphs.edge_attr, actions.features])
                graphs.edge_index = torch.cat(
                    [graphs.edge_index, actions.edge_index], dim=1
                )

        return graphs

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
                torch.all(graphs.x[:, None] == actions.features, dim=-1), dim=-1
            )
            graphs.x = graphs.x[~is_equal]
        elif actions.action_type == GraphActionType.ADD_EDGE:
            assert actions.edge_index is not None
            is_equal = torch.all(
                graphs.edge_index[:, None] == actions.edge_index[:, :, None], dim=0
            )
            is_equal = torch.any(is_equal, dim=0)
            graphs.edge_attr = graphs.edge_attr[~is_equal]
            graphs.edge_index = graphs.edge_index[:, ~is_equal]

        return self.States(graphs)

    def is_action_valid(
        self, states: GraphStates, actions: GraphActions, backward: bool = False
    ) -> bool:
        add_node_mask = actions.action_type == GraphActionType.ADD_NODE
        if not torch.any(add_node_mask):
            add_node_out = True
        else:
            equal_nodes_per_batch = torch.all(
                states[add_node_mask].data.x == actions[add_node_mask].features[:, None], dim=-1
            ).reshape(states.data.batch_size, -1)
            equal_nodes_per_batch = torch.sum(equal_nodes_per_batch, dim=-1)
            if backward:  # TODO: check if no edge are connected?
                add_node_out = torch.all(equal_nodes_per_batch == 1)
            else:
                add_node_out = torch.all(equal_nodes_per_batch == 0)
        
        add_edge_mask = actions.action_type == GraphActionType.ADD_EDGE
        if not torch.any(add_edge_mask):
            add_edge_out = True
        else:
            add_edge_states = states[add_edge_mask]
            add_edge_actions = actions[add_edge_mask]

            if torch.any(add_edge_actions.edge_index[0] == add_edge_actions.edge_index[1]):
                return False
            if add_edge_states.data.num_nodes == 0:
                return False
            if torch.any(add_edge_actions.edge_index > add_edge_states.data.num_nodes):
                return False

            batch_dim = add_edge_actions.features.shape[0]
            batch_idx = add_edge_actions.edge_index % batch_dim
            if torch.any(batch_idx != torch.arange(batch_dim)):
                return False

            equal_edges_per_batch_attr = torch.all(
                add_edge_states.data.edge_attr == add_edge_actions.features[:, None], dim=-1
            ).reshape(add_edge_states.data.batch_size, -1)
            equal_edges_per_batch_attr = torch.sum(equal_edges_per_batch_attr, dim=-1)
            equal_edges_per_batch_index = torch.all(
                add_edge_states.data.edge_index[:, None] == add_edge_actions.edge_index[:, :, None], dim=0
            ).reshape(add_edge_states.data.batch_size, -1)
            equal_edges_per_batch_index = torch.sum(equal_edges_per_batch_index, dim=-1)
        
            if backward:
                add_edge_out = torch.all(equal_edges_per_batch_attr == 1) and torch.all(
                    equal_edges_per_batch_index == 1
                )
            else:
                add_edge_out = torch.all(equal_edges_per_batch_attr == 0) and torch.all(
                    equal_edges_per_batch_index == 0
                )
            
        return bool(add_node_out) and bool(add_edge_out)

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
        """Generates random states tensor of shape (*batch_shape, feature_dim)."""
        return self.States.from_batch_shape(batch_shape)


class GCNConvEvaluator:
    def __init__(self, num_features):
        self.net = GCNConv(num_features, 1)

    def __call__(self, batch: Batch) -> torch.Tensor:
        out = self.net(batch.x, batch.edge_index)
        out = out.reshape(batch.batch_size, -1)
        return out.mean(-1)
