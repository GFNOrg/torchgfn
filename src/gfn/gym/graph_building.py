from copy import deepcopy
from typing import Callable, Literal, Tuple

import torch
from torch_geometric.nn import GCNConv
from tensordict import TensorDict

from gfn.actions import GraphActions, GraphActionType
from gfn.env import GraphEnv, NonValidActionsError
from gfn.states import GraphStates


class GraphBuilding(GraphEnv):
    def __init__(
        self,
        feature_dim: int,
        state_evaluator: Callable[[GraphStates], torch.Tensor] | None = None,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        s0 = TensorDict({
            "node_feature": torch.zeros((0, feature_dim), dtype=torch.float32),
            "edge_feature": torch.zeros((0, feature_dim), dtype=torch.float32),
            "edge_index": torch.zeros((0, 2), dtype=torch.long),
        }, device=device_str)
        sf = TensorDict({
            "node_feature": torch.ones((1, feature_dim), dtype=torch.float32) * float("inf"),
            "edge_feature": torch.ones((1, feature_dim), dtype=torch.float32) * float("inf"),
            "edge_index": torch.zeros((0, 2), dtype=torch.long),
        }, device=device_str)

        if state_evaluator is None:
            state_evaluator = GCNConvEvaluator(feature_dim)
        self.state_evaluator = state_evaluator

        super().__init__(
            s0=s0,
            sf=sf,
            device_str=device_str,
        )

    def step(self, states: GraphStates, actions: GraphActions) -> TensorDict:
        """Step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to add.

        Returns the next graph the new GraphStates.
        """
        if not self.is_action_valid(states, actions):
            raise NonValidActionsError("Invalid action.")
        state_tensor = deepcopy(states.tensor)
        if len(actions) == 0:
            return state_tensor

        action_type = actions.action_type[0]
        assert torch.all(actions.action_type == action_type)
        if action_type == GraphActionType.EXIT:
            return self.States.make_sink_states_tensor(states.batch_shape)

        if action_type == GraphActionType.ADD_NODE:
            batch_indices = torch.arange(len(states))[actions.action_type == GraphActionType.ADD_NODE]
            state_tensor = self._add_node(state_tensor, batch_indices, actions.features)

        if action_type == GraphActionType.ADD_EDGE:
            assert len(state_tensor) == len(actions)
            state_tensor["edge_feature"] = torch.cat([state_tensor["edge_feature"], actions.features], dim=0)
            state_tensor["edge_index"] = torch.cat(
                [state_tensor["edge_index"], actions.edge_index], dim=0
            )
        return state_tensor

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
            node_feature = states[add_node_mask].tensor["node_feature"]
            equal_nodes_per_batch = torch.all(
                node_feature == actions[add_node_mask].features[:, None], dim=-1
            ).reshape(-1)
            if backward:  # TODO: check if no edge are connected?
                add_node_out = torch.all(equal_nodes_per_batch == 1)
            else:
                add_node_out = torch.all(equal_nodes_per_batch == 0)
        
        add_edge_mask = actions.action_type == GraphActionType.ADD_EDGE
        if not torch.any(add_edge_mask):
            add_edge_out = True
        else:
            add_edge_states = states[add_edge_mask].tensor
            add_edge_actions = actions[add_edge_mask]

            if torch.any(add_edge_actions.edge_index[:, 0] == add_edge_actions.edge_index[:, 1]):
                return False
            if add_edge_states["node_feature"].shape[0] == 0:
                return False
            if torch.any(add_edge_actions.edge_index > add_edge_states["node_feature"].shape[0]):
                return False

            batch_dim = add_edge_actions.features.shape[0]
            batch_idx = add_edge_actions.edge_index % batch_dim
            if torch.any(batch_idx != torch.arange(batch_dim)):
                return False

            equal_edges_per_batch_attr = torch.all(
                add_edge_states["edge_feature"] == add_edge_actions.features[:, None], dim=-1
            ).reshape(len(add_edge_states), -1)
            equal_edges_per_batch_attr = torch.sum(equal_edges_per_batch_attr, dim=-1)
            equal_edges_per_batch_index = torch.all(
                add_edge_states["edge_index"] == add_edge_actions.edge_index, dim=0
            ).reshape(len(add_edge_states), -1)
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
    
    def _add_node(self, tensor_dict: TensorDict, batch_indices: torch.Tensor, nodes_to_add: torch.Tensor) -> TensorDict:
        if isinstance(batch_indices, list):
            batch_indices = torch.tensor(batch_indices)
        if len(batch_indices) != len(nodes_to_add):
            raise ValueError("Number of batch indices must match number of node feature lists")
        
        modified_dict = tensor_dict.clone()
        node_feature_dim = modified_dict['node_feature'].shape[1]
        edge_feature_dim = modified_dict['edge_feature'].shape[1]
        
        for graph_idx, new_nodes in zip(batch_indices, nodes_to_add):
            start_ptr = tensor_dict['batch_ptr'][graph_idx]
            end_ptr = tensor_dict['batch_ptr'][graph_idx + 1]
            num_original_nodes = end_ptr - start_ptr

            if new_nodes.ndim == 1:
                new_nodes = new_nodes.unsqueeze(0)
            if new_nodes.shape[1] != node_feature_dim:
                raise ValueError(f"Node features must have dimension {node_feature_dim}")
            
            # Update batch pointers for subsequent graphs
            shift = new_nodes.shape[0]
            modified_dict['batch_ptr'][graph_idx + 1:] += shift
            
            # Expand node features
            original_nodes = modified_dict['node_feature'][start_ptr:end_ptr]
            modified_dict['node_feature'] = torch.cat([
                modified_dict['node_feature'][:end_ptr],
                new_nodes,
                modified_dict['node_feature'][end_ptr:]
            ])
            
            # Update edge indices
            # Increment indices for edges after the current graph
            edge_mask_0 = modified_dict['edge_index'][:, 0] >= end_ptr
            edge_mask_1 = modified_dict['edge_index'][:, 1] >= end_ptr
            modified_dict['edge_index'][edge_mask_0, 0] += shift
            modified_dict['edge_index'][edge_mask_1, 1] += shift
        
        return modified_dict

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        return self.state_evaluator(final_states)

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

    def __call__(self, state: GraphStates) -> torch.Tensor:
        node_feature = state.tensor["node_feature"]
        edge_index = state.tensor["edge_index"].T
        if len(node_feature) == 0:
            return torch.zeros(len(state))

        out = self.net(node_feature, edge_index)
        out = out.reshape(*state.batch_shape, -1)
        return out.mean(-1)