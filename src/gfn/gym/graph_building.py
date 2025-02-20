from typing import Callable, Literal, Tuple

import torch
from tensordict import TensorDict

from gfn.actions import GraphActions, GraphActionType
from gfn.env import GraphEnv, NonValidActionsError
from gfn.states import GraphStates


class GraphBuilding(GraphEnv):
    """Environment for incrementally building graphs.

    This environment allows constructing graphs by:
    - Adding nodes with features
    - Adding edges between existing nodes with features
    - Terminating construction (EXIT)

    Args:
        feature_dim: Dimension of node and edge features
        state_evaluator: Callable that computes rewards for final states.
            If None, uses default GCNConvEvaluator
        device_str: Device to run computations on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        feature_dim: int,
        state_evaluator: Callable[[GraphStates], torch.Tensor],
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        s0 = TensorDict(
            {
                "node_feature": torch.zeros((0, feature_dim), dtype=torch.float32),
                "edge_feature": torch.zeros((0, feature_dim), dtype=torch.float32),
                "edge_index": torch.zeros((0, 2), dtype=torch.long),
            },
            device=device_str,
        )
        sf = TensorDict(
            {
                "node_feature": torch.ones((1, feature_dim), dtype=torch.float32)
                * float("inf"),
                "edge_feature": torch.ones((0, feature_dim), dtype=torch.float32)
                * float("inf"),
                "edge_index": torch.zeros((0, 2), dtype=torch.long),
            },
            device=device_str,
        )

        self.state_evaluator = state_evaluator
        self.feature_dim = feature_dim

        super().__init__(
            s0=s0,
            sf=sf,
            device_str=device_str,
        )

    def reset(self, batch_shape: Tuple | int) -> GraphStates:
        """Reset the environment to a new batch of graphs."""
        states = super().reset(batch_shape)
        assert isinstance(states, GraphStates)
        return states

    def step(self, states: GraphStates, actions: GraphActions) -> TensorDict:
        """Step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to add.

        Returns the next graph the new GraphStates.
        """
        if not self.is_action_valid(states, actions):
            raise NonValidActionsError("Invalid action.")
        if len(actions) == 0:
            return states.tensor

        action_type = actions.action_type[0]
        assert torch.all(actions.action_type == action_type)
        if action_type == GraphActionType.EXIT:
            return self.States.make_sink_states_tensor(states.batch_shape)

        if action_type == GraphActionType.ADD_NODE:
            batch_indices = torch.arange(len(states))[
                actions.action_type == GraphActionType.ADD_NODE
            ]
            states.tensor = self._add_node(
                states.tensor, batch_indices, actions.features
            )

        if action_type == GraphActionType.ADD_EDGE:
            states.tensor["edge_feature"] = torch.cat(
                [states.tensor["edge_feature"], actions.features], dim=0
            )
            states.tensor["edge_index"] = torch.cat(
                [
                    states.tensor["edge_index"],
                    actions.edge_index,
                ],
                dim=0,
            )

        return states.tensor

    def backward_step(self, states: GraphStates, actions: GraphActions) -> TensorDict:
        """Backward step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to remove.

        Returns the previous graph as a new GraphStates.
        """
        if not self.is_action_valid(states, actions, backward=True):
            raise NonValidActionsError("Invalid action.")
        if len(actions) == 0:
            return states.tensor

        action_type = actions.action_type[0]
        assert torch.all(actions.action_type == action_type)
        if action_type == GraphActionType.ADD_NODE:
            is_equal = torch.any(
                torch.all(
                    states.tensor["node_feature"][:, None] == actions.features, dim=-1
                ),
                dim=-1,
            )
            states.tensor["node_feature"] = states.tensor["node_feature"][~is_equal]
        elif action_type == GraphActionType.ADD_EDGE:
            assert actions.edge_index is not None
            is_equal = torch.all(
                states.tensor["edge_index"] == actions.edge_index[:, None], dim=-1
            )
            is_equal = torch.any(is_equal, dim=0)
            states.tensor["edge_feature"] = states.tensor["edge_feature"][~is_equal]
            states.tensor["edge_index"] = states.tensor["edge_index"][~is_equal]

        return states.tensor

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
            ).sum(dim=-1)
            if backward:  # TODO: check if no edge is connected?
                add_node_out = torch.all(equal_nodes_per_batch == 1)
            else:
                add_node_out = torch.all(equal_nodes_per_batch == 0)

        add_edge_mask = actions.action_type == GraphActionType.ADD_EDGE
        if not torch.any(add_edge_mask):
            add_edge_out = True
        else:
            add_edge_states = states.tensor
            add_edge_actions = actions[add_edge_mask].edge_index
            if torch.any(add_edge_actions[:, 0] == add_edge_actions[:, 1]):
                return False
            if add_edge_states["node_feature"].shape[0] == 0:
                return False
            node_exists = torch.isin(add_edge_actions, add_edge_states["node_index"])
            if not torch.all(node_exists):
                return False

            equal_edges_per_batch = torch.all(
                add_edge_states["edge_index"] == add_edge_actions[:, None],
                dim=-1,
            ).sum(dim=-1)
            if backward:
                add_edge_out = torch.all(equal_edges_per_batch != 0)
            else:
                add_edge_out = torch.all(equal_edges_per_batch == 0)

        return bool(add_node_out) and bool(add_edge_out)

    def _add_node(
        self,
        tensor_dict: TensorDict,
        batch_indices: torch.Tensor,
        nodes_to_add: torch.Tensor,
    ) -> TensorDict:
        if isinstance(batch_indices, list):
            batch_indices = torch.tensor(batch_indices)
        if len(batch_indices) != len(nodes_to_add):
            raise ValueError(
                "Number of batch indices must match number of node feature lists"
            )

        modified_dict = tensor_dict.clone()
        node_feature_dim = modified_dict["node_feature"].shape[1]

        for graph_idx, new_nodes in zip(batch_indices, nodes_to_add):
            tensor_dict["batch_ptr"][graph_idx]
            end_ptr = tensor_dict["batch_ptr"][graph_idx + 1]
            new_nodes = torch.atleast_2d(new_nodes)
            if new_nodes.shape[1] != node_feature_dim:
                raise ValueError(
                    f"Node features must have dimension {node_feature_dim}"
                )

            # Update batch pointers for subsequent graphs
            num_new_nodes = new_nodes.shape[0]
            modified_dict["batch_ptr"][graph_idx + 1 :] += num_new_nodes

            # Expand node features
            modified_dict["node_feature"] = torch.cat(
                [
                    modified_dict["node_feature"][:end_ptr],
                    new_nodes,
                    modified_dict["node_feature"][end_ptr:],
                ]
            )
            modified_dict["node_index"] = torch.cat(
                [
                    modified_dict["node_index"][:end_ptr],
                    GraphStates.unique_node_indices(num_new_nodes),
                    modified_dict["node_index"][end_ptr:],
                ]
            )

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
