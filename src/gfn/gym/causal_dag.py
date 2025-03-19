from typing import Callable, Literal, Optional, Tuple

import torch
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData

from gfn.actions import GraphActions, GraphActionType
from gfn.env import GraphEnv
from gfn.states import GraphStates


class Causal_DAG(GraphEnv):
    """Environment for incrementally building graphs.

    This environment replicates the setting of the Causal DAG construction task (Bayesian
    Structure Learning). We assume that we have the nodes of interest in our initial state.
    The environment allows the following actions:
    - Adding edges between existing nodes with features
    - Terminating construction (EXIT)

    Args:
        nodes (num_nodes*feature_dim): states for which the casual relationship needs to be
            established. This will be the intial state of env where all nodes are disconnected.
        state_evaluator: Callable that computes rewards for final states.
            If None, uses default GCNConvEvaluator
        device_str: Device to run computations on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        nodes: torch.Tensor,
        state_evaluator: Callable[[GraphStates], torch.Tensor],
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):

        self.feature_dim = nodes.shape[1]
        s0 = GeometricData(
            x=nodes,
            edge_attr=torch.zeros((0, self.feature_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            device=device_str,
        )
        sf = GeometricData(
            x=torch.ones((1, self.feature_dim), dtype=torch.float32) * float("inf"),
            edge_attr=torch.ones((0, self.feature_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            device=device_str,
        )
        self.state_evaluator = state_evaluator

        super().__init__(
            s0=s0,
            sf=sf,
        )

    def reset(
        self,
        batch_shape: int | Tuple[int, ...],
        seed: Optional[int] = None,
    ) -> GraphStates:
        """Reset the environment to a new batch of graphs."""
        states = super().reset(batch_shape, seed=seed, random=False, sink=False)
        assert isinstance(states, GraphStates)
        return states

    def step(self, states: GraphStates, actions: GraphActions) -> GeometricBatch:
        """Step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph states.
            actions: Actions to apply to each graph state.

        Returns:
            The updated graph states after applying the actions.
        """
        if len(actions) == 0:
            return states.tensor

        if torch.any(actions.action_type == GraphActionType.ADD_NODE):
            raise ValueError(
                "ADD_NODE action is not supported in Causal DAG environment."
            )

        # Get the data list from the batch for processing individual graphs
        data_list = states.tensor.to_data_list()

        # Create masks for different action types
        exit_mask = actions.action_type == GraphActionType.EXIT
        add_edge_mask = actions.action_type == GraphActionType.ADD_EDGE

        # Handle EXIT actions
        if torch.any(exit_mask):
            # For graphs with EXIT action, replace them with sink states
            exit_indices = torch.where(exit_mask)[0]
            sink_data = self.sf.clone()
            for idx in exit_indices:
                data_list[idx] = sink_data

        # Handle ADD_EDGE actions
        if torch.any(add_edge_mask):
            edge_indices = torch.where(add_edge_mask)[0]

            for i in edge_indices:
                # Get source and destination nodes for this edge
                src, dst = actions.edge_index[i]
                graph = data_list[i]

                # Add the new edge
                graph.edge_index = torch.cat(
                    [
                        graph.edge_index,
                        torch.tensor([[src], [dst]], device=graph.edge_index.device),
                    ],
                    dim=1,
                )
                # Add the edge feature
                graph.edge_attr = torch.cat(
                    [graph.edge_attr, actions.features[i].unsqueeze(0)], dim=0
                )
        # Create a new batch from the updated data list
        new_tensor = GeometricBatch.from_data_list(data_list)
        new_tensor.batch_shape = states.tensor.batch_shape
        return new_tensor

    def backward_step(
        self, states: GraphStates, actions: GraphActions
    ) -> GeometricBatch:
        """Backward step function for the Causal DAG environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to remove.

        Returns:
            The previous graph states after reversing the actions.
        """
        if len(actions) == 0:
            return states.tensor

        # Check that there are no ADD_NODE actions (not supported in Causal DAG)
        if torch.any(actions.action_type == GraphActionType.ADD_NODE):
            raise ValueError(
                "ADD_NODE action is not supported in Causal DAG environment."
            )

        # Get the data list from the batch for processing individual graphs
        data_list = states.tensor.to_data_list()

        add_edge_mask = actions.action_type == GraphActionType.ADD_EDGE

        # Handle ADD_EDGE actions
        if torch.any(add_edge_mask):
            edge_indices = torch.where(add_edge_mask)[0]

            for i in edge_indices:
                # Get source and destination nodes for the edge to remove
                src, dst = actions.edge_index[i]
                graph = data_list[i]
                # Find the edge to remove
                edge_mask = ~(
                    (graph.edge_index[0] == src) & (graph.edge_index[1] == dst)
                )
                # Remove the edge
                graph.edge_index = graph.edge_index[:, edge_mask]
                # Also remove the edge feature
                if graph.edge_attr is not None and graph.edge_attr.shape[0] > 0:
                    graph.edge_attr = graph.edge_attr[edge_mask]

        # Create a new batch from the updated data list
        new_tensor = GeometricBatch.from_data_list(data_list)
        new_tensor.batch_shape = states.tensor.batch_shape
        return new_tensor

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        return self.state_evaluator(final_states)
