from typing import Literal, Optional, Tuple

import torch
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData
from torch_geometric.utils import to_dense_adj

from gfn.actions import GraphActions, GraphActionType
from gfn.env import GraphEnv
from gfn.states import GraphStates


class BayesianStructure(GraphEnv):
    """Environment for incrementally building a directed acyclic graph (DAG) for
    Bayesian structure learning (Deleu et al., 2022).

    The environment allows the following actions:
    - Adding edges between existing nodes with features
    - Terminating construction (EXIT)

    Args:
        n_nodes: Number of nodes in the graph.
        state_evaluator: Callable that computes rewards for final states.
            If None, uses default GCNConvEvaluator
        device_str: Device to run computations on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        n_nodes: int,
        state_evaluator: callable,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
    ):

        self.n_nodes = n_nodes

        s0 = GeometricData(
            x=torch.arange(n_nodes, dtype=torch.float)[:, None],  # Node ids
            edge_attr=torch.ones((0, 1)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            device=device,
        )
        sf = GeometricData(
            x=-torch.ones(n_nodes, dtype=torch.float)[:, None],
            edge_attr=torch.zeros((0, 1)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            device=device,
        )

        self.state_evaluator = state_evaluator

        super().__init__(
            s0=s0,
            sf=sf,
        )

    def make_actions_class(self) -> type[GraphActions]:

        class BayesianStructureActions(GraphActions):
            """Actions for building DAGs for Bayesian structure

            Actions are represented as discrete indices where:
            - 0 to (n_nodes ** 2) - 1: Add edge between nodes i and j
                where i = index // n_nodes, j = index % n_nodes
            - n_nodes ** 2: Exit action
            - n_nodes ** 2 + 1: dummy action (used for padding)
            """

            action_shape = (1,)
            exit_action = torch.tensor([self.n_nodes**2])
            dummy_action = torch.tensor([self.n_nodes**2 + 1])

        return BayesianStructureActions

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class BayesianStructureStates(GraphStates):
            """Represents the state for building DAGs for Bayesian structure.

            Each state is a graph with a fixed number of nodes wierhe edges
            are being addd incrementally to form a DAG.

            The state representation consists of:
            - x: Node IDs (shape: [n_nodes, 1])
            - edge_index: Edge indices (shape: [2, n_edges])

            Special states:
            - s0: Initial state with no edges
            - sf: Terminal dummy state

            The class also provides masks for allowed actions.
            """

            s0 = env.s0
            sf = env.sf
            n_actions = env.n_nodes**2 + 1

            @property
            def forward_masks(self) -> dict:
                """Returns masks denoting allowed forward actions.

                Returns:
                    A dictionary containing masks for different action types.
                """
                # Get the data list from the batch
                data_list = self.tensor.to_data_list()
                num_nodes = self.tensor.x.size(0)

                # Initialize masks
                action_type_mask = torch.ones(
                    self.batch_shape + (3,), dtype=torch.bool, device=self.device
                )
                dense_masks = torch.ones(
                    (len(data_list), num_nodes, num_nodes),
                    dtype=torch.bool,
                    device=self.device,
                )

                # For each graph in the batch
                for i, data in enumerate(data_list):
                    # Flatten the batch index
                    flat_idx = i
                    # ADD_NODE is not allowed in Bayesian Structure Learning setting
                    action_type_mask[flat_idx, GraphActionType.ADD_NODE] = False
                    # ADD_EDGE is allowed only if there are at least 2 nodes
                    assert data.num_nodes is not None
                    action_type_mask[flat_idx, GraphActionType.ADD_EDGE] = (
                        data.num_nodes > 1
                    )
                    # EXIT is always allowed
                    action_type_mask[flat_idx, GraphActionType.EXIT] = True

                    # For each graph, create a dense mask for potential edges
                    sparse_edge_index = data.edge_index
                    adjacency = (
                        to_dense_adj(sparse_edge_index, max_num_nodes=num_nodes)
                        .squeeze(0)
                        .to(torch.bool)
                    )
                    # Create self-loop mask
                    self_loops = torch.eye(
                        num_nodes, dtype=torch.bool, device=self.device
                    )
                    # Compute transitive closure using the Floyd–Warshall style update:
                    # reach[u, v] is True if there is a path from u to v.
                    reach = adjacency.clone()
                    for k in range(num_nodes):
                        reach = reach | (reach[:, k : k + 1] & reach[k : k + 1, :])
                    # An edge u -> v is allowed if:
                    # 1. There is no existing edge (i.e. not in adjacency)
                    # 2. It won't create a cycle (i.e. no path from v back to u: reach[v, u] is False)
                    # 3. u and v are different (avoid self-loops)
                    allowed = (~adjacency) & (~reach.T) & (~self_loops)
                    dense_masks[i, :num_nodes, :num_nodes] = allowed

                return {"action_type": action_type_mask, "dense_mask": dense_masks}

            def backward_masks(self):
                raise NotImplementedError

        return BayesianStructureStates

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
            states: BayesianStructureStates object representing the current graph states.
            actions: Actions to apply to each graph state.

        Returns:
            The updated graph states after applying the actions.
        """
        if len(actions) == 0:
            return states.tensor

        if torch.any(actions.action_type == GraphActionType.ADD_NODE):
            raise ValueError(
                "ADD_NODE action is not supported in BayesianStructure environment."
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
        """Backward step function for the Bayesian structure learning environment.

        Args:
            states: BayesianStructureStates object representing the current graph.
            actions: Actions indicating which edge to remove.

        Returns:
            The previous graph states after reversing the actions.
        """
        if len(actions) == 0:
            return states.tensor

        # Check that there are no ADD_NODE actions (not supported in this environment)
        if torch.any(actions.action_type == GraphActionType.ADD_NODE):
            raise ValueError(
                "ADD_NODE action is not supported in BayesianStructure environment."
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

    def is_action_valid(
        self,
        states: GraphStates,
        actions: GraphActions,
    ) -> torch.Tensor:
        # TODO
        raise NotImplementedError

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        # Clone the final states to create a reverted graph
        reverted_states = final_states.clone()
        # Get the data list from the batch for processing individual graphs
        data_list = reverted_states.tensor.to_data_list()
        for i in range(len(data_list)):
            graph = data_list[i]
            # Remove the edge
            graph.edge_index = graph.edge_index[:, -1]
            # Also remove the edge feature
            if graph.edge_attr is not None and graph.edge_attr.shape[0] > 0:
                graph.edge_attr = graph.edge_attr[:-1]
        # Create a new batch from the updated data list
        reverted_states.tensor = GeometricBatch.from_data_list(data_list)
        # Compute the BDe score for the reverted graph (before the last action)
        score_before = self.state_evaluator(self.States(reverted_states.tensor))
        # Compute the BDe score for the current graph (after the last action)
        score_after = self.state_evaluator(final_states)
        # Compute the delta score (reward)
        delta_score = score_after - score_before

        return delta_score
