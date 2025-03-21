from typing import Literal, Optional, Tuple

import torch
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData
from torch_geometric.utils import to_dense_adj

from gfn.actions import Actions, GraphActions, GraphActionType
from gfn.env import GraphEnv
from gfn.states import GraphStates


class BayesianStructure(GraphEnv):
    """Environment for incrementally building a directed acyclic graph (DAG) for
    Bayesian structure learning (Deleu et al., 2022).

    The environment allows the following actions:
    - Adding edges between existing nodes with features
    - Terminating construction (EXIT)

    Args:
        num_nodes: Number of nodes in the graph.
        state_evaluator: Callable that computes rewards for final states.
            If None, uses default GCNConvEvaluator
        device_str: Device to run computations on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        num_nodes: int,
        state_evaluator: callable,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
    ):

        self.num_nodes = num_nodes
        self.n_actions = num_nodes**2 + 1

        s0 = GeometricData(
            x=torch.arange(num_nodes, dtype=torch.float)[:, None],  # Node ids
            edge_attr=torch.ones((0, 1)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            device=device,
        )
        sf = GeometricData(
            x=-torch.ones(num_nodes, dtype=torch.float)[:, None],
            edge_attr=torch.zeros((0, 1)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            device=device,
        )

        self.state_evaluator = state_evaluator

        super().__init__(
            s0=s0,
            sf=sf,
        )

    def make_actions_class(self) -> type[Actions]:
        env = self

        class BayesianStructureActions(Actions):
            """Actions for building DAGs for Bayesian structure

            Actions are represented as discrete indices where:
            - 0 to (num_nodes ** 2) - 1: Add edge between nodes i and j
                where i = index // num_nodes, j = index % num_nodes
            - n_actions - 1 = num_nodes ** 2: Exit action
            - n_actions = num_nodes ** 2 + 1: dummy action (used for padding)
            """

            action_shape = (1,)
            exit_action = torch.tensor([env.n_actions - 1])
            dummy_action = torch.tensor([env.n_actions])

        return BayesianStructureActions

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class BayesianStructureStates(GraphStates):
            """Represents the state for building DAGs for Bayesian structure

            Each state is a graph with a fixed number of nodes wierhe edges
            are being addd incrementally to form a DAG.

            The state representation consists of:
            - x: Node IDs (shape: [num_nodes, 1])
            - edge_index: Edge indices (shape: [2, n_edges])

            Special states:
            - s0: Initial state with no edges
            - sf: Terminal dummy state

            The class also provides masks for allowed actions.
            """

            s0 = env.s0
            sf = env.sf
            num_nodes = env.num_nodes
            n_actions = env.n_actions

            @property
            def num_edges(self) -> torch.Tensor:
                """Returns the number of edges in each graph."""
                return torch.tensor(
                    [data.num_edges for data in self.tensor.to_data_list()]
                ).view(*self.batch_shape)

            @property
            def forward_masks(self) -> torch.Tensor:
                """Returns forward action mask for the current state.

                Returns:
                    A tensor of shape [batch_size, n_actions] with True for valid actions.
                """
                assert (
                    ~self.is_sink_state
                ).all(), "No valid forward actions for sink states."

                # Allow all actions.
                forward_mask = torch.ones(len(self), self.n_actions, dtype=torch.bool)

                # For each graph in the batch
                for i, data in enumerate(self.tensor.to_data_list()):
                    # For each graph, create a dense mask for potential edges
                    sparse_edge_index = data.edge_index
                    adjacency = (
                        to_dense_adj(sparse_edge_index, max_num_nodes=self.num_nodes)
                        .squeeze(0)
                        .to(torch.bool)
                    )
                    # Create self-loop mask
                    self_loops = torch.eye(
                        self.num_nodes, dtype=torch.bool, device=self.device
                    )
                    # Compute transitive closure using the Floyd–Warshall style update:
                    # reach[u, v] is True if there is a path from u to v.
                    reach = adjacency.clone()
                    for k in range(self.num_nodes):
                        reach = reach | (reach[:, k : k + 1] & reach[k : k + 1, :])
                    # An edge u -> v is allowed if:
                    # 1. There is no existing edge (i.e. not in adjacency)
                    # 2. It won't create a cycle (i.e. no path from v back to u: reach[v, u] is False)
                    # 3. u and v are different (avoid self-loops)
                    allowed = (~adjacency) & (~reach.T) & (~self_loops)
                    forward_mask[i, : self.num_nodes**2] = allowed.flatten()

                return forward_mask.view(*self.batch_shape, self.n_actions)

            @property
            def backward_masks(self) -> torch.Tensor:
                """Compute masks for valid backward actions from the current state (a DAG).
                All existing edges are considered for removal.

                The EXIT action is not included in backward masks.

                Returns:
                    A tensor of shape [batch_size, n_actions - 1] with True for valid backward actions.
                """
                assert (
                    ~self.is_initial_state
                ).all(), "No valid backward actions for initial states."

                # Disable all actions.
                backward_masks = torch.zeros(
                    len(self), self.n_actions - 1, dtype=torch.bool
                )

                # Get the data list from the batch
                data_list = self.tensor.to_data_list()

                # For each graph in the batch
                for i, data in enumerate(data_list):
                    # For each graph, create a dense mask for potential edges
                    backward_masks[i] = to_dense_adj(
                        data.edge_index, max_num_nodes=self.num_nodes
                    ).flatten()

                return backward_masks.view(*self.batch_shape, self.n_actions - 1)

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

    def _step(self, states: GraphStates, actions: Actions) -> GraphStates:
        graph_actions = self.convert_actions(actions)
        new_states = super()._step(states, graph_actions)
        assert isinstance(new_states, self.States)
        return new_states

    def convert_actions(self, actions: Actions) -> GraphActions:
        """Convert actions from the Actions class to the GraphActions class.

        This method maps discrete action indices to specific graph operations:
        - GraphActionType.ADD_EDGE: Add an edge between specific nodes
        - GraphActionType.EXIT: Terminate trajectory
        - GraphActionType.DUMMY: No-op action (for padding)

        Args:
            actions: Discrete actions to convert

        Returns:
            Equivalent actions in the GraphActions format
        """
        # TODO: factor out into utility function.
        action_tensor = actions.tensor.squeeze(-1).clone()

        is_exit = action_tensor == (self.n_actions - 1)
        action_type = torch.where(
            is_exit, GraphActionType.EXIT, GraphActionType.ADD_EDGE
        )
        action_type[action_tensor == self.n_actions] = GraphActionType.DUMMY

        edge_index = torch.zeros(
            (action_type.shape[0], 2), dtype=torch.long, device=action_type.device
        )
        edge_index[~is_exit, 0] = action_tensor[~is_exit] // self.num_nodes
        edge_index[~is_exit, 1] = action_tensor[~is_exit] % self.num_nodes

        graph_actions = GraphActions(
            TensorDict(
                {
                    "action_type": action_type,
                    "features": torch.ones(action_type.shape + (1,)),
                    "edge_index": edge_index,
                },
                batch_size=action_type.shape,
            )
        )
        return graph_actions

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
        backward: bool = False,
    ) -> bool:
        # TODO
        return True

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        # Clone the final states to create a reverted graph
        batch_size = final_states.batch_shape[0]
        reverted_states = final_states.clone()
        data_list = []
        for i in range(batch_size):
            graph = reverted_states[i]
            # Remove the edge
            graph.tensor.edge_index = graph.tensor.edge_index[:, :-1]
            # Also remove the edge feature
            if (
                graph.tensor.edge_attr is not None
                and graph.tensor.edge_attr.shape[0] > 0
            ):
                graph.tensor.edge_attr = graph.tensor.edge_attr[:-1]
            data_list.append(graph.tensor)
        # Create a new batch from the updated data list
        reverted_states = GeometricBatch.from_data_list(data_list)
        reverted_states.batch_shape = final_states.batch_shape
        # Compute the local score for the reverted graph (before the last action)
        score_before = self.state_evaluator(GraphStates(reverted_states))
        # Compute the local score for the current graph (after the last action)
        score_after = self.state_evaluator(final_states)
        # Compute the delta score (reward)
        print(score_before)
        print(score_after)
        delta_score = score_after - score_before

        return delta_score
