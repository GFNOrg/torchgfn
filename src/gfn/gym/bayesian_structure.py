from math import prod
from typing import Callable, List, Literal, Tuple, cast

import numpy as np
import torch
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData
from torch_geometric.data.data import BaseData
from torch_geometric.utils import to_dense_adj

from gfn.actions import GraphActions, GraphActionType
from gfn.gym.graph_building import GraphBuilding
from gfn.states import GraphStates


class BayesianStructure(GraphBuilding):
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
        state_evaluator: Callable[[GraphStates], torch.Tensor],
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)

        self.n_nodes = n_nodes
        self.n_actions = n_nodes**2 + 1

        s0 = GeometricData(
            x=torch.arange(n_nodes, dtype=torch.float)[:, None].to(device),
            edge_attr=torch.ones((0, 1), dtype=torch.float).to(device),
            edge_index=torch.zeros((2, 0), dtype=torch.long).to(device),
            device=device,
        )
        sf = GeometricData(
            x=-torch.ones(n_nodes, dtype=torch.float)[:, None].to(device),
            edge_attr=torch.zeros((0, 1), dtype=torch.float).to(device),
            edge_index=torch.zeros((2, 0), dtype=torch.long).to(device),
            device=device,
        )

        super().__init__(
            num_node_classes=1,
            num_edge_classes=1,
            state_evaluator=state_evaluator,
            is_directed=True,
            device=device,
            s0=s0,
            sf=sf,
        )

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class BayesianStructureStates(GraphStates):
            """Represents the state for building DAGs for Bayesian structure

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

            num_node_classes = env.num_node_classes
            num_edge_classes = env.num_edge_classes

            s0 = env.s0
            sf = env.sf
            n_nodes = env.n_nodes
            n_actions = env.n_actions

            @property
            def num_edges(self) -> torch.Tensor:
                """Returns the number of edges in each graph."""
                return torch.tensor(
                    [data.num_edges for data in self.tensor.to_data_list()]
                ).view(*self.batch_shape)

            @property
            def forward_masks(self) -> TensorDict:
                """Returns forward action mask for the current state.

                Returns:
                    A TensorDict with the following keys:
                    - action_type: Tensor of shape [*batch_shape, 3] with True for valid action types
                    - node_class: Tensor of shape [*batch_shape, num_node_classes] (unused for this environment)
                    - edge_class: Tensor of shape [*batch_shape, n_nodes, n_nodes] (unused for this environment)
                    - edge_index: Tensor of shape [*batch_shape, n_nodes, n_nodes] with True for valid edge index to add
                """

                batch_adjacency = (
                    to_dense_adj(self.tensor.edge_index, self.tensor.batch)
                    .squeeze(0)
                    .to(torch.bool)
                )
                if batch_adjacency.ndim == 2:
                    batch_adjacency = batch_adjacency.unsqueeze(0)

                # Create self-loop mask
                self_loops = torch.eye(
                    self.n_nodes, dtype=torch.bool, device=self.device
                ).repeat(prod(self.batch_shape), 1, 1)
                # Compute transitive closure using the Floyd–Warshall style update:
                # reach[u, v] is True if there is a path from u to v.
                reach = batch_adjacency.clone()
                for k in range(self.n_nodes):
                    reach = reach | (reach[:, :, k : k + 1] & reach[:, k : k + 1, :])
                # An edge u -> v is allowed if:
                # 1. It is not already in the graph (i.e. not in adjacency)
                # 2. It won't create a cycle (i.e. no path from v back to u: reach[v, u] is False)
                # 3. It is not a self-loop (i.e. u and v are different)
                allowed = (~batch_adjacency) & (~reach.transpose(1, 2)) & (~self_loops)
                batch_edge_masks = allowed.flatten(1, 2)
                edge_masks = batch_edge_masks.reshape(*self.batch_shape, -1)

                # There are 3 action types: ADD_NODE, ADD_EDGE, EXIT
                action_type = torch.zeros(
                    *self.batch_shape, 3, dtype=torch.bool, device=self.device
                )
                action_type[..., GraphActionType.ADD_EDGE] = torch.any(
                    edge_masks, dim=-1
                )
                action_type[..., GraphActionType.EXIT] = 1

                return TensorDict(
                    {
                        "action_type": action_type,
                        "node_class": torch.ones(
                            *self.batch_shape,
                            self.num_node_classes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        "edge_class": torch.ones(
                            *self.batch_shape,
                            self.num_edge_classes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        "edge_index": edge_masks,
                    },
                    batch_size=self.batch_shape,
                )

            @property
            def backward_masks(self) -> TensorDict:
                """Compute masks for valid backward actions from the current state (a DAG).
                All existing edges are considered for removal.

                The EXIT action is not included in backward masks.

                Returns:
                    A TensorDict with the following keys:
                    - action_type: Tensor of shape [*batch_shape, 3] with True for valid action types
                    - node_class: Tensor of shape [*batch_shape, num_node_classes] (unused for this environment)
                    - edge_class: Tensor of shape [*batch_shape, n_nodes, n_nodes] (unused for this environment)
                    - edge_index: Tensor of shape [*batch_shape, n_nodes, n_nodes] with True for valid edge index to remove
                """

                batch_adjacency = (
                    to_dense_adj(self.tensor.edge_index, self.tensor.batch)
                    .squeeze(0)
                    .to(torch.bool)
                )
                if batch_adjacency.ndim == 2:
                    batch_adjacency = batch_adjacency.unsqueeze(0)
                edge_masks = batch_adjacency.flatten(1, 2).reshape(*self.batch_shape, -1)

                # There are 3 action types: ADD_NODE, ADD_EDGE, EXIT
                action_type = torch.zeros(
                    *self.batch_shape, 3, dtype=torch.bool, device=self.device
                )
                action_type[..., GraphActionType.ADD_EDGE] = torch.any(
                    edge_masks, dim=-1
                )

                return TensorDict(
                    {
                        "action_type": action_type,
                        "node_class": torch.ones(
                            *self.batch_shape,
                            self.num_node_classes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        "edge_class": torch.ones(
                            *self.batch_shape,
                            self.num_edge_classes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        "edge_index": edge_masks,
                    },
                    batch_size=self.batch_shape,
                )

        return BayesianStructureStates

    def make_random_states_tensor(
        self, batch_shape: int | Tuple, device: torch.device
    ) -> GeometricBatch:
        """Makes a batch of random DAG states with fixed number of nodes.

        Args:
            batch_shape: Shape of the batch dimensions.

        Returns:
            A PyG Batch object containing random DAG states.
        """
        assert self.s0.edge_attr is not None
        assert self.s0.x is not None

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = prod(batch_shape)

        data_list = []
        for _ in range(num_graphs):
            # Create a random DAG with the given number of nodes
            n_nodes = self.n_nodes

            # Create node features
            x = self.s0.x.clone()

            # Create the random number of edges
            n_edges = np.random.randint(0, n_nodes**2)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            adjacency = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=device)
            for i in range(n_edges):
                # Create self-loop mask
                self_loops = torch.eye(
                    self.n_nodes, dtype=torch.bool, device=self.device
                )
                # Compute transitive closure using the Floyd–Warshall style update:
                # reach[u, v] is True if there is a path from u to v.
                reach = adjacency.clone()
                for k in range(self.n_nodes):
                    reach = reach | (reach[:, k : k + 1] & reach[k : k + 1, :])
                # An edge u -> v is allowed if:
                # 1. It is not already in the graph (i.e. not in adjacency)
                # 2. It won't create a cycle (i.e. no path from v back to u: reach[v, u] is False)
                # 3. It is not a self-loop (i.e. u and v are different)
                allowed = (~adjacency) & (~reach.T) & (~self_loops)
                edge_mask = allowed.flatten()

                # sample a random edge
                src, dst = torch.where(edge_mask)
                if (n_valid := len(src)) > 0:
                    rand_idx = np.random.randint(0, n_valid)
                    src, dst = src[rand_idx], dst[rand_idx]
                else:
                    n_edges = i
                    break

                edge_index = torch.cat(
                    [edge_index, torch.tensor([[src], [dst]], device=device)], dim=1
                )
                adjacency[src, dst] = True

            # Create random edge attributes
            edge_attr = torch.rand(n_edges, self.s0.edge_attr.size(1), device=device)

            data = GeometricData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            data_list.append(data)

        if len(data_list) == 0:
            data_list = [
                GeometricData(
                    x=torch.zeros(0, self.s0.x.size(1)),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, self.s0.edge_attr.size(1)),
                )
            ]

        # Create a batch from the list
        batch = GeometricBatch.from_data_list(cast(List[BaseData], data_list))

        # Store the batch shape for later reference
        batch.batch_shape = batch_shape

        return batch

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
        # Flatten each mask, from (*batch_shape) to (prod(batch_shape),)
        exit_mask = (actions.action_type == GraphActionType.EXIT).flatten()
        add_edge_mask = (actions.action_type == GraphActionType.ADD_EDGE).flatten()

        # Handle ADD_EDGE actions
        if torch.any(add_edge_mask):
            add_edge_index = torch.where(add_edge_mask)[0]
            action_edge_index_flat = actions.edge_index.flatten()
            action_edge_class_flat = actions.edge_class.flatten()

            for i in add_edge_index:
                graph = data_list[i]
                edge_idx = action_edge_index_flat[i]

                # Get source and destination nodes for this edge
                src, dst = edge_idx // self.n_nodes, edge_idx % self.n_nodes

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
                    [graph.edge_attr, action_edge_class_flat[i].reshape(1, 1)], dim=0
                )

        # Handle EXIT actions
        if torch.any(exit_mask):
            # For graphs with EXIT action, replace them with sink states
            exit_indices = torch.where(exit_mask)[0]
            for idx in exit_indices:
                data_list[idx] = self.sf  # TODO: should we clone?

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

        # Get the data list from the batch for processing individual graphs
        data_list = states.tensor.to_data_list()

        add_edge_mask = (actions.action_type == GraphActionType.ADD_EDGE).flatten()
        assert (
            add_edge_mask.all()
        ), "Only ADD_EDGE actions are valid for backward step in this environment"

        # Handle ADD_EDGE actions
        add_edge_index = torch.where(add_edge_mask)[0]
        action_edge_index_flat = actions.edge_index.flatten()

        for i in add_edge_index:
            graph = data_list[i]
            edge_idx = action_edge_index_flat[i]

            # Get source and destination nodes for the edge to remove
            src, dst = edge_idx // self.n_nodes, edge_idx % self.n_nodes

            # Find the edge to remove
            edge_mask = ~((graph.edge_index[0] == src) & (graph.edge_index[1] == dst))
            # Remove the edge
            graph.edge_index = graph.edge_index[:, edge_mask]
            graph.edge_attr = graph.edge_attr[edge_mask]

        # Create a new batch from the updated data list
        new_tensor = GeometricBatch.from_data_list(data_list)
        new_tensor.batch_shape = states.tensor.batch_shape
        return new_tensor

    def is_action_valid(
        self, states: GraphStates, actions: GraphActions, backward: bool = False
    ) -> bool:
        """Check if actions are valid for the given states.

        Args:
            states: Current graph states.
            actions: Actions to validate.
            backward: Whether this is a backward step.

        Returns:
            True if all actions are valid, False otherwise.
        """
        if not backward and (actions.action_type == GraphActionType.ADD_NODE).any():
            return False
        if backward and (actions.action_type != GraphActionType.ADD_EDGE).any():
            return False

        # Get the data list from the batch
        data_list = states.tensor.to_data_list()
        action_type_flat = actions.action_type.flatten()
        edge_index_flat = actions.edge_index.flatten()

        for i in range(len(actions)):
            action_type = action_type_flat[i]
            if action_type == GraphActionType.EXIT:
                continue

            graph = data_list[i]
            assert isinstance(graph.num_nodes, int) and graph.num_nodes == self.n_nodes

            if action_type_flat[i] == GraphActionType.ADD_EDGE:
                edge_idx = edge_index_flat[i]
                src, dst = edge_idx // self.n_nodes, edge_idx % self.n_nodes

                # Check if the edge already exists
                edge_exists = torch.any(
                    (graph.edge_index[0] == src) & (graph.edge_index[1] == dst)
                )

                if backward:
                    # For backward actions, the edge must exist
                    if not edge_exists:
                        return False
                else:
                    # For forward actions, the edge must not exist
                    if edge_exists:
                        return False

        return True

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        raise NotImplementedError(
            "Use log_reward instead of reward for BayesianStructure environment."
        )

    def log_reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        return self.state_evaluator(final_states).to(self.device)
