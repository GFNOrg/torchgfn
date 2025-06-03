from math import prod
from typing import Callable, List, Literal, Tuple, cast

import numpy as np
import torch
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData
from torch_geometric.data.data import BaseData

from gfn.actions import GraphActions, GraphActionType
from gfn.env import GraphEnv
from gfn.states import GraphStates
from gfn.utils.graphs import get_edge_indices


class GraphBuilding(GraphEnv):
    """Environment for incrementally building graphs.

    This environment allows constructing graphs by:
    - Adding nodes of a given class
    - Adding edges of a given class between existing nodes
    - Terminating construction (EXIT)

    Args:
        state_evaluator: Callable that computes rewards for final states.
            If None, uses default GCNConvEvaluator
        device_str: Device to run computations on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        num_node_classes: int,
        num_edge_classes: int,
        state_evaluator: Callable[[GraphStates], torch.Tensor],
        is_directed: bool = True,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        s0: GeometricData | None = None,
        sf: GeometricData | None = None,
    ):
        if s0 is None:
            s0 = GeometricData(
                x=torch.zeros((0, 1), dtype=torch.int64).to(device),
                edge_attr=torch.zeros((0, 1), dtype=torch.int64).to(device),
                edge_index=torch.zeros((2, 0), dtype=torch.long).to(device),
            )
        if sf is None:
            sf = GeometricData(
                x=torch.full((1, 1), -1, dtype=torch.int64).to(device),
                edge_attr=torch.full((0, 1), -1, dtype=torch.int64).to(device),
                edge_index=torch.zeros((2, 0), dtype=torch.long).to(device),
            )

        self.state_evaluator = state_evaluator
        super().__init__(
            s0=s0,
            sf=sf,
            num_node_classes=num_node_classes,
            num_edge_classes=num_edge_classes,
            is_directed=is_directed,
        )

    def step(self, states: GraphStates, actions: GraphActions) -> GeometricBatch:
        """Step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to add.

        Returns the next graph the new GraphStates.
        """
        if len(actions) == 0:
            return states.tensor

        data_list = states.tensor.to_data_list()

        # Create masks for different action types
        # Flatten each mask, from (*batch_shape) to (prod(batch_shape),)
        add_node_mask = (actions.action_type == GraphActionType.ADD_NODE).flatten()
        add_edge_mask = (actions.action_type == GraphActionType.ADD_EDGE).flatten()
        exit_mask = (actions.action_type == GraphActionType.EXIT).flatten()

        # Handle ADD_NODE action
        if torch.any(add_node_mask):
            batch_indices_flat = torch.arange(len(states))[add_node_mask]
            node_class_action_flat = actions.node_class.flatten()
            data_list = self._add_node(
                data_list, batch_indices_flat, node_class_action_flat
            )

        # Handle ADD_EDGE action
        if torch.any(add_edge_mask):
            add_edge_index = torch.where(add_edge_mask)[0]
            edge_index_action_flat = actions.edge_index.flatten()
            edge_class_action_flat = actions.edge_class.flatten()

            # Add edges to each graph
            for i in add_edge_index:
                graph = data_list[i]
                edge_idx = edge_index_action_flat[i]

                assert isinstance(graph.num_nodes, int)
                src, dst = self.Actions.edge_index_action_to_src_dst(
                    edge_idx, graph.num_nodes
                )
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
                    [graph.edge_attr, edge_class_action_flat[i].reshape(1, 1)], dim=0
                )

        # Handle EXIT action
        if torch.any(exit_mask):
            # For graphs with EXIT action, replace them with sink states
            exit_indices = torch.where(exit_mask)[0]
            for idx in exit_indices:
                data_list[idx] = self.sf

        # Create a new batch from the updated data list
        new_tensor = GeometricBatch.from_data_list(data_list)
        new_tensor.batch_shape = states.tensor.batch_shape
        return new_tensor

    def backward_step(
        self, states: GraphStates, actions: GraphActions
    ) -> GeometricBatch:
        """Backward step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to remove.

        Returns the previous graph as a new GraphStates.
        """
        if len(actions) == 0:
            return states.tensor

        # Get the data list from the batch
        data_list = states.tensor.to_data_list()

        # Create masks for different action types
        # Flatten each mask, from (*batch_shape) to (prod(batch_shape),)
        add_node_mask = (actions.action_type == GraphActionType.ADD_NODE).flatten()
        add_edge_mask = (actions.action_type == GraphActionType.ADD_EDGE).flatten()

        # Handle ADD_NODE action
        if torch.any(add_node_mask):
            add_node_index = torch.where(add_node_mask)[0]
            node_class_action_flat = actions.node_class.flatten()

            # Remove nodes with matching features
            for i in add_node_index:
                graph = data_list[i]
                assert isinstance(graph.num_nodes, int)

                # Find nodes with matching features
                is_equal = torch.all(
                    graph.x == node_class_action_flat[i].unsqueeze(0), dim=1
                )

                # Remove the first matching node
                node_idx = int(torch.where(is_equal)[0][0].item())

                # Remove the node
                mask = torch.ones(
                    graph.num_nodes, dtype=torch.bool, device=graph.x.device
                )
                mask[node_idx] = False

                # Update node features
                graph.x = graph.x[mask]

        # Handle ADD_EDGE action
        if torch.any(add_edge_mask):
            add_edge_index = torch.where(add_edge_mask)[0]
            edge_index_action_flat = actions.edge_index.flatten()

            # Remove edges with matching indices
            for i in add_edge_index:
                graph = data_list[i]
                edge_idx = edge_index_action_flat[i]

                assert isinstance(graph.num_nodes, int)
                src, dst = self.Actions.edge_index_action_to_src_dst(
                    edge_idx, graph.num_nodes
                )

                # Find the edge to remove
                edge_mask = ~(
                    (graph.edge_index[0] == src) & (graph.edge_index[1] == dst)
                )

                # Remove the edge
                graph.edge_index = graph.edge_index[:, edge_mask]
                graph.edge_attr = graph.edge_attr[edge_mask]

        # Create a new batch from the updated data list
        new_batch = GeometricBatch.from_data_list(data_list)
        new_batch.batch_shape = states.batch_shape
        return new_batch

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
        # Get the data list from the batch
        data_list = states.tensor.to_data_list()
        action_type_action_flat = actions.action_type.flatten()
        node_class_action_flat = actions.node_class.flatten()
        edge_index_action_flat = actions.edge_index.flatten()

        for i in range(len(actions)):
            action_type = action_type_action_flat[i]
            if action_type == GraphActionType.EXIT:
                continue

            graph = data_list[i]

            if action_type == GraphActionType.ADD_NODE:
                # Check if a node with these features already exists
                equal_nodes = torch.all(
                    graph.x == node_class_action_flat[i].unsqueeze(0), dim=1
                )

                if backward:
                    # For backward actions, we need at least one matching node
                    if not torch.any(equal_nodes):
                        return False
                else:
                    # For forward actions, we should not have any matching nodes
                    if torch.any(equal_nodes):
                        return False

            elif action_type == GraphActionType.ADD_EDGE:
                assert isinstance(graph.num_nodes, int)
                edge_idx = edge_index_action_flat[i]
                src, dst = self.Actions.edge_index_action_to_src_dst(
                    edge_idx, graph.num_nodes
                )

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

    def _add_node(
        self,
        data_list: List[BaseData],
        batch_indices_flat: torch.Tensor,
        node_class: torch.Tensor,
    ) -> List[BaseData]:
        """Add nodes to graphs in a batch.

        Args:
            data_list: The current batch of graphs.
            batch_indices: Indices of graphs to add nodes to.
            node_class: Class of nodes to add.

        Returns:
            Updated batch of graphs.
        """
        if len(batch_indices_flat) != len(node_class):
            raise ValueError(
                "Number of batch indices must match number of node feature lists"
            )

        # Add nodes to the specified graphs
        for graph_idx, new_node_class in zip(batch_indices_flat, node_class):
            # Get the graph to modify
            graph = data_list[graph_idx]
            assert graph.x is not None

            # Ensure new_nodes is 2D
            new_node_class = torch.atleast_2d(new_node_class)

            # Check feature dimension
            if new_node_class.shape[1] != graph.x.shape[1]:
                raise ValueError(f"Node features must have dimension {graph.x.shape[1]}")

            # Add new nodes to the graph
            graph.x = torch.cat([graph.x, new_node_class], dim=0)

        return data_list

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        return self.state_evaluator(final_states)

    def make_random_states_tensor(
        self, batch_shape: Tuple, device: torch.device | None = None
    ) -> GeometricBatch:
        """Generates random states tensor of shape (*batch_shape, feature_dim).

        Args:
            batch_shape: Shape of the batch dimensions.

        Returns:
            A PyG Batch object containing random graph states.
        """
        assert self.s0.edge_attr is not None
        assert self.s0.x is not None
        device = self.device if device is None else device

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = prod(batch_shape)

        data_list = []
        for _ in range(num_graphs):
            # Create a random graph with random number of nodes
            n_nodes = np.random.randint(1, 10)  # TODO: make the max n_nodes a parameter
            n_possible_edges = (
                n_nodes**2 - n_nodes if self.is_directed else (n_nodes**2 - n_nodes) // 2
            )

            # Create random node features
            x = torch.rand(n_nodes, self.s0.x.size(1), device=device)

            # Create random edges (not all possible edges to keep it sparse)
            n_edges = np.random.randint(0, n_possible_edges + 1)
            # Get all possible edge indices
            src, dst = get_edge_indices(n_nodes, self.is_directed, device)
            # Randomly select n_edges from all possible edges
            selected_indices = torch.randperm(len(src), device=device)[:n_edges]
            edge_index = torch.stack([src[selected_indices], dst[selected_indices]])

            # Create random edge attributes
            edge_attr = torch.rand(n_edges, self.s0.edge_attr.size(1), device=device)

            data = GeometricData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            data_list.append(data)

        if len(data_list) == 0:  # If batch_shape is 0, create a single empty graph
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

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class GraphBuildingStates(GraphStates):
            """Represents the state of a graph building process.

            The state representation consists of:
            - node_class: Class of each node (shape: [n_nodes, 1])
            - edge_class: Class of each edge (shape: [n_edges, 1])
            - edge_index: Indices representing the source and target nodes for each edge

            Special states:
            - s0: Initial state with n_nodes and no edges
            - sf: Terminal state (used as a placeholder)

            The class provides masks for both forward and backward actions to determine
            which actions are valid from the current state.
            """

            num_node_classes = env.num_node_classes
            num_edge_classes = env.num_edge_classes
            is_directed = env.is_directed

            s0 = env.s0
            sf = env.sf
            make_random_states_tensor = env.make_random_states_tensor

        return GraphBuildingStates

    def make_actions_class(self) -> type[GraphActions]:
        env = self

        class GraphBuildingActions(GraphActions):
            @classmethod
            def edge_index_action_to_src_dst(
                cls, edge_index_action: torch.Tensor, n_nodes: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Converts the edge index action to source and destination node indices."""
                assert edge_index_action.ndim == 0  # TODO: support vector actions
                src, dst = get_edge_indices(
                    n_nodes, env.is_directed, edge_index_action.device
                )
                return src[edge_index_action], dst[edge_index_action]

        return GraphBuildingActions


class GraphBuildingOnEdges(GraphBuilding):
    """Environment for building graphs edge by edge with discrete action space.

    The environment supports both directed and undirected graphs.

    In each state, the policy can:
    1. Add an edge between existing nodes.
    2. Use the exit action to terminate graph building.

    The action space is discrete, with size:
    - For directed graphs: n_nodes^2 - n_nodes + 1 (all possible directed edges + exit).
    - For undirected graphs: (n_nodes^2 - n_nodes)/2 + 1 (upper triangle + exit).

    Args:
        n_nodes: The number of nodes in the graph.
        state_evaluator: A function that evaluates a state and returns a reward.
        directed: Whether the graph should be directed.
    """

    def __init__(
        self,
        n_nodes: int,
        state_evaluator: callable,
        directed: bool,
        device: Literal["cpu", "cuda"] | torch.device,
    ):
        self.n_nodes = n_nodes
        if directed:
            # all off-diagonal edges.
            self.n_possible_edges = n_nodes**2 - n_nodes
        else:
            # bottom triangle.
            self.n_possible_edges = (n_nodes**2 - n_nodes) // 2

        s0 = GeometricData(
            x=torch.arange(self.n_nodes)[:, None].to(device),
            edge_attr=torch.ones((0, 1), device=device),
            edge_index=torch.ones((2, 0), dtype=torch.long, device=device),
        )
        sf = GeometricData(
            x=-torch.ones(self.n_nodes)[:, None].to(device),
            edge_attr=torch.zeros((0, 1), device=device),
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=device),
        )
        super().__init__(
            num_node_classes=1,
            num_edge_classes=1,
            state_evaluator=state_evaluator,
            is_directed=directed,
            device=device,
            s0=s0,
            sf=sf,
        )

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class GraphBuildingOnEdgesStates(GraphStates):
            """Represents the state of an edge-by-edge graph building process.

            This class extends GraphStates to specifically handle edge-by-edge graph
            building states. Each state represents a graph with a fixed number of nodes
            where edges are being added incrementally.

            The state representation consists of:
            - node_feature: Node IDs for each node in the graph (shape: [n_nodes, 1])
            - edge_feature: Features for each edge (shape: [n_edges, 1])
            - edge_index: Indices representing the source and target nodes for each edge
                (shape: [n_edges, 2])

            Special states:
            - s0: Initial state with n_nodes and no edges
            - sf: Terminal state (used as a placeholder)

            The class provides masks for both forward and backward actions to determine
            which actions are valid from the current state.
            """

            num_node_classes = env.num_node_classes
            num_edge_classes = env.num_edge_classes
            is_directed = env.is_directed
            n_nodes = env.n_nodes
            n_possible_edges = env.n_possible_edges
            s0 = env.s0
            sf = env.sf
            make_random_states_tensor = env.make_random_states_tensor

            @property
            def forward_masks(self) -> TensorDict:
                """Compute masks for valid forward actions from the current state.

                A forward action is valid if:
                1. The edge doesn't already exist in the graph
                2. The edge connects two distinct nodes

                For directed graphs, all possible src->dst edges are considered.
                For undirected graphs, only the upper triangular portion of the
                    adjacency matrix is used.

                Returns:
                    TensorDict: Boolean mask where True indicates valid actions
                """
                forward_masks = super().forward_masks
                forward_masks[GraphActions.ACTION_TYPE_KEY][
                    ..., GraphActionType.ADD_NODE
                ] = False
                return forward_masks

            @property
            def backward_masks(self) -> TensorDict:
                """Compute masks for valid backward actions from the current state.

                A backward action is valid if:
                1. The edge exists in the current graph (i.e., can be removed)

                For directed graphs, all existing edges are considered for removal.
                For undirected graphs, only the upper triangular edges are considered.

                The EXIT action is not included in backward masks.

                Returns:
                    TensorDict: Boolean mask where True indicates valid actions
                """
                backward_masks = super().backward_masks
                backward_masks[GraphActions.ACTION_TYPE_KEY][
                    ..., GraphActionType.ADD_NODE
                ] = False
                return backward_masks

        return GraphBuildingOnEdgesStates

    def make_random_states_tensor(
        self, batch_shape: Tuple, device: torch.device | None = None
    ) -> GeometricBatch:
        """Makes a batch of random graph states with fixed number of nodes.

        Args:
            batch_shape: Shape of the batch dimensions.

        Returns:
            A PyG Batch object containing random graph states.
        """
        assert self.s0.edge_attr is not None
        assert self.s0.x is not None
        device = self.device if device is None else device

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = prod(batch_shape)

        data_list = []
        for _ in range(num_graphs):
            # Create a graph with the given number of nodes
            n_nodes = self.n_nodes

            # Create node features
            x = self.s0.x.clone()

            # Create random edges using get_edge_indices
            n_edges = np.random.randint(0, self.n_possible_edges + 1)
            # Get all possible edge indices
            src, dst = get_edge_indices(n_nodes, self.is_directed, device)
            # Randomly select n_edges from all possible edges
            selected_indices = torch.randperm(len(src), device=device)[:n_edges]
            edge_index = torch.stack([src[selected_indices], dst[selected_indices]])

            # Create random edge attributes
            edge_attr = torch.rand(n_edges, self.s0.edge_attr.size(1), device=device)

            data = GeometricData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            data_list.append(data)

        if len(data_list) == 0:  # If batch_shape is 0, create a single empty graph
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

    def is_action_valid(
        self, states: GraphStates, actions: GraphActions, backward: bool = False
    ) -> bool:
        if not backward and (actions.action_type == GraphActionType.ADD_NODE).any():
            return False
        if backward and (actions.action_type != GraphActionType.ADD_EDGE).any():
            return False
        return super().is_action_valid(states, actions, backward)
