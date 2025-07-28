from math import prod
from typing import Callable, Literal, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torch_geometric.data import Data as GeometricData

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

    Attributes:
        num_node_classes: The number of node classes.
        num_edge_classes: The number of edge classes.
        state_evaluator: A callable that computes rewards for final states.
        is_directed: Whether the graph is directed.
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
        """Initializes the GraphBuilding environment.

        Args:
            num_node_classes: The number of node classes.
            num_edge_classes: The number of edge classes.
            state_evaluator: A callable that computes rewards for final states.
            is_directed: Whether the graph is directed.
            device: The device to run computations on.
            s0: The initial state.
            sf: The sink state.
        """
        if s0 is None:
            s0 = GeometricData(
                x=torch.zeros((0, 1), dtype=torch.long).to(
                    device
                ),  # TODO: should dtype be allowed to be float?
                edge_attr=torch.zeros((0, 1), dtype=torch.long).to(
                    device
                ),  # TODO: should dtype be allowed to be float?
                edge_index=torch.zeros((2, 0), dtype=torch.long).to(device),
            )
        if sf is None:
            sf = GeometricData(
                x=torch.full((1, 1), -1, dtype=torch.long).to(
                    device
                ),  # TODO: should dtype be allowed to be float?
                edge_attr=torch.full((0, 1), -1, dtype=torch.long).to(
                    device
                ),  # TODO: should dtype be allowed to be float?
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

    def step(self, states: GraphStates, actions: GraphActions) -> GraphStates:
        """Performs a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The next states.
        """
        if len(actions) == 0:
            return states

        data_array = states.data.flatten()
        # Create masks for different action types
        # Flatten each mask, from (*batch_shape) to (prod(batch_shape),)
        add_node_mask = (actions.action_type == GraphActionType.ADD_NODE).flatten()
        add_edge_mask = (actions.action_type == GraphActionType.ADD_EDGE).flatten()
        exit_mask = (actions.action_type == GraphActionType.EXIT).flatten()

        # Handle ADD_NODE action
        if torch.any(add_node_mask):
            batch_indices_flat = torch.arange(len(states))[add_node_mask]
            node_class_action_flat = actions.node_class.flatten()

            # Add nodes to the specified graphs
            for graph_idx, new_node_class in zip(
                batch_indices_flat, node_class_action_flat
            ):
                # Get the graph to modify
                graph = data_array[graph_idx]

                # Ensure new_nodes is 2D
                new_node_class = torch.atleast_2d(new_node_class)

                # Check feature dimension
                if new_node_class.shape[1] != graph.x.shape[1]:
                    raise ValueError(
                        f"Node features must have dimension {graph.x.shape[1]}"
                    )

                # Add new nodes to the graph
                graph.x = torch.cat([graph.x, new_node_class], dim=0)

        # Handle ADD_EDGE action
        if torch.any(add_edge_mask):
            add_edge_index = torch.where(add_edge_mask)[0]
            edge_index_action_flat = actions.edge_index.flatten()
            edge_class_action_flat = actions.edge_class.flatten()

            # Add edges to each graph
            for i in add_edge_index:
                graph = data_array[i]
                edge_idx = edge_index_action_flat[i]
                src, dst = self.Actions.edge_index_action_to_src_dst(
                    edge_idx, graph.x.size(0)
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
                data_array[idx] = self.sf

        # Create a new batch from the updated data list
        return self.States(data_array, device=states.device)

    def backward_step(self, states: GraphStates, actions: GraphActions) -> GraphStates:
        """Performs a backward step in the environment.

        Args:
            states: The current states.
            actions: The actions to undo.

        Returns:
            The previous states.
        """
        if len(actions) == 0:
            return states

        # Get the data list from the batch
        data_array = states.data.flatten()
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
                graph = data_array[i]

                # Find nodes with matching features
                is_equal = torch.all(
                    graph.x == node_class_action_flat[i].unsqueeze(0), dim=1
                )

                # Remove the first matching node
                node_idx = int(torch.where(is_equal)[0][0].item())

                # Remove the node
                mask = torch.ones(
                    graph.x.size(0), dtype=torch.bool, device=graph.x.device
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
                graph = data_array[i]
                edge_idx = edge_index_action_flat[i]
                src, dst = self.Actions.edge_index_action_to_src_dst(
                    edge_idx, graph.x.size(0)
                )
                # Find the edge to remove
                edge_mask = ~(
                    (graph.edge_index[0] == src) & (graph.edge_index[1] == dst)
                )

                # Remove the edge
                graph.edge_index = graph.edge_index[:, edge_mask]
                graph.edge_attr = graph.edge_attr[edge_mask]

        # Create a new batch from the updated data list
        return self.States(data_array, device=states.device)

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
        data_array = states.data.flat
        action_type_action_flat = actions.action_type.flatten()
        node_class_action_flat = actions.node_class.flatten()
        edge_index_action_flat = actions.edge_index.flatten()

        for i in range(len(actions)):
            action_type = action_type_action_flat[i]
            if action_type == GraphActionType.EXIT:
                continue

            graph = data_array[i]
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
                edge_idx = edge_index_action_flat[i]
                src, dst = self.Actions.edge_index_action_to_src_dst(
                    edge_idx, graph.x.size(0)
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

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.

        Args:
            final_states: A batch of final states.

        Returns:
            A tensor of shape `(batch_size,)` containing the rewards.
        """
        return self.state_evaluator(final_states)

    def make_random_states(
        self, batch_shape: Tuple, device: torch.device | None = None
    ) -> GraphStates:
        """Generates random states.

        Args:
            batch_shape: The shape of the batch.
            device: The device to use.

        Returns:
            A `GraphStates` object with random states.
        """
        assert self.s0.edge_attr is not None
        assert self.s0.x is not None
        device = self.device if device is None else device

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = prod(batch_shape)

        data_array = np.empty(batch_shape, dtype=object)
        for i in range(num_graphs):
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
            data_array.flat[i] = data

        return self.States(data_array, device=device)

    def make_states_class(self) -> type[GraphStates]:
        """Creates a `GraphStates` class for this environment."""
        env = self

        class GraphBuildingStates(GraphStates):
            """Represents the state of a graph building process.

            The state representation consists of:
            - node_class: Class of each node (shape: `[n_nodes, 1]`)
            - edge_class: Class of each edge (shape: `[n_edges, 1]`)
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
            make_random_states = env.make_random_states

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
    - For directed graphs: `n_nodes^2 - n_nodes + 1` (all possible directed edges + exit).
    - For undirected graphs: `(n_nodes^2 - n_nodes)/2 + 1` (upper triangle + exit).

    Attributes:
        n_nodes (int): The number of nodes in the graph.
        n_possible_edges (int): The number of possible edges.
    """

    def __init__(
        self,
        n_nodes: int,
        state_evaluator: callable,
        directed: bool,
        device: Literal["cpu", "cuda"] | torch.device,
    ):
        """Initializes the `GraphBuildingOnEdges` environment.

        Args:
            n_nodes: The number of nodes in the graph.
            state_evaluator: A function that evaluates a state and returns a reward.
            directed: Whether the graph should be directed.
            device: The device to use.
        """
        self.n_nodes = n_nodes
        if directed:
            # all off-diagonal edges.
            self.n_possible_edges = n_nodes**2 - n_nodes
        else:
            # bottom triangle.
            self.n_possible_edges = (n_nodes**2 - n_nodes) // 2

        s0 = GeometricData(
            x=torch.arange(self.n_nodes, dtype=torch.long)[:, None].to(
                device
            ),  # TODO: should dtype be allowed to be float?
            edge_attr=torch.ones(
                (0, 1), dtype=torch.long, device=device
            ),  # TODO: should dtype be allowed to be float?
            edge_index=torch.ones((2, 0), dtype=torch.long, device=device),
        )
        sf = GeometricData(
            x=-torch.ones(self.n_nodes, dtype=torch.long)[:, None].to(
                device
            ),  # TODO: should dtype be allowed to be float?
            edge_attr=torch.zeros(
                (0, 1), dtype=torch.long, device=device
            ),  # TODO: should dtype be allowed to be float?
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
        """Creates a `GraphStates` class for this environment."""
        env = self

        class GraphBuildingOnEdgesStates(GraphStates):
            """Represents the state of an edge-by-edge graph building process.

            This class extends `GraphStates` to specifically handle edge-by-edge graph
            building states. Each state represents a graph with a fixed number of nodes
            where edges are being added incrementally.

            The state representation consists of:
            - node_feature: Node IDs for each node in the graph (shape: `[n_nodes, 1]`)
            - edge_feature: Features for each edge (shape: `[n_edges, 1]`)
            - edge_index: Indices representing the source and target nodes for each edge
                (shape: `[n_edges, 2]`)

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
            make_random_states = env.make_random_states

            @property
            def forward_masks(self) -> TensorDict:
                """Compute masks for valid forward actions from the current state.

                A forward action is valid if:
                1. The edge doesn't already exist in the graph
                2. The edge connects two distinct nodes

                For directed graphs, all possible `src->dst` edges are considered.
                For undirected graphs, only the upper triangular portion of the
                adjacency matrix is used.

                Returns:
                    A boolean mask where `True` indicates valid actions.
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
                    A boolean mask where `True` indicates valid actions.
                """
                backward_masks = super().backward_masks
                backward_masks[GraphActions.ACTION_TYPE_KEY][
                    ..., GraphActionType.ADD_NODE
                ] = False
                return backward_masks

            @property
            def is_sink_state(self) -> torch.Tensor:
                """Returns a tensor that is `True` for states that are `sf`."""
                xs = torch.cat([g.x for g in self.data.flat], dim=1)  # type: ignore
                return (xs == self.sf.x).all(dim=0).view(self.batch_shape)

            @property
            def is_initial_state(self) -> torch.Tensor:
                """Returns a tensor that is `True` for states that are `s0`."""
                is_not_sink = ~self.is_sink_state
                has_edges = torch.tensor(
                    [g.edge_index.shape[1] > 0 for g in self.data.flat],  # type: ignore
                    device=self.device,
                ).view(self.batch_shape)
                return is_not_sink & ~has_edges

        return GraphBuildingOnEdgesStates

    def make_random_states(
        self, batch_shape: Tuple, device: torch.device | None = None
    ) -> GraphStates:
        """Makes a batch of random graph states with fixed number of nodes.

        Args:
            batch_shape: Shape of the batch dimensions.
            device: The device to use.

        Returns:
            A `GraphStates` object containing random graph states.
        """
        assert self.s0.edge_attr is not None
        assert self.s0.x is not None
        device = self.device if device is None else device

        batch_shape = batch_shape if isinstance(batch_shape, Tuple) else (batch_shape,)
        num_graphs = prod(batch_shape)

        data_array = np.empty(batch_shape, dtype=object)
        for i in range(num_graphs):
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

            data = GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_array.flat[i] = data

        return self.States(data_array, device=device)

    def is_action_valid(
        self, states: GraphStates, actions: GraphActions, backward: bool = False
    ) -> bool:
        """Checks if the actions are valid.

        Args:
            states: The current states.
            actions: The actions to validate.
            backward: Whether the actions are backward actions.

        Returns:
            `True` if the actions are valid, `False` otherwise.
        """
        if not backward and (actions.action_type == GraphActionType.ADD_NODE).any():
            return False
        if backward and (actions.action_type != GraphActionType.ADD_EDGE).any():
            return False
        return super().is_action_valid(states, actions, backward)
