from typing import Callable, Literal, Optional, Tuple

import torch
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData

from gfn.actions import Actions, GraphActions, GraphActionType
from gfn.env import GraphEnv
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
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        s0 = GeometricData(
            x=torch.zeros((0, feature_dim), dtype=torch.float32).to(device),
            edge_attr=torch.zeros((0, feature_dim), dtype=torch.float32).to(device),
            edge_index=torch.zeros((2, 0), dtype=torch.long).to(device),
            device=device,
        )
        sf = GeometricData(
            x=torch.ones((1, feature_dim), dtype=torch.float32).to(device)
            * float("inf"),
            edge_attr=torch.ones((0, feature_dim), dtype=torch.float32).to(device)
            * float("inf"),
            edge_index=torch.zeros((2, 0), dtype=torch.long).to(device),
            device=device,
        )

        self.state_evaluator = state_evaluator
        self.feature_dim = feature_dim

        super().__init__(
            s0=s0,
            sf=sf,
        )

    def reset(
        self,
        batch_shape: int | Tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        seed: Optional[int] = None,
    ) -> GraphStates:
        """Reset the environment to a new batch of graphs."""
        states = super().reset(batch_shape, random, sink, seed)
        assert isinstance(states, GraphStates)
        return states

    def step(self, states: GraphStates, actions: GraphActions) -> GeometricBatch:
        """Step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to add.

        Returns the next graph the new GraphStates.
        """
        if len(actions) == 0:
            return states.tensor

        action_type = actions.action_type[0]
        assert torch.all(
            actions.action_type == action_type
        )  # TODO: allow different action types
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
            # Get the data list from the batch
            data_list = states.tensor.to_data_list()

            # Add edges to each graph
            for i, (src, dst) in enumerate(actions.edge_index):
                # Get the graph to modify
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
            states.tensor = new_tensor

        return states.tensor

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

        action_type = actions.action_type[0]
        assert torch.all(actions.action_type == action_type)

        # Get the data list from the batch
        data_list = states.tensor.to_data_list()

        if action_type == GraphActionType.ADD_NODE:
            # Remove nodes with matching features
            for i, features in enumerate(actions.features):
                graph = data_list[i]
                assert isinstance(graph.num_nodes, int)

                # Find nodes with matching features
                is_equal = torch.all(graph.x == features.unsqueeze(0), dim=1)

                if torch.any(is_equal):
                    # Remove the first matching node
                    node_idx = int(torch.where(is_equal)[0][0].item())

                    # Remove the node
                    mask = torch.ones(
                        graph.num_nodes,
                        dtype=torch.bool,
                        device=graph.x.device,
                    )
                    mask[node_idx] = False

                    # Update node features
                    graph.x = graph.x[mask]

        elif action_type == GraphActionType.ADD_EDGE:
            # Remove edges with matching indices
            for i, (src, dst) in enumerate(actions.edge_index):
                graph = data_list[i]

                # Find the edge to remove
                edge_mask = ~(
                    (graph.edge_index[0] == src) & (graph.edge_index[1] == dst)
                )

                # Remove the edge
                graph.edge_index = graph.edge_index[:, edge_mask]
                graph.edge_attr = graph.edge_attr[edge_mask]

        # Create a new batch from the updated data list
        new_batch = GeometricBatch.from_data_list(data_list)

        # Preserve the batch shape
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

        for i in range(len(actions)):
            graph = data_list[i]
            assert isinstance(graph.num_nodes, int)

            if actions.action_type[i] == GraphActionType.ADD_NODE:
                # Check if a node with these features already exists
                equal_nodes = torch.all(
                    graph.x == actions.features[i].unsqueeze(0), dim=1
                )

                if backward:
                    # For backward actions, we need at least one matching node
                    if not torch.any(equal_nodes):
                        return False
                else:
                    # For forward actions, we should not have any matching nodes
                    if torch.any(equal_nodes):
                        return False

            elif actions.action_type[i] == GraphActionType.ADD_EDGE:
                src, dst = actions.edge_index[i]

                # Check if src and dst are valid node indices
                if src >= graph.num_nodes or dst >= graph.num_nodes or src == dst:
                    return False

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
        tensor: GeometricBatch,
        batch_indices: torch.Tensor | list[int],
        nodes_to_add: torch.Tensor,
    ) -> GeometricBatch:
        """Add nodes to graphs in a batch.

        Args:
            tensor_dict: The current batch of graphs.
            batch_indices: Indices of graphs to add nodes to.
            nodes_to_add: Features of nodes to add.

        Returns:
            Updated batch of graphs.
        """
        batch_indices = (
            torch.tensor(batch_indices)
            if isinstance(batch_indices, list)
            else batch_indices
        )
        if len(batch_indices) != len(nodes_to_add):
            raise ValueError(
                "Number of batch indices must match number of node feature lists"
            )

        # Get the data list from the batch
        data_list = tensor.to_data_list()

        # Add nodes to the specified graphs
        for graph_idx, new_nodes in zip(batch_indices, nodes_to_add):
            # Get the graph to modify
            graph = data_list[graph_idx]

            # Ensure new_nodes is 2D
            new_nodes = torch.atleast_2d(new_nodes)

            # Check feature dimension
            if new_nodes.shape[1] != graph.x.shape[1]:
                raise ValueError(f"Node features must have dimension {graph.x.shape[1]}")

            # Add new nodes to the graph
            graph.x = torch.cat([graph.x, new_nodes], dim=0)

        # Create a new batch from the updated data list
        new_batch = GeometricBatch.from_data_list(data_list)

        # Preserve the batch shape
        new_batch.batch_shape = tensor.batch_shape
        return new_batch

    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        return self.state_evaluator(final_states)

    def make_random_states_tensor(self, batch_shape: Tuple) -> GraphStates:
        """Generates random states tensor of shape (*batch_shape, feature_dim)."""
        random_states_tensor = self.States.from_batch_shape(batch_shape)
        assert isinstance(random_states_tensor, GraphStates)
        return random_states_tensor


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
        device: torch.device,
    ):
        self.n_nodes = n_nodes
        if directed:
            # all off-diagonal edges + exit.
            self.n_actions = (n_nodes**2 - n_nodes) + 1
        else:
            # bottom triangle + exit.
            self.n_actions = ((n_nodes**2 - n_nodes) // 2) + 1
        super().__init__(
            feature_dim=n_nodes,
            state_evaluator=state_evaluator,
            device=device,
        )
        self.is_discrete = True  # actions here are discrete, needed for FlowMatching
        self.is_directed = directed

    def make_actions_class(self) -> type[Actions]:
        env = self

        class EdgeActions(Actions):
            """Actions for building graphs with a fixed number of nodes.

            Actions are represented as discrete indices where:
            - 0 to n_actions-2: Adding an edge between specific nodes
            - n_actions-1: EXIT action to terminate the trajectory
            - n_actions: DUMMY action (used for padding)
            """

            action_shape = (1,)
            dummy_action = torch.tensor([env.n_actions]).to(env.device)
            exit_action = torch.tensor([env.n_actions - 1]).to(env.device)

        return EdgeActions

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

            s0 = GeometricData(
                x=torch.arange(env.n_nodes)[:, None].to(env.device),
                edge_attr=torch.ones((0, 1), device=env.device),
                edge_index=torch.ones((2, 0), dtype=torch.long, device=env.device),
            )
            sf = GeometricData(
                x=-torch.ones(env.n_nodes)[:, None].to(env.device),
                edge_attr=torch.zeros((0, 1), device=env.device),
                edge_index=torch.zeros((2, 0), dtype=torch.long, device=env.device),
            )

            def __init__(self, tensor: GeometricBatch):
                self.tensor = tensor.to(env.device)
                self.node_features_dim = tensor.x.shape[-1]
                self.edge_features_dim = tensor.edge_attr.shape[-1]
                self._log_rewards: Optional[float] = None

                self.n_nodes = env.n_nodes
                self.n_actions = env.n_actions

            @property
            def forward_masks(self):
                """Compute masks for valid forward actions from the current state.

                A forward action is valid if:
                1. The edge doesn't already exist in the graph
                2. The edge connects two distinct nodes

                For directed graphs, all possible src->dst edges are considered.
                For undirected graphs, only the upper triangular portion of the
                    adjacency matrix is used.

                The last action is always the EXIT action, which is always valid.

                Returns:
                    Tensor: Boolean mask of shape [batch_size, n_actions] where
                        True indicates valid actions
                """
                # Allow all actions.
                forward_masks = torch.ones(
                    len(self), self.n_actions, dtype=torch.bool, device=self.device
                )

                if env.is_directed:
                    i_up, j_up = torch.triu_indices(
                        self.n_nodes, self.n_nodes, offset=1, device=self.device
                    )  # Upper triangle.
                    i_lo, j_lo = torch.tril_indices(
                        self.n_nodes, self.n_nodes, offset=-1, device=self.device
                    )  # Lower triangle.

                    # Combine them
                    ei0 = torch.cat([i_up, i_lo])
                    ei1 = torch.cat([j_up, j_lo])
                else:
                    ei0, ei1 = torch.triu_indices(
                        self.n_nodes, self.n_nodes, offset=1, device=self.device
                    )

                # Remove existing edges.
                for i in range(len(self)):
                    existing_edges = self[i].tensor.edge_index
                    assert torch.all(existing_edges >= 0)  # TODO: convert to test.

                    if existing_edges.numel() == 0:
                        edge_idx = torch.zeros(0, dtype=torch.bool, device=self.device)
                    else:
                        edge_idx = torch.logical_and(
                            existing_edges[0][..., None] == ei0[None],
                            existing_edges[1][..., None] == ei1[None],
                        ).to(self.device)

                        # Collapse across the edge dimension.
                        if len(edge_idx.shape) == 2:
                            edge_idx = edge_idx.sum(0).bool()

                        # Adds an unmasked exit action.
                        edge_idx = torch.cat(
                            (edge_idx, torch.tensor([False], device=self.device))
                        )
                        forward_masks[i, edge_idx] = (
                            False  # Disallow the addition of this edge.
                        )

                return forward_masks.view(*self.batch_shape, self.n_actions)

            @forward_masks.setter
            def forward_masks(self, value: torch.Tensor):
                pass  # fwd masks is computed on the fly

            @property
            def backward_masks(self):
                """Compute masks for valid backward actions from the current state.

                A backward action is valid if:
                1. The edge exists in the current graph (i.e., can be removed)

                For directed graphs, all existing edges are considered for removal.
                For undirected graphs, only the upper triangular edges are considered.

                The EXIT action is not included in backward masks.

                Returns:
                    Tensor: Boolean mask of shape [batch_size, n_actions-1] where True indicates valid actions
                """
                # Disallow all actions.
                backward_masks = torch.zeros(
                    len(self), self.n_actions - 1, dtype=torch.bool, device=self.device
                )

                for i in range(len(self)):
                    existing_edges = self[i].tensor.edge_index

                    if env.is_directed:
                        i_up, j_up = torch.triu_indices(
                            self.n_nodes, self.n_nodes, offset=1, device=self.device
                        )  # Upper triangle.
                        i_lo, j_lo = torch.tril_indices(
                            self.n_nodes, self.n_nodes, offset=-1, device=self.device
                        )  # Lower triangle.

                        # Combine them
                        ei0 = torch.cat([i_up, i_lo])
                        ei1 = torch.cat([j_up, j_lo])
                    else:
                        ei0, ei1 = torch.triu_indices(
                            self.n_nodes, self.n_nodes, offset=1, device=self.device
                        )

                    if len(existing_edges) == 0:
                        edge_idx = torch.zeros(0, dtype=torch.bool)
                    else:
                        edge_idx = torch.logical_and(
                            existing_edges[0][..., None] == ei0[None],
                            existing_edges[1][..., None] == ei1[None],
                        )
                        # Collapse across the edge dimension.
                        if len(edge_idx.shape) == 2:
                            edge_idx = edge_idx.sum(0).bool()

                        backward_masks[i, edge_idx] = (
                            True  # Allow the removal of this edge.
                        )

                return backward_masks.view(*self.batch_shape, self.n_actions - 1)

            @backward_masks.setter
            def backward_masks(self, value: torch.Tensor):
                pass  # bwd masks is computed on the fly

        return GraphBuildingOnEdgesStates

    def _step(self, states: GraphStates, actions: Actions) -> GraphStates:
        """Take a step in the environment by applying actions to states.

        Args:
            states: Current states batch
            actions: Actions to apply

        Returns:
            New states after applying the actions
        """
        actions = self.convert_actions(states, actions)
        new_states = super()._step(states, actions)
        assert isinstance(new_states, GraphStates)
        return new_states

    def _backward_step(self, states: GraphStates, actions: Actions) -> GraphStates:
        """Take a backward step in the environment.

        Args:
            states: Current states batch
            actions: Actions to apply in reverse

        Returns:
            New states after applying the backward actions
        """
        actions = self.convert_actions(states, actions)
        new_states = super()._backward_step(states, actions)
        assert isinstance(new_states, GraphStates)
        return new_states

    def convert_actions(self, states: GraphStates, actions: Actions) -> GraphActions:
        """Converts the action from discrete space to graph action space.

        This method maps discrete action indices to specific graph operations:
        - GraphActionType.ADD_EDGE: Add an edge between specific nodes
        - GraphActionType.EXIT: Terminate trajectory
        - GraphActionType.DUMMY: No-op action (for padding)

        Args:
            states: Current states batch
            actions: Discrete actions to convert

        Returns:
            Equivalent actions in the GraphActions format
        """
        # TODO: factor out into utility function.
        action_tensor = actions.tensor.squeeze(-1).clone()
        action_type = torch.where(
            action_tensor == self.n_actions - 1,
            GraphActionType.EXIT,
            GraphActionType.ADD_EDGE,
        )
        action_type[action_tensor == self.n_actions] = GraphActionType.DUMMY

        # Convert action indices to source-target node pairs
        # TODO: factor out into utility function.
        if self.is_directed:
            i_up, j_up = torch.triu_indices(
                self.n_nodes, self.n_nodes, offset=1
            )  # Upper triangle.
            i_lo, j_lo = torch.tril_indices(
                self.n_nodes, self.n_nodes, offset=-1
            )  # Lower triangle.

            # Combine them
            ei0 = torch.cat([i_up, i_lo])
            ei1 = torch.cat([j_up, j_lo])

        else:
            ei0, ei1 = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)

        # Adds -1 "edge" representing exit, -2 "edge" representing dummy.
        ei0 = torch.cat((ei0, torch.tensor([-1, -2])), dim=0).to(self.device)
        ei1 = torch.cat((ei1, torch.tensor([-1, -2])), dim=0).to(self.device)

        # Indexes either the second last element (exit) or la
        # action_tensor[action_tensor >= (self.n_actions - 1)] = 0
        ei0, ei1 = ei0[action_tensor], ei1[action_tensor]

        out = GraphActions(
            TensorDict(
                {
                    "action_type": action_type,
                    "features": torch.ones(action_tensor.shape + (1,)),
                    "edge_index": torch.stack([ei0, ei1], dim=-1),
                },
                batch_size=action_tensor.shape,
            ).to(self.device)
        )
        return out
