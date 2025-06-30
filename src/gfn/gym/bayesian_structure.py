from math import prod
from typing import Callable, Literal, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torch_geometric.data import Data as GeometricData

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

            Each state is a graph with a fixed number of nodes where edges
            are being added incrementally to form a DAG.

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

                batch_adjacency = torch.zeros(
                    (len(self), self.n_nodes, self.n_nodes),
                    dtype=torch.bool,
                    device=self.device,
                )
                for i, graph in enumerate(self.data.flat):
                    src, dst = graph.edge_index
                    batch_adjacency[i, src, dst] = True

                # Create self-loop mask
                self_loops = torch.eye(
                    self.n_nodes, dtype=torch.bool, device=self.device
                ).repeat(len(self), 1, 1)
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
                        GraphActions.ACTION_TYPE_KEY: action_type,
                        GraphActions.NODE_CLASS_KEY: torch.ones(
                            *self.batch_shape,
                            self.num_node_classes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        GraphActions.EDGE_CLASS_KEY: torch.ones(
                            *self.batch_shape,
                            self.num_edge_classes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        GraphActions.EDGE_INDEX_KEY: edge_masks,
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

                batch_adjacency = torch.zeros(
                    (len(self), self.n_nodes, self.n_nodes),
                    dtype=torch.bool,
                    device=self.device,
                )
                for i, graph in enumerate(self.data.flat):
                    src, dst = graph.edge_index
                    batch_adjacency[i, src, dst] = True

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
                        GraphActions.ACTION_TYPE_KEY: action_type,
                        GraphActions.NODE_CLASS_KEY: torch.ones(
                            *self.batch_shape,
                            self.num_node_classes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        GraphActions.EDGE_CLASS_KEY: torch.ones(
                            *self.batch_shape,
                            self.num_edge_classes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        GraphActions.EDGE_INDEX_KEY: edge_masks,
                    },
                    batch_size=self.batch_shape,
                )

            @property
            def is_sink_state(self) -> torch.Tensor:
                """Returns a tensor that is True for states that are sf."""
                xs = torch.cat([graph.x for graph in self.data.flat], dim=1)  # type: ignore
                return (xs == self.sf.x).all(dim=0).view(self.batch_shape)

            @property
            def is_initial_state(self) -> torch.Tensor:
                """Returns a tensor that is True for states that are s0."""
                is_not_sink = ~self.is_sink_state
                has_edges = torch.tensor(
                    [graph.edge_index.shape[1] > 0 for graph in self.data.flat],  # type: ignore
                    device=self.device,
                ).view(self.batch_shape)
                return is_not_sink & ~has_edges

        return BayesianStructureStates

    def make_actions_class(self) -> type[GraphActions]:
        class GraphBuildingActions(GraphActions):
            @classmethod
            def edge_index_action_to_src_dst(
                cls, edge_index_action: torch.Tensor, n_nodes: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Converts the edge index action to source and destination node indices."""
                assert edge_index_action.ndim == 0  # TODO: support vector actions
                return edge_index_action // n_nodes, edge_index_action % n_nodes

        return GraphBuildingActions

    def make_random_states_tensor(
        self, batch_shape: int | Tuple, device: torch.device
    ) -> GraphStates:
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

        data_array = np.empty(batch_shape, dtype=object)
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

            data = GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_array.flat[i] = data

        return self.States(data_array, device=device)

    def step(self, states: GraphStates, actions: GraphActions) -> GraphStates:
        if torch.any(actions.action_type == GraphActionType.ADD_NODE):
            raise ValueError(
                "ADD_NODE action is not supported in BayesianStructure environment."
            )
        return super().step(states, actions)

    def backward_step(self, states: GraphStates, actions: GraphActions) -> GraphStates:
        if torch.any(actions.action_type == GraphActionType.ADD_NODE):
            raise ValueError(
                "ADD_NODE action is not supported in BayesianStructure environment."
            )
        return super().backward_step(states, actions)

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
        return super().is_action_valid(states, actions, backward)

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
