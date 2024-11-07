from copy import copy
from typing import Callable, Literal, Tuple

import torch
from gfn.actions import Actions
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from gfn.env import GraphEnv
from gfn.states import GraphStates


class GraphBuilding(GraphEnv):

    def __init__(self,
        num_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        state_evaluator: Callable[[Batch], torch.Tensor] | None = None,
        device_str: Literal["cpu", "cuda"] = "cpu"
    ):
        s0 = Data(x=torch.zeros((num_nodes, node_feature_dim)).to(device_str))
        exit_action = torch.tensor(
            [-float("inf"), -float("inf")], device=torch.device(device_str)
        )
        dummy_action = torch.tensor(
            [float("inf"), float("inf")], device=torch.device(device_str)
        )
        if state_evaluator is None:
            state_evaluator = GCNConvEvaluator(node_feature_dim)
        self.state_evaluator = state_evaluator

        super().__init__(
            s0=s0,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,           
            action_shape=(2,),
            dummy_action=dummy_action,
            exit_action=exit_action,
            device_str=device_str,
        )


    def step(self, states: GraphStates, actions: Actions) -> GraphStates:
        """Step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to add.

        Returns the next graph the new GraphStates.
        """
        graphs: Batch = copy.deepcopy(states.data)
        assert len(graphs) == len(actions)

        for i, act in enumerate(actions.tensor):
            edge_index = torch.cat([graphs[i].edge_index, act.unsqueeze(1)], dim=1)
            graphs[i].edge_index = edge_index
        
        return GraphStates(graphs)

    def backward_step(self, states: GraphStates, actions: Actions) -> GraphStates:
        """Backward step function for the GraphBuilding environment.

        Args:
            states: GraphStates object representing the current graph.
            actions: Actions indicating which edge to remove.

        Returns the previous graph as a new GraphStates.
        """
        graphs: Batch = copy.deepcopy(states.data)
        assert len(graphs) == len(actions)

        for i, act in enumerate(actions.tensor):
            edge_index = graphs[i].edge_index
            edge_index = edge_index[:, edge_index[1] != act]
            graphs[i].edge_index = edge_index
        
        return GraphStates(graphs)

    def is_action_valid(
        self, states: GraphStates, actions: Actions, backward: bool = False
    ) -> bool:
        for i, act in enumerate(actions.tensor):
            if backward and len(states.data[i].edge_index[1]) == 0:
                return False
            if not backward and torch.any(states.data[i].edge_index[1] == act):
                return False
        return True
        
    def reward(self, final_states: GraphStates) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        return self.state_evaluator(final_states.data).sum(dim=1)

    @property
    def log_partition(self) -> float:
        "Returns the logarithm of the partition function."
        raise NotImplementedError

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        "Returns a one-dimensional tensor representing the true distribution."
        raise NotImplementedError


class GCNConvEvaluator:
    def __init__(self, num_features):
        self.net = GCNConv(num_features, 1)

    def __call__(self, batch: Batch) -> torch.Tensor:
        return self.net(batch.x, batch.edge_index)