"""Write ane xamples where we want to create graphs that are rings."""

import torch
from torch import nn
from gfn.actions import Actions, GraphActionType, GraphActions
from gfn.gflownet.flow_matching import FMGFlowNet
from gfn.gym import GraphBuilding
from gfn.modules import DiscretePolicyEstimator
from gfn.preprocessors import Preprocessor
from gfn.states import GraphStates
from tensordict import TensorDict
from torch_geometric.nn import GCNConv


def state_evaluator(states: GraphStates) -> torch.Tensor:
    if states.tensor["edge_index"].shape[0] == 0:
        return torch.zeros(states.batch_shape)
    if states.tensor["edge_index"].shape[0] != states.tensor["node_feature"].shape[0]:
        return torch.zeros(states.batch_shape)
    
    i0 = torch.unique(states.tensor["edge_index"][0], sorted=False)
    i1 = torch.unique(states.tensor["edge_index"][1], sorted=False)

    if len(i0) == len(i1) == states.tensor["node_feature"].shape[0]:
        return torch.ones(states.batch_shape)
    return torch.zeros(states.batch_shape)


class RingPolicyEstimator(nn.Module):
    def __init__(self, n_nodes: int):
        super().__init__()
        self.action_type_conv = GCNConv(1, 1)
        self.edge_index_conv = GCNConv(1, 8)
        self.n_nodes = n_nodes

    def _group_sum(self, tensor: torch.Tensor, batch_ptr: torch.Tensor) -> torch.Tensor:
        cumsum = torch.zeros((len(tensor) + 1, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        cumsum[1:] = torch.cumsum(tensor, dim=0)
        return cumsum[batch_ptr[1:]] - cumsum[batch_ptr[:-1]]

    def forward(self, states_tensor: TensorDict) -> torch.Tensor:
        node_feature = states_tensor["node_feature"].reshape(-1, 1)
        edge_index = states_tensor["edge_index"].T
        batch_ptr = states_tensor["batch_ptr"]

        action_type = self.action_type_conv(node_feature, edge_index)
        action_type = self._group_sum(action_type, batch_ptr)

        edge_index = self.edge_index_conv(node_feature, edge_index)
        #edge_index = self._group_sum(edge_index, batch_ptr)
        edge_index = edge_index.reshape(*states_tensor["batch_shape"], -1, 8)
        edge_index = torch.einsum("bnf,bmf->bnm", edge_index, edge_index)
        torch.diagonal(edge_index, dim1=-2, dim2=-1).fill_(float("-inf"))
        edge_actions = edge_index.reshape(*states_tensor["batch_shape"], -1)

        return torch.cat([action_type, edge_actions], dim=-1)

class RingGraphBuilding(GraphBuilding):
    def __init__(self, nodes: int = 10):
        self.nodes = nodes
        self.n_actions = 1 + nodes * nodes
        super().__init__(feature_dim=1, state_evaluator=state_evaluator)

    
    def make_actions_class(self) -> type[Actions]:
        env = self
        class RingActions(Actions):
            action_shape = (1,)
            dummy_action = torch.tensor([env.n_actions])
            exit_action = torch.zeros(1,)

        return RingActions
    

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class RingStates(GraphStates):
            s0 = TensorDict({
                "node_feature": torch.zeros((env.nodes, 1)),
                "edge_feature": torch.zeros((0, 1)),
                "edge_index": torch.zeros((0, 2), dtype=torch.long),
            }, batch_size=())
            sf = TensorDict({
                "node_feature": torch.ones((env.nodes, 1)),
                "edge_feature": torch.zeros((0, 1)),
                "edge_index": torch.zeros((0, 2), dtype=torch.long),
            }, batch_size=())
            n_actions = env.n_actions

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.forward_masks = torch.ones(self.batch_shape + (self.n_actions,), dtype=torch.bool)
                self.backward_masks = torch.ones(self.batch_shape + (self.n_actions,), dtype=torch.bool)
        return RingStates
    
    def _step(self, states: GraphStates, actions: Actions) -> GraphStates:
        actions = self.convert_actions(actions)
        return super()._step(states, actions)

    def _backward_step(self, states: GraphStates, actions: Actions) -> GraphStates:
        actions = self.convert_actions(actions)
        return super()._backward_step(states, actions)

    def convert_actions(self, actions: Actions) -> GraphActions:
        action_tensor = actions.tensor.squeeze(-1)
        action_type = torch.where(action_tensor == 0, GraphActionType.EXIT, GraphActionType.ADD_EDGE)
        edge_index_i0 = (action_tensor - 1) // (self.nodes)
        edge_index_i1 = (action_tensor - 1) % (self.nodes)
        # edge_index_i1 = edge_index_i1 + (edge_index_i1 >= edge_index_i0)

        edge_index = torch.stack([edge_index_i0, edge_index_i1], dim=-1)
        return GraphActions(TensorDict({
            "action_type": action_type,
            "features": torch.ones(action_tensor.shape + (1,)),
            "edge_index": edge_index,
        }, batch_size=action_tensor.shape))


class GraphPreprocessor(Preprocessor):

    def __init__(self, feature_dim: int = 1):
        super().__init__(output_dim=feature_dim)

    def preprocess(self, states: GraphStates) -> TensorDict:
        return states.tensor

    def __call__(self, states: GraphStates) -> torch.Tensor:
        return self.preprocess(states)


if __name__ == "__main__":
    torch.random.manual_seed(42)
    env = RingGraphBuilding(nodes=10)
    module = RingPolicyEstimator(env.nodes)

    pf_estimator = DiscretePolicyEstimator(module=module, n_actions=env.n_actions, preprocessor=GraphPreprocessor())

    gflownet = FMGFlowNet(pf_estimator)
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=1e-3)

    visited_terminating_states = env.States.from_batch_shape((0,))
    losses = []

    for iteration in range(100):
        print(f"Iteration {iteration}")
        trajectories = gflownet.sample_trajectories(env, n=128)
        samples = gflownet.to_training_samples(trajectories)
        optimizer.zero_grad()
        loss = gflownet.loss(env, samples)
        loss.backward()
        optimizer.step()

        visited_terminating_states.extend(trajectories.last_states)
        losses.append(loss.item())



