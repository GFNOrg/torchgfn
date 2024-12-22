"""Write ane xamples where we want to create graphs that are rings."""

from typing import Optional
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
    eps = 1e-6
    if states.tensor["edge_index"].shape[0] == 0:
        return torch.full(states.batch_shape, eps)
    if states.tensor["edge_index"].shape[0] != states.tensor["node_feature"].shape[0]:
        return torch.full(states.batch_shape, eps)

    out = torch.zeros(len(states))
    for i in range(len(states)):
        start, end = states.tensor["batch_ptr"][i], states.tensor["batch_ptr"][i + 1]
        edge_index_mask = torch.all(states.tensor["edge_index"] >= start, dim=-1) & torch.all(states.tensor["edge_index"] < end, dim=-1)
        edge_index = states.tensor["edge_index"][edge_index_mask]
        arange = torch.arange(start, end)
        # TODO: not correct, accepts multiple rings
        if torch.all(torch.sort(edge_index[:, 0])[0] == arange) and torch.all(torch.sort(edge_index[:, 1])[0] == arange):
            out[i] = 1
        else:
            out[i] = eps
    return out.view(*states.batch_shape)


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
        edge_index = edge_index.reshape(*states_tensor["batch_shape"], self.n_nodes, 8)
        edge_index = torch.einsum("bnf,bmf->bnm", edge_index, edge_index)
        edge_actions = edge_index.reshape(*states_tensor["batch_shape"], self.n_nodes * self.n_nodes)

        return torch.cat([action_type, edge_actions], dim=-1)

class RingGraphBuilding(GraphBuilding):
    def __init__(self, n_nodes: int = 10):
        self.n_nodes = n_nodes
        self.n_actions = 1 + n_nodes * n_nodes
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
                "node_feature": torch.zeros((env.n_nodes, 1)),
                "edge_feature": torch.zeros((0, 1)),
                "edge_index": torch.zeros((0, 2), dtype=torch.long),
            }, batch_size=())
            sf = TensorDict({
                "node_feature": torch.ones((env.n_nodes, 1)),
                "edge_feature": torch.zeros((0, 1)),
                "edge_index": torch.zeros((0, 2), dtype=torch.long),
            }, batch_size=())

            def __init__(self, tensor: TensorDict):
                self.tensor = tensor
                self.node_features_dim = tensor["node_feature"].shape[-1]
                self.edge_features_dim = tensor["edge_feature"].shape[-1]
                self._log_rewards: Optional[float] = None
    
                self.n_nodes = env.n_nodes
                self.n_actions = env.n_actions

            @property
            def forward_masks(self):
                forward_masks = torch.ones(len(self), self.n_actions, dtype=torch.bool)
                forward_masks[:, 1::self.n_nodes + 1] = False
                for i in range(len(self)):
                    existing_edges = self[i].tensor["edge_index"]
                    forward_masks[i, 1 + existing_edges[:, 0] * self.n_nodes + existing_edges[:, 1]] = False
                
                return forward_masks.view(*self.batch_shape, self.n_actions)
        
            @forward_masks.setter
            def forward_masks(self, value: torch.Tensor):
                pass # fwd masks is computed on the fly

            @property
            def backward_masks(self):
                backward_masks = torch.zeros(len(self), self.n_actions, dtype=torch.bool)
                for i in range(len(self)):
                    existing_edges = self[i].tensor["edge_index"]
                    backward_masks[i, 1 + existing_edges[:, 0] * self.n_nodes + existing_edges[:, 1]] = True
                
                return backward_masks.view(*self.batch_shape, self.n_actions)
        
            @backward_masks.setter
            def backward_masks(self, value: torch.Tensor):
                pass # bwd masks is computed on the fly
    
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
        edge_index_i0 = (action_tensor - 1) // (self.n_nodes)
        edge_index_i1 = (action_tensor - 1) % (self.n_nodes)

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
    env = RingGraphBuilding(n_nodes=3)
    module = RingPolicyEstimator(env.n_nodes)

    pf_estimator = DiscretePolicyEstimator(module=module, n_actions=env.n_actions, preprocessor=GraphPreprocessor())

    gflownet = FMGFlowNet(pf_estimator)
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=1e-3)

    visited_terminating_states = env.States.from_batch_shape((0,))
    losses = []

    for iteration in range(128):
        trajectories = gflownet.sample_trajectories(env, n=32)
        samples = gflownet.to_training_samples(trajectories)
        optimizer.zero_grad()
        loss = gflownet.loss(env, samples)
        loss.backward()
        optimizer.step()

        visited_terminating_states.extend(trajectories.last_states)
        losses.append(loss.item())



