"""Write ane xamples where we want to create graphs that are rings."""

import math
import time
from typing import Optional

import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from gfn.actions import Actions, GraphActions, GraphActionType
from gfn.gflownet.flow_matching import FMGFlowNet
from gfn.gym import GraphBuilding
from gfn.modules import DiscretePolicyEstimator
from gfn.preprocessors import Preprocessor
from gfn.states import GraphStates


def state_evaluator(states: GraphStates) -> torch.Tensor:
    eps = 1e-6
    if states.tensor["edge_index"].shape[0] == 0:
        return torch.full(states.batch_shape, eps)

    out = torch.full((len(states),), eps)  # Default reward.

    for i in range(len(states)):
        start, end = states.tensor["batch_ptr"][i], states.tensor["batch_ptr"][i + 1]
        nodes_index_range = states.tensor["node_index"][start:end]
        edge_index_mask = torch.all(
            states.tensor["edge_index"] >= nodes_index_range[0], dim=-1
        ) & torch.all(states.tensor["edge_index"] <= nodes_index_range[-1], dim=-1)
        masked_edge_index = states.tensor["edge_index"][edge_index_mask] - nodes_index_range[0]

        n_nodes = nodes_index_range.shape[0]
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        adj_matrix[masked_edge_index[:, 0], masked_edge_index[:, 1]] = 1

        # # Matrix must be symmetric (undirected graph).
        # if not torch.all(adj_matrix == adj_matrix.T):
        #     continue
        # Each vertex must have exactly degree 2 (sum of each row = 2).
        if not torch.all(adj_matrix.sum(axis=1) == 1):
            continue
       
        # Connectivity check: Start at vertex 0 and follow edges, keep track
        # of visited edges, visit all edges once, end at vertex 0.
        visited = []
        current = 0
        while current not in visited:
            visited.append(current)

            def set_diff(tensor1, tensor2):
                mask = ~torch.isin(tensor1, tensor2)
                return tensor1[mask]

            # Find an unvisited neighbor
            neighbors = torch.where(adj_matrix[current] == 1)[0]
            valid_neighbours = set_diff(neighbors, torch.tensor(visited))

            # Visit the fir
            if len(valid_neighbours) == 1:
                current = valid_neighbours[0]
            elif len(valid_neighbours) == 0:
                break
            else:
                break  # TODO: This actually should never happen, should be caught on line 45.
                
        # Check if we visited all vertices and the last vertex connects back to start.
        if len(visited) == n_nodes and adj_matrix[current][0] == 1:
            out[i] = 1.0

    return out.view(*states.batch_shape)


class RingPolicyEstimator(nn.Module):
    def __init__(
            self, 
            n_nodes: int, 
            action_hidden_dim: int = 16, 
            edge_hidden_dim: int = 16,
        ):
        super().__init__()
        self.action_type_conv = GCNConv(n_nodes, action_hidden_dim)
        self.edge_index_conv = GCNConv(n_nodes, edge_hidden_dim)
        self.n_nodes = n_nodes
        self.edge_hidden_dim = edge_hidden_dim

    def _group_sum(self, tensor: torch.Tensor, batch_ptr: torch.Tensor) -> torch.Tensor:
        cumsum = torch.zeros(
            (len(tensor) + 1, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        cumsum[1:] = torch.cumsum(tensor, dim=0)

        # Subtract the end val from each batch idx fom the start val of each batch idx. 
        return cumsum[batch_ptr[1:]] - cumsum[batch_ptr[:-1]]

    def forward(self, states_tensor: TensorDict) -> torch.Tensor:
        node_feature, batch_ptr = states_tensor["node_feature"], states_tensor["batch_ptr"]

        edge_index = torch.where(
            states_tensor["edge_index"][..., None] == states_tensor["node_index"]
        )[2].reshape(states_tensor["edge_index"].shape)   # (M, 2)

        action_type = self.action_type_conv(node_feature, edge_index.T)
        action_type = self._group_sum(torch.mean(action_type, dim=-1, keepdim=True), batch_ptr)

        edge_index = self.edge_index_conv(node_feature, edge_index.T)
        edge_index = edge_index.reshape(
            *states_tensor["batch_shape"], self.n_nodes, self.edge_hidden_dim
        )
        edge_index = torch.einsum("bnf,bmf->bnm", edge_index, edge_index)

        #
        edge_actions = edge_index.reshape(
            *states_tensor["batch_shape"], self.n_nodes * self.n_nodes
        )

        return torch.cat([action_type, edge_actions], dim=-1)


class RingGraphBuilding(GraphBuilding):
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.n_actions = 1 + n_nodes * n_nodes
        super().__init__(feature_dim=n_nodes, state_evaluator=state_evaluator)
        self.is_discrete = True  # actions here are discrete, needed for FlowMatching

    def make_actions_class(self) -> type[Actions]:
        env = self

        class RingActions(Actions):
            action_shape = (1,)
            dummy_action = torch.tensor([env.n_actions])
            exit_action = torch.zeros(
                1,
            )

        return RingActions

    def make_states_class(self) -> type[GraphStates]:
        env = self

        class RingStates(GraphStates):
            s0 = TensorDict(
                {
                    "node_feature": F.one_hot(torch.arange(env.n_nodes), num_classes=env.n_nodes).float(),
                    "edge_feature": torch.ones((0, 1)),
                    "edge_index": torch.ones((0, 2), dtype=torch.long),
                },
                batch_size=(),
            )
            sf = TensorDict(
                {
                    "node_feature": torch.zeros((env.n_nodes, env.n_nodes)),
                    "edge_feature": torch.zeros((0, 1)),
                    "edge_index": torch.zeros((0, 2), dtype=torch.long),
                },
                batch_size=(),
            )

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
                forward_masks[:, 1 :: self.n_nodes + 1] = False
                for i in range(len(self)):
                    existing_edges = self[i].tensor["edge_index"] - self.tensor['node_index'][self.tensor['batch_ptr'][i]]
                    forward_masks[
                        i,
                        1 + existing_edges[:, 0] * self.n_nodes + existing_edges[:, 1],
                    ] = False

                return forward_masks.view(*self.batch_shape, self.n_actions)

            @forward_masks.setter
            def forward_masks(self, value: torch.Tensor):
                pass  # fwd masks is computed on the fly

            @property
            def backward_masks(self):
                backward_masks = torch.zeros(
                    len(self), self.n_actions, dtype=torch.bool
                )
                for i in range(len(self)):
                    existing_edges = self[i].tensor["edge_index"] - self.tensor['node_index'][self.tensor['batch_ptr'][i]]
                    backward_masks[
                        i,
                        1 + existing_edges[:, 0] * self.n_nodes + existing_edges[:, 1],
                    ] = True

                return backward_masks.view(*self.batch_shape, self.n_actions)

            @backward_masks.setter
            def backward_masks(self, value: torch.Tensor):
                pass  # bwd masks is computed on the fly

        return RingStates

    def _step(self, states: GraphStates, actions: Actions) -> GraphStates:
        actions = self.convert_actions(states, actions)
        out = super()._step(states, actions)
        return out

    def _backward_step(self, states: GraphStates, actions: Actions) -> GraphStates:
        actions = self.convert_actions(states, actions)
        return super()._backward_step(states, actions)

    def convert_actions(self, states: GraphStates, actions: Actions) -> GraphActions:
        action_tensor = actions.tensor.squeeze(-1)
        action_type = torch.where(
            action_tensor == 0, GraphActionType.EXIT, GraphActionType.ADD_EDGE
        )
        edge_index_i0 = (action_tensor - 1) // (self.n_nodes)
        edge_index_i1 = (action_tensor - 1) % (self.n_nodes)

        edge_index = torch.stack([edge_index_i0, edge_index_i1], dim=-1)
        offset = states.tensor["node_index"][states.tensor["batch_ptr"][:-1]]
        return GraphActions(
            TensorDict(
                {
                    "action_type": action_type,
                    "features": torch.ones(action_tensor.shape + (1,)),
                    "edge_index": edge_index + offset[:, None],
                },
                batch_size=action_tensor.shape,
            )
        )


class GraphPreprocessor(Preprocessor):
    def __init__(self, feature_dim: int = 1):
        super().__init__(output_dim=feature_dim)

    def preprocess(self, states: GraphStates) -> TensorDict:
        return states.tensor

    def __call__(self, states: GraphStates) -> torch.Tensor:
        return self.preprocess(states)


def render_states(states: GraphStates):
    rewards = state_evaluator(states)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    for i in range(8):
        current_ax = ax[i // 4, i % 4]
        state = states[i]
        n_circles = state.tensor["node_feature"].shape[0]
        radius = 5
        xs, ys = [], []
        for j in range(n_circles):
            angle = 2 * math.pi * j / n_circles
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            xs.append(x)
            ys.append(y)
            current_ax.add_patch(
                plt.Circle((x, y), 0.5, facecolor="none", edgecolor="black")
            )

        edge_index = states[i].tensor["edge_index"]
        edge_index = torch.where(
            edge_index[..., None] == states[i].tensor["node_index"]
        )[2].reshape(edge_index.shape)
        for edge in edge_index:
            start_x, start_y = xs[edge[0]], ys[edge[0]]
            end_x, end_y = xs[edge[1]], ys[edge[1]]
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx**2 + dy**2)
            dx, dy = dx / length, dy / length

            circle_radius = 0.5
            head_thickness = 0.2
            start_x += dx * (circle_radius)
            start_y += dy * (circle_radius)
            end_x -= dx * (circle_radius + head_thickness)
            end_y -= dy * (circle_radius + head_thickness)

            current_ax.arrow(
                start_x,
                start_y,
                end_x - start_x,
                end_y - start_y,
                head_width=head_thickness,
                head_length=head_thickness,
                fc="black",
                ec="black",
            )

        current_ax.set_title(f"State {i}, $r={rewards[i]:.2f}$")
        current_ax.set_xlim(-(radius + 1), radius + 1)
        current_ax.set_ylim(-(radius + 1), radius + 1)
        current_ax.set_aspect("equal")
        current_ax.set_xticks([])
        current_ax.set_yticks([])

    plt.show()


if __name__ == "__main__":
    # ring_state = GraphStates(
    #     TensorDict(
    #         {
    #             "node_feature": torch.tensor([[0], [1], [2]]),
    #             "node_index": torch.tensor([0, 1, 2]),
    #             "edge_feature": torch.ones((3, 1)),
    #             "edge_index": torch.tensor([[1, 0], [1, 2], [2, 0]]),
    #             "batch_ptr": torch.tensor([0, 3]),
    #             "batch_shape": torch.ones((1,), dtype=torch.long),
    #         },
    #         batch_size=(),
    #     )
    # )
    # print(state_evaluator(ring_state))

    N_NODES = 3
    N_ITERATIONS = 128
    torch.random.manual_seed(7)
    env = RingGraphBuilding(n_nodes=N_NODES)
    module = RingPolicyEstimator(env.n_nodes)

    logf_estimator = DiscretePolicyEstimator(
        module=module, n_actions=env.n_actions, preprocessor=GraphPreprocessor()
    )

    gflownet = FMGFlowNet(logf_estimator)
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=1e-2)
    batch_size = 32
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=N_ITERATIONS, eta_min=1e-4
    # )

    losses = []

    t1 = time.time()
    for iteration in range(N_ITERATIONS):
        trajectories = gflownet.sample_trajectories(env, n=batch_size)
        samples = gflownet.to_training_samples(trajectories)
        optimizer.zero_grad()
        loss = gflownet.loss(env, samples)
        print("Iteration", iteration, "Loss:", loss.item())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # scheduler.step()

    t2 = time.time()
    print("Time:", t2 - t1)
    render_states(trajectories.last_states[:8])
