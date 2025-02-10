from gfn.states import GraphStates
from tensordict import TensorDict
import torch


def make_graph_states(n_graphs, n_nodes, n_edges):
    batch_ptr = torch.cat([torch.zeros(1), n_nodes.cumsum(0)]).int()
    node_feature = torch.randn(batch_ptr[-1].item(), 10)
    node_index = torch.arange(0, batch_ptr[-1].item())

    edge_features = torch.randn(n_edges.sum(), 10)
    edge_index = []
    for i, (start, end) in enumerate(zip(batch_ptr[:-1], batch_ptr[1:])):
        edge_index.append(torch.randint(start, end, (n_edges[i], 2)))
    edge_index = torch.cat(edge_index)

    return GraphStates(
        TensorDict(
            {
                "node_feature": node_feature,
                "edge_feature": edge_features,
                "edge_index": edge_index,
                "node_index": node_index,
                "batch_ptr": batch_ptr,
                "batch_shape": torch.tensor([n_graphs]),
            }
        )
    )


def test_get_set():
    n_graphs = 10
    n_nodes = torch.randint(1, 10, (n_graphs,))
    n_edges = torch.randint(1, 10, (n_graphs,))
    graphs = make_graph_states(10, n_nodes, n_edges)
    assert not graphs[0]._compare(graphs[9].tensor)
    last_graph = graphs[9]
    graphs = graphs[:-1]
    graphs[0] = last_graph
    assert graphs[0]._compare(last_graph.tensor)


def test_stack():
    GraphStates.s0 = make_graph_states(1, torch.tensor([1]), torch.tensor([0])).tensor
    n_graphs = 10
    n_nodes = torch.randint(1, 10, (n_graphs,))
    n_edges = torch.randint(1, 10, (n_graphs,))
    graphs = make_graph_states(10, n_nodes, n_edges)
    stacked_graphs = GraphStates.stack([graphs[0], graphs[1]])
    assert stacked_graphs.batch_shape == (2, 1)
    assert stacked_graphs[0]._compare(graphs[0].tensor)
    assert stacked_graphs[1]._compare(graphs[1].tensor)


if __name__ == "__main__":
    test_get_set()
