import pandas as pd
import torch
from torch_geometric.data import Data


def topological_sort_data(graph: Data) -> list[int]:
    # Compute topological order for a torch_geometric graph DAG
    assert graph.num_nodes is not None, "Graph must have nodes"
    indegree = torch.zeros(graph.num_nodes, dtype=torch.int)
    # Assert edge_index is present
    assert graph.edge_index is not None, "Graph must have a valid edge_index attribute."
    edge_index = graph.edge_index
    for child in edge_index[1]:
        indegree[child] += 1
    order = []
    queue = [int(i) for i in torch.nonzero(indegree == 0, as_tuple=False)]
    while queue:
        node = queue.pop(0)
        order.append(node)
        mask = edge_index[0] == node
        children = edge_index[1][mask]
        for child in children.tolist():
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return order


def sample_from_linear_gaussian(
    graph: Data,
    num_samples: int,
    rng: torch.Generator,
    obs_noise=0.1,
) -> pd.DataFrame:
    """
    Sample from a linear-Gaussian graph represented as a torch_geometric graph.
    Each node's value is computed as a weighted sum of its parents' samples plus noise,
    following an ancestral sampling method similar in spirit to the provided NumPy/NetworkX code.

    Args:
        graph (torch_geometric.data.Data): Graph structure.
        rng (torch.Generator): Random number generator for reproducibility.
        num_samples (int): Number of graphs to sample.
    """
    assert graph.num_nodes is not None, "graph must have nodes"
    assert graph.edge_index is not None, "graph must have a valid edge_index attribute."
    # Initialize samples as a DataFrame with columns representing nodes.
    column_names = [str(i) for i in range(graph.num_nodes)]
    samples = pd.DataFrame(
        index=range(num_samples), columns=pd.Index(column_names), dtype=float
    )
    order = topological_sort_data(graph)
    edge_index = graph.edge_index
    for node in order:
        mask = edge_index[1] == node
        parent_nodes = edge_index[0][mask]
        if parent_nodes.numel() == 0:
            parent_mean = torch.zeros(num_samples, dtype=torch.float32)
        else:
            parent_mean = torch.zeros(num_samples, dtype=torch.float32)
            for idx, parent in enumerate(parent_nodes.tolist()):
                parent_col = samples.get(str(parent))
                if parent_col is None:
                    parent_samples = torch.zeros(num_samples, dtype=torch.float32)
                else:
                    parent_samples = torch.tensor(
                        parent_col.to_numpy(), dtype=torch.float32
                    )
                parent_mean += parent_samples
        noise_std = obs_noise
        noise = torch.normal(
            mean=0.0, std=float(noise_std), size=(num_samples,), generator=rng
        )
        samples[str(node)] = (parent_mean + noise).numpy()

    # Ensure the final return is a DataFrame with columns as nodes.
    return pd.DataFrame(samples, columns=pd.Index(column_names))
