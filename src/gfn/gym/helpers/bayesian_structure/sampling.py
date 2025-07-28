import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as GeometricData


def topological_sort_data(graph: GeometricData) -> list[int]:
    # Compute topological order for a torch_geometric graph DAG
    assert graph.node_names is not None, "Graph must have nodes"
    in_degree = torch.zeros(len(graph.node_names), dtype=torch.long)
    # Assert edge_index is present
    assert graph.edge_index is not None, "Graph must have a valid edge_index attribute."
    edge_index = graph.edge_index
    for child in edge_index[1]:
        in_degree[child] += 1
    order = []
    queue = [int(i) for i in torch.nonzero(in_degree == 0, as_tuple=False)]
    while queue:
        node = queue.pop(0)
        order.append(node)
        mask = edge_index[0] == node
        children = edge_index[1][mask]
        for child in children.tolist():
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order


def sample_from_linear_gaussian(
    graph: GeometricData,
    num_samples: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> pd.DataFrame:
    """Sample from a linear-Gaussian model using ancestral sampling."""
    samples = pd.DataFrame(columns=graph.node_names)
    for node in topological_sort_data(graph):
        cpd = graph.cpds[node]

        if cpd.evidence:
            values = samples.loc[:, cpd.evidence].values.T
            mean = cpd.mean[0] + np.dot(cpd.mean[1:], values)
            samples[graph.node_names[node]] = rng.normal(mean, cpd.variance)
        else:
            samples[graph.node_names[node]] = rng.normal(
                cpd.mean[0], cpd.variance, size=(num_samples,)
            )

    return samples
