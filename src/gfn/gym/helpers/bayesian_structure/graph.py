import string
from itertools import chain, count, islice, product
from typing import List, Optional

import numpy as np
import torch
from pgmpy.factors.continuous import LinearGaussianCPD
from torch_geometric.data import Data as GeometricData


def sample_erdos_renyi_graph(
    num_nodes: int,
    rng: np.random.Generator,
    p: Optional[float] = None,
    num_edges: Optional[int] = None,
    node_names: Optional[List[str]] = None,
) -> GeometricData:
    """
    Sample an Erdos-Renyi graph.

    Args:
        num_nodes (int): Number of nodes in the graph.
        rng (np.random.Generator): Numpy random number generator.
        p (Optional[float]): Probability of creating an edge.
        num_edges (Optional[int]): Total number of edges (used to compute p if p is None).
        node_names (Optional[List[str]]): Optional list of node names.

    Returns:
        Data: A PyTorch Geometric Data object representing the sampled graph with
              an attribute 'node_names' mapping indices to node names.

    Raises:
        ValueError: If both p and num_edges are None.
    """
    if p is None:
        if num_edges is None:
            raise ValueError("One of p or num_edges must be specified.")
        p = num_edges / (num_nodes * (num_nodes - 1) / 2.0)

    # Generate node names if not provided
    if node_names is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(product(uppercase, repeat=r) for r in count(1))
        node_names = ["".join(letters) for letters in islice(iterator, num_nodes)]

    # Generate adjacency matrix using torch and keep lower triangular matrix
    adjacency = rng.binomial(1, p=p, size=(num_nodes, num_nodes))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_nodes)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]
    adjacency = torch.tensor(adjacency, dtype=torch.long)

    # Get edge indices using torch.nonzero and transpose to shape (2, num_edges)
    edge_index = torch.nonzero(adjacency, as_tuple=False).t().contiguous()

    return GeometricData(edge_index=edge_index, node_names=node_names)


def sample_erdos_renyi_linear_gaussian(
    num_nodes: int,
    rng: np.random.Generator,
    p: Optional[float] = None,
    num_edges: Optional[int] = None,
    node_names: Optional[List[str]] = None,
    loc_edges: float = 0.0,
    scale_edges: float = 1.0,
    obs_noise: float = 0.1,
) -> GeometricData:
    """
    Sample a linear Gaussian Bayesian network based on an Erdos-Renyi graph.

    Creates graph structure using torch-geometric and assigns CPD factors for each node.
    Each CPD factor is constructed by sampling a parameter vector theta (with bias fixed to zero)
    based on the node's parent set determined from the graph structure.

    Args:
        num_nodes (int): Number of nodes.
        rng (np.random.Generator): Random number generator for reproducibility.
        p (Optional[float]): Probability of creating an edge.
        num_edges (Optional[int]): Total number of edges (used to compute p if p is None).
        node_names (Optional[List[str]]): Optional list of node names.
        loc_edges (float): Mean value for edge parameters.
        scale_edges (float): Standard deviation for edge parameters.
        obs_noise (float): Observation noise for each node.

    Returns:
        Data: A PyTorch Geometric Data object with additional attributes 'nodes' and 'cpds'.
              'nodes' is a list of node names.
              'cpds' contains a list of LinearGaussianCPD factors for each node.
    """
    # Create graph structure using existing function
    graph = sample_erdos_renyi_graph(
        num_nodes, rng=rng, p=p, num_edges=num_edges, node_names=node_names
    )
    # Create CPD factors for each node
    factors = []
    for i, node in enumerate(graph.node_names):
        # For each node, its parents are those nodes j such that edge_index[1]==i
        if graph.edge_index is not None:
            parents_tensor = graph.edge_index[0][graph.edge_index[1] == i]
            parents = [graph.node_names[parent] for parent in parents_tensor.tolist()]
        else:
            parents = []

        # Sample random parameters (from Normal distribution)
        theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
        theta[0] = 0.0  # There is no bias term

        # Create LinearGaussianCPD factor
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    graph.cpds = factors
    return graph
