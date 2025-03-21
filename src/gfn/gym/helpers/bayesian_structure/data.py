from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data as GeometricData

from gfn.gym.helpers.bayesian_structure import priors
from gfn.gym.helpers.bayesian_structure.graph import sample_erdos_renyi_linear_gaussian
from gfn.gym.helpers.bayesian_structure.sampling import sample_from_linear_gaussian


def get_prior(name, **kwargs):
    prior = {
        "uniform": priors.UniformPrior,
        "erdos_renyi": priors.ErdosRenyiPrior,
        "edge": priors.EdgePrior,
        "fair": priors.FairPrior,
    }
    return prior[name](**kwargs)


def get_data(
    name: str,
    num_nodes: int,
    num_edges: int,
    num_samples: int,
    node_names: Optional[list[str]] = None,
    rng: Optional[torch.Generator] = None,
) -> tuple[GeometricData, pd.DataFrame, str]:
    """
    Generate Bayesian linear Gaussian data.

    Parameters:
        name (str): Data generation method type.
        num_nodes (int): Number of variables in the graph.
        num_edges (int): Number of edges to sample in the graph.
        num_samples (int): Number of samples to generate.
        node_names (Optional[List[str]]): Optional list of node names.
        rng (Optional[torch.Generator]): Optional random generator instance.

    Returns:
        tuple: (graph, data, score) where 'score' indicates the scoring method used.
    """
    if rng is None:
        rng = torch.Generator()
    if name == "erdos_renyi_lingauss":
        graph = sample_erdos_renyi_linear_gaussian(
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_names=node_names,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng,
        )
        data = sample_from_linear_gaussian(graph, num_samples=num_samples, rng=rng)
        score = "bge"
        return graph, data, score
    else:
        raise ValueError(f"Data generation method '{name}' not implemented.")
