from typing import Optional

import torch

from src.gfn.gym.helpers.bayesian_structure import priors
from src.gfn.gym.helpers.bayesian_structure.data import get_data
from src.gfn.gym.helpers.bayesian_structure.scores import BGeScore


def get_prior(name: str) -> priors.BasePrior:
    prior = {
        "uniform": priors.UniformPrior,
        "erdos_renyi": priors.ErdosRenyiPrior,
        "edge": priors.EdgePrior,
        "fair": priors.FairPrior,
    }
    return prior[name]()


def get_scorer(
    graph_name: str,
    prior_name: str,
    num_nodes: int,
    num_edges: int,
    num_samples: int,
    node_names: Optional[list[str]] = None,
    rng: Optional[torch.Generator] = None,
):
    # Get the data
    graph, data, score = get_data(
        graph_name, num_nodes, num_edges, num_samples, node_names, rng=rng
    )
    # Get the prior
    prior = get_prior(name=prior_name)
    scores = {"bge": BGeScore}
    scorer = scores[score](data=data, prior=prior)

    return scorer, data, graph, prior
