"""
The goal of this script is to reproduce the results of DAG-GFlowNet for
Bayesian structure learning (Deleu et al., 2022) using the GraphEnv.

Specifically, we consider a randomly generated (under the Erdős-Rényi model) linear-Gaussian
Bayesian network over `n_nodes` nodes. We generate 100 datapoints from it, and use them to
calculate the BGe score. The GFlowNet is learned to generate directed acyclic graphs (DAGs)
proportionally to their BGe score, using the modified DB loss.

Key components:
- BayesianStructure: Environment for Bayesian structure learning
- LinearTransformerPolicyModule: Linear transformer policy module
- ModifiedDBGFlowNet: GFlowNet with modified detailed balance loss
"""

import torch
from numpy.random import default_rng

from gfn.gym.bayesian_structure import BayesianStructure
from gfn.utils.common import set_seed

DEFAULT_SEED = 4444


def main(args):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    default_rng(seed)
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Create the environment
    env = BayesianStructure(
        n_nodes=args.n_nodes,
        state_evaluator=lambda x: torch.zeros(x.batch_size, device=x.device),  # TODO
        device=device_str,
    )
    env.reset(args.batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
