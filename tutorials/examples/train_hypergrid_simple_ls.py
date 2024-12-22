#!/usr/bin/env python
import argparse

import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import LocalSearchSampler
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP


def main(args):
    set_seed(args.seed)
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Setup the Environment.
    env = HyperGrid(ndim=args.ndim, height=args.height, device_str=device_str)

    # Build the GFlowNet.
    module_PF = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    module_PB = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        trunk=module_PF.trunk,
    )
    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor
    )
    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)

    # Feed pf to the sampler.
    sampler = LocalSearchSampler(pf_estimator=pf_estimator, pb_estimator=pb_estimator)

    # Move the gflownet to the GPU.
    gflownet = gflownet.to(device_str)

    # Policy parameters have their own LR. Log Z gets dedicated learning rate
    # (typically higher).
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group(
        {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
    )

    for _ in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = sampler.sample_trajectories(
            env,
            n=(args.batch_size // args.n_local_search_loops),
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
            n_local_search_loops=args.n_local_search_loops,
            back_ratio=args.back_ratio,
            use_metropolis_hastings=args.use_metropolis_hastings,
        )
        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=16, help="Height of the environment"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_logz",
        type=float,
        default=1e-1,
        help="Learning rate for the logZ parameter",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon for the sampler"
    )

    # Local search parameters.
    parser.add_argument(
        "--n_local_search_loops",
        type=int,
        default=4,
        help="Number of local search loops",
    )
    parser.add_argument(
        "--back_ratio",
        type=float,
        default=0.5,
        help="The ratio of the number of backward steps to the length of the trajectory",
    )
    parser.add_argument(
        "--use_metropolis_hastings",
        action="store_true",
        help="Use Metropolis-Hastings acceptance criterion",
    )

    args = parser.parse_args()

    main(args)
