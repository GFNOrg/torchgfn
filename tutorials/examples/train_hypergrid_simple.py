#!/usr/bin/env python
import argparse

import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP
from gfn.utils.training import validate


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
    sampler = Sampler(estimator=pf_estimator)

    # Move the gflownet to the GPU.
    gflownet = gflownet.to(device_str)

    # Policy parameters have their own LR. Log Z gets dedicated learning rate
    # (typically higher).
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group(
        {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
    )

    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            save_estimator_outputs=True,
            epsilon=args.epsilon,
        )
        visited_terminating_states.extend(trajectories.last_states)

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories)
        loss.backward()
        optimizer.step()
        if (it + 1) % args.validation_interval == 0:
            validation_info = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )
            print(f"Iter {it + 1}: L1 distance {validation_info['l1_dist']:.8f}")
        pbar.set_postfix({"loss": loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument(
        "--ndim", type=int, default=4, help="Number of dimensions in the environment"
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
    parser.add_argument(
        "--validation_interval", type=int, default=100, help="Validation interval"
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=100000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon for the sampler"
    )

    args = parser.parse_args()

    main(args)
