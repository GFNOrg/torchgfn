#!/usr/bin/env python
r"""
In this example, we will use backward sampling from the terminating states to train a GFlowNet.
To do so, we will use the `TerminatingStateBuffer` to store the terminating states and sample from them.
"""

import argparse
from typing import cast

import torch
from tqdm import tqdm

from gfn.containers import TerminatingStateBuffer
from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform
from gfn.utils.training import validate


def main(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Setup the Environment.
    env = HyperGrid(
        ndim=args.ndim,
        height=args.height,
        device=device,
        calculate_partition=True,
        store_all_states=True,
    )
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    # Build the GFlowNet.
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    if not args.uniform_pb:
        module_PB = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            trunk=module_PF.trunk,
        )
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)
    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )
    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
    gflownet = gflownet.to(device)

    # Feed pf to the sampler.
    forward_sampler = Sampler(estimator=pf_estimator)
    backward_sampler = Sampler(estimator=pb_estimator)

    # Policy parameters have their own LR. Log Z gets dedicated learning rate
    # (typically higher).
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    replay_buffer = TerminatingStateBuffer(env, capacity=args.buffer_capacity)

    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = forward_sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
        )
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        with torch.no_grad():
            replay_buffer.add(trajectories)  # This will add only the terminating states.

        if it < args.prefill or len(replay_buffer) < args.batch_size:
            continue

        # First, train with on-policy trajectories
        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        loss.backward()
        optimizer.step()

        # Second, train with off-policy trajectories
        # Terminating states are sampled from the replay buffer, and then
        # backward trajectories are sampled starting from them using pb_estimator
        terminating_states_container = replay_buffer.sample(n_samples=args.batch_size)
        terminating_states = terminating_states_container.states
        bwd_trajectories = backward_sampler.sample_trajectories(
            env,
            states=terminating_states,
            save_logprobs=False,
            save_estimator_outputs=False,
            # TODO: log rewards, conditioning, ...
        )
        optimizer.zero_grad()
        loss = gflownet.loss(
            env,
            bwd_trajectories.reverse_backward_trajectories(),
            recalculate_all_logprobs=False,
        )
        loss.backward()
        optimizer.step()

        if (it + 1) % args.validation_interval == 0:
            validation_info, _ = validate(
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
        "--uniform_pb", action="store_true", help="Use a uniform backward policy"
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
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=10000,
        help="Capacity of the replay buffer",
    )
    parser.add_argument(
        "--prefill",
        type=int,
        default=10,
        help="Number of iterations to prefill the replay buffer",
    )
    args = parser.parse_args()

    main(args)
