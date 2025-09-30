#!/usr/bin/env python
"""
Example script for training a GFlowNet using replay buffers.

This script demonstrates two approaches for off-policy training:
1. Trajectory buffer: Stores and samples entire trajectories for training.
2. Terminating state buffer: Stores terminating states, from which backward trajectories are sampled.

Both buffer types can be selected to improve training efficiency and diversity.
"""

import argparse
from typing import cast

import torch
from tqdm import tqdm

from gfn.containers import ReplayBuffer, TerminatingStateBuffer, Trajectories
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

    buffer_args = {
        "capacity": args.buffer_capacity,
        "prioritized_capacity": args.prioritized_capacity,
        "prioritized_sampling": args.prioritized_sampling,
    }
    if args.buffer_type == "trajectory":
        replay_buffer = ReplayBuffer(env, **buffer_args)
    elif args.buffer_type == "terminating_state":
        replay_buffer = TerminatingStateBuffer(env, **buffer_args)
    else:
        raise ValueError(f"Invalid buffer type: {args.buffer_type}")

    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    discovered_modes = set()
    n_pixels_per_mode = round(env.height / 10) ** env.ndim
    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = forward_sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
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

        if args.buffer_type == "trajectory":
            buffer_trajectories = cast(
                Trajectories, replay_buffer.sample(n_samples=args.batch_size)
            )
        else:  # args.buffer_type == "terminating_state"
            # Second, train with off-policy trajectories
            # Terminating states are sampled from the replay buffer, and then
            # backward trajectories are sampled starting from them using pb_estimator
            terminating_states_container = replay_buffer.sample(
                n_samples=args.batch_size
            )
            terminating_states = terminating_states_container.states
            bwd_trajectories = backward_sampler.sample_trajectories(
                env,
                states=terminating_states,
                save_logprobs=False,  # TODO: enable this
                save_estimator_outputs=False,
                # TODO: log rewards, conditioning, ...
            )
            buffer_trajectories = bwd_trajectories.reverse_backward_trajectories()
            buffer_trajectories._log_rewards = terminating_states_container.log_rewards

        optimizer.zero_grad()
        loss = gflownet.loss(env, buffer_trajectories, recalculate_all_logprobs=True)
        loss.backward()
        optimizer.step()

        if (it + 1) % args.validation_interval == 0:
            validation_info, _ = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )
            # Modes will have a reward greater than R2+R1+R0.
            mode_reward_threshold = sum(
                [
                    env.reward_fn_kwargs["R2"],
                    env.reward_fn_kwargs["R1"],
                    env.reward_fn_kwargs["R0"],
                ]
            )

            assert isinstance(visited_terminating_states, DiscreteStates)
            modes = visited_terminating_states[
                env.reward(visited_terminating_states) >= mode_reward_threshold
            ].tensor
            # Finds all the unique modes in visited_terminating_states.
            modes_found = set([tuple(s.tolist()) for s in modes])
            discovered_modes.update(modes_found)
            # torch.tensor(list(modes_found)).shape ==[batch_size, 2]
            str_info = f"Iter {it + 1}: "
            if "l1_dist" in validation_info:
                str_info += f"L1 distance={validation_info['l1_dist']:.8f} "
            str_info += (
                f"modes discovered={len(discovered_modes) / n_pixels_per_mode:.3f} "
            )
            str_info += f"n terminating states {len(visited_terminating_states)}"
            print(str_info)

        pbar.set_postfix(
            {"loss": loss.item(), "trajectories_sampled": (it + 1) * args.batch_size}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=64, help="Height of the environment"
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
        default=200000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration parameter for the sampler",
    )
    parser.add_argument(
        "--buffer_type",
        type=str,
        default="trajectory",
        choices=["trajectory", "terminating_state"],
        help="Type of buffer to use",
    )
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=10000,
        help="Capacity of the replay buffer",
    )
    parser.add_argument(
        "--prioritized_capacity",
        action="store_true",
        help="Use prioritized capacity",
    )
    parser.add_argument(
        "--prioritized_sampling",
        action="store_true",
        help="Use prioritized sampling",
    )
    parser.add_argument(
        "--prefill",
        type=int,
        default=10,
        help="Number of iterations to prefill the replay buffer",
    )
    args = parser.parse_args()

    main(args)
