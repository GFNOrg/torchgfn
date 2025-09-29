#!/usr/bin/env python
r"""
A simplified version of GFlowNet training on the HyperGrid environment, focusing on the core concepts.
This script implements Trajectory Balance (TB) training with minimal features to aid understanding.

Example usage:
python train_hypergrid_simple.py --ndim 2 --height 8 --epsilon 0.1

Key differences from the full version:
- Only implements TB loss
- No wandb integration
- Simpler architecture with shared trunks
- Basic command line options
"""

import argparse
import os
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from gfn.containers import ReplayBuffer
from gfn.containers.trajectories import Trajectories
from gfn.env import DiscreteEnv, Env
from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform
from gfn.utils.training import validate


def print_final_results(all_results: pd.DataFrame, width: int = 80):
    """Print final results in a pretty formatted table."""
    # Get the final (last) results for each experiment and seed
    final_results = all_results.groupby(["experiment_name", "seed"]).last().reset_index()

    # Calculate summary statistics across seeds for each experiment
    summary = (
        final_results.groupby("experiment_name")  # type: ignore
        .agg(
            {
                "l1_dist": ["mean", "std"],
                "logZ_diff": ["mean", "std"],
                "loss": ["mean", "std"],
                "modes_discovered": ["mean", "std"],
            }
        )
        .round(4)  # type: ignore
    )

    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]

    print("\n" + "=" * width)
    print("FINAL RESULTS SUMMARY")
    print("=" * width)
    print(
        f"{'Experiment':<20} {'L1 Dist':<15} {'LogZ Diff':<15} {'Loss':<15} {'Modes':<15}"
    )
    print(
        f"{'Name':<20} {'(mean±std)':<15} {'(mean±std)':<15} {'(mean±std)':<15} {'(mean±std)':<15}"
    )
    print("-" * width)

    for exp_name in summary.index:
        l1_mean = summary.loc[exp_name, "l1_dist_mean"]
        l1_std = summary.loc[exp_name, "l1_dist_std"]
        logz_mean = summary.loc[exp_name, "logZ_diff_mean"]
        logz_std = summary.loc[exp_name, "logZ_diff_std"]
        loss_mean = summary.loc[exp_name, "loss_mean"]
        loss_std = summary.loc[exp_name, "loss_std"]
        modes_mean = summary.loc[exp_name, "modes_discovered_mean"]
        modes_std = summary.loc[exp_name, "modes_discovered_std"]

        print(
            f"{exp_name:<20} {l1_mean:.3f}±{l1_std:.3f}    "
            f"{logz_mean:.3f}±{logz_std:.3f}    "
            f"{loss_mean:.3f}±{loss_std:.3f}    "
            f"{modes_mean:.1f}±{modes_std:.1f}"
        )

    print("=" * width)

    # Find best performing experiment for each metric
    best_l1 = summary["l1_dist_mean"].idxmin()  # type: ignore
    best_logz = summary["logZ_diff_mean"].idxmin()  # type: ignore
    best_loss = summary["loss_mean"].idxmin()  # type: ignore
    best_modes = summary["modes_discovered_mean"].idxmax()  # type: ignore

    print("BEST PERFORMERS:")
    print(f"  Lowest L1 Distance: {best_l1}")
    print(f"  Lowest LogZ Diff:   {best_logz}")
    print(f"  Lowest Loss:        {best_loss}")
    print(f"  Most Modes Found:   {best_modes}")
    print("=" * width)


def count_modes(env: Env, visited_terminating_states: DiscreteStates):
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

    return set([tuple(s.tolist()) for s in modes])


def calculate_mode_stats(env: Env, verbose: bool = False):
    """Calculate the number of pixels per mode to normalize results."""
    n_pixels_in_all_modes = len(count_modes(env, env.all_states))
    n_modes = 2**env.ndim
    n_pixels_per_mode = n_pixels_in_all_modes / n_modes

    if verbose:
        print("\nMode Stats:")
        print(f"+ Number of pixels per mode: {n_pixels_per_mode}")
        print(f"+ Number of modes: {n_modes}")
        print(f"+ Number of pixels in all modes: {n_pixels_in_all_modes}\n")

    return n_pixels_per_mode, n_modes, n_pixels_in_all_modes


def build_gflownet(
    preprocessor: KHotPreprocessor,
    env: Env,
    uniform_pb: bool = False,
    n_hidden_layers: int = 2,
    n_noisy_layers: int = 0,
    std_init: float = 0.5,
):

    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        n_hidden_layers=n_hidden_layers,
        n_noisy_layers=n_noisy_layers,
        std_init=std_init,
    )
    if not uniform_pb:
        module_PB = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            trunk=module_PF.trunk,
            n_noisy_layers=1 if n_noisy_layers > 0 else 0,
            std_init=std_init,
        )
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    return TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)


def train(
    env: Env,
    preprocessor: KHotPreprocessor,
    device: torch.device,
    lr: float,
    lr_logz: float,
    batch_size: int,
    n_iterations: int,
    epsilon: float,
    temperature: float,
    use_noisy_layers: bool,
    use_replay_buffer: bool,
    seed: int,
    uniform_pb: bool,
    validation_interval: int,
    validation_samples: int,
):

    set_seed(seed)
    off_policy = (
        epsilon > 0.0 or temperature != 1.0 or use_noisy_layers or use_replay_buffer
    )

    if use_replay_buffer:
        replay_buffer = ReplayBuffer(
            env,
            capacity=batch_size,
            prioritized_capacity=True,
            prioritized_sampling=True,
        )
    else:
        replay_buffer = None

    # Move the gflownet to the GPU.
    gflownet = build_gflownet(
        preprocessor,
        env,
        uniform_pb=uniform_pb,
        n_hidden_layers=2,
        n_noisy_layers=1 if use_noisy_layers else 0,
        std_init=0.5,  # ignored if n_noisy_layers == 0.
    )
    gflownet = gflownet.to(device)

    # Policy parameters have their own LR. Log Z gets dedicated learning rate
    # (typically higher).
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": lr_logz})

    val_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    discovered_modes = set()

    n_pixels_per_mode, _, _ = calculate_mode_stats(env, verbose=False)

    # Training loop.
    results = {
        "step": [],
        "loss": [],
        "modes_discovered": [],
        "l1_dist": [],
        "logZ_diff": [],
    }
    n_unique_modes_discovered = 0
    n_terminating = 0
    l1_dist = float("inf")
    logZ_diff = float("inf")

    for it in (pbar := tqdm(range(n_iterations), dynamic_ncols=True)):
        trajectories = gflownet.sample_trajectories(
            env,
            n=batch_size // 2 if use_replay_buffer else batch_size,
            save_logprobs=False if off_policy else True,
            # When training off-policy, we can re-use the estimator outputs during
            # the loss calculation (for the calculation of on-policy log probs).
            save_estimator_outputs=True if off_policy else False,
            epsilon=epsilon,
            temperature=temperature,
        )

        # Possibly add trajectories to the replay buffer and sample from it.
        if isinstance(replay_buffer, ReplayBuffer):
            replay_buffer.add(trajectories)
            buffer_trajectories = replay_buffer.sample(n_samples=batch_size // 2)
            assert isinstance(buffer_trajectories, Trajectories)
            trajectories.extend(buffer_trajectories)

        assert isinstance(trajectories, Trajectories)
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        optimizer.zero_grad()
        loss = gflownet.loss(
            env,
            trajectories,
            # When training off-policy, we need to recalculate all the logprobs.
            recalculate_all_logprobs=True if off_policy else False,
        )
        loss.backward()
        optimizer.step()

        if (it + 1) % validation_interval == 0:
            assert isinstance(visited_terminating_states, DiscreteStates)
            assert isinstance(env, DiscreteEnv)

            val_info, _ = validate(
                env,
                gflownet,
                validation_samples,
                visited_terminating_states,
            )
            modes_found = count_modes(env, visited_terminating_states)
            discovered_modes.update(modes_found)
            n_unique_modes_discovered = len(discovered_modes) / n_pixels_per_mode

            # Format training progress information.
            l1_dist = val_info["l1_dist"]
            logZ_diff = val_info["logZ_diff"]
            n_terminating = len(visited_terminating_states)
            # l1_info = f"L1 dist={val_info['l1_dist']:.8f} " if "l1_dist" in val_info else ""
            # print(
            #     f"Iter {it + 1}: {l1_info}modes discovered={n_unique_modes_discovered} "
            #     f"n terminating states {n_terminating}"
            # )

            # Store results.
            results["step"].append(it + 1)
            results["loss"].append(loss.item())
            results["modes_discovered"].append(n_unique_modes_discovered)
            results["l1_dist"].append(l1_dist)
            results["logZ_diff"].append(logZ_diff)

        pbar.set_postfix(
            {
                "loss": loss.item(),
                "trajectories_sampled": (it + 1) * batch_size,
                "modes_discovered": n_unique_modes_discovered,
                "n_terminating": n_terminating,
                "l1_dist": l1_dist,
                "logZ_diff": logZ_diff,
            }
        )

    return results


def main(args):
    # Setup the Environment.
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    env = HyperGrid(
        ndim=args.ndim,
        height=args.height,
        reward_fn_str="original",
        reward_fn_kwargs={
            "R0": args.R0,
            "R1": args.R1,
            "R2": args.R2,
        },
        device=device,
        calculate_partition=True,
        store_all_states=True,
        check_action_validity=__debug__,
    )
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    _, _, _ = calculate_mode_stats(env, verbose=True)

    common_kwargs = {
        "env": env,
        "preprocessor": preprocessor,
        "device": device,
        "lr": args.lr,
        "lr_logz": args.lr_logz,
        "batch_size": args.batch_size,
        "n_iterations": args.n_iterations,
        "uniform_pb": args.uniform_pb,
        "validation_interval": args.validation_interval,
        "validation_samples": args.validation_samples,
    }

    # Intalize our four configurations.
    # 1. On-policy training.
    # 2. Use a replay buffer.
    # 3. Epsilon-greedy training on a schedule.
    # 4. Using Noisy Layers.
    experiments = {
        "on_policy": {
            **common_kwargs,
            "epsilon": 0.0,
            "temperature": 1.0,
            "use_noisy_layers": False,
            "use_replay_buffer": False,
        },
        "replay_buffer": {
            **common_kwargs,
            "epsilon": 0.0,
            "temperature": 1.0,
            "use_noisy_layers": False,
            "use_replay_buffer": True,
        },
        "epsilon_greedy_0.1": {
            **common_kwargs,
            "epsilon": 0.1,
            "temperature": 1.0,
            "use_noisy_layers": False,
            "use_replay_buffer": False,
        },
        "epsilon_greedy_0.2": {
            **common_kwargs,
            "epsilon": 0.2,
            "temperature": 1.0,
            "use_noisy_layers": False,
            "use_replay_buffer": False,
        },
        "noisy_layers": {
            **common_kwargs,
            "epsilon": 0,
            "temperature": 1.0,
            "use_noisy_layers": True,
            "use_replay_buffer": False,
        },
        "temperature_2.0": {
            **common_kwargs,
            "epsilon": 0,
            "temperature": 2.0,
            "use_noisy_layers": False,
            "use_replay_buffer": False,
        },
        "temperature_1.5": {
            **common_kwargs,
            "epsilon": 0,
            "temperature": 1.5,
            "use_noisy_layers": False,
            "use_replay_buffer": False,
        },
        "temp=1.5_noisy": {
            **common_kwargs,
            "epsilon": 0,
            "temperature": 1.5,
            "use_noisy_layers": True,
            "use_replay_buffer": False,
        },
        "temp=1.5_epsilon=0.1_buffer": {
            **common_kwargs,
            "epsilon": 0.1,
            "temperature": 1.5,
            "use_noisy_layers": False,
            "use_replay_buffer": True,
        },
    }

    cols = [
        "experiment_name",
        "seed",
        "step",
        "loss",
        "modes_discovered",
        "l1_dist",
        "logZ_diff",
    ]
    all_results = pd.DataFrame(columns=cols)  # type: ignore
    for i, seed in enumerate(range(1234, 1234 + (42 * args.n_seeds), 42)):
        for experiment_name, experiment_kwargs in experiments.items():
            these_results = pd.DataFrame(columns=cols)  # type: ignore
            this_experiments_kwargs = {**experiment_kwargs, "seed": seed}
            if i == 0:
                print(
                    f"Running experiment {experiment_name} with config {this_experiments_kwargs}"
                )
            results = train(**this_experiments_kwargs)

            # Store results.
            these_results["step"] = results["step"]
            these_results["loss"] = results["loss"]
            these_results["modes_discovered"] = results["modes_discovered"]
            these_results["l1_dist"] = results["l1_dist"]
            these_results["logZ_diff"] = results["logZ_diff"]
            these_results["experiment_name"] = experiment_name
            these_results["seed"] = seed

            all_results = pd.concat([all_results, these_results])

    # Adjust layout and save to home directory.
    if args.plot:
        # Create a figure with 3 subplots arranged horizontally
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Modes Discovered
        sns.lineplot(
            data=all_results,
            x="step",
            y="modes_discovered",
            hue="experiment_name",
            marker="o",
            ax=ax1,
        )
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Modes Discovered")
        ax1.set_title("Modes Discovered Over Time")

        # Plot 2: L1 Distance
        sns.lineplot(
            data=all_results,
            x="step",
            y="l1_dist",
            hue="experiment_name",
            marker="o",
            ax=ax2,
        )
        ax2.set_xlabel("Step")
        ax2.set_ylabel("L1 Distance")
        ax2.set_title("L1 Distance Over Time")

        # Plot 3: LogZ Diff
        sns.lineplot(
            data=all_results,
            x="step",
            y="logZ_diff",
            hue="experiment_name",
            marker="o",
            ax=ax3,
        )
        ax3.set_xlabel("Step")
        ax3.set_ylabel("LogZ Diff")
        ax3.set_title("LogZ Diff Over Time")

        plt.tight_layout()

        home_dir = os.path.expanduser("~")
        output_path = os.path.join(home_dir, "exploration_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Print the result path
        print(f"\nPlot saved successfully to: {output_path}")
        print(
            f"The figure shows comparison of exploration methods across {args.n_seeds} seeds."
        )

    else:
        print_final_results(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    # Environment settings.
    parser.add_argument(
        "--ndim", type=int, default=3, help="Number of dimensions in the environment"
    )

    parser.add_argument(
        "--height", type=int, default=32, help="Height of the environment"
    )
    parser.add_argument(
        "--R0",
        type=float,
        default=0.1,
        help="Environment's R0",
    )
    parser.add_argument(
        "--R1",
        type=float,
        default=0.5,
        help="Environment's R1",
    )
    parser.add_argument(
        "--R2",
        type=float,
        default=2.0,
        help="Environment's R2",
    )

    # Optimization settings.
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

    # Policy settings.
    parser.add_argument(
        "--uniform_pb", action="store_true", help="Use a uniform backward policy"
    )

    # Training settings.
    parser.add_argument(
        "--n_iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument(
        "--validation_interval", type=int, default=200, help="Validation interval"
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Whether to plot the results"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--n_seeds", type=int, default=5, help="Number of seeds per experiment."
    )

    args = parser.parse_args()

    main(args)
