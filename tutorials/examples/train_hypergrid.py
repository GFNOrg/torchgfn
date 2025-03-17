r"""
The goal of this script is to reproduce some of the published results on the HyperGrid
environment. Run one of the following commands to reproduce some of the results in
[Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

python train_hypergrid.py --ndim 4 --height 8 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}
python train_hypergrid.py --ndim 2 --height 64 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}

And run one of the following to reproduce some of the results in
[Learning GFlowNets from partial episodes for improved convergence and stability](https://arxiv.org/abs/2209.12782)
python train_hypergrid.py --ndim {2, 4} --height 12 --R0 {1e-3, 1e-4} --tied --loss {TB, DB, SubTB}

This script also provides a function `get_exact_P_T` that computes the exact terminating state
distribution for the HyperGrid environment, which is useful for evaluation and visualization.
"""

from argparse import ArgumentParser
from typing import cast

import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.gridspec import GridSpec
from tqdm import tqdm, trange

from gfn.containers import NormBasedDiversePrioritizedReplayBuffer, ReplayBuffer
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    GFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator, GFNModule, ScalarEstimator
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform, Tabular
from gfn.utils.training import validate

DEFAULT_SEED = 4444


def get_exact_P_T(env: HyperGrid, gflownet: GFlowNet) -> torch.Tensor:
    r"""Evaluates the exact terminating state distribution P_T for HyperGrid.

    For each state s', the terminating state probability is computed as:

    .. math::
        P_T(s') = u(s') P_F(s_f | s')

    where u(s') satisfies the recursion:

    .. math::
        u(s') = \sum_{s \in \text{Par}(s')} u(s) P_F(s' | s)

    with the base case u(s_0) = 1.

    Args:
        env: The HyperGrid environment
        gflownet: The GFlowNet model

    Returns:
        The exact terminating state distribution as a tensor
    """
    grid = env.build_grid()

    # Get the forward policy distribution for all states
    with torch.no_grad():
        # Handle both FM and other GFlowNet types
        policy: GFNModule = cast(
            GFNModule, gflownet.logF if isinstance(gflownet, FMGFlowNet) else gflownet.pf
        )

        estimator_outputs = policy(grid)
        dist = policy.to_probability_distribution(grid, estimator_outputs)
        probabilities = torch.exp(dist.logits)  # Get raw probabilities

    u = torch.ones(grid.batch_shape)

    indices = env.all_indices()
    for index in indices[1:]:
        parents = [
            tuple(list(index[:i]) + [index[i] - 1] + list(index[i + 1 :]) + [i])
            for i in range(len(index))
            if index[i] > 0
        ]
        parents_tensor = torch.tensor(parents)
        parents_indices = parents_tensor[:, :-1].long()  # All but last column for u
        action_indices = parents_tensor[:, -1].long()  # Last column for probabilities

        # Compute u values for parent states
        parent_u_values = torch.stack([u[tuple(p.tolist())] for p in parents_indices])

        # Compute probabilities for parent transitions
        parent_probs = torch.stack(
            [
                probabilities[tuple(list(p.tolist()) + [a.item()])]
                for p, a in zip(parents_indices, action_indices)
            ]
        )

        u[tuple(index)] = torch.sum(parent_u_values * parent_probs)

    return (u * probabilities[..., -1]).view(-1).detach().cpu()


def main(args):  # noqa: C901
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    use_wandb = len(args.wandb_project) > 0
    if use_wandb:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    # 1. Create the environment
    env = HyperGrid(
        args.ndim, args.height, args.R0, args.R1, args.R2, device_str=device_str
    )

    # 2. Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    gflownet = None
    if args.loss == "FM":
        # We need a LogEdgeFlowEstimator
        if args.tabular:
            module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        else:
            module = MLP(
                input_dim=env.ndim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
        estimator = DiscretePolicyEstimator(
            module=module,
            n_actions=env.n_actions,
        )
        gflownet = FMGFlowNet(estimator)
    else:
        pb_module = None
        # We need a DiscretePFEstimator and a DiscretePBEstimator
        if args.tabular:
            pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
            if not args.uniform_pb:
                pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
        else:
            pf_module = MLP(
                input_dim=env.ndim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
            if not args.uniform_pb:
                pb_module = MLP(
                    input_dim=env.ndim,
                    output_dim=env.n_actions - 1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    trunk=pf_module.trunk if args.tied else None,
                )
        if args.uniform_pb:
            pb_module = DiscreteUniform(env.n_actions - 1)

        assert (
            pf_module is not None
        ), f"pf_module is None. Command-line arguments: {args}"
        assert (
            pb_module is not None
        ), f"pb_module is None. Command-line arguments: {args}"

        pf_estimator = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
        )
        pb_estimator = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
        )

        if args.loss == "ModifiedDB":
            gflownet = ModifiedDBGFlowNet(
                pf_estimator,
                pb_estimator,
            )

        elif args.loss in ("DB", "SubTB"):
            # We need a LogStateFlowEstimator
            assert (
                pf_estimator is not None
            ), f"pf_estimator is None. Command-line arguments: {args}"
            assert (
                pb_estimator is not None
            ), f"pb_estimator is None. Command-line arguments: {args}"

            if isinstance(pf_module, Tabular):
                module = Tabular(n_states=env.n_states, output_dim=1)
            else:
                module = MLP(
                    input_dim=env.ndim,
                    output_dim=1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    trunk=pf_module.trunk if args.tied else None,
                )

            logF_estimator = ScalarEstimator(module=module)
            if args.loss == "DB":
                gflownet = DBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                )
            else:
                gflownet = SubTBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    weighting=args.subTB_weighting,
                    lamda=args.subTB_lambda,
                )
        elif args.loss == "TB":
            gflownet = TBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
            )
        elif args.loss == "ZVar":
            gflownet = LogPartitionVarianceGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
            )

    assert gflownet is not None, f"No gflownet for loss {args.loss}"

    # Create replay buffer if needed
    replay_buffer = None
    if args.replay_buffer_size > 0:
        if args.replay_buffer_prioritized:
            replay_buffer = NormBasedDiversePrioritizedReplayBuffer(
                env,
                capacity=args.replay_buffer_size,
                cutoff_distance=args.cutoff_distance,
                p_norm_distance=args.p_norm_distance,
            )
        else:
            replay_buffer = ReplayBuffer(
                env,
                capacity=args.replay_buffer_size,
            )

    # Move the gflownet to the GPU.
    gflownet = gflownet.to(device_str)

    # 3. Create the optimizer
    # Policy parameters have their own LR.
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]

    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.states_from_batch_shape((0,))

    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size
    validation_info = {"l1_dist": float("inf")}
    l1_distances = []  # Track l1 distances over time
    validation_steps = []  # Track corresponding steps
    for iteration in trange(n_iterations):
        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=args.replay_buffer_size == 0,
            save_estimator_outputs=False,
        )
        training_samples = gflownet.to_training_samples(trajectories)
        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                training_objects = replay_buffer.sample(n_trajectories=args.batch_size)
        else:
            training_objects = training_samples

        optimizer.zero_grad()
        gflownet = cast(GFlowNet, gflownet)
        loss = gflownet.loss(
            env, training_objects, recalculate_all_logprobs=args.replay_buffer_size > 0
        )
        loss.backward()
        optimizer.step()
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        states_visited += len(trajectories)

        to_log = {"loss": loss.item(), "states_visited": states_visited}
        if use_wandb:
            wandb.log(to_log, step=iteration)
        if iteration % args.validation_interval == 0:
            validation_info = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )
            if use_wandb:
                wandb.log(validation_info, step=iteration)
            to_log.update(validation_info)
            tqdm.write(f"{iteration}: {to_log}")
            l1_distances.append(validation_info["l1_dist"])  # Store l1 distance
            validation_steps.append(iteration)  # Store corresponding step

    if args.plot:
        if args.wandb_project:
            raise ValueError("plot argument is incompatible with wandb_project")
        if args.ndim != 2:
            raise ValueError("plotting is only supported for 2D environments")

        # Create figure with 3 subplots with proper spacing
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 0.1, 1.2])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])  # Colorbar axis
        ax3 = fig.add_subplot(gs[3])

        # Get distributions and find global min/max for consistent color scaling
        true_dist = env.true_dist_pmf.reshape(args.height, args.height).cpu().numpy()
        learned_dist = (
            get_exact_P_T(env, gflownet).reshape(args.height, args.height).numpy()
        )

        # Ensure consistent orientation by transposing
        true_dist = true_dist.T
        learned_dist = learned_dist.T

        vmin = min(true_dist.min(), learned_dist.min())
        vmax = max(true_dist.max(), learned_dist.max())

        # True reward distribution
        im1 = ax1.imshow(
            true_dist,
            cmap="viridis",
            interpolation="none",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax1.set_title("True Distribution")

        # Learned reward distribution
        _ = ax2.imshow(
            learned_dist,
            cmap="viridis",
            interpolation="none",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax2.set_title("Learned Distribution")

        # Add colorbar in its own axis
        plt.colorbar(im1, cax=cax)

        # L1 distances over time
        states_per_validation = args.batch_size * args.validation_interval
        validation_states = [i * states_per_validation for i in range(len(l1_distances))]
        ax3.plot(validation_states, l1_distances)
        ax3.set_xlabel("States Visited")
        ax3.set_ylabel("L1 Distance")
        ax3.set_title("L1 Distance Evolution")
        ax3.set_yscale("log")  # Set log scale for y-axis

        plt.tight_layout()
        plt.show()
        plt.close()

    return validation_info["l1_dist"]


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=8, help="Height of the environment"
    )
    parser.add_argument("--R0", type=float, default=0.1, help="Environment's R0")
    parser.add_argument("--R1", type=float, default=0.5, help="Environment's R1")
    parser.add_argument("--R2", type=float, default=2.0, help="Environment's R2")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed, if 0 then a random seed is used",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=1000,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )
    parser.add_argument(
        "--replay_buffer_prioritized",
        action="store_true",
        help="If set and replay_buffer_size > 0, use a prioritized replay buffer.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["FM", "TB", "DB", "SubTB", "ZVar", "ModifiedDB"],
        default="FM",
        help="Loss function to use",
    )
    parser.add_argument(
        "--subTB_weighting",
        type=str,
        default="geometric_within",
        help="weighting scheme for SubTB",
    )
    parser.add_argument(
        "--subTB_lambda", type=float, default=0.9, help="Lambda parameter for SubTB"
    )

    parser.add_argument(
        "--tabular",
        action="store_true",
        help="Use a lookup table for F, PF, PB instead of an estimator",
    )
    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform PB")
    parser.add_argument(
        "--tied", action="store_true", help="Tie the parameters of PF, PB, and F"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the estimators' neural network modules.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=2,
        help="Number of hidden layers (of size `hidden_dim`) in the estimators'"
        + " neural network modules",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_Z",
        type=float,
        default=0.1,
        help="Specific learning rate for Z (only used for TB loss)",
    )

    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(1e6),
        help="Total budget of trajectories to train on. "
        + "Training iterations = n_trajectories // batch_size",
    )

    parser.add_argument(
        "--validation_interval",
        type=int,
        default=100,
        help="How often (in training steps) to validate the gflownet",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of true and learned distributions (only works for 2D, incompatible with wandb)",
    )

    args = parser.parse_args()

    print(main(args))
