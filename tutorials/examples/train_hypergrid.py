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

import logging
import os
import time
from argparse import ArgumentParser
from typing import cast

import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
from tqdm import trange

from gfn.containers import NormBasedDiversePrioritizedReplayBuffer, ReplayBuffer
from gfn.containers.replay_buffer import ContainerUnion
from gfn.containers.replay_buffer_manager import ReplayBufferManager
from gfn.estimators import DiscretePolicyEstimator, Estimator, ScalarEstimator
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
from gfn.preprocessors import KHotPreprocessor
from gfn.states import DiscreteStates
from gfn.utils.common import Timer, set_seed
from gfn.utils.modules import MLP, DiscreteUniform, Tabular

logger = logging.getLogger(__name__)


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
    if env.ndim != 2:
        raise ValueError("plotting is only supported for 2D environments")

    grid = env.all_states
    assert grid is not None, "all_states is not implemented in the environment"

    # Get the forward policy distribution for all states
    with torch.no_grad():
        # Handle both FM and other GFlowNet types
        policy: Estimator = cast(
            Estimator, gflownet.logF if isinstance(gflownet, FMGFlowNet) else gflownet.pf
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

        # Compute u values for parent states.
        parent_u_values = []
        for p in parents_indices:
            grid_idx = torch.all(grid.tensor == p, 1)  # index along flattened grid.
            parent_u_values.append(u[grid_idx])
        parent_u_values = torch.stack(parent_u_values)

        # Compute probabilities for parent transitions.
        parent_probs = []
        for p, a in zip(parents_indices, action_indices):
            grid_idx = torch.all(grid.tensor == p, 1)  # index along flattened grid.
            parent_probs.append(probabilities[grid_idx, a])
        parent_probs = torch.stack(parent_probs)

        u[indices.index(index)] = torch.sum(parent_u_values * parent_probs)

    return (u * probabilities[..., -1]).detach().cpu()


def _make_optimizer_for(gflownet, args) -> torch.optim.Optimizer:
    """Build a fresh AdamW optimizer for a (re)built GFlowNet with logZ group."""
    named = dict(gflownet.named_parameters())
    non_logz = [v for k, v in named.items() if k != "logZ"]
    logz = [named["logZ"]] if "logZ" in named else []

    return torch.optim.AdamW(
        [
            {"params": non_logz, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": logz, "lr": args.lr_Z, "weight_decay": 0.0},
        ]
    )


def set_up_fm_gflownet(args, env, preprocessor):
    """Returns a FM GFlowNet."""
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
    else:
        module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )

    estimator = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    return FMGFlowNet(estimator)


def set_up_pb_pf_estimators(args, env, preprocessor):
    """Returns a pair of estimators for the forward and backward policies."""
    if args.tabular:
        pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        if not args.uniform_pb:
            pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
    else:
        pf_module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )
        if not args.uniform_pb:
            pb_module = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk=(
                    pf_module.trunk
                    if args.tied and isinstance(pf_module.trunk, torch.nn.Module)
                    else None
                ),
            )
    if args.uniform_pb:
        pb_module = DiscreteUniform(env.n_actions - 1)

    for v in ["pf_module", "pb_module"]:
        assert locals()[v] is not None, f"{v} is None, Args: {args}"

    assert pf_module is not None
    assert pb_module is not None
    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )

    return (pf_estimator, pb_estimator)


def set_up_logF_estimator(args, env, preprocessor, pf_module):
    """Returns a LogStateFlowEstimator."""
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=1)
    else:
        module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=(
                pf_module.trunk
                if args.tied and isinstance(pf_module.trunk, torch.nn.Module)
                else None
            ),
        )

    return ScalarEstimator(module=module, preprocessor=preprocessor)


def set_up_gflownet(args, env, preprocessor):
    """Returns a GFlowNet complete with the required estimators."""
    if args.loss == "FM":
        gflownet = set_up_fm_gflownet(args, env, preprocessor)
        return gflownet

    # We need a DiscretePFEstimator and a DiscretePBEstimator.
    pf_estimator, pb_estimator = set_up_pb_pf_estimators(args, env, preprocessor)
    assert pf_estimator is not None
    assert pb_estimator is not None

    if args.loss == "ModifiedDB":
        return ModifiedDBGFlowNet(pf_estimator, pb_estimator)

    elif args.loss == "TB":
        return TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)

    elif args.loss == "ZVar":
        return LogPartitionVarianceGFlowNet(pf=pf_estimator, pb=pb_estimator)

    elif args.loss in ("DB", "SubTB"):
        logF_estimator = set_up_logF_estimator(args, env, preprocessor, pf_estimator)

        if args.loss == "DB":
            return DBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                logF=logF_estimator,
            )
        elif args.loss == "SubTB":
            return SubTBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                logF=logF_estimator,
                weighting=args.subTB_weighting,
                lamda=args.subTB_lambda,
            )


def plot_results(env, gflownet, l1_distances, args):
    # Create figure with 3 subplots with proper spacing
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 0.1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])  # Colorbar axis
    ax3 = fig.add_subplot(gs[3])

    # Get distributions and find global min/max for consistent color scaling
    true_dist = env.true_dist()
    assert isinstance(true_dist, torch.Tensor)
    true_dist = true_dist.reshape(args.height, args.height).cpu().numpy()
    learned_dist = (
        get_exact_P_T(env, gflownet).reshape(args.height, args.height).cpu().numpy()
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


def main(args) -> dict:  # noqa: C901
    """Trains a GFlowNet on the Hypergrid Environment."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Check if plotting is allowed.
    if args.plot:
        if args.wandb_project:
            raise ValueError("plot argument is incompatible with wandb_project")
        if args.ndim != 2:
            raise ValueError("plotting is only supported for 2D environments")

    set_seed(args.seed)

    # Initialize the environment.
    env = HyperGrid(
        args.ndim,
        args.height,
        device=device,
        reward_fn_str="original",
        reward_fn_kwargs={
            "R0": args.R0,
            "R1": args.R1,
            "R2": args.R2,
        },
        calculate_partition=args.validate_environment,
        store_all_states=args.validate_environment,
        debug=__debug__,
    )

    # Initialize WandB.
    use_wandb = args.wandb_project != ""
    if use_wandb:
        if args.wandb_local:
            os.environ["WANDB_MODE"] = "offline"

        import wandb

        wandb.init(
            project=args.wandb_project,
            group=wandb.util.generate_id(),
            entity=args.wandb_entity,
            config=vars(args),
        )

    # Initialize the preprocessor.
    preprocessor = KHotPreprocessor(height=args.height, ndim=args.ndim)

    # Build the model and optimizer.
    gflownet = set_up_gflownet(args, env, preprocessor)
    assert gflownet is not None
    gflownet = gflownet.to(device)
    optimizer = _make_optimizer_for(gflownet, args)

    # Create replay buffer if needed.
    replay_buffer = None
    if args.replay_buffer_size > 0:
        replay_buffer = ReplayBuffer(
            env,
            capacity=args.replay_buffer_size,
        )

    n_iterations = args.n_trajectories // args.batch_size
    modes_found: set = set()

    logger.info("n_iterations = %d", n_iterations)

    # Initialize some variables before the training loop.
    timing: dict = {}
    time_start = time.time()
    l1_distances = []

    # Used for calculating the L1 distance across all nodes.
    all_visited_terminating_states = env.states_from_batch_shape((0,))
    to_log: dict = {}

    # Training loop.
    pbar = trange(n_iterations)
    for iteration in pbar:
        # Keep track of visited terminating states.
        visited_terminating_states = env.states_from_batch_shape((0,))

        # Determine on-policy for this iteration.
        is_on_policy = (
            (args.replay_buffer_size == 0)
            and (args.epsilon == 0.0)
            and (args.temperature == 1.0)
        )

        # Sample trajectories.
        with Timer(timing, "generate_samples", enabled=args.timing):
            trajectories = gflownet.sample_trajectories(
                env,
                n=args.batch_size,
                save_logprobs=is_on_policy,
                save_estimator_outputs=not is_on_policy,
                epsilon=args.epsilon,
                temperature=args.temperature,
            )

        # Training objects (incl. possible replay buffer sampling).
        with Timer(timing, "to_training_samples", enabled=args.timing):
            training_samples = gflownet.to_training_samples(trajectories)

            if replay_buffer is not None:
                with torch.no_grad():
                    replay_buffer.add(training_samples)
                    training_objects = replay_buffer.sample(n_samples=args.batch_size)
            else:
                training_objects = training_samples

        # Loss.
        with Timer(timing, "calculate_loss", enabled=args.timing):
            optimizer.zero_grad()
            loss = gflownet.loss(
                env,
                training_objects,  # type: ignore
                recalculate_all_logprobs=(not is_on_policy),
                reduction="sum" if args.loss == "SubTB" else "mean",  # type: ignore
            )

        # Backpropagation.
        with Timer(timing, "loss_backward", enabled=args.timing):
            loss.backward()

        # Optimization.
        with Timer(timing, "optimizer", enabled=args.timing):
            optimizer.step()

        log_this_iter = (
            iteration % args.validation_interval == 0
        ) or iteration == n_iterations - 1

        # Keep track of trajectories / states.
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )
        all_visited_terminating_states.extend(visited_terminating_states)

        to_log = {
            "loss": loss.item(),
            "l1_dist": None,
        }

        if log_this_iter:
            modes_found.update(env.modes_found(all_visited_terminating_states))
            n_modes_found = len(modes_found)
            to_log["n_modes_found"] = n_modes_found

            if args.validate_environment:
                with Timer(timing, "validation", enabled=args.timing):
                    validation_info, _ = env.validate(
                        gflownet,
                        args.validation_samples,
                    )
                    to_log.update(validation_info)

            pbar.set_postfix(
                loss=to_log["loss"],
                l1_dist=to_log["l1_dist"],
                n_modes_found=to_log.get("n_modes_found", 0),
            )

            if use_wandb:
                wandb.log(to_log, step=iteration)

    logger.info("Finished all iterations")
    total_time = time.time() - time_start

    if args.timing:
        timing["total_time"] = [total_time]
        logger.info("\n" + "=" * 80)
        logger.info("\n Timing information:")
        logger.info("=" * 80)
        logger.info("%-25s %12s", "Step Name", "Time (s)")
        logger.info("-" * 80)
        for k, v in timing.items():
            logger.info("%-25s %10.4fs", k, sum(v))

    # Plot the results if requested & possible.
    if args.plot:
        plot_results(env, gflownet, l1_distances, args)

    print("Training complete, logs:", to_log)

    return to_log


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = ArgumentParser()

    # Machine setting.
    parser.add_argument("--seed", type=int, default=4444, help="Random seed.")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Prevent CUDA usage",
    )

    # Environment settings.
    parser.add_argument(
        "--ndim",
        type=int,
        default=2,
        help="Number of dimensions in the environment",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=8,
        help="Height of the environment",
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

    # Training settings.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=2048,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["FM", "TB", "DB", "SubTB", "ZVar", "ModifiedDB"],
        default="TB",
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
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(1e6),
        help=(
            "Total budget of trajectories to train on. "
            "Training iterations = n_trajectories // batch_size"
        ),
    )

    # Exploration settings.
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon for epsilon-greedy exploration (default: 0.0).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0).",
    )

    # Policy architecture.
    parser.add_argument(
        "--tabular",
        action="store_true",
        help="Use a lookup table for F, PF, PB instead of an estimator",
    )
    parser.add_argument(
        "--uniform_pb",
        action="store_true",
        help="Use a uniform PB",
    )
    parser.add_argument(
        "--tied",
        action="store_true",
        help="Tie the parameters of PF, PB, and F",
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
        default=3,
        help=(
            "Number of hidden layers incl. input projection (of size `hidden_dim`)"
            " in the estimators' neural network modules"
        ),
    )

    # Validation settings.
    parser.add_argument(
        "--validate_environment",
        action="store_true",
        help="Validate the environment at the end of training",
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

    # WandB settings.
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Name of the wandb entity. If empty, don't use wandb",
    )
    parser.add_argument(
        "--wandb_local",
        action="store_true",
        help="Stores wandb results locally, to be uploaded later.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of true and learned distributions (only works for 2D, incompatible with wandb)",
    )

    parser.add_argument(
        "--timing",
        action="store_true",
        default=True,
        help="Report timing information at the end of training",
    )

    args = parser.parse_args()
    main(args)
