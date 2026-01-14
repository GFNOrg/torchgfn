r"""
The goal of this script is to train a GFlowNet on the Box environment using
Cartesian per-dimension increments.

Example usage:
    python train_box.py --delta 0.25 --tied --loss TB
    python train_box.py --delta 0.1 --loss DB --n_components 5

Based on results from:
[A theory of continuous generative flow networks](https://arxiv.org/abs/2301.12594)
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from numpy.typing import NDArray
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from tqdm import tqdm, trange

from gfn.estimators import ScalarEstimator
from gfn.gflownet import (
    DBGFlowNet,
    LogPartitionVarianceGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import Box
from gfn.gym.helpers.box_utils import (
    BoxCartesianPBEstimator,
    BoxCartesianPBMLP,
    BoxCartesianPFEstimator,
    BoxCartesianPFMLP,
    BoxStateFlowModule,
)
from gfn.preprocessors import IdentityPreprocessor
from gfn.samplers import LocalSearchSampler, Sampler
from gfn.utils.common import set_seed

DEFAULT_SEED: int = 4444


def sample_from_reward(env: Box, n_samples: int) -> NDArray[np.float64]:
    """Samples states from the true reward distribution

    Implement rejection sampling, with proposal being uniform distribution in [0, 1]^2
    Returns:
        A numpy array of shape (n_samples, 2) containing the sampled states
    """
    samples = []
    while len(samples) < n_samples:
        sample = env.reset(batch_shape=(n_samples,), random=True)
        rewards = env.reward(sample)
        rand_n = torch.rand(n_samples).to(env.device)
        mask = rand_n * (env.R0 + max(env.R1, env.R2)) < rewards
        true_samples = sample[mask]
        samples.extend(true_samples[-(n_samples - len(samples)) :].tensor.cpu().numpy())
    return np.array(samples)


def get_test_states(n: int = 100, maxi: float = 1.0) -> NDArray[np.float64]:
    """Create a list of states from [0, 1]^2 by discretizing it into n x n grid.

    Returns:
        A numpy array of shape (n^2, 2) containing the test states,
    """
    x = np.linspace(0.001, maxi, n)
    y = np.linspace(0.001, maxi, n)
    xx, yy = np.meshgrid(x, y)
    test_states = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    return test_states


def estimate_jsd(kde1: KernelDensity, kde2: KernelDensity) -> float:
    """Estimate Jensen-Shannon divergence between two distributions defined by KDEs

    Returns:
        A float value of the estimated JSD
    """
    test_states = get_test_states()
    log_dens1 = kde1.score_samples(test_states)
    log_dens1 = log_dens1 - logsumexp(log_dens1)
    log_dens2 = kde2.score_samples(test_states)
    log_dens2 = log_dens2 - logsumexp(log_dens2)
    log_dens = np.log(0.5 * np.exp(log_dens1) + 0.5 * np.exp(log_dens2))
    jsd = np.sum(np.exp(log_dens1) * (log_dens1 - log_dens))
    jsd += np.sum(np.exp(log_dens2) * (log_dens2 - log_dens))
    return jsd / 2.0


def plot_trajectories(
    env: Box,
    sampler: Sampler,
    n_trajectories: int = 100,
    output_path: Optional[str] = None,
    alpha: float = 0.1,
) -> None:
    """Plot sampled trajectories on the Box environment.

    Each trajectory is plotted as a line from s0 to the terminal state,
    with transparency to visualize overlapping paths.

    Args:
        env: The Box environment.
        sampler: The sampler to use for generating trajectories.
        n_trajectories: Number of trajectories to sample and plot.
        output_path: Path to save the output plot. If None, defaults to
            'output/train_box_trajectories.png' relative to this script.
        alpha: Transparency for each trajectory line.
    """
    # Default output path relative to script location
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = str(script_dir / "output" / "train_box_trajectories.png")

    # Sample trajectories
    trajectories = sampler.sample_trajectories(env, n=n_trajectories)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Get all states: shape is (max_length+1, n_trajectories, state_dim)
    all_states = trajectories.states.tensor.cpu().numpy()
    terminating_idx = trajectories.terminating_idx.cpu().numpy()

    # Plot reward contours (corners at (0,0), (0,1), (1,0), (1,1))
    # Create a grid for the reward landscape
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    grid_states = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
    )
    rewards = env.reward(env.States(grid_states.to(env.device)))
    Z = rewards.cpu().numpy().reshape(X.shape)

    # Plot reward contours
    contour = ax.contourf(X, Y, Z, levels=20, alpha=0.3, cmap="Blues")
    plt.colorbar(contour, ax=ax, label="Reward")

    # Plot each trajectory
    for i in range(n_trajectories):
        # Get the states for this trajectory up to its terminating index
        term_idx = terminating_idx[i]
        traj_states = all_states[: term_idx + 1, i, :]  # (length, state_dim)

        # Plot the trajectory path (black lines, low alpha)
        ax.plot(
            traj_states[:, 0],
            traj_states[:, 1],
            "k-",
            alpha=alpha,
            linewidth=0.5,
        )

        # Mark the terminal state (red dots, full alpha, larger than lines)
        ax.scatter(
            traj_states[-2, 0],
            traj_states[-2, 1],
            c="red",
            s=2,
            alpha=0.2,
            zorder=5,
            marker="D",
            edgecolors="darkred",
            linewidths=1,
        )

    # Mark the source state
    ax.scatter([0], [0], c="green", s=100, marker="*", zorder=10, label="s0")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"Sampled Trajectories (n={n_trajectories})")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved to: {output_path}")


def main(args: Namespace) -> float:  # noqa: C901
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    use_wandb = len(args.wandb_project) > 0
    if use_wandb:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    n_iterations = args.n_trajectories // args.batch_size

    # 1. Create the environment
    env = Box(delta=args.delta, epsilon=1e-10, device=device, debug=__debug__)
    preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])

    # 2. Create the gflownet.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #    Using Cartesian estimators (per-dimension increments, simpler and faster)
    pf_module = BoxCartesianPFMLP(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        n_components=args.n_components,
    )
    pb_module = BoxCartesianPBMLP(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        n_components=args.n_components,
        trunk=pf_module.trunk if args.tied else None,
    )

    pf_estimator = BoxCartesianPFEstimator(
        env,
        pf_module,
        n_components=args.n_components,
        min_concentration=args.min_concentration,
        max_concentration=args.max_concentration,
    )
    pb_estimator = BoxCartesianPBEstimator(
        env,
        pb_module,
        n_components=args.n_components,
        min_concentration=args.min_concentration,
        max_concentration=args.max_concentration,
    )
    module: Optional[BoxStateFlowModule] = None
    logZ: Optional[torch.Tensor] = None

    assert args.loss in ("DB", "SubTB", "TB", "ZVar"), f"Invalid loss: {args.loss}"
    assert args.subTB_weighting in (
        "geometric_within",
        "geometric_between",
    ), f"Invalid subTB weighting: {args.subTB_weighting}"

    if args.loss in ("DB", "SubTB"):
        # We always need a LogZEstimator
        logZ = torch.tensor(0.0, device=env.device, requires_grad=True)
        # We need a LogStateFlowEstimator

        module = BoxStateFlowModule(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=None,  # We do not tie the parameters of the flow function to PF
            logZ_value=logZ,
        )
        logF_estimator = ScalarEstimator(module=module, preprocessor=preprocessor)

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
                weighting=args.subTB_weighting,  # type: ignore
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

    gflownet = gflownet.to(device)

    if not args.use_local_search:
        sampler = Sampler(estimator=pf_estimator)
        local_search_params = {}
    else:
        sampler = LocalSearchSampler(
            pf_estimator=pf_estimator, pb_estimator=pb_estimator
        )
        local_search_params = {
            "n_local_search_loops": args.n_local_search_loops,
            "back_ratio": args.back_ratio,
            "use_metropolis_hastings": args.use_metropolis_hastings,
        }

    # 3. Create the optimizer and scheduler

    optimizer = torch.optim.Adam(pf_module.parameters(), lr=args.lr)
    assert isinstance(pb_module.last_layer, torch.nn.Module)
    optimizer.add_param_group(
        {
            "params": (
                pb_module.last_layer.parameters()
                if args.tied
                else pb_module.parameters()
            ),
            "lr": args.lr,
        }
    )
    if args.loss in ("DB", "SubTB"):
        assert module is not None
        optimizer.add_param_group(
            {
                "params": module.parameters(),
                "lr": args.lr_F,
            }
        )
    if "logZ" in dict(gflownet.named_parameters()):
        logZ = dict(gflownet.named_parameters())["logZ"]
    if args.loss != "ZVar":
        assert logZ is not None
        optimizer.add_param_group({"params": [logZ], "lr": args.lr_Z})

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            i * args.scheduler_milestone
            for i in range(1, 1 + int(n_iterations / args.scheduler_milestone))
        ],
        gamma=args.gamma_scheduler,
    )

    # 4. Sample from the true reward distribution, and fit a KDE to the samples
    samples_from_reward = sample_from_reward(env, n_samples=args.validation_samples)
    true_kde = KernelDensity(kernel="exponential", bandwidth=0.1).fit(
        samples_from_reward
    )

    states_visited = 0

    jsd = float("inf")
    for iteration in trange(n_iterations, dynamic_ncols=True):
        if iteration % 1000 == 0:
            print(f"current optimizer LR: {optimizer.param_groups[0]['lr']}")

        # Sampling on-policy, so we save logprobs for faster computation.
        trajectories = sampler.sample_trajectories(
            env, save_logprobs=True, n=args.batch_size, **local_search_params
        )

        optimizer.zero_grad()
        loss = gflownet.loss_from_trajectories(
            env, trajectories, recalculate_all_logprobs=False
        )
        loss.backward()
        for p in gflownet.parameters():
            if p.ndim > 0 and p.grad is not None:  # We do not clip logZ grad.
                p.grad.data.clamp_(-10, 10).nan_to_num_(0.0)
        optimizer.step()
        scheduler.step()

        states_visited += len(trajectories)

        to_log = {"loss": loss.item(), "states_visited": states_visited}
        logZ_info = ""
        if args.loss != "ZVar":
            assert logZ is not None
            to_log.update({"logZdiff": env.log_partition - logZ.item()})
            logZ_info = f"logZ: {logZ.item():.2f}, "
        if use_wandb:
            wandb.log(to_log, step=iteration)
        if iteration % (args.validation_interval // 5) == 0:
            tqdm.write(
                f"States: {states_visited}, "
                f"Loss: {loss.item():.3f}, {logZ_info}"
                f"true logZ: {env.log_partition:.2f}, JSD: {jsd:.4f}"
            )

        if iteration % args.validation_interval == 0:
            validation_samples = gflownet.sample_terminating_states(
                env, args.validation_samples
            )
            kde = KernelDensity(kernel="exponential", bandwidth=0.1).fit(
                validation_samples.tensor.detach().cpu().numpy()
            )
            jsd = estimate_jsd(kde, true_kde)

            if use_wandb:
                wandb.log({"JSD": jsd}, step=iteration)

            to_log.update({"JSD": jsd})

    # Plot trajectories at the end of training
    print("\nGenerating trajectory visualization...")
    plot_trajectories(
        env=env,
        sampler=sampler,
        n_trajectories=1000,
    )

    return jsd


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="maximum distance between two successive states (min_incr in gflownet)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed, if 0 then a random seed is used",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["TB", "DB", "SubTB", "ZVar"],
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
        "--min_concentration",
        type=float,
        default=0.1,
        help="minimal value for the Beta concentration parameters",
    )

    parser.add_argument(
        "--max_concentration",
        type=float,
        default=100.0,
        help="maximal value for the Beta concentration parameters",
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=5,
        help="Number of mixture components for Beta distributions",
    )
    parser.add_argument(
        "--tied",
        action="store_true",
        help="Tie the parameters of PF, PB. F is never tied.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
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
        default=1e-4,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_Z",
        type=float,
        default=1e-2,
        help="Specific learning rate for logZ (should be higher than base lr)",
    )
    parser.add_argument(
        "--lr_F",
        type=float,
        default=1e-2,
        help="Specific learning rate for the state flow function (only used for DB and SubTB losses)",
    )
    parser.add_argument(
        "--gamma_scheduler",
        type=float,
        default=0.5,
        help="Every scheduler_milestone steps, multiply the learning rate by gamma_scheduler",
    )
    parser.add_argument(
        "--scheduler_milestone",
        type=int,
        default=2500,
        help="Every scheduler_milestone steps, multiply the learning rate by gamma_scheduler",
    )

    parser.add_argument(
        "--use_local_search",
        action="store_true",
        help="Use local search to sample the next state",
    )

    # Local search parameters.
    parser.add_argument(
        "--n_local_search_loops",
        type=int,
        default=2,
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

    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(3e6),
        help="Total budget of trajectories to train on. "
        + "Training iterations = n_trajectories // batch_size",
    )

    parser.add_argument(
        "--validation_interval",
        type=int,
        default=500,
        help="How often (in training steps) to validate the gflownet",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=10000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )

    args = parser.parse_args()

    print(main(args))
