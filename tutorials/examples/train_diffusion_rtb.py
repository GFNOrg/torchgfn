#!/usr/bin/env python
"""
Minimal end-to-end Relative Trajectory Balance (RTB) training script for diffusion.

Uses the 25→9 GMM posterior target (`gmm25_posterior9`) with a learnable
posterior forward policy and a fixed prior forward policy. Loss is RTB
(no backward policy). At the end of training, saves a scatter plot of sampled
states to the user's home directory.
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gfn.estimators import PinnedBrownianMotionForward
from gfn.gflownet import RelativeTrajectoryBalanceGFlowNet
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.gym.helpers.diffusion_utils import viz_2d_slice
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.modules import DiffusionPISGradNetForward


def get_exploration_std(
    iteration: int,
    exploration_factor: float = 0.1,
    warm_down_start: int = 500,
    warm_down_end: int = 4500,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a callable exploration std schedule for state-space noise.

    When exploration is enabled, return a step-index function that emits a fixed
    std for the current training iteration, optionally linearly warmed down
    after warm_down_start iters toward 0 by warm_down_end iters.
    """
    device = device or torch.get_default_device()
    dtype = dtype or torch.get_default_dtype()

    # Tensor ops only (torch.compile-friendly): no Python branching on iteration.
    iter_t = torch.tensor(iteration, device=device, dtype=dtype)
    # Clamp negatives to zero to avoid Python-side checks/overhead.
    factor_t = torch.clamp(
        torch.tensor(exploration_factor, device=device, dtype=dtype), min=0.0
    )
    start_t = torch.tensor(warm_down_start, device=device, dtype=dtype)
    end_t = torch.tensor(warm_down_end, device=device, dtype=dtype)

    # Phase indicator: 1 before warm_down_start, linear decay afterward.
    progress = torch.clamp(iter_t / end_t, min=0.0, max=1.0)
    decay = torch.where(
        iter_t < start_t, torch.ones_like(progress), torch.clamp(1.0 - progress, min=0.0)
    )
    exploration_std = factor_t * decay

    return exploration_std


def build_forward_estimator(
    s_dim: int,
    num_steps: int,
    sigma: float,
    harmonics_dim: int,
    t_emb_dim: int,
    s_emb_dim: int,
    hidden_dim: int,
    joint_layers: int,
    zero_init: bool,
    device: torch.device,
) -> PinnedBrownianMotionForward:
    pf_module = DiffusionPISGradNetForward(
        s_dim=s_dim,
        harmonics_dim=harmonics_dim,
        t_emb_dim=t_emb_dim,
        s_emb_dim=s_emb_dim,
        hidden_dim=hidden_dim,
        joint_layers=joint_layers,
        zero_init=zero_init,
    )
    return PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=sigma,
        num_discretization_steps=num_steps,
    ).to(device)


def plot_samples(
    xs: torch.Tensor,
    target,
    save_path: str,
    return_fig: bool = False,
):
    """Contour + scatter plot of samples against the posterior density."""

    assert target.plot_border is not None, "Target must define plot_border for plotting."

    # If target exposes a posterior density, build a lightweight shim with the same
    # interface that viz_2d_slice expects (log_reward, dim, device, plot_border).
    if hasattr(target, "posterior"):
        # Use a shallow copy and replace log_reward to return posterior density
        viz_target = target

        def _posterior_log_reward(x: torch.Tensor) -> torch.Tensor:
            return viz_target.posterior.log_prob(x).flatten()

        viz_target.log_reward = _posterior_log_reward  # type: ignore[attr-defined]
    else:
        viz_target = target

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    viz_2d_slice(
        ax,
        viz_target,
        (0, 1),
        samples=xs,
        plot_border=viz_target.plot_border,
        use_log_reward=True,
        grid_width_n_points=200,
        max_n_samples=2000,
    )
    ax.set_title("RTB posterior samples")
    fig.tight_layout()
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    fig.savefig(save_path)
    if return_fig:
        return fig
    plt.close(fig)
    return None


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Environment / target
    env = DiffusionSampling(
        target_str=args.target,
        target_kwargs=None,
        num_discretization_steps=args.num_steps,
        device=device,
        debug=__debug__,
    )
    s_dim = env.dim

    # Posterior forward (trainable)
    pf_post = build_forward_estimator(
        s_dim=s_dim,
        num_steps=args.num_steps,
        sigma=args.sigma,
        harmonics_dim=args.harmonics_dim,
        t_emb_dim=args.t_emb_dim,
        s_emb_dim=args.s_emb_dim,
        hidden_dim=args.hidden_dim,
        joint_layers=args.joint_layers,
        zero_init=args.zero_init,
        device=device,
    )

    # Prior forward (fixed, no grad)
    pf_prior = build_forward_estimator(
        s_dim=s_dim,
        num_steps=args.num_steps,
        sigma=args.sigma,
        harmonics_dim=args.harmonics_dim,
        t_emb_dim=args.t_emb_dim,
        s_emb_dim=args.s_emb_dim,
        hidden_dim=args.hidden_dim,
        joint_layers=args.joint_layers,
        zero_init=args.zero_init,
        device=device,
    )
    pf_prior.eval()
    for p in pf_prior.parameters():
        p.requires_grad_(False)

    gflownet = RelativeTrajectoryBalanceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        init_logZ=0.0,
        beta=args.beta,
        log_reward_clip_min=args.log_reward_clip_min,
    ).to(device)

    sampler = Sampler(estimator=pf_post)
    optimizer = torch.optim.Adam(
        [
            {"params": gflownet.pf_pb_parameters(), "lr": args.lr},
            {"params": gflownet.logz_parameters(), "lr": args.lr_logz},
        ]
    )

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,  # if args.exploration_factor > 0 else True,
            save_estimator_outputs=False,
            # Extra exploration noise (combined with base PF variance in estimator).
            exploration_std=get_exploration_std(
                iteration=it,
                exploration_factor=args.exploration_factor,
                warm_down_start=args.exploration_warm_down_start,
                warm_down_end=args.exploration_warm_down_end,
            ),
        )

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=True)
        loss.backward()
        optimizer.step()

        if (it + 1) % args.log_interval == 0 or it == args.n_iterations - 1:
            with torch.no_grad():
                term_states = gflownet.sample_terminating_states(env, n=args.eval_n)
                rewards = env.target.log_reward(term_states.tensor[:, :-1])
                avg_reward = rewards.mean().item()
            pbar.set_postfix({"loss": float(loss.item()), "avg_reward": avg_reward})
        else:
            pbar.set_postfix({"loss": float(loss.item())})

    # Final visualization
    with torch.no_grad():
        samples_states = gflownet.sample_terminating_states(env, n=args.vis_n)
        xs = samples_states.tensor[:, :-1]
    save_path = os.path.expanduser(args.save_fig_path)
    plot_samples(
        xs,
        env.target,
        save_path,
        return_fig=False,
    )
    print(f"Saved final samples scatter to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # System
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Target / environment
    parser.add_argument(
        "--target",
        type=str,
        default="gmm25_posterior9",
        help="Diffusion target (default: gmm25_posterior9)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=256, help="number of discretization steps"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="diffusion coefficient for the pinned Brownian motion",
    )

    # Model (DiffusionPISGradNetForward)
    parser.add_argument("--harmonics_dim", type=int, default=64)
    parser.add_argument("--t_emb_dim", type=int, default=64)
    parser.add_argument("--s_emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--joint_layers", type=int, default=2)
    parser.add_argument("--zero_init", action="store_true")

    # Training
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_logz", type=float, default=1e-1)
    parser.add_argument("--beta", type=float, default=1.0, help="RTB beta multiplier")
    parser.add_argument(
        "--log_reward_clip_min",
        type=float,
        default=-float("inf"),
        help="Min clip for log reward",
    )
    # Exploration noise (state-space Gaussian added in quadrature to PF std)
    parser.add_argument(
        "--exploration_factor",
        type=float,
        default=5.0,
        help="Base exploration std applied per step when exploratory is enabled",
    )
    parser.add_argument(
        "--exploration_warm_down_start",
        type=float,
        default=0,
        help="Linearly warm down exploration after n iters (to 0 by exploration_warm_down_end iters)",
    )
    parser.add_argument(
        "--exploration_warm_down_end",
        type=float,
        default=3000,
        help="Linearly warm down exploration after n iters (to 0 by exploration_warm_down_end iters)",
    )

    # Logging / eval
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_n", type=int, default=500)
    parser.add_argument(
        "--vis_n", type=int, default=2000, help="Number of samples for final plot"
    )
    parser.add_argument(
        "--save_fig_path",
        type=str,
        default="~/rtb_final_samples.png",
        help="Path to save final samples plot",
    )

    args = parser.parse_args()
    main(args)
