#!/usr/bin/env python
"""
Minimal end-to-end Relative Trajectory Balance (RTB) fine-tuning training script for
diffusion models.

- Prior is pre-trained (auto-runs if the prior checkpoint is missing), so
  finetuning starts from a learned prior.
- Posterior is fine-tuned from this prior (pf).

By default, uses the 25→9 GMM posterior target (`gmm25_posterior9`) by default with a
learnable posterior forward policy and a fixed prior forward policy. Loss is RTB (no
backward policy). This script outputs the prior weights alongside plots of samples
from both the prior and posterior distributions.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gfn.estimators import PinnedBrownianMotionBackward, PinnedBrownianMotionForward
from gfn.gflownet import RelativeTrajectoryBalanceGFlowNet
from gfn.gflownet.mle import MLEDiffusion
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.gym.helpers.diffusion_utils import viz_2d_slice
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.modules import (
    DiffusionFixedBackwardModule,
    DiffusionPISGradNetBackward,
    DiffusionPISGradNetForward,
)


def resolve_output_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve all output paths relative to this script's directory."""
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    args.output_dir = output_dir
    args.prior_ckpt_path = output_dir / "train_diffusion_rtb_prior_ckpt.pt"
    args.pretrain_save_fig_path = output_dir / "train_diffusion_rtb_prior_samples.png"
    args.save_fig_path = output_dir / "train_diffusion_rtb_posterior_samples.png"

    return args


def get_debug_metrics(estimator: torch.nn.Module) -> tuple[torch.Tensor, bool]:
    """Compute gradient norm for a module; return (total_norm, has_nan)."""
    grad_list = [p.grad.norm() for p in estimator.parameters() if p.grad is not None]
    if grad_list:
        total_norm = torch.norm(torch.stack(grad_list))
    else:
        total_norm = torch.tensor(0.0, device=next(estimator.parameters()).device)
    has_nan = torch.isnan(total_norm)
    return total_norm, bool(has_nan)


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
    learn_variance: bool,
    clipping: bool,
    gfn_clip: float,
    t_scale: float,
    log_var_range: float,
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
        clipping=clipping,
        gfn_clip=gfn_clip,
        t_scale=t_scale,
        log_var_range=log_var_range,
        learn_variance=learn_variance,
    )

    return PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=sigma,
        num_discretization_steps=num_steps,
        n_variance_outputs=1 if learn_variance else 0,
    ).to(device)


def pretrain_prior(args: argparse.Namespace, device: torch.device, s_dim: int) -> None:
    """
    Auto-pretrain the prior if the checkpoint is missing.
    Saves to args.prior_ckpt_path and returns the resolved path.
    """
    ckpt_path = Path(args.prior_ckpt_path)

    if ckpt_path.exists():
        if args.clobber_pretrained_prior:
            print(f"[pretrain] Clobbering existing prior checkpoint at {ckpt_path}")
            ckpt_path.unlink()
        else:
            return

    print(f"[pretrain] Prior checkpoint missing at {ckpt_path}, starting pretraining...")

    env_prior = DiffusionSampling(
        target_str=args.pretrain_target,
        target_kwargs=None,
        num_discretization_steps=args.pretrain_num_steps,
        device=device,
        debug=__debug__,
    )

    pf_prior = build_forward_estimator(
        s_dim=s_dim,
        num_steps=args.pretrain_num_steps,
        sigma=args.pretrain_sigma,
        harmonics_dim=args.harmonics_dim,
        t_emb_dim=args.t_emb_dim,
        s_emb_dim=args.s_emb_dim,
        hidden_dim=args.hidden_dim,
        joint_layers=args.joint_layers,
        zero_init=args.zero_init,
        learn_variance=args.learn_variance,
        clipping=args.clipping,
        gfn_clip=args.gfn_clip,
        t_scale=args.t_scale,
        log_var_range=args.log_var_range,
        device=device,
    )

    # Build backward estimator: learned pb if enabled, else fixed Brownian bridge.
    if args.pretrain_learn_pb:
        pb_module = DiffusionPISGradNetBackward(
            s_dim=s_dim,
            harmonics_dim=args.harmonics_dim,
            t_emb_dim=args.t_emb_dim,
            s_emb_dim=args.s_emb_dim,
            hidden_dim=args.hidden_dim,
            joint_layers=args.joint_layers,
            zero_init=args.zero_init,
            clipping=args.clipping,
            gfn_clip=args.gfn_clip,
            pb_scale_range=args.pb_scale_range,
            log_var_range=args.log_var_range,
            learn_variance=args.learn_variance,
        )
        n_var_outputs = 1 if args.learn_variance else 0
        pb_scale_range = args.pb_scale_range
    else:
        pb_module = DiffusionFixedBackwardModule(s_dim)
        n_var_outputs = 0
        pb_scale_range = 0.0

    pb_prior = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=args.pretrain_sigma,
        num_discretization_steps=args.pretrain_num_steps,
        n_variance_outputs=n_var_outputs,
        pb_scale_range=pb_scale_range,
    ).to(device)

    optim_params = [{"params": pf_prior.parameters(), "lr": args.lr}]
    if args.pretrain_learn_pb:
        optim_params.append({"params": pb_prior.parameters(), "lr": args.lr})
    optimizer = torch.optim.Adam(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # MLE trainer (uses forward PF and optional PB).
    mle_trainer = MLEDiffusion(
        pf=pf_prior,
        pb=pb_prior,
        num_steps=args.pretrain_num_steps,
        sigma=args.pretrain_sigma,
        t_scale=args.t_scale,
        pb_scale_range=args.pb_scale_range,
        learn_variance=args.learn_variance,
        debug=__debug__,
    )

    def _save_checkpoint(pf_prior, pb_prior, optimizer, it, ckpt_path):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pf_state_dict": pf_prior.state_dict(),
                "pb_state_dict": pb_prior.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": it + 1,
            },
            ckpt_path,
        )

    pf_prior.train()
    pbar = tqdm(range(args.pretrain_steps), dynamic_ncols=True, desc="pretrain_prior")

    for it in pbar:
        with torch.no_grad():
            batch = env_prior.target.sample(args.batch_size)
        optimizer.zero_grad()
        loss = mle_trainer.loss(batch, exploration_std=args.pretrain_exploration_factor)
        loss.backward()
        if __debug__:
            total_norm, has_nan = get_debug_metrics(pf_prior)
            print(
                f"[pretrain][debug] step={it} loss={loss.item():.4e} grad_norm={total_norm.item():.4e}"
            )
            if has_nan:
                raise ValueError("NaN grad norm in pretrain.")

        optimizer.step()

        # Log progress only.
        if (it + 1) % args.pretrain_log_interval == 0 or it == args.pretrain_steps - 1:
            pbar.set_postfix({"loss": float(loss.item())})

    # Final checkpoint after pretraining (no intermediate resume support).
    _save_checkpoint(pf_prior, pb_prior, optimizer, it, ckpt_path)
    print(f"[pretrain] Saved prior to {ckpt_path}")

    # Quick visual check of the learned prior.
    with torch.no_grad():
        sampler_prior = Sampler(estimator=pf_prior)
        trajectories = sampler_prior.sample_trajectories(
            env=env_prior,
            n=args.pretrain_vis_n,
        )
        xs = trajectories.terminating_states.tensor[:, :-1]
        plot_samples(
            xs,
            env_prior.target,
            "RTB Prior Samples",
            args.pretrain_save_fig_path,
            return_fig=False,
        )
        print(f"[pretrain] Saved prior samples plot to {args.pretrain_save_fig_path}")


def plot_samples(
    xs: torch.Tensor,
    target,
    title: str,
    save_path: Path | str,
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

    ax.set_title(title)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)

    if return_fig:
        return fig

    plt.close(fig)

    return None


def main(args: argparse.Namespace) -> None:
    """Runs the posterior finetuning pipeline, including prior pretraining if required."""
    args = resolve_output_paths(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    torch.set_default_device(device)

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
        learn_variance=args.learn_variance,
        clipping=args.clipping,
        gfn_clip=args.gfn_clip,
        t_scale=args.t_scale,
        log_var_range=args.log_var_range,
        device=device,
    )

    # Prior forward.
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
        learn_variance=args.learn_variance,
        clipping=args.clipping,
        gfn_clip=args.gfn_clip,
        t_scale=args.t_scale,
        log_var_range=args.log_var_range,
        device=device,
    )

    # Pretrain prior if needed, then load weights into both prior and posterior so
    # finetuning starts from the learned prior.
    pretrain_prior(args, device, s_dim)

    if args.prior_ckpt_path.exists():
        ckpt = torch.load(args.prior_ckpt_path, map_location=device)
        state = ckpt.get("pf_state_dict", ckpt)
        missing, unexpected = pf_prior.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] prior load missing={missing}, unexpected={unexpected}")
        # Initialize posterior from the same prior weights.
        pf_post.load_state_dict(pf_prior.state_dict(), strict=False)
    else:
        raise Exception(
            f"pretrained weights not found at {args.prior_ckpt_path}, pretraining failed"
        )

    # During finetuning, the prior is fixed, no grad,
    pf_prior.eval()
    for p in pf_prior.parameters():
        p.requires_grad_(False)

    gflownet = RelativeTrajectoryBalanceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        init_logZ=0.0,
        beta=args.beta,
    ).to(device)

    sampler = Sampler(estimator=pf_post)

    param_groups = [
        {"params": gflownet.pf.parameters(), "lr": args.lr},
        {"params": gflownet.logz_parameters(), "lr": args.lr_logz},
    ]
    optimizer = torch.optim.Adam(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
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
    plot_samples(
        xs,
        env.target,
        "RTB Posterior Samples",
        args.save_fig_path,
        return_fig=False,
    )
    print(f"Saved final samples scatter to {args.save_fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for training.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Target / environment
    parser.add_argument(
        "--target",
        type=str,
        default="gmm25_posterior9",
        help="Diffusion target (default: gmm25_posterior9)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="number of discretization steps (reference=100)",
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
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--joint_layers", type=int, default=2)
    parser.add_argument("--zero_init", action="store_true", default=True)
    parser.add_argument(
        "--learn_variance",
        action="store_true",
        default=False,
        help="Use learned scalar variance in the diffusion forward policy (ref default: off)",
    )
    parser.add_argument(
        "--clipping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip model outputs (reference default: off)",
    )
    parser.add_argument(
        "--gfn_clip",
        type=float,
        default=1e4,
        help="Clipping value for drift outputs (reference: 1e4)",
    )
    parser.add_argument(
        "--t_scale",
        type=float,
        default=5.0,
        help="Scale diffusion std to mirror reference (reference: 5.0)",
    )
    parser.add_argument(
        "--log_var_range",
        type=float,
        default=4.0,
        help="Range to bound learned log-std when learn_variance is enabled (reference: 4.0)",
    )

    # Training
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_logz", type=float, default=1e-1)
    parser.add_argument("--beta", type=float, default=1.0, help="RTB beta multiplier")
    # Exploration noise (state-space Gaussian added in quadrature to PF std)
    parser.add_argument(
        "--exploration_factor",
        type=float,
        default=0.5,
        help="Base exploration std applied per step when exploratory is enabled (reference ~0.5)",
    )
    parser.add_argument(
        "--exploration_warm_down_start",
        type=float,
        default=500,
        help="Linearly warm down exploration after n iters (to 0 by exploration_warm_down_end iters)",
    )
    parser.add_argument(
        "--exploration_warm_down_end",
        type=float,
        default=4500,
        help="Linearly warm down exploration after n iters (to 0 by exploration_warm_down_end iters)",
    )

    # Logging / eval
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_n", type=int, default=500)
    parser.add_argument(
        "--vis_n", type=int, default=2000, help="Number of samples for final plot"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Base output dir (resolved relative to this script)",
    )

    # Prior pretraining / loading
    parser.add_argument(
        "--clobber_pretrained_prior",
        action="store_true",
        default=False,
        help="Overwrite existing prior checkpoint and re-run pretraining",
    )
    parser.add_argument(
        "--pretrain_learn_pb",
        action="store_true",
        default=False,
        help="Enable learned backward policy corrections (pb) during pretrain",
    )
    parser.add_argument(
        "--pb_scale_range",
        type=float,
        default=0.1,
        help="Tanh scaling for backward mean/var corrections (reference: 0.1)",
    )
    parser.add_argument(
        "--pretrain_target",
        type=str,
        default="gmm25_prior",
        help="Target used for prior pretraining (matches reference prior)",
    )
    parser.add_argument(
        "--pretrain_num_steps",
        type=int,
        default=100,
        help="Discretization steps for prior pretraining (reference=100)",
    )
    parser.add_argument(
        "--pretrain_sigma",
        type=float,
        default=2.0,
        help="Diffusion coefficient for prior pretraining",
    )
    parser.add_argument(
        "--pretrain_exploration_factor",
        type=float,
        default=0.0,
        help="Exploration std for pretrain backward MLE (reference: off by default)",
    )
    parser.add_argument(
        "--pretrain_steps",
        type=int,
        default=10000,
        help="Training steps for prior pretraining",
    )

    # Optimizer extras
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for the RTB optimizer (policy/logZ)",
    )

    args = parser.parse_args()
    main(args)
