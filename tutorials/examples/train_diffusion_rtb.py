#!/usr/bin/env python
"""
Minimal end-to-end Relative Trajectory Balance (RTB) training script for diffusion.

Now includes:
- Optional prior pretraining (auto-runs if the prior checkpoint is missing), so
  finetuning starts from the same learned prior used in the reference scripts.
- An optimizer helper that mirrors the reference param grouping (policy vs. logZ).
- Hooks to add additional posterior targets (keep existing defaults).

Uses the 25→9 GMM posterior target (`gmm25_posterior9`) by default with a learnable
posterior forward policy and a fixed prior forward policy. Loss is RTB (no backward
policy). At the end of training, saves a scatter plot of sampled states to the user's
home directory.
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gfn.estimators import PinnedBrownianMotionBackward, PinnedBrownianMotionForward
from gfn.gflownet import RelativeTrajectoryBalanceGFlowNet
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.gym.helpers.diffusion_utils import viz_2d_slice
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.modules import (
    DiffusionFixedBackwardModule,
    DiffusionPISGradNetBackward,
    DiffusionPISGradNetForward,
)


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


def build_backward_estimator(
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
    pb_scale_range: float,
    log_var_range: float,
    device: torch.device,
) -> PinnedBrownianMotionBackward:
    """Build learnable backward policy (pb) with optional variance correction."""
    pb_module = DiffusionPISGradNetBackward(
        s_dim=s_dim,
        harmonics_dim=harmonics_dim,
        t_emb_dim=t_emb_dim,
        s_emb_dim=s_emb_dim,
        hidden_dim=hidden_dim,
        joint_layers=joint_layers,
        zero_init=zero_init,
        clipping=clipping,
        gfn_clip=gfn_clip,
        pb_scale_range=pb_scale_range,
        log_var_range=log_var_range,
        learn_variance=learn_variance,
    )
    return PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=sigma,
        num_discretization_steps=num_steps,
        n_variance_outputs=1 if learn_variance else 0,
        pb_scale_range=pb_scale_range,
    ).to(device)


def _backward_mle_loss(
    pf: PinnedBrownianMotionForward,
    pb: PinnedBrownianMotionBackward,
    samples: torch.Tensor,
    num_steps: int,
    sigma: float,
    t_scale: float,
    exploration_std: float = 0.0,
    debug: bool = False,
) -> torch.Tensor:
    """
    Backward MLE:
      1) Sample backward path via Brownian bridge + optional learned pb corrections.
      2) Evaluate forward log-prob of observed increments under pf (with learned var).
      3) Minimize negative sum of logpf.
    """
    device = samples.device
    dtype = samples.dtype
    bsz, dim = samples.shape
    dt = 1.0 / num_steps
    base_std_fixed = sigma * math.sqrt(dt) * math.sqrt(t_scale)
    log_2pi = math.log(2 * math.pi)

    # Start from terminal states (data samples).
    s_curr = samples
    logpf_sum = torch.zeros(bsz, device=device, dtype=dtype)

    exploration_std_t = torch.as_tensor(
        exploration_std, device=device, dtype=dtype
    ).clamp(min=0.0)

    for i in range(num_steps):
        # Forward time index for transition s_prev -> s_curr.
        t_fwd = torch.full((bsz, 1), 1.0 - (i + 1) * dt, device=device, dtype=dtype)
        t_curr = torch.full((bsz, 1), 1.0 - i * dt, device=device, dtype=dtype)

        # Backward sampler (Brownian bridge base + optional corrections).
        pb_inp = torch.cat([s_curr, t_curr], dim=1)
        pb_out = pb.module(pb_inp)

        is_s0 = (t_curr - dt) < dt * 1e-2
        # Brownian bridge (t_prev = t_curr - dt), conditioned to hit 0 at t=0:
        #   mean_bb = s_curr * (1 - dt / t_curr)
        #   std_bb  = sigma * sqrt(dt * (t_curr - dt) / t_curr)
        # At t_prev=0, both mean and std collapse to 0.
        base_mean = torch.where(
            is_s0,
            torch.zeros_like(s_curr),
            s_curr * (1.0 - dt / t_curr),
        )
        base_std = torch.where(
            is_s0,
            torch.zeros_like(t_curr),
            sigma * (dt * (t_curr - dt) / t_curr).sqrt(),
        )

        mean_corr = pb_out[..., :dim] * pb.pb_scale_range

        # Learned variance case.
        if pb_out.shape[-1] == dim + 1:
            log_std_corr = pb_out[..., [-1]] * pb.pb_scale_range
            corr_std = torch.exp(log_std_corr)
        else:
            corr_std = torch.zeros_like(base_std)

        # Combine bridge variance with optional learned correction (no t_scale here; forward handles it).
        bwd_std = (base_std**2 + corr_std**2).sqrt()
        noise = torch.randn_like(s_curr, device=device, dtype=dtype)
        s_prev = base_mean + mean_corr + bwd_std * noise

        # Forward log-prob under model for observed increment (s_prev -> s_curr).
        model_inp = torch.cat([s_prev, t_fwd], dim=1)
        module_out = pf.module(model_inp)
        increment = s_curr - s_prev

        # Forward log p(s_prev -> s_curr).
        # If model predicts variance (s_dim + 1 output): σ_i = exp(log_std_i)*sqrt(dt*t_scale)
        # log p = -0.5 * Σ_i [ ((Δ - dt μ)_i / σ_i)^2 + 2 log σ_i + log 2π ]
        if module_out.shape[-1] == dim + 1:
            drift = module_out[..., :dim]
            log_std = module_out[..., [-1]]
            std = torch.exp(log_std) * math.sqrt(dt) * math.sqrt(t_scale)
            if exploration_std_t.item() > 0:
                std = torch.sqrt(std**2 + exploration_std_t**2)
            diff = increment - dt * drift
            logpf_step = -0.5 * ((diff / std) ** 2 + 2 * std.log() + log_2pi).sum(dim=1)
        else:
            # Fixed variance: σ = sigma*sqrt(dt*t_scale); same log p form with shared σ.
            drift = module_out
            std = base_std_fixed
            if exploration_std_t.item() > 0:
                std = math.sqrt(base_std_fixed**2 + float(exploration_std_t.item()) ** 2)
            diff = increment - dt * drift
            logpf_step = -0.5 * ((diff / std) ** 2).sum(dim=1) - 0.5 * dim * (
                log_2pi + 2 * math.log(std)
            )

        logpf_sum += logpf_step
        s_curr = s_prev

    # Negative log-likelihood (mean over batch).
    if debug and torch.isnan(logpf_sum).any():
        raise ValueError("NaNs in logpf_sum during pretrain loss.")

    return -(logpf_sum.mean())


def pretrain_prior_if_needed(
    args: argparse.Namespace,
    device: torch.device,
    s_dim: int,
) -> Path:
    """
    Auto-pretrain the prior if the checkpoint is missing.
    Saves to args.prior_ckpt_path and returns the resolved path.
    """
    ckpt_path = Path(os.path.expanduser(args.prior_ckpt_path))
    if ckpt_path.exists() or not args.pretrain_if_missing:
        return ckpt_path

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
    if args.learn_pb:
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

    optim_params = [{"params": pf_prior.parameters(), "lr": args.pretrain_lr}]
    if args.learn_pb:
        optim_params.append(
            {"params": pb_prior.parameters(), "lr": args.pretrain_lr_back}
        )
    optimizer = torch.optim.Adam(
        optim_params,
        lr=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
    )

    pf_prior.train()
    pbar = tqdm(range(args.pretrain_steps), dynamic_ncols=True, desc="pretrain_prior")

    for it in pbar:
        with torch.no_grad():
            batch = env_prior.target.sample(args.pretrain_batch_size)
        optimizer.zero_grad()
        loss = _backward_mle_loss(
            pf_prior,
            pb_prior,
            batch,
            num_steps=args.pretrain_num_steps,
            sigma=args.pretrain_sigma,
            t_scale=args.t_scale,
            exploration_std=args.pretrain_exploration_factor,
            debug=args.debug_pretrain,
        )
        loss.backward()
        if args.debug_pretrain:
            grad_list = [
                p.grad.norm() for p in pf_prior.parameters() if p.grad is not None
            ]
            total_norm = (
                torch.norm(torch.stack(grad_list)) if grad_list else torch.tensor(0.0)
            )
            print(
                f"[pretrain][debug] step={it} loss={loss.item():.4e} grad_norm={total_norm.item():.4e}"
            )
            if torch.isnan(total_norm):
                raise ValueError("NaN grad norm in pretrain.")

        optimizer.step()

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

        if (it + 1) % args.pretrain_log_interval == 0 or it == args.pretrain_steps - 1:
            pbar.set_postfix({"loss": float(loss.item())})
            if (
                it + 1
            ) % args.pretrain_ckpt_interval == 0 or it == args.pretrain_steps - 1:
                _save_checkpoint(pf_prior, pb_prior, optimizer, it, ckpt_path)

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
            os.path.expanduser(args.pretrain_save_fig_path),
            return_fig=False,
        )
        print(f"[pretrain] Saved prior samples plot to {args.pretrain_save_fig_path}")

    return ckpt_path


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

    # Prior forward (fixed, no grad). Will be loaded from checkpoint if available.
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
    # finetuning starts from the learned prior (mirrors reference behavior).
    prior_ckpt_path = pretrain_prior_if_needed(args, device, s_dim)
    if prior_ckpt_path.exists():
        ckpt = torch.load(prior_ckpt_path, map_location=device)
        state = ckpt.get("pf_state_dict", ckpt)
        missing, unexpected = pf_prior.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] prior load missing={missing}, unexpected={unexpected}")
        # Initialize posterior from the same prior weights.
        pf_post.load_state_dict(pf_prior.state_dict(), strict=False)
    else:
        raise Exception(
            f"pretrained weights not found at {prior_ckpt_path}, pretraining failed"
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
        action=argparse.BooleanOptionalAction,
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
        default=1.0,  # 5.0
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
        default=0.5,
        help="Base exploration std applied per step when exploratory is enabled (reference ~0.5)",
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
        "--save_fig_path",
        type=str,
        default="output/rtb_final_samples.png",
        help="Path to save final samples plot",
    )

    # Prior pretraining / loading
    parser.add_argument(
        "--prior_ckpt_path",
        type=str,
        default="output/prior.pt",
        help="Path to save/load the pretrained prior checkpoint",
    )
    parser.add_argument(
        "--pretrain_if_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-run prior pretraining if the checkpoint is missing",
    )
    parser.add_argument(
        "--pretrain_use_bwd_mle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exact backward MLE (reference) instead of surrogate bridge loss",
    )
    parser.add_argument(
        "--learn_pb",
        action=argparse.BooleanOptionalAction,
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
        "--pretrain_batch_size",
        type=int,
        default=500,
        help="Batch size for prior pretraining",
    )
    parser.add_argument(
        "--pretrain_steps",
        type=int,
        default=10000,
        help="Training steps for prior pretraining",
    )
    parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3, help="LR for prior pretraining"
    )
    parser.add_argument(
        "--pretrain_lr_back",
        type=float,
        default=1e-3,
        help="LR for backward policy during pretrain",
    )
    parser.add_argument(
        "--pretrain_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for prior pretraining",
    )
    parser.add_argument(
        "--pretrain_log_interval",
        type=int,
        default=100,
        help="Logging interval (steps) during prior pretraining (reference: 100)",
    )
    parser.add_argument(
        "--pretrain_ckpt_interval",
        type=int,
        default=1000,
        help="Checkpoint interval during prior pretraining (reference: 1000)",
    )
    parser.add_argument(
        "--pretrain_vis_n",
        type=int,
        default=2000,
        help="Number of samples to plot after prior pretraining",
    )
    parser.add_argument(
        "--pretrain_save_fig_path",
        type=str,
        default="output/prior_pretrain.png",
        help="Path to save prior samples plot after pretraining",
    )
    parser.add_argument(
        "--debug_pretrain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable extra NaN/grad checks during pretrain loss",
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
