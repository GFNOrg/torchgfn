#!/usr/bin/env python
"""
Example script for training a diffusion sampler with a GFlowNet loss (Trajectory Balance)
with SimpleGaussianMixtureTarget as the target unnormalized distribution.

Here, we use the pinned Brownian motion as the reference process; see https://arxiv.org/abs/2402.05098
for more details, and see https://arxiv.org/abs/2302.13834 or https://arxiv.org/abs/2211.01364 for
Ornstein-Uhlenbeck process as the reference process.

Reference: https://github.com/GFNOrg/gfn-diffusion
"""

import argparse

import torch
from tqdm import tqdm

from gfn.estimators import PinnedBrownianMotionBackward, PinnedBrownianMotionForward
from gfn.gflownet import TBGFlowNet
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.modules import DiffusionFixedBackwardModule, DiffusionPISGradNetForward


def main(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Pass command-line arguments to the target class for the diffusion sampling
    # environment.
    target_kwargs = {"seed": args.target_seed}
    if args.dim is not None:
        target_kwargs["dim"] = args.dim
    if args.num_components is not None:
        target_kwargs["num_components"] = args.num_components

    # Set up environment.
    env = DiffusionSampling(
        target_str=args.target,
        target_kwargs=target_kwargs,
        num_discretization_steps=args.num_steps,
        device=device,
        check_action_validity=False,
    )

    # Build forward/backward modules and estimators
    s_dim = env.dim
    pf_module = DiffusionPISGradNetForward(
        s_dim=s_dim,
        harmonics_dim=args.harmonics_dim,
        t_emb_dim=args.t_emb_dim,
        s_emb_dim=args.s_emb_dim,
        hidden_dim=args.hidden_dim,
        joint_layers=args.joint_layers,
        zero_init=args.zero_init,
    )
    pb_module = DiffusionFixedBackwardModule(s_dim=s_dim)

    pf_estimator = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=args.sigma,
        num_discretization_steps=args.num_steps,
    )
    pb_estimator = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=args.sigma,
        num_discretization_steps=args.num_steps,
    )

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
    gflownet = gflownet.to(device)

    sampler = Sampler(estimator=pf_estimator)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        # On-policy sampling
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
        )

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        loss.backward()
        optimizer.step()

        if args.visualize:
            if (
                it == 0
                or (it + 1) % args.vis_interval == 0
                or it == args.n_iterations - 1
            ):
                with torch.no_grad():
                    samples_states = gflownet.sample_terminating_states(env, args.vis_n)
                    xs = samples_states.tensor[:, :-1]
                    env.target.visualize(
                        samples=xs,
                        prefix=f"it{it}_",
                        show=False,
                    )  # type: ignore[attr-defined]

        pbar.set_postfix({"loss": float(loss.item())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # System
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Target / environment
    parser.add_argument("--target", type=str, default="gmm2", help="Target name")
    parser.add_argument(
        "--dim", type=int, default=None, help="State dimension override for the target"
    )
    parser.add_argument(
        "--num_components", type=int, default=None, help="Mixture components"
    )
    parser.add_argument("--target_seed", type=int, default=2, help="Target RNG seed")

    # Discretization / diffusion params
    parser.add_argument(
        "--num_steps", type=int, default=32, help="number of discretization steps"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,  # 5.0 is for simple_gmm target, you might need to change this for other targets
        help="diffusion coefficient for the pinned Brownian motion",
    )

    # Model (PISGradNet)
    parser.add_argument("--harmonics_dim", type=int, default=64)
    parser.add_argument("--t_emb_dim", type=int, default=64)
    parser.add_argument("--s_emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--joint_layers", type=int, default=2)
    parser.add_argument("--zero_init", action="store_true")

    # Training
    parser.add_argument("--n_iterations", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_logz", type=float, default=1e-1)

    # Visualization
    parser.add_argument("--vis_interval", type=int, default=200)
    parser.add_argument("--vis_n", type=int, default=2000)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    main(args)
