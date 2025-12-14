#!/usr/bin/env python
"""
Example script for training a diffusion sampler with a GFlowNet loss (Trajectory Balance)
with SimpleGaussianMixtureTarget as the target unnormalized distribution.

Here, we use the pinned Brownian motion as the reference process; see https://arxiv.org/abs/2402.05098
for more details, and see https://arxiv.org/abs/2302.13834 or https://arxiv.org/abs/2211.01364 for
examples of using the Ornstein-Uhlenbeck process as the reference process.

Reference: https://github.com/GFNOrg/gfn-diffusion
"""

import argparse
import math

import torch
from tqdm import tqdm

from gfn.estimators import PinnedBrownianMotionBackward, PinnedBrownianMotionForward
from gfn.gflownet import PFBasedGFlowNet, TBGFlowNet
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.modules import DiffusionFixedBackwardModule, DiffusionPISGradNetForward
from gfn.utils.prob_calculations import get_trajectory_pfs_and_pbs


def evaluate_density_metrics(
    gflownet: PFBasedGFlowNet,
    env: DiffusionSampling,
    eval_n: int,
    eval_batch_size: int,
) -> dict:
    assert gflownet.pb is not None
    fwd_sampler = Sampler(estimator=gflownet.pf)

    n_batches = math.ceil(eval_n / eval_batch_size)
    fwd_log_pfs_list = []
    fwd_log_pbs_list = []
    fwd_log_rewards_list = []
    for i in range(n_batches):
        batch_size = (
            eval_batch_size if i < n_batches - 1 else eval_n - i * eval_batch_size
        )
        trajectories = fwd_sampler.sample_trajectories(
            env,
            n=batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
        )
        fwd_log_pfs, fwd_log_pbs = get_trajectory_pfs_and_pbs(
            gflownet.pf,
            gflownet.pb,
            trajectories,
            recalculate_all_logprobs=False,
        )
        fwd_log_rewards = trajectories.log_rewards
        fwd_log_pfs_list.append(fwd_log_pfs)
        fwd_log_pbs_list.append(fwd_log_pbs)
        fwd_log_rewards_list.append(fwd_log_rewards)
    fwd_log_pfs = torch.cat(fwd_log_pfs_list, dim=0)
    fwd_log_pbs = torch.cat(fwd_log_pbs_list, dim=0)
    fwd_log_rewards = torch.cat(fwd_log_rewards_list, dim=0)

    gt_xs, gt_xs_log_rewards = env.target.cached_sample(batch_size=eval_n)
    if gt_xs is not None and gt_xs_log_rewards is not None:
        bwd_sampler = Sampler(estimator=gflownet.pb)
        bwd_log_pfs_list = []
        bwd_log_pbs_list = []
        bwd_log_rewards_list = []
        for i in range(n_batches):
            xs_batch = gt_xs[i * eval_batch_size : (i + 1) * eval_batch_size]
            xs_batch_with_time = torch.cat(
                [xs_batch, torch.ones(xs_batch.shape[0], 1, device=xs_batch.device)],
                dim=1,
            )
            states_batch = env.states_from_tensor(xs_batch_with_time)
            bwd_trajectories = bwd_sampler.sample_trajectories(
                env,
                states=states_batch,
                save_logprobs=False,  # backward logprobs can't be saved (TODO: fix this)
                save_estimator_outputs=False,
            )
            bwd_trajectories_reversed = bwd_trajectories.reverse_backward_trajectories()
            bwd_log_pfs, bwd_log_pbs = get_trajectory_pfs_and_pbs(
                gflownet.pf,
                gflownet.pb,
                bwd_trajectories_reversed,
                recalculate_all_logprobs=False,
            )
            bwd_log_rewards = bwd_trajectories_reversed.log_rewards
            bwd_log_pfs_list.append(bwd_log_pfs)
            bwd_log_pbs_list.append(bwd_log_pbs)
            bwd_log_rewards_list.append(bwd_log_rewards)
        bwd_log_pfs = torch.cat(bwd_log_pfs_list, dim=0)
        bwd_log_pbs = torch.cat(bwd_log_pbs_list, dim=0)
        bwd_log_rewards = torch.cat(bwd_log_rewards_list, dim=0)

    assert isinstance(gflownet.logZ, torch.nn.Parameter)  # TODO: support other cases
    try:
        gt_logz = env.target.gt_logz()
    except NotImplementedError:
        gt_logz = None
    return env.density_metrics(
        fwd_log_pfs=fwd_log_pfs,
        fwd_log_pbs=fwd_log_pbs,
        fwd_log_rewards=fwd_log_rewards,
        log_Z_learned=gflownet.logZ.item(),
        bwd_log_pfs=bwd_log_pfs,
        bwd_log_pbs=bwd_log_pbs,
        bwd_log_rewards=bwd_log_rewards,
        gt_log_Z=gt_logz,
    )


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
        debug=__debug__,
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

        if it == 0 or (it + 1) % args.eval_interval == 0 or it == args.n_iterations - 1:
            density_metrics = evaluate_density_metrics(
                gflownet, env, args.eval_n, args.eval_batch_size
            )
            print(f"Evaluation metrics at iteration {it}:")
            for key, value in density_metrics.items():
                print(f"{key}: {value:.4f}")
            print("-" * 40)

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

    # Model (DiffusionPISGradNetForward)
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

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_n", type=int, default=2000)
    parser.add_argument("--eval_batch_size", type=int, default=2000)

    # Visualization
    parser.add_argument("--vis_interval", type=int, default=200)
    parser.add_argument("--vis_n", type=int, default=2000)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    main(args)
