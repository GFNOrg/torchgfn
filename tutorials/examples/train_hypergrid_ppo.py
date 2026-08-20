#!/usr/bin/env python
r"""Soft policy-gradient GFlowNet training (VPG and Entropic PPO) on HyperGrid.

Reproduces Algorithms 1 and 2 of Zykova-Myzina et al., "Proximal Policy Optimization for
Amortized Discrete Sampling" (arXiv:2606.15793), which train the forward policy as an
entropy-regularized RL agent instead of enforcing a flow-balance condition.

The outer loop follows Algorithm 2 exactly:

    1. Roll out `batch_size` trajectories under the current policy.
    2. Freeze the soft rewards, GAE advantages and value-fit targets for that batch
       (`gflownet.to_training_samples`).
    3. Take `K` policy-gradient steps on the frozen batch (Ent-PPO's importance ratio and
       KL trust region are what make K > 1 safe).
    4. Fit the soft value function for `E` epochs of `S` mini-batches each.
    5. Optionally update the backward policy by Trajectory Likelihood Maximization —
       the soft-RL objectives admit no joint forward/backward loss, because changing
       P_B changes the MDP reward itself.

Defaults are the paper's tuned Hypergrid configuration (Appendix G): batch size 16,
policy lr 1e-3, value lr = policy lr / 3, lambda = 0.7, E = 4, S = 2, clip eps = 0.2.

Example usage:

    # Ent-PPO with 4 update epochs per batch (the paper's headline setting).
    python train_hypergrid_ppo.py --method ent_ppo --K 4

    # VPG, comparing advantage estimators (their Section 5.2).
    python train_hypergrid_ppo.py --method vpg --advantage reward_to_go

    # The ablations of their Figs. 7 and 8.
    python train_hypergrid_ppo.py --method ent_ppo --K 8 --no_kl
    python train_hypergrid_ppo.py --method ent_ppo --K 8 --no_clip

Reported `l1_dist` is twice the Total Variation distance used in the paper's figures.
"""

import argparse
from typing import Iterator

import torch
from tqdm import tqdm

from gfn.containers import PolicyGradientTrajectories
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import EntPPOGFlowNet, PolicyGradientGFlowNet, tlm_loss
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform


def batch_splits(
    batch: PolicyGradientTrajectories, n_splits: int
) -> Iterator[PolicyGradientTrajectories]:
    """Yields `n_splits` random mini-batches of a frozen rollout.

    Slicing a PolicyGradientTrajectories carries its frozen advantages and value targets
    along, so every mini-batch sees the same targets the full batch would.

    Args:
        batch: The prepared rollout to split.
        n_splits: Number of mini-batches per epoch (the paper's `S`).

    Yields:
        Mini-batches of trajectories.
    """
    perm = torch.randperm(len(batch), device=batch.device)
    for chunk in perm.chunk(n_splits):
        if len(chunk) > 0:
            yield batch[chunk]


def build_gflownet(args, env, preprocessor):
    """Builds the policy-gradient GFlowNet and its estimators.

    Args:
        args: Parsed command-line arguments.
        env: The HyperGrid environment.
        preprocessor: The state preprocessor shared by all estimators.

    Returns:
        The configured GFlowNet.
    """
    module_pf = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    # P_B is the per-step soft-MDP reward, r(s_t, s_{t+1}) = log P_B(s_t | s_{t+1}).
    # An MLP that no optimizer ever updates would shape that reward by its random
    # initialization, so it is only built when TLM actually trains it.
    if args.uniform_pb or args.learn_pb == "none":
        module_pb = DiscreteUniform(output_dim=env.n_actions - 1)
    else:
        module_pb = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1)

    pf = DiscretePolicyEstimator(
        module_pf, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb = DiscretePolicyEstimator(
        module_pb, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    # The paper uses a separate value network mirroring the policy encoder, with a
    # scalar head and no shared parameters.
    needs_value = args.method == "ent_ppo" or args.advantage in ("baseline", "gae")
    logV = (
        ScalarEstimator(
            MLP(input_dim=preprocessor.output_dim, output_dim=1),
            preprocessor=preprocessor,
        )
        if needs_value
        else None
    )

    if args.method == "ent_ppo":
        assert logV is not None
        return EntPPOGFlowNet(
            pf=pf,
            pb=pb,
            logV=logV,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            use_kl=not args.no_kl,
            use_clipping=not args.no_clip,
            kl_estimator=args.kl_estimator,
            critic_loss=args.critic,
        )
    return PolicyGradientGFlowNet(
        pf=pf,
        pb=pb,
        logV=logV,
        advantage=args.advantage,
        gae_lambda=args.gae_lambda,
        critic_loss=args.critic,
    )


def main(args) -> dict:
    """Runs soft policy-gradient training on HyperGrid.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A dict with the final validation metrics and the number of reward evaluations.
    """
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    env = HyperGrid(
        ndim=args.ndim,
        height=args.height,
        reward_fn_str="original",
        reward_fn_kwargs={"R0": args.R0, "R1": args.R1, "R2": args.R2},
        device=device,
        calculate_partition=True,
        store_all_states=True,
        validate_modes=False,
        debug=__debug__,
    )
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    gflownet = build_gflownet(args, env, preprocessor).to(device)

    # Separate optimizers: the policy and the critic take a different number of steps
    # per rollout, and the paper tunes their learning rates independently.
    policy_optimizer = torch.optim.Adam([*gflownet.pf.parameters()], lr=args.lr)
    value_optimizer = (
        torch.optim.Adam(gflownet.logV_parameters(), lr=args.lr_value or args.lr / 3)
        if gflownet.logV is not None
        else None
    )
    # build_gflownet always provides a backward policy estimator.
    assert gflownet.pb is not None
    pb_optimizer = (
        torch.optim.Adam(gflownet.pb.parameters(), lr=args.lr)
        if args.learn_pb != "none" and not args.uniform_pb
        else None
    )

    sampler = Sampler(estimator=gflownet.pf)
    reward_evaluations = 0
    validation_info = {"l1_dist": float("inf")}

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        # 1. Roll out under the current policy. The estimator outputs are the pi_old
        #    logits that Ent-PPO's analytic KL needs.
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            save_estimator_outputs=True,
            epsilon=args.epsilon,
        )
        reward_evaluations += args.batch_size

        # 2. Freeze the advantages and value targets for this rollout.
        batch = gflownet.to_training_samples(trajectories)

        # 3. K policy-gradient epochs on the frozen batch.
        for _ in range(args.K):
            policy_optimizer.zero_grad()
            policy_loss = gflownet.policy_loss(batch)
            policy_loss.backward()
            gflownet.assert_finite_gradients()
            torch.nn.utils.clip_grad_norm_(gflownet.pf.parameters(), 1.0)
            policy_optimizer.step()

        # 4. E value epochs of S mini-batches each.
        value_loss = torch.zeros((), device=device)
        if value_optimizer is not None:
            for _ in range(args.E):
                for split in batch_splits(batch, args.S):
                    value_optimizer.zero_grad()
                    step_loss = gflownet.value_loss(env, split)
                    step_loss.backward()
                    gflownet.assert_finite_gradients()
                    value_loss = step_loss.detach()
                    value_optimizer.step()

        # 5. Backward-policy learning by TLM (Eq. 24). These rollouts consume no reward
        #    evaluations: the TLM objective never touches R.
        if pb_optimizer is not None:
            n_pb_updates = args.K if args.learn_pb == "tlm_k" else 1
            for _ in range(n_pb_updates):
                pb_trajectories = (
                    trajectories
                    if args.learn_pb == "tlm"
                    else sampler.sample_trajectories(
                        env, n=args.batch_size, save_logprobs=False
                    )
                )
                pb_optimizer.zero_grad()
                tlm_loss(gflownet.pb, pb_trajectories).backward()
                pb_optimizer.step()

        gflownet.assert_finite_parameters()

        if (it + 1) % args.validation_interval == 0:
            validation_info, _ = env.validate(gflownet, args.validation_samples)
            print(
                f"Iter {it + 1}: TV={validation_info['l1_dist'] / 2:.6f} "
                f"JSD={validation_info['jsd']:.6f} "
                f"reward evaluations={reward_evaluations}"
            )

        pbar.set_postfix(
            {
                "policy_loss": f"{policy_loss.detach().item():.4f}",
                "value_loss": f"{value_loss.item():.4f}",
                "reward_evals": reward_evaluations,
            }
        )

    return {**validation_info, "reward_evaluations": reward_evaluations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Environment.
    parser.add_argument("--ndim", type=int, default=2, help="HyperGrid dimensions")
    parser.add_argument("--height", type=int, default=16, help="HyperGrid height")
    parser.add_argument("--R0", type=float, default=0.1, help="Environment's R0")
    parser.add_argument("--R1", type=float, default=0.5, help="Environment's R1")
    parser.add_argument("--R2", type=float, default=2.0, help="Environment's R2")

    # Objective.
    parser.add_argument(
        "--method",
        type=str,
        choices=["vpg", "ent_ppo"],
        default="ent_ppo",
        help="Vanilla policy gradient, or Entropic PPO",
    )
    parser.add_argument(
        "--advantage",
        type=str,
        choices=["total", "reward_to_go", "baseline", "gae"],
        default="gae",
        help="Advantage estimator for VPG (Ent-PPO always uses GAE)",
    )
    parser.add_argument(
        "--gae_lambda", type=float, default=0.7, help="GAE lambda (paper's best value)"
    )
    parser.add_argument(
        "--clip_eps", type=float, default=0.2, help="PPO clipping parameter"
    )
    parser.add_argument(
        "--no_kl",
        action="store_true",
        help="Ablation: drop the analytic KL, reducing Ent-PPO to naive PPO (Fig. 7)",
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Ablation: drop the ratio clipping (Fig. 8)",
    )
    parser.add_argument(
        "--kl_estimator",
        type=str,
        choices=["analytic", "importance"],
        default="analytic",
        help="Exact KL, or the single-sample importance-weighted estimator",
    )
    parser.add_argument(
        "--critic",
        type=str,
        choices=["mse", "sub_eb"],
        default="mse",
        help="Value-function objective: MSE regression, or Sub-EB balance (Eq. 15)",
    )

    # Optimization.
    parser.add_argument("--batch_size", type=int, default=16, help="Trajectories/batch")
    parser.add_argument(
        "--K", type=int, default=4, help="Policy update epochs per rollout batch"
    )
    parser.add_argument(
        "--E", type=int, default=4, help="Value-network epochs per rollout batch"
    )
    parser.add_argument(
        "--S", type=int, default=2, help="Mini-batch splits within a value epoch"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Policy learning rate")
    parser.add_argument(
        "--lr_value",
        type=float,
        default=None,
        help="Value learning rate (defaults to lr / 3, as in the paper)",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=5000, help="Number of outer iterations"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.0, help="Sampler exploration parameter"
    )

    # Backward policy.
    parser.add_argument(
        "--uniform_pb",
        action="store_true",
        help="Use a fixed uniform backward policy. Implied by --learn_pb none",
    )
    parser.add_argument(
        "--learn_pb",
        type=str,
        choices=["none", "tlm", "tlm_k"],
        default="none",
        help="Backward-policy learning: off, one TLM update per iteration, or K "
        "updates on freshly sampled rollouts",
    )

    # Validation.
    parser.add_argument(
        "--validation_interval", type=int, default=500, help="Validation interval"
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Trajectories sampled to estimate the terminating distribution. The "
        "estimator has a noise floor of its own: on a small grid a *perfect* model "
        "scores TV ~ 0.06 at 10k samples, so under-sampling makes every method look "
        "equally good. See tutorials/notebooks/policy_gradient_gflownets.ipynb for an "
        "exact, sampling-free alternative.",
    )

    main(parser.parse_args())
