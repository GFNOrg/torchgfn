#!/usr/bin/env python
"""
Conditional GFlowNet training on the HyperGrid environment.

This script demonstrates how to train conditional GFlowNets that learn different
distributions based on a continuous condition variable on the HyperGrid environment.
The condition interpolates between two extremes:

- Condition = 0: Uniform distribution (all states get reward R0+R1+R2)
- Condition = 1: Original HyperGrid multi-modal distribution
- Condition ∈ (0,1): Linear interpolation between uniform and original

During training:
- Condition values are sampled uniformly from [0, 1] for each batch
- The GFlowNet learns to generate different distributions based on the condition
- LogZ is modeled as a function of condition only (not states)

During validation:
- Fresh trajectories are sampled for multiple condition values [0, 0.25, 0.5, 0.75, 1]
- L1 distance is computed between empirical and true distributions
- Mode discovery is tracked for condition=1

Example usage:
python train_conditional.py --ndim 2 --height 8 --epsilon 0.1
"""
from argparse import ArgumentParser
from typing import cast

import torch
from torch.optim import Adam
from tqdm import tqdm

from gfn.estimators import (
    ConditionalDiscretePolicyEstimator,
    ConditionalLogZEstimator,
    ConditionalScalarEstimator,
    ScalarEstimator,
)
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import ConditionalHyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP
from gfn.utils.training import get_terminating_state_dist

DEFAULT_SEED: int = 4444


def build_conditional_pf_pb(
    env: ConditionalHyperGrid,
) -> tuple[ConditionalDiscretePolicyEstimator, ConditionalDiscretePolicyEstimator]:
    """Build conditional policy forward and backward estimators.

    Args:
        env: The ConditionalHyperGrid environment

    Returns:
        A tuple of (forward policy estimator, backward policy estimator)
    """
    # Create preprocessor for the environment
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    CONCAT_SIZE = 16
    input_dim = (
        preprocessor.output_dim
        if preprocessor.output_dim is not None
        else env.state_shape[-1]
    )
    module_PF = MLP(
        input_dim=input_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_PB = MLP(
        input_dim=input_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        trunk=module_PF.trunk,
    )

    # Encoder for the condition information.
    module_cond = MLP(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=16,
    )

    # Modules post-concatenation.
    module_final_PF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions,
    )
    module_final_PB = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions - 1,
        trunk=module_final_PF.trunk,
    )

    pf_estimator = ConditionalDiscretePolicyEstimator(
        module_PF,
        module_cond,
        module_final_PF,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=False,
    )
    pb_estimator = ConditionalDiscretePolicyEstimator(
        module_PB,
        module_cond,
        module_final_PB,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=True,
    )

    return pf_estimator, pb_estimator


def build_conditional_logF_scalar_estimator(
    env: ConditionalHyperGrid,
) -> ConditionalScalarEstimator:
    """Build conditional log flow estimator.

    Args:
        env: The ConditionalHyperGrid environment

    Returns:
        A conditional scalar estimator for log flow
    """
    # Create preprocessor for the environment
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    CONCAT_SIZE = 16
    module_state_logF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        n_hidden_layers=1,
    )
    module_condition_logF = MLP(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=16,
        n_hidden_layers=1,
    )
    module_final_logF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=1,
        hidden_dim=256,
        n_hidden_layers=1,
    )

    logF_estimator = ConditionalScalarEstimator(
        module_state_logF,
        module_condition_logF,
        module_final_logF,
        preprocessor=preprocessor,
    )

    return logF_estimator


# Build the GFlowNet -- Modules pre-concatenation.
def build_tb_gflownet(env: ConditionalHyperGrid) -> TBGFlowNet:
    """Build a Trajectory Balance GFlowNet.

    Args:
        env: The ConditionalHyperGrid environment

    Returns:
        A TBGFlowNet instance
    """
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)

    # Create conditional logZ estimator that only depends on conditions
    # LogZ should be a function of condition only, not states
    module_logZ = MLP(
        input_dim=1,  # Only condition input
        output_dim=1,  # Scalar output
        hidden_dim=64,
        n_hidden_layers=2,
    )

    logZ_estimator = ConditionalLogZEstimator(module_logZ, reduction="sum")
    gflownet = TBGFlowNet(logZ=logZ_estimator, pf=pf_estimator, pb=pb_estimator)

    return gflownet


def build_db_gflownet(env):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)
    logF_estimator = build_conditional_logF_scalar_estimator(env)
    gflownet = DBGFlowNet(logF=logF_estimator, pf=pf_estimator, pb=pb_estimator)

    return gflownet


def build_db_mod_gflownet(env):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)
    gflownet = ModifiedDBGFlowNet(pf=pf_estimator, pb=pb_estimator)

    return gflownet


def build_fm_gflownet(env):
    # Create preprocessor for the environment
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    CONCAT_SIZE = 16
    module_logF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_cond = MLP(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_final_logF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions,
    )
    logF_estimator = ConditionalDiscretePolicyEstimator(
        module_logF,
        module_cond,
        module_final_logF,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=False,
    )

    gflownet = FMGFlowNet(logF=logF_estimator)

    return gflownet


def build_subTB_gflownet(env):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)
    logF_estimator = build_conditional_logF_scalar_estimator(env)
    gflownet = SubTBGFlowNet(logF=logF_estimator, pf=pf_estimator, pb=pb_estimator)

    return gflownet


def train(
    env: ConditionalHyperGrid,
    gflownet,
    seed,
    device,
    n_iterations=10,
    batch_size=1000,
    validation_interval=100,
    validation_samples=20000,
    lr=1e-3,
    lr_logz=1e-2,
    epsilon=0.0,
):
    torch.manual_seed(seed)

    # Policy parameters and logZ/logF get independent LRs (logF/Z typically higher).
    if type(gflownet) is TBGFlowNet:
        optimizer = Adam(gflownet.pf_pb_parameters(), lr=lr)
        optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": lr_logz})
    elif type(gflownet) is DBGFlowNet or type(gflownet) is SubTBGFlowNet:
        optimizer = Adam(gflownet.pf_pb_parameters(), lr=lr)
        optimizer.add_param_group({"params": gflownet.logF_parameters(), "lr": lr_logz})
    elif type(gflownet) is FMGFlowNet or type(gflownet) is ModifiedDBGFlowNet:
        optimizer = Adam(gflownet.parameters(), lr=lr)
    else:
        print("unknown gflownet type: {}".format(type(gflownet)))

    print("+ Training Conditional {}!".format(type(gflownet)))
    print(
        f"+ Environment: ndim={env.ndim}, height={env.height}, n_states={env.n_states}"
    )
    print(
        f"+ Training parameters: n_iter={n_iterations}, batch_size={batch_size}, lr={lr}, epsilon={epsilon}"
    )

    # Track discovered modes for condition=1
    discovered_modes = set()
    mode_reward_threshold = (
        env.reward_fn_kwargs.get("R2", 2.0)
        + env.reward_fn_kwargs.get("R1", 0.5)
        + env.reward_fn_kwargs.get("R0", 0.1)
    )

    final_loss = None
    for it in (pbar := tqdm(range(n_iterations), dynamic_ncols=True)):
        # Sample trajectories with conditions
        trajectories = gflownet.sample_trajectories(
            env,
            n=batch_size,
            save_logprobs=False,
            save_estimator_outputs=True,
            epsilon=epsilon,
        )

        # Training step
        optimizer.zero_grad()
        loss = gflownet.loss_from_trajectories(
            env, trajectories, recalculate_all_logprobs=False
        )
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{final_loss:.3e}",
                "trajectories": (it + 1) * batch_size,
            }
        )

        # Validation at regular intervals
        if (it + 1) % validation_interval == 0:
            # Test multiple condition values
            test_cond_values = [0.0, 0.25, 0.5, 0.75, 1.0]

            l1_dists = []
            logZ_diffs = []
            for cond_val in test_cond_values:
                # Set conditions for this validation
                conditions_val = torch.full(
                    (validation_samples, 1), cond_val, device=device
                )

                # Sample fresh trajectories for this conditions value
                # This follows the validate function's approach but with conditions support
                with torch.no_grad():
                    sampled_trajectories = gflownet.sample_trajectories(
                        env,
                        n=validation_samples,
                        conditions=conditions_val,
                        save_logprobs=False,
                        save_estimator_outputs=False,
                        epsilon=0.0,  # No exploration during validation
                    )
                    sampled_states = cast(
                        DiscreteStates, sampled_trajectories.terminating_states
                    )

                # Update discovered modes for condition=1
                if cond_val == 1.0:
                    # Conditions are already in sampled_states from trajectory sampling
                    # But we verify by computing with explicit conditions
                    states_for_reward = sampled_states.clone()
                    states_for_reward.conditions = torch.full(
                        (len(sampled_states), 1), cond_val, device=device
                    )
                    rewards = env.reward(states_for_reward)
                    modes = sampled_states[rewards >= mode_reward_threshold].tensor
                    modes_found = set([tuple(s.tolist()) for s in modes])
                    discovered_modes.update(modes_found)

                # Compute log partition function for this condition value
                if isinstance(gflownet, TBGFlowNet):
                    true_log_Z = env.log_partition(
                        torch.tensor([cond_val], device=device)
                    )
                    assert isinstance(gflownet.logZ, ScalarEstimator)
                    learned_log_Z = gflownet.logZ(
                        torch.tensor([cond_val], device=device)
                    ).item()
                    logZ_diffs.append(true_log_Z - learned_log_Z)

                # Compute empirical distribution using validate's helper function
                empirical_dist = env.get_terminating_state_dist(sampled_states)
                # Compute true distribution for this condition value
                true_conditional_dist = env.true_dist(
                    torch.tensor([cond_val], device=device)
                )
                # L1 distance as computed in validate function
                l1_dist = (empirical_dist - true_conditional_dist).abs().mean().item()
                l1_dists.append(l1_dist)

            # Print concise results
            log_str = f"[Iter {it + 1}]"
            log_str += f"\n\tNum modes discovered: {len(discovered_modes)}"
            if len(l1_dists) > 0:
                l1_dists_str = [f"{l1_dist:.6f}" for l1_dist in l1_dists]
                l1_dists_str = ", ".join(l1_dists_str)
                log_str += f"\n\tL1: [{l1_dists_str}]"
            if len(logZ_diffs) > 0:
                logZ_diffs_str = [f"{logZ_diff:.6f}" for logZ_diff in logZ_diffs]
                logZ_diffs_str = ", ".join(logZ_diffs_str)
                log_str += f"\n\tTrue logZ - Learned logZ: [{logZ_diffs_str}]"
            print(log_str)

    print("\n" + "=" * 60)
    print("+ Training complete!")
    print("=" * 60)

    return final_loss


def evaluate_conditional_sampling(env, gflownet, device, n_eval_samples=10000):
    """Evaluate the conditional sampling distributions with detailed metrics."""
    print("\n" + "=" * 60)
    print("FINAL EVALUATION OF CONDITIONAL SAMPLING")
    print("=" * 60)

    results = {}

    # Test a range of condition values
    test_cond_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for cond_value in test_cond_values:
        print(f"\n{'=' * 60}")
        print(f"Evaluating condition={cond_value}")
        print(f"{'=' * 60}")

        # Set fixed condition for evaluation
        conditions = torch.full(
            (n_eval_samples, 1), cond_value, dtype=torch.float, device=device
        )

        # Sample without exploration
        print(f"Sampling {n_eval_samples} trajectories with condition={cond_value}...")
        trajectories = gflownet.sample_trajectories(
            env,
            n=n_eval_samples,
            conditions=conditions,
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=0.0,  # No exploration
        )

        # Get terminal states
        term_states = cast(DiscreteStates, trajectories.terminating_states)

        empirical_dist = get_terminating_state_dist(env, term_states)
        true_dist = env.true_dist(torch.tensor([cond_value], device=device))

        if cond_value == 0:
            dist_type = "Uniform"
        elif cond_value == 1:
            dist_type = "Original HyperGrid"
        else:
            dist_type = f"Interpolated (α={cond_value:.2f})"

        # Compute L1 distance
        # L1 distance: mean of absolute differences (not sum) to match gfn.utils.training.validate
        l1_dist = torch.abs(empirical_dist - true_dist).mean().item()

        # Find top modes
        top_k = min(10, env.n_states)
        emp_topk_vals, _ = torch.topk(empirical_dist, top_k)
        true_topk_vals, _ = torch.topk(true_dist, top_k)

        print(f"\n{dist_type} Distribution:")
        print(f"  L1 Distance: {l1_dist:.6f}")
        print(
            f"  Top {min(5, top_k)} empirical probs: {[f'{p:.4f}' for p in emp_topk_vals[:5].tolist()]}"
        )
        print(
            f"  Top {min(5, top_k)} true probs:      {[f'{p:.4f}' for p in true_topk_vals[:5].tolist()]}"
        )

        # Additional metrics for each condition
        if cond_value == 0.0:
            # For uniform, check uniformity
            max_prob = empirical_dist.max().item()
            min_prob = empirical_dist.min().item()
            expected_uniform_prob = 1.0 / env.n_states
            uniformity_ratio = max_prob / expected_uniform_prob

            print("\nUniformity Metrics:")
            print(
                f"  Max probability: {max_prob:.6f} (expected: {expected_uniform_prob:.6f})"
            )
            print(
                f"  Min probability: {min_prob:.6f} (expected: {expected_uniform_prob:.6f})"
            )
            print(f"  Uniformity ratio (max/expected): {uniformity_ratio:.3f}")

            if uniformity_ratio > 1.5:
                print("  Warning: Distribution deviates from uniform")
            else:
                print("  Successfully learned uniform distribution")

        elif cond_value == 1.0:
            print("\nOriginal HyperGrid Distribution:")
            # Check concentration on high-probability states
            mode_prob_mass = empirical_dist[empirical_dist > 0.01].sum().item()
            print(
                f"  Probability mass on significant states (p>0.01): {mode_prob_mass:.4f}"
            )

            if l1_dist < 0.1:
                print("  Successfully learned multi-modal distribution")
            else:
                print(f"  L1 distance {l1_dist:.3f} > 0.1")

        else:  # Intermediate condition values
            # Show interpolation quality metrics
            print("\nInterpolation Metrics:")
            # Compute variance as a measure of spread
            emp_var = empirical_dist.var().item()
            true_var = true_dist.var().item()
            print(f"  Variance - empirical: {emp_var:.6f}, true: {true_var:.6f}")

            # Check KL divergence for quality of interpolation
            eps = 1e-10  # Small epsilon to avoid log(0)
            kl_div = (
                (
                    empirical_dist * (empirical_dist + eps).log()
                    - empirical_dist * (true_dist + eps).log()
                )
                .sum()
                .item()
            )
            print(f"  KL divergence: {kl_div:.6f}")

        results[cond_value] = {
            "l1_dist": l1_dist,
            "empirical_dist": empirical_dist,
            "true_dist": true_dist,
            "top_probs": emp_topk_vals[:5].tolist(),
        }

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for cond_val in test_cond_values:
        print(
            f"Condition={cond_val:.2f}: L1 distance = {results[cond_val]['l1_dist']:.6f}"
        )

    return results


GFN_FNS = {
    "tb": build_tb_gflownet,
    "db": build_db_gflownet,
    "db_mod": build_db_mod_gflownet,
    "subtb": build_subTB_gflownet,
    "fm": build_fm_gflownet,
}


def main(args):
    set_seed(args.seed if args.seed is not None else DEFAULT_SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Use ConditionalHyperGrid instead of regular HyperGrid
    environment = ConditionalHyperGrid(
        ndim=args.ndim,
        height=args.height,
        device=device,
        calculate_partition=True,  # Need this for validation
        store_all_states=True,  # Need this for validation
        debug=__debug__,
    )

    seed = int(args.seed) if args.seed is not None else DEFAULT_SEED
    n_iterations = args.n_iterations
    batch_size = args.batch_size
    validation_interval = args.validation_interval

    if args.gflownet == "all":
        print("Note: Evaluation will only be shown for the last trained GFlowNet")
        final_losses = []
        for fn_name, fn in GFN_FNS.items():
            print(f"\n{'=' * 50}")
            print(f"Training {fn_name.upper()} GFlowNet")
            print("=" * 50)
            gflownet = fn(environment)
            gflownet = gflownet.to(device)
            final_loss = train(
                environment,
                gflownet,
                seed,
                device,
                n_iterations,
                batch_size,
                validation_interval,
                args.validation_samples,
                args.lr,
                args.lr_logz,
                args.epsilon,
            )
            final_losses.append(final_loss)

        # Evaluate the last trained model
        if hasattr(args, "evaluate") and args.evaluate:
            evaluate_conditional_sampling(
                environment, gflownet, device, args.n_eval_samples
            )

        return sum(final_losses) / len(final_losses)  # Return average loss
    else:
        assert args.gflownet in GFN_FNS, "invalid gflownet name\n{}".format(GFN_FNS)
        gflownet = GFN_FNS[args.gflownet](environment)
        gflownet = gflownet.to(device)
        final_loss = train(
            environment,
            gflownet,
            seed,
            device,
            n_iterations,
            batch_size,
            validation_interval,
            args.validation_samples,
            args.lr,
            args.lr_logz,
            args.epsilon,
        )

        # Final evaluation
        if hasattr(args, "evaluate") and args.evaluate:
            evaluate_conditional_sampling(
                environment, gflownet, device, args.n_eval_samples
            )

        return final_loss


if __name__ == "__main__":
    parser = ArgumentParser()

    # Machine settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, if not set, then {} is used".format(DEFAULT_SEED),
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Prevent CUDA usage",
    )

    # Environment settings
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

    # Training settings
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=200,
        help="Interval for validation during training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_logz",
        type=float,
        default=1e-2,
        help="Learning rate for the logZ estimator neural network (for TB)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration parameter for the sampler",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=20000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    # GFlowNet settings
    parser.add_argument(
        "--gflownet",
        "-g",
        type=str,
        default="tb",
        help="Name of the gflownet. From {}".format(list(GFN_FNS.keys())),
    )

    # Evaluation settings
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to perform final evaluation of conditional distributions",
    )
    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=10000,
        help="Number of samples for final evaluation",
    )

    args = parser.parse_args()
    main(args)
