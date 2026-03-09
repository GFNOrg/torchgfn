"""Train a GFlowNet on the non-autoregressive BitSequence environment.

This example trains a Trajectory Balance GFlowNet on the non-autoregressive
BitSequence environment, where actions encode (position, word) pairs and
positions can be filled in any order.

The environment generates binary sequences and rewards those close (in Hamming
distance) to a set of target "mode" sequences. The non-autoregressive
formulation creates a richer DAG structure compared to the standard
autoregressive (left-to-right) version, since the same terminal state can
be reached via multiple orderings of position fills.

Usage:
    python tutorials/examples/train_bitsequence_non_autoregressive.py
    python tutorials/examples/train_bitsequence_non_autoregressive.py --seq_size 8 --word_size 2 --n_modes 4
"""

from argparse import ArgumentParser

import torch
from tqdm import tqdm

from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import PFBasedGFlowNet, TBGFlowNet
from gfn.gym import NonAutoregressiveBitSequence
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP

DEFAULT_SEED = 4444


def evaluate_l1(gflownet: PFBasedGFlowNet, env: NonAutoregressiveBitSequence) -> float:
    """Compute L1 distance between learned and true distributions.

    Only feasible for small environments (seq_size <= ~12).

    Args:
        gflownet: Trained GFlowNet.
        env: The environment.

    Returns:
        Mean absolute difference between estimated and true distributions.
    """
    sampler = Sampler(estimator=gflownet.pf)
    all_states = env.terminating_states
    n_states = all_states.tensor.shape[0]

    # Sample many trajectories and estimate the distribution
    n_samples = max(n_states * 100, 10000)
    with torch.no_grad():
        trajectories = sampler.sample_trajectories(
            env, n=n_samples, save_logprobs=False, save_estimator_outputs=False
        )

    # Count how often each terminal state is reached
    terminal_states = trajectories.terminating_states.tensor
    counts = torch.zeros(n_states, device=env.device)
    for i in range(n_states):
        matches = (terminal_states == all_states.tensor[i]).all(dim=-1)
        counts[i] = matches.sum()

    estimated_dist = counts / counts.sum()
    true_dist = env.true_dist()

    return torch.abs(estimated_dist - true_dist).mean().item()


def main(args):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Create the non-autoregressive BitSequence environment
    env = NonAutoregressiveBitSequence(
        word_size=args.word_size,
        seq_size=args.seq_size,
        n_modes=args.n_modes,
        reward_exponent=args.reward_exponent,
        device_str=str(device),
        seed=seed,
        debug=__debug__,
    )

    print("Environment: NonAutoregressiveBitSequence")
    print(f"  seq_size={args.seq_size}, word_size={args.word_size}")
    print(f"  words_per_seq={env.words_per_seq}, n_words={env.n_words}")
    print(f"  n_actions={env.n_actions} ({env.words_per_seq}*{env.n_words} + 1 exit)")
    print(f"  n_modes={args.n_modes}, reward_exponent={args.reward_exponent}")
    print(f"  n_terminating_states={env.n_terminating_states}")
    print(f"  device={device}")

    # Build policy networks with shared trunk
    pf = MLP(
        input_dim=env.words_per_seq,
        output_dim=env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_layers,
    )
    pb = MLP(
        input_dim=env.words_per_seq,
        output_dim=env.n_actions - 1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_layers,
        trunk=pf.trunk,
    )

    pf_estimator = DiscretePolicyEstimator(pf, n_actions=env.n_actions)
    pb_estimator = DiscretePolicyEstimator(pb, n_actions=env.n_actions, is_backward=True)

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0).to(device)

    # Optimizer with separate learning rates for network params and logZ
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_Z})

    n_params = sum(p.numel() for p in gflownet.parameters() if p.requires_grad)
    print(f"  n_params={n_params:,}")

    sampler = Sampler(estimator=pf_estimator)

    # Training loop
    for i in tqdm(range(args.n_iterations), desc="Training"):
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            save_estimator_outputs=False,
        )

        optimizer.zero_grad()
        loss = gflownet.loss_from_trajectories(
            env, trajectories, recalculate_all_logprobs=True
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 1.0)
        optimizer.step()

        if (i + 1) % args.print_every == 0:
            print(f"  Step {i+1}: loss={loss.item():.4f}")

    # Evaluate
    if env.n_terminating_states <= 4096:
        l1 = evaluate_l1(gflownet, env)
        print(f"\nL1 distance to true distribution: {l1:.6f}")
        return l1
    else:
        print("\nTraining complete (environment too large for exact L1 evaluation).")
        return None


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train GFlowNet on non-autoregressive BitSequence"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--n_iterations", type=int, default=2000, help="Training iterations"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lr_Z", type=float, default=1e-1, help="Learning rate for logZ"
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer size")
    parser.add_argument(
        "--n_layers", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument("--word_size", type=int, default=1, help="Bits per word")
    parser.add_argument("--seq_size", type=int, default=4, help="Total bits in sequence")
    parser.add_argument("--n_modes", type=int, default=2, help="Number of target modes")
    parser.add_argument(
        "--reward_exponent", type=float, default=2.0, help="Reward sharpness"
    )
    parser.add_argument(
        "--print_every", type=int, default=200, help="Print loss every N steps"
    )

    args = parser.parse_args()
    main(args)
