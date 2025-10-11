#!/usr/bin/env python
"""
Minimal TB training on BitSequence with a recurrent policy.

Key choices:
- RecurrentDiscretePolicyEstimator + RecurrentDiscreteSequenceModel
- Sampler uses RecurrentEstimatorAdapter (saves on-policy log-probs)
- TBGFlowNet with constant_pb=True (tree DAG), pb=None

This is intentionally small and mirrors train_hypergrid_simple.py structure.
"""

import argparse
from typing import cast

import torch
from tqdm import tqdm

from gfn.estimators import RecurrentDiscretePolicyEstimator
from gfn.gflownet import PFBasedGFlowNet, TBGFlowNet
from gfn.gym.bitSequence import BitSequence
from gfn.samplers import RecurrentEstimatorAdapter
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import RecurrentDiscreteSequenceModel
from gfn.utils.prob_calculations import get_trajectory_pfs


def estimated_dist(gflownet: PFBasedGFlowNet, env: BitSequence):
    states = env.terminating_states
    trajectories = env.trajectory_from_terminating_states(states.tensor)
    log_pf_trajectories = get_trajectory_pfs(
        pf=gflownet.pf,
        trajectories=trajectories,
        recalculate_all_logprobs=True,
        adapter=gflownet.pf_adapter,
    )
    pf = torch.exp(log_pf_trajectories.sum(dim=0))

    l1_dist = torch.abs(pf - env.true_dist).mean().item()

    return l1_dist


def main(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Environment
    H = torch.randint(
        0, 2, (args.n_modes, args.seq_size), dtype=torch.long, device=device
    )
    env = BitSequence(
        word_size=args.word_size,
        seq_size=args.seq_size,
        n_modes=args.n_modes,
        temperature=args.temperature,
        H=H,
        device_str=str(device),
        seed=args.seed,
        check_action_validity=__debug__,
    )

    # Model + Estimator
    # Set vocab_size so projection outputs env.n_actions logits (includes exit).
    model = RecurrentDiscreteSequenceModel(
        vocab_size=env.n_actions,  # projection -> env.n_actions
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    ).to(device)

    pf_estimator = RecurrentDiscretePolicyEstimator(
        module=model,
        n_actions=env.n_actions,
        is_backward=False,
    ).to(device)

    # GFlowNet (Trajectory Balance), tree DAG -> pb=None, constant_pb=True,
    # Use a recurrent adapter for the PF.
    gflownet = TBGFlowNet(
        pf=pf_estimator,
        pb=None,
        init_logZ=0.0,
        constant_pb=True,
        pf_adapter=RecurrentEstimatorAdapter(pf_estimator),
    )
    gflownet = gflownet.to(device)

    # Optimizer: policy params + logZ
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    visited_terminating_states = env.states_from_batch_shape((0,))
    l1_distances = []
    eval_freq = args.n_iterations // 10  # 10% of the iterations.
    l1_dist = float("inf")

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,  # crucial: avoid recalculation, use adapter path
            save_estimator_outputs=False,
            epsilon=args.epsilon,  # Off-policy sampling.
        )

        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        optimizer.zero_grad()
        # Use saved log-probs from sampler; no need to recalc
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        loss.backward()

        gflownet.assert_finite_gradients()
        torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 1.0)
        optimizer.step()
        gflownet.assert_finite_parameters()

        if (it + 1) % eval_freq == 0 or it == 0:
            l1_dist = estimated_dist(gflownet, env)
            l1_distances.append(l1_dist)

        pbar.set_postfix({"loss": loss.item(), "l1_dist": l1_dist})

    # Final validation.
    l1_dist = estimated_dist(gflownet, env)
    l1_distances.append(l1_dist)
    print(f"L1_dist training curve: {[f'{l1:.5f}' for l1 in l1_distances]}")
    print(f"Final L1_dist: {l1_dist:.5f}")

    return l1_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA use")

    # BitSequence config (keep small by default)
    parser.add_argument("--word_size", type=int, default=3, help="Word size")
    parser.add_argument("--seq_size", type=int, default=9, help="Sequence size")
    parser.add_argument("--n_modes", type=int, default=5, help="Number of modes")
    parser.add_argument("--temperature", type=float, default=1.0)

    # Model config
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--rnn_type", type=str, choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training config
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_logz", type=float, default=1e-1)
    parser.add_argument("--n_iterations", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.05)

    args = parser.parse_args()
    main(args)
