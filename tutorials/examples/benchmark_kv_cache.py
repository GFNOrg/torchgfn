#!/usr/bin/env python
r"""Instructional benchmark: KV-caching for autoregressive GFlowNet sampling.

Run me:

    python tutorials/examples/benchmark_kv_cache.py
    python tutorials/examples/benchmark_kv_cache.py --seq_sizes 16 32 64 128 --plot out.png

I sample the *same* transformer policy on ``BitSequence`` two ways and time both, so the
wall-clock difference is exactly the KV-cache. No GPU needed -- the win is a FLOP
reduction and shows up on CPU (and grows with sequence length).

================================================================================
WHY KV-CACHING HELPS
================================================================================
A GFlowNet policy over sequences is *autoregressive*: to choose token ``t`` it runs a
transformer over the tokens chosen so far and reads the logits at the last position.

Inside each attention layer, the query at position ``t`` attends to the keys/values
(K/V) of every earlier position ``0..t``. Crucially, the K/V of positions ``0..t-1``
are a function of tokens that are *already fixed* -- they do not change as the sequence
grows. Yet naive sampling recomputes the whole prefix from scratch at every step:

    step t=3, NO cache:  embed+attend [w0 w1 w2]  <- recompute K/V for w0,w1 again
    step t=3, cache:     embed+attend [      w2]  <- compute K/V for w2 only,
                                                     reuse cached K/V for w0,w1

So a length-``L`` rollout without a cache re-embeds an ever-growing prefix each step:
roughly ``sum_t O(t^2) = O(L^3)`` attention FLOPs. A cache stores each position's K/V
the first time it is computed and reuses it, so step ``t`` only computes the new
token's Q/K/V and attends it against the ``t`` cached keys: ``sum_t O(t) = O(L^2)``.
That is a ~``L``x speedup that *widens with sequence length* -- the effect this script
makes visible.

================================================================================
HOW torchgfn IMPLEMENTS IT
================================================================================
The seam is the *recurrent carry* threaded through sampling by ``Sampler``:

    RecurrentDiscretePolicyEstimator.forward(states, carry) -> (logits, carry)

The carry is a ``dict[str, Tensor]`` holding, per transformer layer, the running K/V
cache, plus a scalar write cursor and the absolute position. Two forward paths (both
build the same ``[BOS, w0, ..., w_{t-1}]`` token stream and are numerically identical
to one full-sequence forward):

  * ``use_kv_cache=False`` (default): re-embed the full committed prefix with a fresh
    carry each step. Correct and grad-bearing, but O(L^3) -- the honest baseline.

  * ``use_kv_cache=True``: feed only the newest token; ``TransformerDiscreteSequenceModel``
    writes its K/V *in place* into a preallocated ``[B, n_heads, max_len, head_dim]``
    buffer at the cursor (no ``torch.cat``, no realloc) and attends against the filled
    prefix. This is the O(L^2) fast path.

The autograd catch: writing K/V in place mutates a buffer whose earlier slice was saved
for a previous step's backward, which invalidates the autograd graph. So the cached path
runs under ``torch.no_grad()`` -- which is fine, because *sampling only needs sampled
actions, not gradients*. This makes ``use_kv_cache=True`` ideal for inference/rollout;
to *train* with it, recompute the loss log-probs with a grad-bearing forward
(``recalculate_all_logprobs=True``). A companion "teacher-forced" parallel recompute
makes that efficient and turns this sampling win into an end-to-end training win.

================================================================================
WHAT THIS SCRIPT MEASURES
================================================================================
For each sequence length it (1) asserts the two configs produce identical per-step
logits (``|Δ logits| ~ 1e-7`` -- proof it is the same model), then times pure rollout
under ``no_grad`` for each -- isolating the cache (O(L^3) vs O(L^2)).
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch

from gfn.estimators import RecurrentDiscretePolicyEstimator
from gfn.gflownet import TBGFlowNet
from gfn.gym.bitSequence import BitSequence
from gfn.utils.common import set_seed
from gfn.utils.modules import TransformerDiscreteSequenceModel


def _sync(device: torch.device) -> None:
    """Block until queued device work finishes so timings are accurate.

    GPU/MPS kernels launch asynchronously, so ``perf_counter`` would otherwise time
    only the Python dispatch, not the compute. A no-op on CPU.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build(args, seq_size: int, use_kv_cache: bool, device: torch.device):
    """Construct ``(env, gflownet)`` for one benchmark configuration.

    This is the only place the KV-cache API appears -- everything else (sampler, loss,
    optimizer) is the ordinary torchgfn stack, unchanged.

    Args:
        args: Parsed CLI arguments (model/env hyperparameters).
        seq_size: BitSequence length; with ``word_size=1`` this equals the number of
            autoregressive decode steps per trajectory (the ``L`` in the docstring).
        use_kv_cache: Selects the sampling path (see module docstring).
        device: Torch device.

    Returns:
        The environment and a ``TBGFlowNet`` wrapping a transformer forward policy.
    """
    env = BitSequence(
        word_size=args.word_size,
        seq_size=seq_size,
        n_modes=args.n_modes,
        device_str=str(device),
        seed=args.seed,
        debug=False,
    )
    # The KV-cache is preallocated to the model's position range, so it must cover the
    # whole rollout: one slot per word plus the leading BOS token (+ a little headroom).
    max_positions = env.words_per_seq + 2
    model = TransformerDiscreteSequenceModel(
        vocab_size=env.n_actions,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        num_layers=args.num_layers,
        max_position_embeddings=max_positions,
        dropout=0.0,
    ).to(device)
    est = RecurrentDiscretePolicyEstimator(
        module=model,
        n_actions=env.n_actions,
        is_backward=False,
        # The single switch that turns on the in-place, feed-newest-token fast path.
        use_kv_cache=use_kv_cache,
    )
    # Tree-structured DAG (each string has a unique parent) => backward policy is
    # constant, hence pb=None + constant_pb=True. Only the forward policy is timed.
    gflownet = TBGFlowNet(pf=est, pb=None, init_logZ=0.0, constant_pb=True).to(device)
    return env, gflownet


def _median_ms(fn: Callable[[], None], device: torch.device, warmup: int, iters: int):
    """Return the median (min, max) wall-clock of ``fn`` in milliseconds.

    Warm-up iterations are discarded so one-time costs (lazy allocations, the first
    preallocation of the cache buffers) do not pollute the measurement.
    """
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    lo, hi = min(samples), max(samples)
    return statistics.median(samples), lo, hi


def time_sampling(gflownet, env, batch: int, device, warmup, iters):
    """Time pure rollout under ``no_grad`` -- the operation the cache directly speeds up.

    Both configs run under ``no_grad`` here, so this isolates the cache: identical model,
    identical outputs, the only difference is O(L^3) re-forward vs O(L^2) cached decode.
    """

    def _step():
        with torch.no_grad():
            gflownet.sample_trajectories(
                env, n=batch, save_logprobs=False, save_estimator_outputs=False
            )

    return _median_ms(_step, device, warmup, iters)


def check_equivalence(args, seq_size: int, device: torch.device) -> float:
    """Prove the cache changes speed, not results, at the per-step logit level.

    We reconstruct the states at every decode step of a fixed batch and run the
    estimator over them both ways (``use_kv_cache`` False and True). Because the cached
    incremental decode equals the full-prefix forward, the logits must agree to
    floating-point tolerance; a large gap would mean the cache computes the wrong thing.
    Returns the max absolute logit difference.
    """
    set_seed(args.seed)
    env, gflownet = build(args, seq_size, use_kv_cache=False, device=device)
    # Both estimators wrap the SAME module weights, so the only difference is the path.
    model = gflownet.pf.module
    est_off = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, use_kv_cache=False
    )
    est_on = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, use_kv_cache=True
    )

    batch, L = args.batch_size, env.words_per_seq
    words = torch.randint(0, env.n_actions - 1, (batch, L), device=device)

    def per_step_logits(est) -> torch.Tensor:
        carry = est.init_carry(batch, device)
        out = []
        with torch.no_grad():
            for t in range(L + 1):  # steps 0..L (L+1 decode steps)
                tensor = words.clone()
                tensor[:, t:] = -1  # only first t words committed
                logits, carry = est(env.States(tensor), carry)
                out.append(logits)
        return torch.stack(out, dim=0)

    return (per_step_logits(est_off) - per_step_logits(est_on)).abs().max().item()


def main(args) -> None:
    device = torch.device(args.device)
    print(
        f"device={device}  batch={args.batch_size}  "
        f"model=(dim={args.embedding_dim}, heads={args.num_heads}, "
        f"layers={args.num_layers})  word_size={args.word_size}\n"
    )

    # Columns: sampling base/cache + speedup, and the equivalence residual proving both
    # columns describe the same model.
    header = (
        f"{'seq_len':>7} | {'sample base':>12} {'sample cache':>13} "
        f"{'speedup':>8} | {'|Δlogits|':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for seq_size in args.seq_sizes:
        set_seed(args.seed)

        # Correctness gate first: never report a speedup for a model that changed.
        max_delta = check_equivalence(args, seq_size, device)
        assert (
            max_delta < 1e-4
        ), f"baseline vs cached logits differ by {max_delta} at seq_size={seq_size}"

        # Fresh models per config so neither warms the other's caches/allocator.
        env_b, gfn_b = build(args, seq_size, use_kv_cache=False, device=device)
        env_c, gfn_c = build(args, seq_size, use_kv_cache=True, device=device)

        s_base = time_sampling(
            gfn_b, env_b, args.batch_size, device, args.warmup, args.iters
        )
        s_cache = time_sampling(
            gfn_c, env_c, args.batch_size, device, args.warmup, args.iters
        )
        speedup = s_base[0] / s_cache[0]
        rows.append((seq_size, s_base[0], s_cache[0], speedup))
        print(
            f"{seq_size:>7} | {s_base[0]:>10.2f}ms {s_cache[0]:>11.2f}ms "
            f"{speedup:>7.1f}x | {max_delta:>10.1e}"
        )

    print(
        "\nBoth configs are numerically identical (see |Δlogits|); the speedup is the "
        "KV-cache.\nIt isolates the sampling win (O(L^3)->O(L^2)) and widens with "
        "seq_len -- that is the point.\nTo turn this into an end-to-end training "
        "speedup, pair it with the teacher-forced loss recompute."
    )

    if args.plot:
        _save_plot(rows, args.plot)


def _save_plot(rows, path: str) -> None:
    """Optionally save a sampling-time / speedup figure (requires matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping --plot")
        return

    seq = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(seq, [r[1] for r in rows], "o-", label="baseline (no cache)")
    ax1.plot(seq, [r[2] for r in rows], "s-", label="KV-cache")
    ax1.set(
        xlabel="sequence length (steps)",
        ylabel="ms / rollout",
        title="Pure sampling time",
    )
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(seq, [r[3] for r in rows], "o-")
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax2.set(
        xlabel="sequence length (steps)",
        ylabel="speedup (x)",
        title="KV-cache sampling speedup",
    )
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved plot to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark KV-cached vs baseline autoregressive GFlowNet sampling.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    parser.add_argument(
        "--seq_sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="BitSequence lengths to sweep (== rollout lengths with word_size=1)",
    )
    parser.add_argument(
        "--word_size",
        type=int,
        default=1,
        help="Bits per action; 1 => one decode step per sequence bit",
    )
    parser.add_argument("--n_modes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional path to save a speedup figure (PNG)",
    )
    main(parser.parse_args())
