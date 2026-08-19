#!/usr/bin/env python
r"""Instructional benchmark: KV-caching for autoregressive GFlowNet sampling.

Run me:

    python tutorials/examples/benchmark_kv_cache.py
    python tutorials/examples/benchmark_kv_cache.py --seq_sizes 16 32 64 128 --plot out.png

I train/sample the *same* transformer policy on ``BitSequence`` two ways and time
both, so the wall-clock difference is exactly the KV-cache. No GPU needed -- the win
is a FLOP reduction and shows up on CPU (and grows with sequence length).

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

The autograd catch (and why the design is split): writing K/V in place mutates a buffer
whose earlier slice was saved for a previous step's backward, which invalidates the
autograd graph. So the cached path runs under ``torch.no_grad()`` -- which is fine,
because *sampling only needs sampled actions, not gradients*. Gradients are instead
produced at loss time by a single **teacher-forced** (parallel, causal-masked) forward
over each committed trajectory. ``use_kv_cache=True`` auto-enables this recompute
(``teacher_forced_loss``); you just request it with
``loss(..., recalculate_all_logprobs=True)``. That recompute is grad-safe and itself
cheaper than back-propagating through an ``L``-step sequential rollout.

    sampling (no_grad, cached, O(L^2))   +   loss (teacher-forced, grad, one forward)

================================================================================
WHAT THIS SCRIPT MEASURES
================================================================================
For each sequence length it (1) asserts both optimizations are numerically inert --
``|Δcache|`` compares the cached decode against the full-prefix forward, ``|Δtf|``
compares the teacher-forced recompute against the per-step one (both ~1e-7, proof it
is the same model) -- then times:
  * **pure sampling** under ``no_grad`` -- isolates the cache (O(L^3) vs O(L^2));
  * **full training step** (sample + loss + backward + optimizer) -- the realistic
    cost, which also folds in the teacher-forced loss on the cached side.
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
    optimizer) is the ordinary torchgfn training stack, unchanged.

    Args:
        args: Parsed CLI arguments (model/env hyperparameters).
        seq_size: BitSequence length; with ``word_size=1`` this equals the number of
            autoregressive decode steps per trajectory (the ``L`` in the docstring).
        use_kv_cache: Selects the sampling path (see module docstring). It also
            auto-enables the teacher-forced loss recompute, since cached sampling is
            ``no_grad`` and gradients must come from that recompute.
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
        # The single switch: turns on the in-place, feed-newest-token fast path and,
        # because cached sampling is no_grad, auto-enables the teacher-forced loss
        # recompute so the policy stays trainable.
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


def time_training_step(
    gflownet, env, batch: int, use_kv_cache: bool, device, warmup, iters
):
    """Time a full training step: sample + loss + backward + optimizer step.

    The two realistic end-to-end configurations differ in *how gradients are obtained*:

      * baseline: grad-bearing sequential sampling saves per-step log-probs, and the
        loss reuses them (``save_logprobs=True``, ``recalculate_all_logprobs=False``);
      * cached: ``no_grad`` cached sampling saves nothing differentiable, so the loss
        recomputes log-probs with the teacher-forced parallel forward
        (``recalculate_all_logprobs=True``; teacher forcing is auto-enabled by
        ``use_kv_cache``).
    """
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=1e-3)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": 1e-1})

    def _step():
        trajectories = gflownet.sample_trajectories(
            env,
            n=batch,
            # Cached sampling is no_grad, so saving its log-probs would be pointless
            # (they carry no gradient); recompute them in the loss instead.
            save_logprobs=not use_kv_cache,
            save_estimator_outputs=False,
        )
        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=use_kv_cache)
        loss.backward()
        optimizer.step()

    return _median_ms(_step, device, warmup, iters)


def check_equivalence(args, seq_size: int, device: torch.device) -> tuple[float, float]:
    """Prove that neither optimization changes results -- only speed.

    There are *two* independent things to gate, and each needs its own comparison,
    because the two columns of the table are produced by different code paths:

    1. ``|Δcache|`` -- the cached incremental decode (``use_kv_cache=True``) against
       the full-prefix forward, at the per-step logit level. This is the only check
       that actually executes ``_forward_cached``, so it is what would catch a broken
       cache. Comparing two *recompute* paths would not: both run with the cache off.
    2. ``|Δtf|`` -- the teacher-forced parallel recompute against the per-step
       recompute, at the ``log P_F`` level. This gates the loss-time path that the
       ``train cache`` column depends on.

    Every estimator below wraps the SAME module weights, so any gap is a bug in a
    path, not a difference in models.

    Returns:
        ``(max |Δ logits| cached-vs-baseline, max |Δ log_pf| teacher-forced-vs-per-step)``
    """
    from gfn.utils.prob_calculations import get_trajectory_pfs

    set_seed(args.seed)
    env, gflownet = build(args, seq_size, use_kv_cache=False, device=device)
    model = gflownet.pf.module

    # (1) Cached decode vs full-prefix forward, step by step over a fixed batch.
    est_off = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, use_kv_cache=False
    )
    est_on = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, use_kv_cache=True
    )
    batch, n_words = args.batch_size, env.words_per_seq
    words = torch.randint(0, env.n_actions - 1, (batch, n_words), device=device)

    def per_step_logits(est) -> torch.Tensor:
        carry = est.init_carry(batch, device)
        out = []
        with torch.no_grad():
            for t in range(n_words + 1):  # steps 0..L (L+1 decode steps)
                tensor = words.clone()
                tensor[:, t:] = -1  # only the first t words committed
                logits, carry = est(env.States(tensor), carry)
                out.append(logits)
        return torch.stack(out, dim=0)

    delta_cache = (per_step_logits(est_off) - per_step_logits(est_on)).abs().max().item()

    # (2) Teacher-forced vs per-step recompute over a sampled batch.
    trajectories = gflownet.sample_trajectories(
        env, n=args.batch_size, save_logprobs=False, save_estimator_outputs=False
    )
    est_tf = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, teacher_forced_loss=True
    )
    log_pf_per_step = get_trajectory_pfs(
        est_off, trajectories, recalculate_all_logprobs=True
    )
    log_pf_tf = get_trajectory_pfs(est_tf, trajectories, recalculate_all_logprobs=True)
    delta_tf = (log_pf_per_step - log_pf_tf).abs().max().item()

    return delta_cache, delta_tf


def main(args) -> None:
    device = torch.device(args.device)
    torch.set_grad_enabled(True)
    print(
        f"device={device}  batch={args.batch_size}  "
        f"model=(dim={args.embedding_dim}, heads={args.num_heads}, "
        f"layers={args.num_layers})  word_size={args.word_size}\n"
    )

    # Columns: sampling base/cache + speedup, training base/cache + speedup, and the
    # two equivalence residuals -- one per optimization -- proving that every column
    # describes the same model (see check_equivalence).
    header = (
        f"{'seq_len':>7} | {'sample base':>12} {'sample cache':>13} {'x':>5} | "
        f"{'train base':>11} {'train cache':>12} {'x':>5} | "
        f"{'|Δcache|':>9} {'|Δtf|':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for seq_size in args.seq_sizes:
        set_seed(args.seed)

        # Correctness gate first: never report a speedup for a model that changed.
        # Each optimization is gated separately, so a failure names the culprit.
        delta_cache, delta_tf = check_equivalence(args, seq_size, device)
        assert delta_cache < 1e-4, (
            f"cached vs full-prefix logits differ by {delta_cache} "
            f"at seq_size={seq_size}"
        )
        assert delta_tf < 1e-4, (
            f"teacher-forced vs per-step log_pf differ by {delta_tf} "
            f"at seq_size={seq_size}"
        )

        # Fresh models per config so neither warms the other's caches/allocator.
        env_b, gfn_b = build(args, seq_size, use_kv_cache=False, device=device)
        env_c, gfn_c = build(args, seq_size, use_kv_cache=True, device=device)

        s_base = time_sampling(
            gfn_b, env_b, args.batch_size, device, args.warmup, args.iters
        )
        s_cache = time_sampling(
            gfn_c, env_c, args.batch_size, device, args.warmup, args.iters
        )
        t_base = time_training_step(
            gfn_b, env_b, args.batch_size, False, device, args.warmup, args.iters
        )
        t_cache = time_training_step(
            gfn_c, env_c, args.batch_size, True, device, args.warmup, args.iters
        )

        sample_speedup = s_base[0] / s_cache[0]
        train_speedup = t_base[0] / t_cache[0]
        rows.append(
            (
                seq_size,
                s_base[0],
                s_cache[0],
                sample_speedup,
                t_base[0],
                t_cache[0],
                train_speedup,
            )
        )
        print(
            f"{seq_size:>7} | {s_base[0]:>10.2f}ms {s_cache[0]:>11.2f}ms "
            f"{sample_speedup:>4.1f}x | {t_base[0]:>9.2f}ms {t_cache[0]:>10.2f}ms "
            f"{train_speedup:>4.1f}x | {delta_cache:>9.1e} {delta_tf:>9.1e}"
        )

    print(
        "\nBoth configs are numerically identical (|Δcache| gates the cached decode, "
        "|Δtf| the\nteacher-forced recompute); the speedup is the optimization, not a "
        "different model.\nSampling isolates the cache (O(L^3)->O(L^2)); training-step "
        "folds in the teacher-forced\nloss. The gap widens with seq_len -- that is the "
        "point."
    )

    if args.plot:
        _save_plot(rows, args.plot)


def _save_plot(rows, path: str) -> None:
    """Optionally save a speedup-vs-length figure (requires matplotlib)."""
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
    ax2.plot(seq, [r[3] for r in rows], "o-", label="sampling")
    ax2.plot(seq, [r[6] for r in rows], "s-", label="training step")
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax2.set(
        xlabel="sequence length (steps)", ylabel="speedup (x)", title="KV-cache speedup"
    )
    ax2.legend()
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
