"""
Micro-benchmark for SubTBGFlowNet.get_scores vectorized vs original loop.

This isolates get_scores by monkeypatching dependencies (calculate_preds/targets,
get_pfs_and_pbs, masks) to synthetic tensors so we can time the core logic.
Run on CPU; adjust sizes below to probe different max_len / batch regimes.
"""

from __future__ import annotations

import argparse
from types import MethodType
from typing import Any, Callable, Tuple

import torch
from torch.utils import benchmark

from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet


class _DummyTrajectories:
    """Minimal trajectories carrier for benchmarking get_scores."""

    def __init__(self, terminating_idx: torch.Tensor, max_length: int):
        self.terminating_idx = terminating_idx
        self.max_length = max_length
        self.n_trajectories = terminating_idx.shape[0]

    def __len__(self) -> int:
        return self.n_trajectories


def build_model_and_data(
    max_len: int, n_traj: int, seed: int = 0, device: str | torch.device | None = None
) -> Tuple[SubTBGFlowNet, _DummyTrajectories, list[torch.Tensor], list[torch.Tensor]]:
    torch.manual_seed(seed)
    device = torch.device(device) if device is not None else torch.device("cpu")
    terminating_idx = torch.randint(1, max_len + 1, (n_traj,), device=device)
    # In the real pipeline, trajectories carry log_rewards computed from the env.
    # The vectorized get_scores now asserts on its presence, so seed a dummy tensor here.
    log_rewards = torch.randn(n_traj, device=device)
    log_pf_trajectories = torch.randn(max_len, n_traj, device=device)
    log_pb_trajectories = torch.randn(max_len, n_traj, device=device)
    log_state_flows = torch.randn(max_len, n_traj, device=device)
    sink_states_mask = torch.zeros(max_len, n_traj, dtype=torch.bool, device=device)
    is_terminal_mask = torch.zeros(max_len, n_traj, dtype=torch.bool, device=device)

    preds_list = [
        torch.randn(max_len + 1 - i, n_traj, device=device)
        for i in range(1, max_len + 1)
    ]
    targets_list = [
        torch.randn(max_len + 1 - i, n_traj, device=device)
        for i in range(1, max_len + 1)
    ]

    trajectories = _DummyTrajectories(
        terminating_idx=terminating_idx, max_length=max_len
    )

    # Build a SubTBGFlowNet instance without running heavy __init__.
    model = SubTBGFlowNet.__new__(SubTBGFlowNet)
    torch.nn.Module.__init__(model)
    model.debug = False
    model.log_reward_clip_min = float("-inf")

    # Monkeypatch the dependencies used inside get_scores to deterministic tensors.
    model.get_pfs_and_pbs = MethodType(
        lambda self, traj, recalculate_all_logprobs=True: (
            log_pf_trajectories,
            log_pb_trajectories,
        ),
        model,
    )
    model.calculate_log_state_flows = MethodType(
        lambda self, _env, _traj, _log_pf: log_state_flows, model
    )
    model.calculate_masks = MethodType(
        lambda self, _log_state_flows, _traj: (sink_states_mask, is_terminal_mask),
        model,
    )
    # Attach log_rewards to the dummy trajectories to mirror real trajectories objects.
    trajectories.log_rewards = log_rewards
    model.calculate_preds = MethodType(
        lambda self, _log_pf_cum, _log_state_flows, i: preds_list[i - 1], model
    )
    model.calculate_targets = MethodType(
        lambda self, _traj, _preds, _log_pb_cum, _log_state_flows, _term_mask, _sink_mask, i: targets_list[
            i - 1
        ],
        model,
    )

    return model, trajectories, preds_list, targets_list


def original_get_scores(
    model: SubTBGFlowNet, env, trajectories
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Reference implementation (pre-vectorized) for comparison."""
    log_pf_trajectories_, log_pb_trajectories_ = model.get_pfs_and_pbs(
        trajectories, recalculate_all_logprobs=True
    )

    log_pf_trajectories_cum = model.cumulative_logprobs(
        trajectories, log_pf_trajectories_
    )
    log_pb_trajectories_cum = model.cumulative_logprobs(
        trajectories, log_pb_trajectories_
    )

    log_state_flows_ = model.calculate_log_state_flows(
        env, trajectories, log_pf_trajectories_
    )
    sink_states_mask_, is_terminal_mask_ = model.calculate_masks(
        log_state_flows_, trajectories
    )

    flattening_masks_orig = []
    scores_orig = []
    for i in range(1, 1 + trajectories.max_length):
        preds = model.calculate_preds(log_pf_trajectories_cum, log_state_flows_, i)
        targets = model.calculate_targets(
            trajectories,
            preds,
            log_pb_trajectories_cum,
            log_state_flows_,
            is_terminal_mask_,
            sink_states_mask_,
            i,
        )

        flattening_mask = trajectories.terminating_idx.lt(
            torch.arange(
                i,
                trajectories.max_length + 1,
                device=trajectories.terminating_idx.device,
            ).unsqueeze(-1)
        )

        flat_preds = preds[~flattening_mask]
        if model.debug and torch.any(torch.isnan(flat_preds)):
            raise ValueError("NaN in preds")

        flat_targets = targets[~flattening_mask]
        if model.debug and torch.any(torch.isnan(flat_targets)):
            raise ValueError("NaN in targets")

        flattening_masks_orig.append(flattening_mask)
        scores_orig.append(preds - targets)

    return scores_orig, flattening_masks_orig


def run_once(
    mode: str,
    max_len: int,
    n_traj: int,
    use_compile: bool = False,
    device: str | torch.device = "cpu",
) -> float:
    """Return median time (seconds) for the chosen mode. Optionally uses torch.compile and device selection."""
    model, trajectories, _, _ = build_model_and_data(max_len, n_traj, device=device)
    env_obj: Any = object()
    bench: Callable[[], Any]
    compiled_get_scores: Callable | None = None

    if mode == "original":

        def bench_original():
            return original_get_scores(model, env_obj, trajectories)  # type: ignore[arg-type]

        bench = bench_original
    elif mode == "vectorized":
        if use_compile:
            # Compile only after monkeypatching, so we capture the correct bound method.
            compiled_get_scores = torch.compile(
                model.get_scores, fullgraph=False, dynamic=False, mode="reduce-overhead"
            )

        def bench_vectorized():
            fn = (
                compiled_get_scores
                if compiled_get_scores is not None
                else model.get_scores
            )
            return fn(env_obj, trajectories)  # type: ignore[arg-type]

        bench = bench_vectorized
    else:
        raise ValueError(mode)

    t = benchmark.Timer(
        stmt="bench()",
        globals={"bench": bench},
        setup="",
        num_threads=torch.get_num_threads(),
    ).blocked_autorange(min_run_time=0.5)
    return t.median


def main():
    parser = argparse.ArgumentParser()
    # Defaults scaled ~10x to stress larger workloads; override with --sizes if needed.
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["80x640", "160x1280", "320x2560"],
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile on the vectorized get_scores.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (e.g., cpu, mps, cuda).",
    )
    args = parser.parse_args()

    print("Benchmarking SubTBGFlowNet.get_scores (CPU)")
    print(f"torch version: {torch.__version__}")
    print(f"num threads: {torch.get_num_threads()}")
    print()
    print(f"{'size':>10}  {'orig (ms)':>12}  {'vec (ms)':>12}  {'speedup':>8}")

    for size in args.sizes:
        max_len_s, n_traj_s = size.lower().split("x")
        max_len = int(max_len_s)
        n_traj = int(n_traj_s)
        t_orig = run_once("original", max_len, n_traj, device=args.device) * 1e3
        t_vec = (
            run_once(
                "vectorized",
                max_len,
                n_traj,
                use_compile=args.compile,
                device=args.device,
            )
            * 1e3
        )
        speedup = t_orig / t_vec if t_vec > 0 else float("inf")
        print(f"{size:>10}  {t_orig:12.3f}  {t_vec:12.3f}  {speedup:8.2f}x")


if __name__ == "__main__":
    main()
