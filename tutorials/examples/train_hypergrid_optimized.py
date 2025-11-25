#!/usr/bin/env python
r"""
Optimized HyperGrid training script with optional torch.compile, vmap, and benchmarking.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, cast

import torch
from torch.func import vmap
from tqdm import tqdm

from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet.detailed_balance import DBGFlowNet
from gfn.gflownet.flow_matching import FMGFlowNet
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.compile import try_compile_gflownet
from gfn.utils.modules import MLP, DiscreteUniform
from gfn.utils.training import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["FM", "TB", "DB"], default="TB")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--R0", type=float, default=0.1)
    parser.add_argument("--R1", type=float, default=0.5)
    parser.add_argument("--R2", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_logz", type=float, default=1e-1)
    parser.add_argument("--uniform_pb", action="store_true")
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--validation_samples", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device to run on; auto prefers CUDA>MPS>CPU.",
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="Mode passed to torch.compile.",
    )
    parser.add_argument("--use-vmap", action="store_true", help="Use vmap TB loss.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode.")
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="hypergrid_benchmark.png",
        help="Output path for benchmark plot.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=50,
        help="Warmup iterations before timing (benchmark mode).",
    )
    return parser.parse_args()


def init_metrics() -> Dict[str, Any]:
    return {
        "validation_info": {"l1_dist": float("inf")},
        "discovered_modes": set(),
        "total_steps": 0,
        "measured_steps": 0,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.benchmark:
        scenarios = [
            ("Baseline", False, False),
            (f"Compile ({args.compile_mode})", True, False),
            ("Vmap", False, True),
            (f"Compile+Vmap ({args.compile_mode})", True, True),
        ]
        results: list[dict[str, Any]] = []
        for label, enable_compile, use_vmap in scenarios:
            result = train_with_options(
                args,
                device,
                enable_compile=enable_compile,
                use_vmap=use_vmap,
                warmup_iters=args.warmup_iters,
                quiet=True,
                timing=True,
                record_history=True,
            )
            result["label"] = label
            results.append(result)

        baseline_elapsed = results[0]["elapsed"]
        print("Benchmark summary (speedups vs baseline):")
        for result in results:
            speedup = (
                baseline_elapsed / result["elapsed"]
                if result["elapsed"]
                else float("inf")
            )
            print(
                f"- {result['label']}: {result['elapsed']:.2f}s "
                f"({speedup:.2f}x) | compile_mode={result['compile_mode']} "
                f"| vmap={'on' if result['effective_vmap'] else 'off'}"
            )

        plot_benchmark(results, args.benchmark_output)
        return

    train_with_options(
        args,
        device,
        enable_compile=args.compile,
        use_vmap=args.use_vmap,
        warmup_iters=0,
        quiet=False,
        timing=False,
        record_history=False,
    )


def train_with_options(
    args: argparse.Namespace,
    device: torch.device,
    *,
    enable_compile: bool,
    use_vmap: bool,
    warmup_iters: int,
    quiet: bool,
    timing: bool,
    record_history: bool,
) -> dict[str, Any]:
    set_seed(args.seed)
    (
        env,
        gflownet,
        sampler,
        optimizer,
        visited_states,
    ) = build_training_components(args, device)
    metrics = init_metrics()

    compile_mode = args.compile_mode if enable_compile else "none"
    if enable_compile:
        compile_results = try_compile_gflownet(
            gflownet,
            mode=args.compile_mode,
        )
        if not quiet:
            formatted = ", ".join(
                f"{name}:{'✓' if success else 'x'}"
                for name, success in compile_results.items()
            )
            print(f"[compile] {formatted}")

    requested_vmap = use_vmap
    if use_vmap and not isinstance(gflownet, TBGFlowNet):
        if not quiet:
            print("vmap is currently only supported for TBGFlowNet; ignoring flag.")
        use_vmap = False
    effective_vmap = use_vmap

    if warmup_iters > 0:
        run_iterations(
            env,
            gflownet,
            sampler,
            optimizer,
            visited_states,
            metrics,
            args,
            n_iters=warmup_iters,
            use_vmap=use_vmap,
            quiet=True,
            collect_metrics=False,
            track_time=False,
            record_history=False,
        )

    elapsed, history = run_iterations(
        env,
        gflownet,
        sampler,
        optimizer,
        visited_states,
        metrics,
        args,
        n_iters=args.n_iterations,
        use_vmap=use_vmap,
        quiet=quiet,
        collect_metrics=True,
        track_time=timing,
        record_history=record_history,
    )

    if not quiet:
        validation_info = metrics["validation_info"]
        l1 = validation_info.get("l1_dist", float("nan"))
        print(
            f"Finished training | iterations={metrics['measured_steps']} | "
            f"modes={len(metrics['discovered_modes'])} / {env.n_modes} | "
            f"L1 distance={l1:.6f}"
        )

    return {
        "elapsed": elapsed or 0.0,
        "losses": history["losses"] if history else None,
        "iter_times": history["iter_times"] if history else None,
        "compile_mode": compile_mode,
        "use_compile": enable_compile,
        "requested_vmap": requested_vmap,
        "effective_vmap": effective_vmap,
    }


def run_iterations(
    env: HyperGrid,
    gflownet: TBGFlowNet | DBGFlowNet | FMGFlowNet,
    sampler: Sampler,
    optimizer: torch.optim.Optimizer,
    visited_states: DiscreteStates,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    *,
    n_iters: int,
    use_vmap: bool,
    quiet: bool,
    collect_metrics: bool,
    track_time: bool,
    record_history: bool,
) -> tuple[float | None, Dict[str, list[float]] | None]:
    if n_iters <= 0:
        empty_history = {"losses": [], "iter_times": []} if record_history else None
        return (0.0 if track_time else None), empty_history

    iterator: Iterable[int]
    if quiet:
        iterator = range(n_iters)
    else:
        iterator = tqdm(range(n_iters), dynamic_ncols=True)

    start_time = time.perf_counter() if track_time else None
    last_loss = 0.0
    losses_history: list[float] | None = [] if record_history else None
    iter_time_history: list[float] | None = [] if record_history else None

    for _ in iterator:
        iter_start = time.perf_counter() if (track_time or record_history) else None
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
        )

        terminating_states = cast(DiscreteStates, trajectories.terminating_states)
        visited_states.extend(terminating_states)

        optimizer.zero_grad()
        loss = compute_loss(gflownet, env, trajectories, use_vmap=use_vmap)
        loss.backward()
        gflownet.assert_finite_gradients()
        torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 1.0)
        optimizer.step()
        gflownet.assert_finite_parameters()

        metrics["total_steps"] += 1
        if collect_metrics:
            metrics["measured_steps"] += 1

        last_loss = loss.item()
        if (
            record_history
            and losses_history is not None
            and iter_time_history is not None
        ):
            losses_history.append(last_loss)
            iter_duration = (
                (time.perf_counter() - iter_start) if iter_start is not None else 0.0
            )
            iter_time_history.append(iter_duration)

        if collect_metrics:
            run_validation_if_needed(
                env,
                gflownet,
                visited_states,
                metrics,
                args,
                quiet=quiet,
            )

        if not quiet and isinstance(iterator, tqdm):
            iterator.set_postfix(
                {
                    "loss": last_loss,
                    "trajectories_sampled": (
                        metrics["measured_steps"] * args.batch_size
                    ),
                }
            )

    if track_time:
        synchronize_if_needed(env.device)
        assert start_time is not None
        elapsed_time = time.perf_counter() - start_time
    else:
        elapsed_time = None

    history = None
    if record_history and losses_history is not None and iter_time_history is not None:
        history = {
            "losses": losses_history,
            "iter_times": iter_time_history,
        }

    return elapsed_time, history


def compute_loss(
    gflownet: TBGFlowNet | DBGFlowNet | FMGFlowNet,
    env: HyperGrid,
    trajectories,
    *,
    use_vmap: bool,
) -> torch.Tensor:
    if use_vmap and isinstance(gflownet, TBGFlowNet):
        return trajectory_balance_loss_vmap(gflownet, trajectories)

    return gflownet.loss_from_trajectories(
        env, trajectories, recalculate_all_logprobs=False
    )


def trajectory_balance_loss_vmap(
    gflownet: TBGFlowNet,
    trajectories,
) -> torch.Tensor:
    log_pf, log_pb = gflownet.get_pfs_and_pbs(
        trajectories, recalculate_all_logprobs=False
    )
    log_rewards = trajectories.log_rewards
    if log_rewards is None:
        raise ValueError("Log rewards required for TB loss.")

    def tb_residual(
        log_pf_seq: torch.Tensor, log_pb_seq: torch.Tensor, log_reward: torch.Tensor
    ) -> torch.Tensor:
        return log_pf_seq.sum() - log_pb_seq.sum() - log_reward

    residuals = vmap(tb_residual)(
        log_pf.transpose(0, 1),
        log_pb.transpose(0, 1),
        log_rewards,
    )

    log_z = gflownet.logZ
    if isinstance(log_z, ScalarEstimator):
        if trajectories.conditions is None:
            raise ValueError("Conditional logZ requires conditions tensor.")
        log_z_value = log_z(trajectories.conditions)
    else:
        log_z_value = log_z

    if isinstance(log_z_value, torch.Tensor):
        log_z_tensor = log_z_value
    else:
        log_z_tensor = torch.as_tensor(log_z_value, device=residuals.device)
    log_z_tensor = log_z_tensor.squeeze()
    scores = (residuals + log_z_tensor).pow(2)

    return scores.mean()


def run_validation_if_needed(
    env: HyperGrid,
    gflownet: TBGFlowNet | DBGFlowNet | FMGFlowNet,
    visited_states: DiscreteStates,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    *,
    quiet: bool,
) -> None:
    if args.validation_interval <= 0:
        return
    measured_steps = metrics["measured_steps"]
    if measured_steps == 0:
        return
    if measured_steps % args.validation_interval != 0:
        return

    validation_info, _ = validate(
        env,
        gflownet,
        args.validation_samples,
        visited_states,
    )
    metrics["validation_info"] = validation_info
    modes_found = env.modes_found(visited_states)
    metrics["discovered_modes"].update(modes_found)

    if not quiet:
        str_info = (
            f"Iter {measured_steps}: "
            f"L1 distance={validation_info.get('l1_dist', float('nan')):.8f} "
            f"modes discovered={len(metrics['discovered_modes'])} / {env.n_modes} "
            f"n terminating states {len(visited_states)}"
        )
        print(str_info)


def build_training_components(args: argparse.Namespace, device: torch.device) -> tuple[
    HyperGrid,
    TBGFlowNet | DBGFlowNet | FMGFlowNet,
    Sampler,
    torch.optim.Optimizer,
    DiscreteStates,
]:
    env = HyperGrid(
        ndim=args.ndim,
        height=args.height,
        reward_fn_str="original",
        reward_fn_kwargs={
            "R0": args.R0,
            "R1": args.R1,
            "R2": args.R2,
        },
        device=device,
        calculate_partition=True,
        store_all_states=True,
        check_action_validity=__debug__,
    )

    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    if not args.uniform_pb:
        module_PB = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            trunk=module_PF.trunk,
        )
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)

    if args.loss == "FM":
        logF_estimator = DiscretePolicyEstimator(
            module=module_PF,
            n_actions=env.n_actions,
            preprocessor=preprocessor,
        )
        gflownet: TBGFlowNet | DBGFlowNet | FMGFlowNet = FMGFlowNet(logF_estimator).to(
            device
        )
        optimizer = torch.optim.Adam(gflownet.logF.parameters(), lr=args.lr)
        sampler = Sampler(estimator=logF_estimator)
    else:
        pf_estimator = DiscretePolicyEstimator(
            module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
        )
        pb_estimator = DiscretePolicyEstimator(
            module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
        )

        if args.loss == "DB":
            logF_module = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=1,
            )
            logF_estimator = ScalarEstimator(
                module=logF_module,
                preprocessor=preprocessor,
            )
            gflownet = DBGFlowNet(pf=pf_estimator, pb=pb_estimator, logF=logF_estimator)
        else:
            gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)

        gflownet = gflownet.to(device)
        optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
        if isinstance(gflownet, DBGFlowNet):
            optimizer.add_param_group(
                {"params": gflownet.logF.parameters(), "lr": args.lr}
            )
        else:
            optimizer.add_param_group(
                {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
            )
        sampler = Sampler(estimator=pf_estimator)

    visited_states = env.states_from_batch_shape((0,))
    return env, gflownet, sampler, optimizer, visited_states


def _mps_backend_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_backend_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "mps" and not _mps_backend_available():
        raise RuntimeError("MPS requested but not available.")
    return device


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and _mps_backend_available() and hasattr(torch, "mps"):
        torch.mps.synchronize()


def plot_benchmark(results: list[Dict[str, Any]], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting; install it or omit --benchmark."
        ) from exc

    def summarize_iteration_times(times: list[float]) -> tuple[float, float]:
        if not times:
            return 0.0, 0.0
        mean_time = statistics.fmean(times)
        std_time = statistics.pstdev(times) if len(times) > 1 else 0.0
        return mean_time, std_time

    labels = [res.get("label", f"Run {idx+1}") for idx, res in enumerate(results)]
    times = [res["elapsed"] for res in results]
    losses_list = [res.get("losses") or [] for res in results]
    iter_times_list = [res.get("iter_times") or [] for res in results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Subplot 1: total time comparison
    colors = ["#6c757d", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bar_colors = [colors[i % len(colors)] for i in range(len(results))]
    bars = axes[0].bar(labels, times, color=bar_colors)
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Total Training Time")
    baseline_time = times[0] if times else 1.0
    for i, (bar, value) in enumerate(zip(bars, times)):
        speedup = baseline_time / value if value else float("inf")
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}s\n{speedup:.2f}x",
            ha="center",
            va="bottom",
        )

    # Subplot 2: training curves
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))]

    for idx, losses in enumerate(losses_list):
        if not losses:
            continue
        axes[1].plot(
            range(1, len(losses) + 1),
            losses,
            label=labels[idx],
            color=bar_colors[idx],
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=2.0,
            alpha=0.5,
        )
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # Subplot 3: per-iteration timing with error bars
    summary_stats = [summarize_iteration_times(times) for times in iter_times_list]
    means_ms = [mean * 1000.0 for mean, _ in summary_stats]
    stds_ms = [std * 1000.0 for _, std in summary_stats]
    axes[2].bar(
        labels,
        means_ms,
        yerr=stds_ms,
        capsize=6,
        color=bar_colors,
    )
    axes[2].set_ylabel("Per-iteration time (ms)")
    axes[2].set_title("Iteration Timing (mean ± std)")

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved benchmark plot to {output}")


if __name__ == "__main__":
    main()
