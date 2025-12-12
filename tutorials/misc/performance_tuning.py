"""
Quick performance benchmark for GFlowNet training.

Runs per-iteration timing across environments (HyperGrid, DiffusionSampling,
DiscreteEBM), losses (TB, SubTB, ModifiedDB), debug on/off, and torch.compile
on/off (compiling the standard sampling loop when possible). Produces a bar plot
saved to ~/performance_tuning.png.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch

from gfn.estimators import (
    DiscretePolicyEstimator,
    PinnedBrownianMotionBackward,
    PinnedBrownianMotionForward,
    ScalarEstimator,
)
from gfn.gflownet.detailed_balance import DBGFlowNet, ModifiedDBGFlowNet
from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym import DiscreteEBM, HyperGrid
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor
from gfn.utils.common import set_seed
from gfn.utils.modules import (
    MLP,
    DiffusionFixedBackwardModule,
    DiffusionPISGradNetForward,
)


@dataclass
class BenchmarkSettings:
    batch_size: int = 32
    warmup_iters: int = 50
    n_iters: int = 200
    device: str = "cpu"
    output_path: Path | None = Path.home() / "performance_tuning.png"
    deterministic_mode: bool = False


LR = 1e-3
LR_LOGZ = 1e-1
LR_LOGF = 1e-3


@dataclass
class BenchmarkConfig:
    env_name: str
    loss_name: str
    debug: bool
    use_compile: bool

    @property
    def label(self) -> str:
        dbg = "dbg" if self.debug else "nodbg"
        comp = "comp" if self.use_compile else "eager"
        return f"{self.env_name}/{self.loss_name}/{dbg}/{comp}"


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    env_name: str
    elapsed: float
    mean_iter_ms: float
    std_iter_ms: float
    compiled: bool
    skipped: bool
    reason: str | None = None


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS requested but not available.")
    return device


def build_components(env_name: str, loss_name: str, debug: bool, device: torch.device):
    """Create env, gflownet, optimizer for a given setup."""
    # Diffusion branch first (no n_actions attribute).
    if env_name == "diffusion":
        env = DiffusionSampling(
            target_str="gmm2",
            target_kwargs={"seed": 2},
            num_discretization_steps=32,
            device=device,
            debug=debug,
        )
        s_dim = env.dim
        pf_module = DiffusionPISGradNetForward(
            s_dim=s_dim,
            harmonics_dim=64,
            t_emb_dim=64,
            s_emb_dim=64,
            hidden_dim=64,
            joint_layers=2,
            zero_init=False,
        )
        pb_module = DiffusionFixedBackwardModule(s_dim=s_dim)
        pf = PinnedBrownianMotionForward(
            s_dim=s_dim,
            pf_module=pf_module,
            sigma=5.0,
            num_discretization_steps=32,
        )
        pb = PinnedBrownianMotionBackward(
            s_dim=s_dim,
            pb_module=pb_module,
            sigma=5.0,
            num_discretization_steps=32,
        )
        logF = ScalarEstimator(
            module=MLP(input_dim=env.state_shape[-1], output_dim=1),
            preprocessor=IdentityPreprocessor(output_dim=env.state_shape[-1]),
        )
    elif env_name == "hypergrid":
        env = HyperGrid(
            ndim=2,
            height=32,
            reward_fn_str="original",
            reward_fn_kwargs={"R0": 0.1, "R1": 0.5, "R2": 2.0},
            calculate_partition=False,
            store_all_states=False,
            device=device,
            debug=debug,
        )
        preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
        assert isinstance(preprocessor.output_dim, int)
        out_dim = preprocessor.output_dim
        pf_module = MLP(input_dim=out_dim, output_dim=env.n_actions)
        pb_module = MLP(input_dim=out_dim, output_dim=env.n_actions - 1)
        pf = DiscretePolicyEstimator(
            module=pf_module, n_actions=env.n_actions, preprocessor=preprocessor
        )
        pb = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            preprocessor=preprocessor,
            is_backward=True,
        )
        logF = ScalarEstimator(
            module=MLP(input_dim=out_dim, output_dim=1), preprocessor=preprocessor
        )
    elif env_name == "discrete_ebm":
        env = DiscreteEBM(ndim=6, device=device, debug=debug)
        preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])
        assert isinstance(preprocessor.output_dim, int)
        out_dim = preprocessor.output_dim
        pf_module = MLP(input_dim=out_dim, output_dim=env.n_actions)
        pb_module = MLP(input_dim=out_dim, output_dim=env.n_actions - 1)
        pf = DiscretePolicyEstimator(
            module=pf_module, n_actions=env.n_actions, preprocessor=preprocessor
        )
        pb = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            preprocessor=preprocessor,
            is_backward=True,
        )
        logF = ScalarEstimator(
            module=MLP(input_dim=out_dim, output_dim=1), preprocessor=preprocessor
        )
    else:
        raise ValueError(f"Unknown environment {env_name}")

    if loss_name == "TB":
        gflownet = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0)
    elif loss_name == "SubTB":
        gflownet = SubTBGFlowNet(
            pf=pf, pb=pb, logF=logF, weighting="ModifiedDB", lamda=0.9
        )
    elif loss_name == "DBG":
        gflownet = DBGFlowNet(pf=pf, pb=pb, logF=logF)
    elif loss_name == "ModifiedDB":
        gflownet = ModifiedDBGFlowNet(pf=pf, pb=pb)
    else:
        raise ValueError(f"Unknown loss {loss_name}")

    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=LR)

    if hasattr(gflownet, "logz_parameters") and callable(gflownet.logz_parameters):
        params = gflownet.logz_parameters()
        if params:
            optimizer.add_param_group({"params": params, "lr": LR_LOGZ})

    if hasattr(gflownet, "logF_parameters") and callable(gflownet.logF_parameters):
        params = gflownet.logF_parameters()
        if params:
            optimizer.add_param_group({"params": params, "lr": LR_LOGF})

    return env, gflownet, optimizer


def make_step_fn(
    env, gflownet, optimizer, batch_size: int, device: torch.device
) -> Callable[[], float]:
    def step():
        trajectories = gflownet.sample_trajectories(
            env,
            n=batch_size,
            save_logprobs=False,
            save_estimator_outputs=False,
        )
        training_samples = gflownet.to_training_samples(trajectories)

        # Simulated off-policy sampling.
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    return step


def maybe_compile(fn: Callable[[], float]) -> tuple[Callable[[], float], bool]:
    if not hasattr(torch, "compile"):
        return fn, False
    try:
        compiled = torch.compile(fn, mode="reduce-overhead")  # type: ignore[arg-type]
        # One dry run to populate graph / catch failures early.
        compiled()
        return compiled, True
    except Exception:
        return fn, False


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps":
        mps_backend = getattr(torch, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "synchronize"):
            # Ensure queued MPS work completes before timing.
            mps_backend.synchronize()


def time_iterations(
    step_fn: Callable[[], float],
    device: torch.device,
    warmup: int,
    n_iters: int,
) -> tuple[float, list[float]]:

    for _ in range(warmup):
        step_fn()

    iter_times: list[float] = []
    start = time.perf_counter()

    for _ in range(n_iters):
        iter_start = time.perf_counter()
        step_fn()
        synchronize(device)  # Measure actual kernel + overhead per step.
        iter_times.append((time.perf_counter() - iter_start) * 1000.0)

    synchronize(device)
    total = time.perf_counter() - start

    return total, iter_times


def run_benchmark(
    config: BenchmarkConfig, device: torch.device, settings: BenchmarkSettings
) -> BenchmarkResult:
    # if config.loss_name == "ModifiedDB" and config.env_name != "hypergrid":
    #     return BenchmarkResult(
    #         config=config,
    #         elapsed=0.0,
    #         mean_iter_ms=0.0,
    #         std_iter_ms=0.0,
    #         compiled=False,
    #         skipped=True,
    #         reason="ModifiedDB supported only on HyperGrid",
    #     )
    try:
        env, gflownet, optimizer = build_components(
            config.env_name, config.loss_name, config.debug, device
        )
    except Exception as exc:
        return BenchmarkResult(
            config=config,
            env_name=config.env_name,
            elapsed=0.0,
            mean_iter_ms=0.0,
            std_iter_ms=0.0,
            compiled=False,
            skipped=True,
            reason=f"init failed: {exc}",
        )

    step_fn = make_step_fn(env, gflownet, optimizer, settings.batch_size, device)
    compiled = False
    if config.use_compile:
        step_fn, compiled = maybe_compile(step_fn)

    try:
        elapsed, iter_times = time_iterations(
            step_fn,
            device=device,
            warmup=settings.warmup_iters,
            n_iters=settings.n_iters,
        )
        mean_ms = statistics.fmean(iter_times) if iter_times else 0.0
        std_ms = statistics.pstdev(iter_times) if len(iter_times) > 1 else 0.0
        return BenchmarkResult(
            config=config,
            env_name=config.env_name,
            elapsed=elapsed,
            mean_iter_ms=mean_ms,
            std_iter_ms=std_ms,
            compiled=compiled,
            skipped=False,
            reason=None,
        )
    except Exception as exc:
        return BenchmarkResult(
            config=config,
            env_name=config.env_name,
            elapsed=0.0,
            mean_iter_ms=0.0,
            std_iter_ms=0.0,
            compiled=compiled,
            skipped=True,
            reason=f"run failed: {exc}",
        )


def plot_results(
    results: list[BenchmarkResult],
    output_path: Path | None = None,
    return_fig: bool = False,
):
    ok_results = [r for r in results if not r.skipped]
    if not ok_results:
        print("No successful runs to plot.")
        return

    # Preserve environment order of appearance.
    env_order: list[str] = []
    for res in ok_results:
        if res.env_name not in env_order:
            env_order.append(res.env_name)

    n_rows = max(1, len(env_order))
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.5 * n_rows))
    if n_rows == 1:
        axes = [axes]  # type: ignore[list-item]

    # Fixed colors by condition for easy cross-env comparison.
    condition_colors = {
        (True, False): "#000000",  # debug, eager
        (False, False): "#8b0000",  # nodebug, eager (dark red)
        (True, True): "#555555",  # debug, compiled (dark grey)
        (False, True): "#e57373",  # nodebug, compiled (light red)
    }

    for row_idx, env_key in enumerate(env_order):
        row_ax = axes[row_idx]
        env_results = [r for r in ok_results if r.env_name == env_key]
        if not env_results:
            continue

        # Baselines per loss: debug=True & use_compile=False when available.
        baselines: dict[str, float] = {}
        for res in env_results:
            if res.config.debug and not res.config.use_compile:
                baselines[res.config.loss_name] = res.elapsed or 1.0
        for res in env_results:
            baselines.setdefault(res.config.loss_name, res.elapsed or 1.0)

        labels: list[str] = []
        times: list[float] = []
        colors: list[str] = []
        speeds: list[float] = []

        for res in env_results:
            labels.append(
                f"{res.config.loss_name} | {'dbg' if res.config.debug else 'nodebug'} | "
                f"{'comp' if res.config.use_compile else 'eager'}"
            )
            times.append(res.elapsed)
            colors.append(
                condition_colors.get(
                    (res.config.debug, res.config.use_compile),
                    "#6c757d",  # fallback neutral
                )
            )
            base = baselines.get(res.config.loss_name, res.elapsed or 1.0)
            speeds.append(base / res.elapsed if res.elapsed else 0.0)

        bars = row_ax.barh(labels, times, color=colors)
        row_ax.set_xlabel("Total time (s)")
        row_ax.set_title(f"{env_key} | runtime; baseline = debug+eager")

        for bar, res, speed in zip(bars, env_results, speeds):
            row_ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{res.elapsed:.2f}s\nx{speed:.2f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        row_ax.invert_yaxis()
        for label in row_ax.get_yticklabels():
            label.set_rotation(0)
            label.set_ha("right")

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if return_fig:
        return fig
    plt.close(fig)


def run_benchmarks(
    settings: BenchmarkSettings,
    *,
    return_fig: bool = False,
    verbose: bool = False,
):
    # Reduce CPU scheduling noise for more stable comparisons.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    set_seed(0, deterministic_mode=settings.deterministic_mode)
    device = resolve_device(settings.device)
    if verbose:
        print(f"Using device: {device}")

    configs: list[BenchmarkConfig] = []
    for env_name in ["hypergrid", "diffusion", "discrete_ebm"]:
        for loss_name in ["TB", "SubTB", "DBG"]:
            for debug in [True, False]:
                for use_compile in [True, False]:
                    configs.append(
                        BenchmarkConfig(
                            env_name=env_name,
                            loss_name=loss_name,
                            debug=debug,
                            use_compile=use_compile,
                        )
                    )

    results: list[BenchmarkResult] = []
    for cfg in configs:
        if verbose:
            print(f"Running {cfg.label} ...")
        res = run_benchmark(cfg, device, settings)
        if verbose:
            status = (
                "skipped"
                if res.skipped
                else f"done ({'compiled' if res.compiled else 'eager'})"
            )
            msg = (
                f"  {status}: elapsed={res.elapsed:.2f}s, "
                f"mean_iter={res.mean_iter_ms:.2f} ms, std={res.std_iter_ms:.2f} ms"
            )
            if res.reason:
                msg += f" | reason: {res.reason}"
            print(msg)
        results.append(res)

    fig = plot_results(results, output_path=settings.output_path, return_fig=return_fig)
    return results, fig


def main() -> None:
    parser = argparse.ArgumentParser(description="GFlowNet performance benchmark")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on; default cpu (auto prefers cuda>mps>cpu).",
    )
    args = parser.parse_args()

    settings = BenchmarkSettings(
        batch_size=32,
        warmup_iters=50,
        n_iters=200,
        device=args.device,
        output_path=Path.home() / "performance_tuning.png",
        deterministic_mode=False,
    )

    results, _ = run_benchmarks(settings=settings, return_fig=False, verbose=True)
    if settings.output_path is not None:
        print(f"Saved plot to {settings.output_path}")

    print("\nSummary (successful runs):")
    for res in results:
        if res.skipped:
            print(f"- {res.config.label}: skipped ({res.reason})")
            continue
        print(
            f"- {res.config.label}: {res.elapsed:.2f}s total | "
            f"{res.mean_iter_ms:.2f}±{res.std_iter_ms:.2f} ms/iter | "
            f"{'compiled' if res.compiled else 'eager'}"
        )


if __name__ == "__main__":
    main()
