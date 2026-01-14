#!/usr/bin/env python
"""Benchmark script for comparing GFlowNet libraries.

This script benchmarks torchgfn, gflownet, and gfnx libraries on
Trajectory Balance training for multiple environments:
- hypergrid: Discrete grid navigation (all libraries)
- ising: Discrete Ising model (all libraries)
- box/ccube: Continuous cube environment (torchgfn, gflownet only)
- bitseq: Bit sequence generation (torchgfn, gfnx only)

Example usage:
    python benchmark/benchmark_libraries.py --scenario tb_hypergrid_small --seeds 0 1 2
    python benchmark/benchmark_libraries.py --libraries torchgfn gfnx --scenario tb_bitseq_small
    python benchmark/benchmark_libraries.py --scenario tb_ising_6x6 --libraries torchgfn gflownet gfnx
"""

# ============================================================================
# OpenMP Conflict Workaround (must be set before any imports)
# ============================================================================
# On macOS, mixed conda/pip environments often have multiple copies of libomp
# (from llvm-openmp, torch, sklearn, etc.) which causes a crash when both are
# loaded. This workaround allows the program to continue despite the conflict.
# See: https://github.com/pytorch/pytorch/issues/78490
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Type  # noqa: E402

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.lib_runners.base import BenchmarkConfig  # noqa: E402
from benchmark.lib_runners.base import BenchmarkResult  # noqa: E402
from benchmark.lib_runners.base import LibraryRunner  # noqa: E402

# ============================================================================
# Lazy Import Functions (to avoid OpenMP conflicts on macOS)
# ============================================================================


def _get_torchgfn_runner() -> Type[LibraryRunner]:
    from benchmark.lib_runners.torchgfn_runner import TorchGFNRunner

    return TorchGFNRunner


def _get_gflownet_runner() -> Type[LibraryRunner]:
    from benchmark.lib_runners.gflownet_runner import GFlowNetRunner

    return GFlowNetRunner


def _get_gfnx_runner() -> Type[LibraryRunner]:
    from benchmark.lib_runners.gfnx_runner import GFNXRunner

    return GFNXRunner


# ============================================================================
# Scenario Configurations
# ============================================================================

SCENARIOS: Dict[str, BenchmarkConfig] = {
    # Hypergrid scenarios (all libraries: torchgfn, gflownet, gfnx)
    "tb_hypergrid_small": BenchmarkConfig(
        env_name="hypergrid",
        env_kwargs={"ndim": 2, "height": 8},
        n_iterations=1000,
        batch_size=16,
        n_warmup=50,
    ),
    "tb_hypergrid_medium": BenchmarkConfig(
        env_name="hypergrid",
        env_kwargs={"ndim": 4, "height": 16},
        n_iterations=2000,
        batch_size=16,
        n_warmup=100,
    ),
    "tb_hypergrid_large": BenchmarkConfig(
        env_name="hypergrid",
        env_kwargs={"ndim": 4, "height": 32},
        n_iterations=5000,
        batch_size=16,
        n_warmup=100,
    ),
    # Ising scenarios (all libraries: torchgfn, gflownet, gfnx)
    "tb_ising_6x6": BenchmarkConfig(
        env_name="ising",
        env_kwargs={"L": 6, "J": 0.44},
        n_iterations=1000,
        batch_size=16,
        n_warmup=50,
    ),
    "tb_ising_10x10": BenchmarkConfig(
        env_name="ising",
        env_kwargs={"L": 10, "J": 0.44},
        n_iterations=2000,
        batch_size=16,
        n_warmup=100,
    ),
    # Box/CCube scenarios (torchgfn, gflownet only - gfnx does not have this env)
    "tb_box_2d": BenchmarkConfig(
        env_name="box",
        env_kwargs={"n_dim": 2, "delta": 0.25, "n_components": 5},
        n_iterations=1000,
        batch_size=16,
        n_warmup=50,
    ),
    # BitSequence scenarios (torchgfn, gfnx only - gflownet does not have this env)
    "tb_bitseq_small": BenchmarkConfig(
        env_name="bitseq",
        env_kwargs={"word_size": 1, "seq_size": 4, "n_modes": 2},
        n_iterations=1000,
        batch_size=16,
        n_warmup=50,
    ),
    "tb_bitseq_medium": BenchmarkConfig(
        env_name="bitseq",
        env_kwargs={"word_size": 2, "seq_size": 8, "n_modes": 4},
        n_iterations=2000,
        batch_size=16,
        n_warmup=100,
    ),
}


# ============================================================================
# Library Registry (using lazy loaders to avoid importing all libraries)
# ============================================================================

# Maps library name to a function that returns the runner class
# This avoids importing all libraries at startup, which causes OpenMP conflicts on macOS
LIBRARY_RUNNERS: Dict[str, callable] = {
    "torchgfn": _get_torchgfn_runner,
    "gflownet": _get_gflownet_runner,
    "gfnx": _get_gfnx_runner,
}


# ============================================================================
# Environment-Library Availability Mapping
# ============================================================================
# Not all libraries support all environments. This mapping defines which
# libraries can run each environment type.

ENV_LIBRARY_SUPPORT: Dict[str, List[str]] = {
    "hypergrid": ["torchgfn", "gflownet", "gfnx"],
    "ising": ["torchgfn", "gflownet", "gfnx"],
    "box": ["torchgfn", "gflownet"],  # gfnx does not have continuous box/ccube
    "bitseq": ["torchgfn", "gfnx"],  # gflownet does not have bitsequence
}


def get_supported_libraries(env_name: str) -> List[str]:
    """Get list of libraries that support the given environment.

    Args:
        env_name: Name of the environment.

    Returns:
        List of library names that support this environment.
    """
    return ENV_LIBRARY_SUPPORT.get(env_name, list(LIBRARY_RUNNERS.keys()))


def get_default_libraries(env_name: str) -> List[str]:
    """Get default libraries to run for a given environment.

    Args:
        env_name: Name of the environment.

    Returns:
        List of library names to use by default for this environment.
    """
    return get_supported_libraries(env_name)


# ============================================================================
# Benchmarking Functions
# ============================================================================


def run_benchmark(
    runner: LibraryRunner,
    config: BenchmarkConfig,
    seed: int,
) -> BenchmarkResult:
    """Run a single benchmark for a library with given config and seed.

    Args:
        runner: The library runner instance.
        config: Benchmark configuration.
        seed: Random seed.

    Returns:
        BenchmarkResult with timing and memory information.
    """
    print(f"  Setting up {runner.name} with seed {seed}...")
    runner.setup(config, seed)

    print(f"  Running {config.n_warmup} warmup iterations...")
    runner.warmup(config.n_warmup)

    print(f"  Running {config.n_iterations} timed iterations...")
    iter_times = []

    # Time each iteration individually
    runner.synchronize()
    total_start = time.perf_counter()

    for i in range(config.n_iterations):
        runner.synchronize()
        iter_start = time.perf_counter()

        runner.run_iteration()

        runner.synchronize()
        iter_end = time.perf_counter()

        iter_times.append(iter_end - iter_start)

        # Progress update every 10%
        if (i + 1) % max(1, config.n_iterations // 10) == 0:
            progress = (i + 1) / config.n_iterations * 100
            mean_time = sum(iter_times) / len(iter_times)
            print(
                f"    Progress: {progress:.0f}% ({i+1}/{config.n_iterations}), "
                f"mean iter time: {mean_time*1000:.2f}ms"
            )

    total_end = time.perf_counter()
    total_time = total_end - total_start

    # Get peak memory
    peak_memory = runner.get_peak_memory()

    # Cleanup
    runner.cleanup()

    return BenchmarkResult(
        library=runner.name,
        seed=seed,
        total_time=total_time,
        iter_times=iter_times,
        peak_memory=peak_memory,
    )


def aggregate_results(results: List[BenchmarkResult]) -> Dict:
    """Aggregate results across seeds for each library.

    Args:
        results: List of benchmark results.

    Returns:
        Dictionary with summary statistics per library.
    """
    import statistics
    from collections import defaultdict

    by_library = defaultdict(list)
    for r in results:
        by_library[r.library].append(r)

    summary = {}
    for library, lib_results in by_library.items():
        all_iter_times = []
        for r in lib_results:
            all_iter_times.extend(r.iter_times)

        mean_iter_time = statistics.mean(all_iter_times) if all_iter_times else 0
        std_iter_time = (
            statistics.stdev(all_iter_times) if len(all_iter_times) > 1 else 0
        )

        total_times = [r.total_time for r in lib_results]
        mean_total_time = statistics.mean(total_times) if total_times else 0

        throughputs = [r.throughput for r in lib_results]
        mean_throughput = statistics.mean(throughputs) if throughputs else 0

        peak_memories = [
            r.peak_memory_mb for r in lib_results if r.peak_memory_mb is not None
        ]
        mean_peak_memory = statistics.mean(peak_memories) if peak_memories else None

        summary[library] = {
            "n_runs": len(lib_results),
            "mean_iter_time_ms": mean_iter_time * 1000,
            "std_iter_time_ms": std_iter_time * 1000,
            "mean_total_time_s": mean_total_time,
            "mean_throughput_iters_per_sec": mean_throughput,
            "mean_peak_memory_mb": mean_peak_memory,
        }

    return summary


def save_results(
    scenario: str,
    config: BenchmarkConfig,
    results: List[BenchmarkResult],
    output_dir: Path,
) -> Path:
    """Save benchmark results to JSON file.

    Args:
        scenario: Scenario name.
        config: Benchmark configuration.
        results: List of benchmark results.
        output_dir: Output directory.

    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    libraries_str = "_".join(sorted(set(r.library for r in results)))
    filename = f"benchmark_{scenario}_{libraries_str}_{timestamp}.json"
    filepath = output_dir / filename

    summary = aggregate_results(results)

    output = {
        "scenario": scenario,
        "timestamp": timestamp,
        "config": config.to_dict(),
        "results": [r.to_dict() for r in results],
        "summary": summary,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def print_summary(summary: Dict) -> None:
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Header
    print(
        f"{'Library':<15} {'Iter Time (ms)':<18} {'Throughput (it/s)':<20} {'Memory (MB)':<15}"
    )
    print("-" * 70)

    for library, stats in sorted(summary.items()):
        iter_time = f"{stats['mean_iter_time_ms']:.2f} ± {stats['std_iter_time_ms']:.2f}"
        throughput = f"{stats['mean_throughput_iters_per_sec']:.1f}"
        memory = (
            f"{stats['mean_peak_memory_mb']:.1f}"
            if stats["mean_peak_memory_mb"]
            else "N/A"
        )

        print(f"{library:<15} {iter_time:<18} {throughput:<20} {memory:<15}")

    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GFlowNet libraries on multiple environments"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="tb_hypergrid_small",
        choices=list(SCENARIOS.keys()),
        help="Benchmark scenario to run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds to use",
    )
    parser.add_argument(
        "--libraries",
        type=str,
        nargs="+",
        default=None,  # Will be set based on environment
        choices=list(LIBRARY_RUNNERS.keys()),
        help="Libraries to benchmark (default: all supported for the environment)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: benchmark/outputs)",
    )

    args = parser.parse_args()

    # Get configuration
    config = SCENARIOS[args.scenario]
    output_dir = Path(args.output) if args.output else Path(__file__).parent / "outputs"

    # Determine which libraries to run
    supported_libs = get_supported_libraries(config.env_name)
    if args.libraries is None:
        # Use all supported libraries for this environment
        libraries = supported_libs
    else:
        # Validate that requested libraries support this environment
        libraries = []
        for lib in args.libraries:
            if lib in supported_libs:
                libraries.append(lib)
            else:
                print(
                    f"Warning: {lib} does not support {config.env_name} environment, skipping."
                )
        if not libraries:
            print(
                f"Error: No valid libraries for {config.env_name}. Supported: {supported_libs}"
            )
            sys.exit(1)

    # Note about running multiple libraries on macOS
    import platform

    if platform.system() == "Darwin" and len(libraries) > 1:
        print(
            "\nNote: Running multiple libraries together. KMP_DUPLICATE_LIB_OK is set\n"
            "to work around macOS OpenMP conflicts. For cleanest results, consider\n"
            "running each library separately.\n"
        )

    print("=" * 70)
    print(f"GFlowNet Library Benchmark: {args.scenario}")
    print("=" * 70)
    print("Configuration:")
    print(f"  env: {config.env_name}, {config.env_kwargs}")
    print(f"  n_iterations={config.n_iterations}, batch_size={config.batch_size}")
    print(f"  n_warmup={config.n_warmup}")
    print(f"  lr={config.lr}, lr_logz={config.lr_logz}")
    print(f"  hidden_dim={config.hidden_dim}, n_layers={config.n_layers}")
    print(f"Libraries: {', '.join(libraries)}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    results = []

    for library in libraries:
        print(f"\n[{library.upper()}]")

        runner_cls = LIBRARY_RUNNERS[library]()  # Call the lazy loader function

        for seed in args.seeds:
            print(f"\nSeed {seed}:")
            try:
                runner = runner_cls()
                result = run_benchmark(runner, config, seed)
                results.append(result)

                print(f"  Total time: {result.total_time:.2f}s")
                print(f"  Mean iter time: {result.mean_iter_time*1000:.2f}ms")
                print(f"  Throughput: {result.throughput:.1f} iter/s")
                if result.peak_memory_mb:
                    print(f"  Peak memory: {result.peak_memory_mb:.1f}MB")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()

    # Save and summarize results
    if results:
        filepath = save_results(args.scenario, config, results, output_dir)
        print(f"\nResults saved to: {filepath}")

        summary = aggregate_results(results)
        print_summary(summary)


if __name__ == "__main__":
    main()
