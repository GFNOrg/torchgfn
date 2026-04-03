#!/usr/bin/env python
"""Collect all benchmark results from JSON output files into a single CSV.

Reads every benchmark_*.json file in the outputs directory and produces
a combined CSV with one row per (scenario, batch_size, library, seed).

Usage:
    python benchmark/collect_results.py                    # default: benchmark/outputs/
    python benchmark/collect_results.py --output-dir path  # custom output dir
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def collect_results(output_dir: Path) -> Path:
    json_files = sorted(output_dir.glob("benchmark_*.json"))
    if not json_files:
        print(f"No benchmark JSON files found in {output_dir}")
        sys.exit(1)

    rows = []
    seen = set()

    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)

        scenario = data["scenario"]
        config = data["config"]
        batch_size = config["batch_size"]
        env_name = config["env_name"]

        for result in data["results"]:
            key = (scenario, batch_size, result["library"], result["seed"])

            if key in seen:
                # Duplicate — keep the latest file (sorted by timestamp)
                # Replace the existing row
                rows = [
                    r
                    for r in rows
                    if (r["scenario"], r["batch_size"], r["library"], r["seed"]) != key
                ]

            seen.add(key)

            # Flatten phase times into columns
            phase_times = result.get("phase_times", {})
            phase_cols = {}
            for phase, stats in phase_times.items():
                if isinstance(stats, dict):
                    phase_cols[f"phase_{phase}_mean_ms"] = stats.get("mean_ms")
                    phase_cols[f"phase_{phase}_total_s"] = stats.get("total_s")

            rows.append(
                {
                    "scenario": scenario,
                    "env_name": env_name,
                    "batch_size": batch_size,
                    "library": result["library"],
                    "seed": result["seed"],
                    "n_iterations": result["n_iterations"],
                    "mean_iter_time_ms": result["mean_iter_time"] * 1000,
                    "std_iter_time_ms": result["std_iter_time"] * 1000,
                    "total_time_s": result["total_time"],
                    "throughput_iters_per_sec": result["throughput_iters_per_sec"],
                    "peak_memory_mb": result["peak_memory_mb"],
                    "n_params": result["n_params"],
                    **phase_cols,
                }
            )

    # Sort by scenario, batch_size, library, seed
    rows.sort(key=lambda r: (r["scenario"], r["batch_size"], r["library"], r["seed"]))

    # Collect all fieldnames (base + any phase columns)
    base_fields = [
        "scenario",
        "env_name",
        "batch_size",
        "library",
        "seed",
        "n_iterations",
        "mean_iter_time_ms",
        "std_iter_time_ms",
        "total_time_s",
        "throughput_iters_per_sec",
        "peak_memory_mb",
        "n_params",
    ]
    phase_fields = sorted({k for row in rows for k in row if k.startswith("phase_")})
    fieldnames = base_fields + phase_fields

    csv_path = output_dir / "benchmark_collected.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect benchmark results from JSON files into a single CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory containing benchmark JSON files (default: benchmark/outputs/)",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(__file__).parent / "outputs"
    )

    csv_path = collect_results(output_dir)

    # Print summary
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    scenarios = sorted(set(r["scenario"] for r in rows))
    libraries = sorted(set(r["library"] for r in rows))
    batch_sizes = sorted(set(int(r["batch_size"]) for r in rows))

    print(f"Collected {len(rows)} results from {output_dir}")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Libraries: {', '.join(libraries)}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Saved to: {csv_path}")


if __name__ == "__main__":
    main()
