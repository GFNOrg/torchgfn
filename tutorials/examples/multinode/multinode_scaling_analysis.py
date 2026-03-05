#!/usr/bin/env python3
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false
# pyright: reportOptionalIterable=false, reportArgumentType=false, reportAssignmentType=false
"""
Multinode Scaling Analysis Script

This script analyzes the results of multinode scaling experiments from the torchgfn project on Weights & Biases.
It focuses on n_modes_found progression to understand how many iterations it takes to discover all modes.
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Strategy definition - easily extensible list of parameters
STRATEGY_PARAMS = [
    "average_every",
    "replacement_ratio",
    "restart_init_mode",
    "use_random_strategies",
    "use_restarts",
    "use_selective_averaging",
]  # Add new parameters here

# Strategy mapping: long-form strategy IDs -> human-readable shorthand
# Add entries here to customize how strategies appear in legends
# Keys are the full strategy strings, values are short display names
STRATEGY_MAPPING = {
    "average_every=100_replacement_ratio=0.2_restart_init_mode=random_use_random_strategies=False_use_restarts=False_use_selective_averaging=False": "Baseline",
    "average_every=100_replacement_ratio=0.2_restart_init_mode=random_use_random_strategies=False_use_restarts=False_use_selective_averaging=True": "Selective Averaging",  # Selective averaging is on, but use_restarts is False?
    "average_every=100_replacement_ratio=0.2_restart_init_mode=random_use_random_strategies=True_use_restarts=False_use_selective_averaging=False": "Random Strategies",
    "average_every=100_replacement_ratio=0.2_restart_init_mode=random_use_random_strategies=True_use_restarts=False_use_selective_averaging=True": "Selective Averaging & Random Strategies",  # Selective averaging is on, but use_restarts is False?
    "average_every=16384000_replacement_ratio=0.2_restart_init_mode=random_use_random_strategies=False_use_restarts=False_use_selective_averaging=False": "Baseline (16M steps Averaging?)",
    "average_every=4294967296_replacement_ratio=0.2_restart_init_mode=random_use_random_strategies=False_use_restarts=False_use_selective_averaging=False": "Baseline (4B steps Averaging?)",
}


def fetch_wandb_runs(project_name="torchgfn/torchgfn"):
    """Fetch all runs from the specified wandb project."""
    print(f"Fetching runs from project: {project_name}")

    # Initialize wandb API with increased timeout
    api = wandb.Api(timeout=60)  # 60 second timeout to handle large projects

    # Fetch all runs from the project
    runs = api.runs(project_name)
    runs_list = list(runs)
    print(f"Found {len(runs_list)} runs in the project")

    print("\nFirst few run names:")
    for run in runs_list[:5]:
        print(f"- {run.name} (ID: {run.id}) - State: {run.state}")

    return runs_list


def analyze_environment_configurations(runs_list):
    """Analyze environment configurations for Hypergrid.

    Config is based on config.R0, config.R1, config.R2, config.height, config.ndim.
    """
    print("\n=== ENVIRONMENT CONFIGURATION ANALYSIS ===")

    env_config_vars = ["R0", "R1", "R2", "height", "ndim"]
    run_to_env_config = {}
    env_config_runs = {}
    env_config_details = {}

    for run in runs_list:
        config = getattr(run, "config", {}) or {}
        env_config = tuple(config.get(var, None) for var in env_config_vars)

        # Create a readable identifier for this environment configuration
        env_config_id = f"R0={env_config[0]}_R1={env_config[1]}_R2={env_config[2]}_height={env_config[3]}_ndim={env_config[4]}"

        run_to_env_config[run.id] = env_config_id

        if env_config_id not in env_config_runs:
            env_config_runs[env_config_id] = []
            env_config_details[env_config_id] = env_config

        env_config_runs[env_config_id].append(run.id)

    print(f"Found {len(env_config_runs)} unique environment configurations:")
    for env_config_id, run_ids in sorted(env_config_runs.items()):
        print(f"- {env_config_id}: {len(run_ids)} runs")
        # Show first few run IDs for this config
        if len(run_ids) <= 3:
            print(f"  Run IDs: {run_ids}")
        else:
            print(f"  Run IDs: {run_ids[:3]}... ({len(run_ids)} total)")

    # Check environment configuration status distribution
    print("\nEnvironment configuration status analysis:")
    for env_config_id, run_ids in sorted(env_config_runs.items()):
        env_config_run_objects = [run for run in runs_list if run.id in run_ids]
        states = [run.state for run in env_config_run_objects]
        state_counts = pd.Series(states).value_counts()
        print(f"- {env_config_id}: {dict(state_counts)}")

    return run_to_env_config, env_config_runs, env_config_details


def create_hierarchical_structure(runs_list, run_to_env_config, run_to_community):
    """
    Create hierarchical structure: Environment Groups → Community Groups → Runs

    Returns:
    - env_to_communities: dict mapping environment_config_id to list of community_ids
    - community_to_runs: dict mapping community_id to list of run_ids
    - env_community_runs: nested dict[env_config_id][community_id] = list of runs
    """
    env_to_communities = {}
    community_to_runs = {}
    env_community_runs = {}

    for run in runs_list:
        env_config_id = run_to_env_config.get(run.id)
        community_id = run_to_community.get(run.id)

        if env_config_id and community_id:
            # Build environment → communities mapping
            if env_config_id not in env_to_communities:
                env_to_communities[env_config_id] = set()
            env_to_communities[env_config_id].add(community_id)

            # Build community → runs mapping
            if community_id not in community_to_runs:
                community_to_runs[community_id] = []
            community_to_runs[community_id].append(run.id)

            # Build nested env → community → runs mapping
            if env_config_id not in env_community_runs:
                env_community_runs[env_config_id] = {}
            if community_id not in env_community_runs[env_config_id]:
                env_community_runs[env_config_id][community_id] = []
            env_community_runs[env_config_id][community_id].append(run)

    # Convert sets to sorted lists for consistency
    for env_id in env_to_communities:
        env_to_communities[env_id] = sorted(env_to_communities[env_id])

    print("\n=== HIERARCHICAL STRUCTURE ANALYSIS ===")
    print(f"Environments: {len(env_to_communities)}")
    print(f"Total Communities: {len(community_to_runs)}")

    for env_id, communities in sorted(env_to_communities.items()):
        print(f"\nEnvironment {env_id}:")
        print(f"  Communities: {len(communities)}")
        for community_id in communities:
            runs_in_community = len(community_to_runs[community_id])
            finished_runs = sum(
                1
                for run in env_community_runs[env_id][community_id]
                if run.state == "finished"
            )
            print(
                f"    {community_id}: {runs_in_community} runs ({finished_runs} finished)"
            )

    return env_to_communities, community_to_runs, env_community_runs


def analyze_groups(runs_list):
    """Analyze group structure from run data."""
    print("\n=== WANDB GROUP ANALYSIS ===")
    run_groups = {}
    group_runs = {}

    for run in runs_list:
        group_id = getattr(run, "group", None)
        if group_id:
            run_groups[run.id] = group_id
            if group_id not in group_runs:
                group_runs[group_id] = []
            group_runs[group_id].append(run.id)

    print(f"Found {len(group_runs)} unique wandb groups:")
    for group_id, run_ids in sorted(group_runs.items()):
        print(f"- Group {group_id}: {len(run_ids)} runs")
        # Show first few run IDs for this group
        if len(run_ids) <= 3:
            print(f"  Run IDs: {run_ids}")
        else:
            print(f"  Run IDs: {run_ids[:3]}... ({len(run_ids)} total)")

    # Check group status distribution
    if group_runs:
        print("\nGroup status analysis:")
        for group_id, run_ids in sorted(group_runs.items()):
            group_run_objects = [run for run in runs_list if run.id in run_ids]
            states = [run.state for run in group_run_objects]
            state_counts = pd.Series(states).value_counts()
            print(f"- Group {group_id}: {dict(state_counts)}")

        # Check for group-specific metrics in history
        print("\nChecking for group-specific metrics in sample run...")
        sample_group = list(group_runs.keys())[0]
        sample_runs = [run for run in runs_list if run.id in group_runs[sample_group]]
        sample_run = sample_runs[0] if sample_runs else None
        if sample_run and sample_run.state in ["finished", "crashed"]:
            try:
                history = sample_run.history()
                if len(history) > 0:
                    # Look for all metrics
                    print(
                        f"  Sample run has {len(history.columns)} metrics: {sorted(list(history.columns))}"
                    )
                    # Look for n_modes_found related metrics
                    n_modes_metrics = [
                        col for col in history.columns if "n_modes_found" in col
                    ]
                    print(f"  n_modes_found related metrics: {n_modes_metrics}")
            except Exception as e:
                print(f"  Error checking history: {e}")

    return run_groups, group_runs


def print_environment_configurations(
    runs_list,
    env_config_runs,
    env_config_details,
    env_to_communities=None,
    community_to_runs=None,
):
    """Print detailed information about environment configurations."""
    print("\n=== ENVIRONMENT CONFIGURATIONS DETAILS ===")

    for env_config_id, run_ids in sorted(env_config_runs.items()):
        print(f"\nEnvironment Configuration: {env_config_id}")
        print(f"Number of runs: {len(run_ids)}")

        # Get run details
        config_runs = [run for run in runs_list if run.id in run_ids]
        states = [run.state for run in config_runs]
        state_counts = pd.Series(states).value_counts()

        print(f"Run states: {dict(state_counts)}")

        # Show community breakdown if available
        if (
            env_to_communities
            and community_to_runs
            and env_config_id in env_to_communities
        ):
            communities = env_to_communities[env_config_id]
            print(f"Communities in this environment: {len(communities)}")
            for community_id in communities:
                community_runs = community_to_runs.get(community_id, [])
                finished_count = sum(
                    1
                    for run_id in community_runs
                    for run in config_runs
                    if run.id == run_id and run.state == "finished"
                )
                print(
                    f"  {community_id}: {len(community_runs)} runs ({finished_count} finished)"
                )

        print(f"All run IDs: {sorted(run_ids)}")

        # Show additional config details if available
        if config_runs:
            sample_config = config_runs[0].config
            if sample_config:
                print("Full config details:")
                for key, value in sorted(sample_config.items()):
                    print(f"  {key}: {value}")


def compare_community_vs_environment_groupings(run_to_community, run_to_env_config):
    """Compare community groups (wandb) vs environment configurations."""
    print("\n=== COMMUNITY vs ENVIRONMENT GROUPING COMPARISON ===")

    # Check if groupings are the same (they shouldn't be)
    same_grouping = True
    for run_id in run_to_community:
        if run_id in run_to_env_config:
            if run_to_community[run_id] != run_to_env_config[run_id]:
                same_grouping = False
                break

    if same_grouping:
        print("⚠ Warning: Community groups and environment configurations are identical")
        print("   This suggests the grouping structure may not be set up correctly.")
    else:
        print(
            "✓ Community groups and environment configurations are different (as expected)"
        )
        print("   - Community groups represent agent collaboration groups")
        print("   - Environment configs represent different hypergrid setups")

    # Show some examples
    print("\nExamples of community vs environment groupings:")
    example_runs = list(run_to_community.keys())[:8]  # Show first 8
    for run_id in example_runs:
        if run_id in run_to_env_config:
            community = run_to_community[run_id]
            env_config = run_to_env_config[run_id]
            print(f"  Run {run_id}: Community={community}, Environment={env_config}")


def extract_run_data(runs_list):
    """Extract basic run information into dataframes."""
    run_data = []
    for run in runs_list:
        data = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "duration": run.summary.get("_runtime", None),
            "tags": run.tags,
            "config": run.config,
            "summary": (
                run.summary._json_dict
                if hasattr(run.summary, "_json_dict")
                else dict(run.summary)
            ),
        }
        run_data.append(data)

    # Create DataFrame
    df_runs = pd.DataFrame(run_data)
    print(f"DataFrame shape: {df_runs.shape}")
    print(f"\nColumns: {list(df_runs.columns)}")
    print("\nRun states:")
    assert isinstance(
        df_runs["state"].value_counts(), pd.Series  # type: ignore
    ), "state.value_counts() should be a pd.Series"
    print(df_runs["state"].value_counts())  # type: ignore

    # Extract all unique keys from summary metrics
    all_summary_keys = set()
    for summary in df_runs["summary"]:  # type: ignore
        if isinstance(summary, dict):
            all_summary_keys.update(summary.keys())
    print(f"Found {len(all_summary_keys)} unique summary metrics:")
    for key in sorted(all_summary_keys):
        print(f"- {key}")
    # Extract all unique config keys
    all_config_keys = set()
    for config in df_runs["config"]:  # type: ignore
        if isinstance(config, dict):
            all_config_keys.update(config.keys())
    print(f"\nFound {len(all_config_keys)} unique config parameters:")
    for key in sorted(all_config_keys):
        print(f"- {key}")

    # Flatten summary and config data
    summary_df = pd.json_normalize(df_runs["summary"])  # type: ignore
    config_df = pd.json_normalize(df_runs["config"])  # type: ignore
    # Combine with run metadata
    df_combined = pd.concat(  # type: ignore
        [
            df_runs[["run_id", "run_name", "state", "created_at", "duration", "tags"]],  # type: ignore
            config_df.add_prefix("config."),
            summary_df.add_prefix("summary."),
        ],
        axis=1,
    )
    print(f"Combined DataFrame shape: {df_combined.shape}")
    print("\nColumns:")
    for col in df_combined.columns:
        print(f"- {col}")

    return df_runs, df_combined


def plot_run_states_distribution(df_runs):
    """Plot distribution of run states."""
    plt.figure(figsize=(10, 6))
    df_runs["state"].value_counts().plot(kind="bar")
    plt.title("Distribution of Run States")
    plt.xlabel("State")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_run_durations(df_combined):
    """Plot run duration analysis for completed runs."""
    completed_runs = df_combined[df_combined["state"] == "finished"]
    if len(completed_runs) > 0 and "duration" in completed_runs.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(completed_runs["duration"].dropna(), bins=20, alpha=0.7)
        plt.title("Distribution of Run Durations (Completed Runs)")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        print(f"Average duration: {completed_runs['duration'].mean():.2f} seconds")
        print(f"Median duration: {completed_runs['duration'].median():.2f} seconds")
        print(f"Min duration: {completed_runs['duration'].min():.2f} seconds")
        print(f"Max duration: {completed_runs['duration'].max():.2f} seconds")
    else:
        print("No completed runs with duration data found")


def print_available_variables(  # noqa: C901
    runs_list, group_runs, check_availability=True
):
    """Print all available variables at run and group level.

    Args:
        runs_list: List of wandb run objects
        group_runs: Dict mapping group_id to list of run_ids
        check_availability: If True, check which variables are available across
                           multiple runs (useful for identifying rank-0 only metrics)
    """
    if not runs_list:
        print("No runs available to analyze variables")
        return

    print("\n=== AVAILABLE VARIABLES ===")

    # Get sample run for run-level variables
    sample_run = runs_list[0]

    print("\n--- RUN-LEVEL VARIABLES ---")

    # Basic run attributes
    print("Basic run attributes:")
    basic_attrs = ["id", "name", "state", "created_at", "group", "tags"]
    for attr in basic_attrs:
        if hasattr(sample_run, attr):
            value = getattr(sample_run, attr)
            if attr == "tags" and isinstance(value, list):
                print(f"  {attr}: {value} (list)")
            else:
                print(f"  {attr}: {type(value).__name__}")

    # Config variables
    print("\nConfig variables:")
    if hasattr(sample_run, "config") and sample_run.config:
        for key in sorted(sample_run.config.keys()):
            value = sample_run.config[key]
            if isinstance(value, (int, float)):
                print(f"  config.{key}: {type(value).__name__} = {value}")
            else:
                print(f"  config.{key}: {type(value).__name__}")

    # Summary variables
    print("\nSummary variables:")
    if hasattr(sample_run, "summary") and sample_run.summary:
        for key in sorted(sample_run.summary.keys()):
            value = sample_run.summary[key]
            if isinstance(value, (int, float)):
                print(f"  summary.{key}: {type(value).__name__} = {value}")
            else:
                print(f"  summary.{key}: {type(value).__name__}")

    # History variables (if available)
    print("\nHistory variables (from sample run):")
    try:
        history = sample_run.history()
        if len(history) > 0:
            for col in sorted(history.columns):
                sample_values = history[col].dropna()
                if len(sample_values) > 0:
                    sample_val = sample_values.iloc[0]
                    print(f"  history.{col}: {type(sample_val).__name__}")
                else:
                    print(f"  history.{col}: (no non-null values)")
    except Exception as e:
        print(f"  Error accessing history: {e}")

    # Check variable availability across multiple runs
    if check_availability and len(runs_list) > 1:
        print("\n--- VARIABLE AVAILABILITY ACROSS RUNS ---")
        print("(Checking first 20 runs to identify rank-0 only vs universal metrics)")

        # Sample up to 20 runs
        sample_runs = runs_list[:20]

        # Track summary key availability
        summary_key_counts = {}
        history_col_counts = {}
        runs_checked = 0

        for run in sample_runs:
            runs_checked += 1

            # Check summary keys
            if hasattr(run, "summary") and run.summary:
                for key in run.summary.keys():
                    summary_key_counts[key] = summary_key_counts.get(key, 0) + 1

            # Check history columns
            try:
                history = run.history()
                if len(history) > 0:
                    for col in history.columns:
                        history_col_counts[col] = history_col_counts.get(col, 0) + 1
            except Exception:
                pass

        # Report summary key availability
        print(f"\nSummary variables availability (checked {runs_checked} runs):")
        universal_summary = []
        partial_summary = []
        for key, count in sorted(summary_key_counts.items()):
            pct = count / runs_checked * 100
            if count == runs_checked:
                universal_summary.append(key)
                print(
                    f"  ✓ summary.{key}: {count}/{runs_checked} ({pct:.0f}%) - UNIVERSAL"
                )
            else:
                partial_summary.append((key, count, pct))

        if partial_summary:
            print("\n  Partial availability (likely rank-0 only):")
            for key, count, pct in partial_summary:
                print(f"    summary.{key}: {count}/{runs_checked} ({pct:.0f}%)")

        # Report history column availability
        print(f"\nHistory columns availability (checked {runs_checked} runs):")
        universal_history = []
        partial_history = []
        for col, count in sorted(history_col_counts.items()):
            pct = count / runs_checked * 100
            if count == runs_checked:
                universal_history.append(col)
                print(
                    f"  ✓ history.{col}: {count}/{runs_checked} ({pct:.0f}%) - UNIVERSAL"
                )
            else:
                partial_history.append((col, count, pct))

        if partial_history:
            print("\n  Partial availability (likely rank-0 only):")
            for col, count, pct in partial_history:
                print(f"    history.{col}: {count}/{runs_checked} ({pct:.0f}%)")

        # Recommend metrics for run length tracking
        print("\n--- RECOMMENDED METRICS FOR RUN LENGTH ---")
        length_candidates = ["_step", "_runtime", "iteration", "step", "epoch"]
        found_candidates = []

        for candidate in length_candidates:
            if candidate in universal_summary:
                found_candidates.append(f"summary.{candidate}")
            if candidate in universal_history:
                found_candidates.append(f"history.{candidate} (use max)")

        if found_candidates:
            print("  Recommended (universally available):")
            for c in found_candidates:
                print(f"    ✓ {c}")
        else:
            print("  No standard iteration metrics found universally.")
            print("  Consider using 'total_history_rows' as a proxy.")

    # Group-level variables (aggregated)
    print("\n--- GROUP-LEVEL VARIABLES ---")
    print("Group-level variables are aggregations of run-level variables:")

    # Get sample group
    if group_runs:
        sample_group_id = list(group_runs.keys())[0]
        sample_group_runs = [
            run for run in runs_list if getattr(run, "group", None) == sample_group_id
        ]

        print(f"\nSample group '{sample_group_id}' with {len(sample_group_runs)} runs:")

        # Group-level aggregations we can compute
        group_vars = [
            "run_count: int (number of runs in group)",
            "finished_runs: int (number of finished runs)",
            "failed_runs: int (number of failed runs)",
            "crashed_runs: int (number of crashed runs)",
            "avg_max_modes: float (average of run-level max_modes)",
            "min_max_modes: float (minimum of run-level max_modes)",
            "max_max_modes: float (maximum of run-level max_modes)",
            "avg_iterations_to_max: float (average iterations to reach max modes)",
            "group_efficiency: float (finished_runs / total_runs)",
        ]

        for var in group_vars:
            print(f"  {var}")

    print("\n--- DERIVED GROUP METRICS ---")
    derived_vars = [
        "convergence_rate: float (runs reaching target modes / total runs)",
        "stability_score: float (consistency across runs in group)",
        "performance_rank: int (relative ranking vs other groups)",
        "time_to_convergence: float (average time to max modes)",
    ]

    for var in derived_vars:
        print(f"  {var}")


def print_experiment_summary(df_combined):
    """Print overall experiment summary."""
    print("=== EXPERIMENT SUMMARY ===")
    print(f"Total runs: {len(df_combined)}")
    print(f"Completed runs: {len(df_combined[df_combined['state'] == 'finished'])}")
    print(f"Running runs: {len(df_combined[df_combined['state'] == 'running'])}")
    print(f"Failed runs: {len(df_combined[df_combined['state'] == 'failed'])}")
    print(f"DataFrame shape: {df_combined.shape}")
    print(
        f"Available config parameters: {len([col for col in df_combined.columns if col.startswith('config.')])}"
    )
    print(
        f"Available summary metrics: {len([col for col in df_combined.columns if col.startswith('summary.')])}"
    )
    # Show date range
    if len(df_combined) > 0:
        df_combined["created_at_dt"] = pd.to_datetime(df_combined["created_at"])
        print(
            f"Date range: {df_combined['created_at_dt'].min()} to {df_combined['created_at_dt'].max()}"
        )


def extract_strategy_from_runs(runs):
    """Extract strategy configuration from a list of runs."""
    # Get strategy from first run (assuming all runs in community have same strategy)
    if not runs:
        return None

    sample_run = runs[0]
    config = getattr(sample_run, "config", {}) or {}

    strategy_values = []
    for param in STRATEGY_PARAMS:
        value = config.get(param)
        # Convert to string for consistent hashing/comparison
        if isinstance(value, bool):
            strategy_values.append(f"{param}={value}")
        elif isinstance(value, (int, float)):
            strategy_values.append(f"{param}={value}")
        elif value is None:
            strategy_values.append(f"{param}=None")
        else:
            strategy_values.append(f"{param}={str(value)}")

    strategy_id = "_".join(strategy_values)
    return strategy_id


def get_community_size(community_runs):
    """Get the size of a community (number of runs/agents)."""
    return len(community_runs)


def format_strategy_for_legend(strategy_id):
    """Format strategy ID for clean legend display.

    Uses STRATEGY_MAPPING if available, otherwise falls back to auto-formatting.
    """
    if not strategy_id:
        return "unknown"

    # First check if there's a manual mapping defined
    if strategy_id in STRATEGY_MAPPING:
        return STRATEGY_MAPPING[strategy_id]

    # Fallback: auto-format with full readable parameter names
    parts = strategy_id.split("_")
    key_values = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            key_values[key] = value

    # Format with full readable parameter names
    formatted_parts = []
    if "average_every" in key_values:
        formatted_parts.append(f'average_every={key_values["average_every"]}')
    if "replacement_ratio" in key_values:
        formatted_parts.append(f'replacement_ratio={key_values["replacement_ratio"]}')
    if "use_selective_averaging" in key_values:
        val = key_values["use_selective_averaging"]
        formatted_parts.append(f"selective_avg={val}")
    if "use_restarts" in key_values:
        formatted_parts.append(f'restarts={key_values["use_restarts"]}')
    if "restart_init_mode" in key_values:
        formatted_parts.append(f'restart_mode={key_values["restart_init_mode"]}')

    if formatted_parts:
        return ", ".join(formatted_parts)
    else:
        # Fallback: return full strategy ID
        return strategy_id


def analyze_communities_within_environments(  # noqa: C901
    env_community_runs, exit_after_printing_strategies=False
):
    """Analyze and compare communities within each environment.

    Args:
        env_community_runs: Dict mapping environment configs to community runs
        exit_after_printing_strategies: If True, exit after printing strategy summary
                                        (useful for updating STRATEGY_MAPPING)
    """
    print("\n=== COMMUNITY ANALYSIS WITHIN ENVIRONMENTS ===")

    # =========================================================================
    # FIRST PASS: Collect all unique strategies across all environments
    # =========================================================================
    all_strategies = set()
    for env_config_id, community_runs in env_community_runs.items():
        for community_id, runs in community_runs.items():
            strategy = extract_strategy_from_runs(runs)
            if strategy:
                all_strategies.add(strategy)

    # Print all strategies and highlight unmapped ones
    print(f"\n{'='*60}")
    print("ALL UNIQUE STRATEGIES FOUND IN WANDB")
    print(f"{'='*60}")
    print(f"Total unique strategies: {len(all_strategies)}")
    print()

    mapped_strategies = []
    unmapped_strategies = []

    for strategy in sorted(all_strategies):
        if strategy in STRATEGY_MAPPING:
            mapped_strategies.append(strategy)
        else:
            unmapped_strategies.append(strategy)

    if mapped_strategies:
        print("✓ MAPPED STRATEGIES (have shorthand names):")
        for strategy in mapped_strategies:
            print(f"  ✓ '{strategy}'")
            print(f"     → '{STRATEGY_MAPPING[strategy]}'")
        print()

    if unmapped_strategies:
        print("⚠ UNMAPPED STRATEGIES (need shorthand names in STRATEGY_MAPPING):")
        for strategy in unmapped_strategies:
            print(f"  ⚠ '{strategy}'")
        print()
        print(
            "Add these to STRATEGY_MAPPING at the top of the script to define shorthand names."
        )
        print()

    print(f"{'='*60}\n")

    if exit_after_printing_strategies:
        print("--print-strategies-only flag set. Exiting after strategy summary.")
        return

    # =========================================================================
    # Create GLOBAL linestyle and marker mappings for consistency across plots
    # =========================================================================
    available_linestyles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 2)),
        (0, (1, 1)),
    ]
    available_markers = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]

    # Sort strategies consistently (by their mapped name if available, else raw)
    sorted_strategies = sorted(
        all_strategies, key=lambda s: STRATEGY_MAPPING.get(s, s) or s
    )

    global_strategy_linestyle_map = {}
    global_strategy_marker_map = {}
    for i, strategy in enumerate(sorted_strategies):
        global_strategy_linestyle_map[strategy] = available_linestyles[
            i % len(available_linestyles)
        ]
        global_strategy_marker_map[strategy] = available_markers[
            i % len(available_markers)
        ]

    print("Strategy visual encoding (consistent across all plots):")
    for strategy in sorted_strategies:
        name = STRATEGY_MAPPING.get(strategy, strategy[:40] + "...")
        ls = global_strategy_linestyle_map[strategy]
        marker = global_strategy_marker_map[strategy]
        ls_name = ls if isinstance(ls, str) else "custom"
        print(f"  • {name}: linestyle='{ls_name}', marker='{marker}'")
    print()
    # =========================================================================

    for env_config_id, community_runs in sorted(env_community_runs.items()):
        print(f"\n{'='*60}")
        print(f"Environment: {env_config_id}")
        print(f"{'='*60}")

        if len(community_runs) <= 1:
            print("Only one community in this environment - skipping comparison")
            continue

        # Collect n_modes_found data and extract community metadata
        community_data = {}
        community_metadata = {}  # community_id -> {'size': int, 'strategy': str}

        for community_id, runs in community_runs.items():
            community_data[community_id] = []
            community_size = get_community_size(runs)
            community_strategy = extract_strategy_from_runs(runs)

            community_metadata[community_id] = {
                "size": community_size,
                "strategy": community_strategy,
            }

            print(
                f"\nCommunity {community_id} (Size: {community_size}, Strategy: {format_strategy_for_legend(community_strategy)}):"
            )

            for run in runs:
                try:
                    history = run.history(keys=["n_modes_found"])
                    if len(history) > 0 and "n_modes_found" in history.columns:
                        n_modes_values = history["n_modes_found"].dropna()
                        if len(n_modes_values) > 0:
                            n_modes_values = n_modes_values[:1000]  # Limit for memory
                            steps = list(range(len(n_modes_values)))

                            run_data = {
                                "run_id": run.id,
                                "run_name": run.name,
                                "run_state": run.state,
                                "steps": steps,
                                "n_modes_found": n_modes_values.tolist(),
                                "max_modes": float(max(n_modes_values)),
                                "iterations_to_max": len(n_modes_values),
                            }
                            community_data[community_id].append(run_data)

                            status_indicator = "✓" if run.state == "finished" else "✗"
                            print(
                                f"  {status_indicator} {run.name}: {len(n_modes_values)} points, max_modes={run_data['max_modes']:.0f}"
                            )

                except Exception as e:
                    print(f"  ✗ {run.name}: Error - {e}")

        # Plot communities within this environment with size/strategy coding
        plot_communities_in_environment(
            env_config_id,
            community_data,
            community_metadata,
            global_strategy_linestyle_map,
            global_strategy_marker_map,
        )


def plot_communities_in_environment(  # noqa: C901
    env_config_id,
    community_data,
    community_metadata,
    strategy_linestyle_map,
    strategy_marker_map,
):
    """Plot n_modes_found progression for communities within a single environment.

    Uses color for community size, linestyle and markers for strategy.
    Linestyle and marker mappings are passed in to ensure consistency across plots.
    """
    if not community_data:
        return

    communities_with_data = {cid: data for cid, data in community_data.items() if data}

    if len(communities_with_data) <= 1:
        print(f"Only {len(communities_with_data)} communities with data - skipping plot")
        return

    # Create figure with explicit layout: 70% plot, 30% legend space
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[70, 30], hspace=0.08)
    ax = fig.add_subplot(gs[0])

    # Extract unique community sizes
    community_sizes = sorted(
        set(community_metadata[cid]["size"] for cid in communities_with_data.keys())
    )

    # Color mapping for community sizes
    size_colors = cm.tab20(np.linspace(0, 1, len(community_sizes)))
    size_color_map = dict(zip(community_sizes, size_colors))

    # Style mapping for run states (secondary modifier)
    state_alphas = {"finished": 1.0, "crashed": 0.7, "failed": 0.5, "running": 0.8}

    max_y = 0
    legend_handles = []
    legend_labels = []

    for community_id in sorted(communities_with_data.keys()):
        runs_data = communities_with_data[community_id]
        if not runs_data:
            continue

        community_size = community_metadata[community_id]["size"]
        community_strategy = community_metadata[community_id]["strategy"]

        # Get color for community size
        base_color = size_color_map[community_size]

        # Get linestyle and marker for strategy (from global mappings)
        base_linestyle = strategy_linestyle_map.get(community_strategy, "-")
        base_marker = strategy_marker_map.get(community_strategy, "o")

        for run_data in runs_data:
            steps = run_data["steps"]
            n_modes = run_data["n_modes_found"]
            run_state = run_data["run_state"]

            # Apply state modifier (alpha)
            alpha = state_alphas.get(run_state, 0.6)

            # Create label for legend
            strategy_short = (
                community_strategy.replace("_", ", ")
                if community_strategy
                else "unknown"
            )
            label = f"Size {community_size}, Strategy: {strategy_short} ({run_state})"

            # Determine marker frequency based on data length (show ~10-15 markers per line)
            markevery = max(1, len(steps) // 12)

            (line,) = ax.plot(
                steps,
                n_modes,
                linestyle=base_linestyle,
                color=base_color,
                alpha=alpha,
                linewidth=1.2,
                label=label,
                marker=base_marker,
                markersize=9,
                markevery=markevery,
            )

            # Only add to legend if we haven't seen this combination before
            legend_key = (
                f"size_{community_size}_strategy_{community_strategy}_{run_state}"
            )
            if legend_key not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(legend_key)

            max_y = max(max_y, max(n_modes) if n_modes else 0)

    # Clean up the environment config ID for title
    env_title = env_config_id.replace("_", ", ")
    ax.set_title(
        f"Mode Discovery: {env_title}\nCommunity Size (📏) × Strategy (📊) Analysis",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Iteration/Step", fontsize=12)
    ax.set_ylabel("Number of Modes Found", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Create separate legends for colors (sizes) and linestyles (strategies)
    if legend_handles:
        # Create color legend for community sizes
        size_handles = []
        size_labels = []
        size_colors_used = set()

        # Create linestyle legend for strategies
        strategy_handles = []
        strategy_labels = []
        strategy_linestyles_used = set()

        # First pass: collect all unique sizes and strategies (prefer finished, but include all)
        all_sizes_seen = {}  # size -> best_state (finished > running > crashed > failed)
        all_strategies_seen = {}  # strategy -> best_state

        state_priority = {"finished": 0, "running": 1, "crashed": 2, "failed": 3}

        for handle, label_key in zip(legend_handles, legend_labels):
            parts = label_key.split("_strategy_")
            if len(parts) != 2:
                continue
            size_part = parts[0].replace("size_", "")
            strategy_state_part = parts[1]
            strategy_state_parts = strategy_state_part.rsplit("_", 1)
            if len(strategy_state_parts) != 2:
                continue
            strategy_part, state_part = strategy_state_parts

            # Track best state for each size
            if size_part not in all_sizes_seen:
                all_sizes_seen[size_part] = state_part
            elif state_priority.get(state_part, 99) < state_priority.get(
                all_sizes_seen[size_part], 99
            ):
                all_sizes_seen[size_part] = state_part

            # Track best state for each strategy
            if strategy_part not in all_strategies_seen:
                all_strategies_seen[strategy_part] = state_part
            elif state_priority.get(state_part, 99) < state_priority.get(
                all_strategies_seen[strategy_part], 99
            ):
                all_strategies_seen[strategy_part] = state_part

        # Build size legend entries for ALL sizes
        for size_part in all_sizes_seen.keys():
            if size_part not in size_colors_used:
                size_key = f"Size {size_part}"
                size_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=size_color_map[int(size_part)],
                        linewidth=3,
                        label=size_key,
                    )
                )
                size_labels.append(size_key)
                size_colors_used.add(size_part)

        # Build strategy legend entries for ALL strategies (with linestyle AND marker)
        # Use longer line segment [0, 0.5, 1] so linestyle pattern is visible
        for strategy_part in all_strategies_seen.keys():
            if (
                strategy_part not in strategy_linestyles_used
                and strategy_part in strategy_linestyle_map
            ):
                strategy_key = format_strategy_for_legend(strategy_part)
                strategy_handles.append(
                    Line2D(
                        [0, 0.5, 1],
                        [0, 0, 0],
                        color="black",
                        linestyle=strategy_linestyle_map[strategy_part],
                        marker=strategy_marker_map.get(strategy_part, "o"),
                        markersize=10,
                        linewidth=1.5,
                        label=strategy_key,
                    )
                )
                strategy_labels.append(strategy_key)
                strategy_linestyles_used.add(strategy_part)

        # Sort size labels in ascending order
        size_order = sorted(size_labels, key=lambda x: int(x.split()[1]))
        size_handles_sorted = []
        size_labels_sorted = []
        for label in size_order:
            idx = size_labels.index(label)
            size_handles_sorted.append(size_handles[idx])
            size_labels_sorted.append(size_labels[idx])
        size_handles = size_handles_sorted
        size_labels = size_labels_sorted

        # Create legend axes in the bottom gridspec slot
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis("off")

        # Create THREE separate legends stacked vertically:
        # 1. Sizes (horizontal, compact) at top
        # 2. Strategies (one per line, full width) in middle
        # 3. Run states (horizontal) at bottom

        # Legend 1: Community Sizes (horizontal row)
        if size_handles:
            size_legend = legend_ax.legend(
                size_handles,
                [f"📏 {label}" for label in size_labels],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                fontsize=10,
                title="Community Sizes",
                title_fontsize=11,
                ncol=len(size_handles),
                frameon=True,
                fancybox=True,
                columnspacing=2.0,
                handletextpad=0.5,
            )
            legend_ax.add_artist(size_legend)

        # Legend 2: Strategies (single column, full width for long labels)
        # Use handlelength=4 to show linestyle pattern clearly alongside markers
        if strategy_handles:
            strategy_legend = legend_ax.legend(
                strategy_handles,
                [f"📊 {label}" for label in strategy_labels],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.65),
                fontsize=10,
                title="Strategies",
                title_fontsize=11,
                ncol=1,
                frameon=True,
                fancybox=True,
                handlelength=4,
                handletextpad=0.8,
            )
            legend_ax.add_artist(strategy_legend)

        # Legend 3: Run states (horizontal row at bottom)
        state_handles = [
            Line2D([0], [0], color="gray", linestyle="-", alpha=1.0, linewidth=3),
            Line2D([0], [0], color="gray", linestyle="-", alpha=0.5, linewidth=3),
        ]
        state_labels = ["Solid opacity: finished", "Faded opacity: crashed/failed"]
        legend_ax.legend(
            state_handles,
            state_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.15),
            fontsize=9,
            title="Run States",
            title_fontsize=10,
            ncol=2,
            frameon=True,
            fancybox=True,
            columnspacing=3.0,
            handletextpad=0.5,
        )

    # Set y-axis limit with some padding
    if max_y > 0:
        ax.set_ylim(0, max_y * 1.1)

    plt.tight_layout()
    plt.show()

    # Print community comparison statistics with size/strategy breakdown
    print(f"\nCommunity Performance Summary for {env_config_id}:")

    # Group by size and strategy
    size_strategy_stats = {}
    for community_id in sorted(communities_with_data.keys()):
        runs_data = communities_with_data[community_id]
        metadata = community_metadata[community_id]

        size = metadata["size"]
        strategy = metadata["strategy"] or "unknown"

        key = f"Size {size}, Strategy: {format_strategy_for_legend(strategy)}"

        if key not in size_strategy_stats:
            size_strategy_stats[key] = {"finished": [], "total": 0}

        size_strategy_stats[key]["total"] += len(runs_data)
        finished_runs = [r for r in runs_data if r["run_state"] == "finished"]
        size_strategy_stats[key]["finished"].extend(
            [r["max_modes"] for r in finished_runs]
        )

    for group_key, stats in sorted(size_strategy_stats.items()):
        finished_count = len(stats["finished"])
        total_count = stats["total"]

        if finished_count > 0:
            avg_max_modes = sum(stats["finished"]) / finished_count
            min_max = min(stats["finished"])
            max_max = max(stats["finished"])
            print(
                f"  {group_key}: {finished_count}/{total_count} finished, "
                f"avg max modes = {avg_max_modes:.1f} (range: {min_max:.0f}-{max_max:.0f})"
            )
        else:
            print(f"  {group_key}: {total_count} runs, 0 finished")


def identify_runs_for_deletion(  # noqa: C901
    runs_list,
    run_to_community,
    min_steps: int = 100,
    include_states: list[str] | None = None,
):
    """Identify runs that should be deleted from wandb.

    This function identifies runs that are likely incomplete or failed experiments
    based on the _step metric from run summary (universally available on all ranks).

    Args:
        runs_list: List of wandb run objects
        run_to_community: Dict mapping run_id to community_id
        min_steps: Minimum number of steps (_step from summary) required.
                   Runs with fewer steps are flagged for deletion.
                   This metric is available on ALL ranks, not just rank-0.
        include_states: List of run states to consider for deletion.
                        Default: ["crashed", "failed"]. Use ["all"] to include
                        all states (including "finished" runs with insufficient data).

    Returns:
        dict with:
            - runs_to_delete: List of run info dicts
            - communities_to_delete: List of community IDs where ALL runs should be deleted
            - summary: Summary statistics
    """
    if include_states is None:
        include_states = ["crashed", "failed"]

    include_all_states = "all" in include_states

    print(f"\n{'='*60}")
    print("IDENTIFYING RUNS FOR DELETION")
    print(f"{'='*60}")
    print(f"Minimum steps threshold (_step): {min_steps}")
    print(f"States to consider: {include_states}")
    print()

    runs_to_delete = []
    runs_analyzed = 0
    runs_with_errors = 0

    # Track per-community statistics
    community_stats = {}  # community_id -> {total: int, flagged: int, run_ids: list}

    for run in runs_list:
        run_id = run.id
        run_name = run.name
        run_state = run.state
        community_id = run_to_community.get(run_id, "unknown")

        # Initialize community stats
        if community_id not in community_stats:
            community_stats[community_id] = {
                "total": 0,
                "flagged": 0,
                "flagged_runs": [],
                "all_runs": [],
            }
        community_stats[community_id]["total"] += 1
        community_stats[community_id]["all_runs"].append(run_id)

        # Check if this run's state should be considered
        if not include_all_states and run_state not in include_states:
            continue

        runs_analyzed += 1

        # Try to get n_modes_found data length and run-level metrics
        n_datapoints = 0
        total_history_rows = 0
        available_columns = []
        error_msg = None

        # Extract run-level iteration metrics from summary (available on all ranks)
        summary_step = None
        summary_runtime = None
        if hasattr(run, "summary") and run.summary:
            # _step is the last step logged (universally available)
            summary_step = run.summary.get("_step")
            # _runtime is the total runtime in seconds
            summary_runtime = run.summary.get("_runtime")

        try:
            # First get full history to see what columns exist
            history = run.history()
            total_history_rows = len(history)
            available_columns = list(history.columns) if len(history) > 0 else []

            if "n_modes_found" in available_columns:
                n_modes_values = history["n_modes_found"].dropna()
                n_datapoints = len(n_modes_values)
        except Exception as e:
            error_msg = str(e)
            runs_with_errors += 1

        # Determine if this run should be deleted based on run length
        # Use _step from summary as it's universally available on all ranks
        should_delete = False
        deletion_reason = []

        if error_msg:
            should_delete = True
            deletion_reason.append(f"error accessing data: {error_msg}")
        elif summary_step is None:
            should_delete = True
            deletion_reason.append("no _step in summary (run may not have started)")
        elif summary_step < min_steps:
            should_delete = True
            deletion_reason.append(f"insufficient steps ({summary_step} < {min_steps})")

        if should_delete:
            run_info = {
                "run_id": run_id,
                "run_name": run_name,
                "state": run_state,
                "community_id": community_id,
                "n_datapoints": n_datapoints,
                "total_history_rows": total_history_rows,
                "summary_step": summary_step,  # Last step logged (universal)
                "summary_runtime": summary_runtime,  # Runtime in seconds (universal)
                "has_n_modes_found": "n_modes_found" in available_columns,
                "available_columns": available_columns,
                "reason": "; ".join(deletion_reason),
            }
            runs_to_delete.append(run_info)
            community_stats[community_id]["flagged"] += 1
            community_stats[community_id]["flagged_runs"].append(run_id)

    # Identify communities where ALL runs should be deleted
    communities_to_delete = []
    for community_id, stats in community_stats.items():
        if stats["total"] > 0 and stats["flagged"] == stats["total"]:
            communities_to_delete.append(community_id)

    # Build summary
    summary = {
        "total_runs_in_project": len(runs_list),
        "runs_analyzed": runs_analyzed,
        "runs_with_errors": runs_with_errors,
        "runs_flagged_for_deletion": len(runs_to_delete),
        "communities_fully_flagged": len(communities_to_delete),
        "total_communities": len(community_stats),
    }

    # Print summary
    print("=" * 60)
    print("DELETION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total runs in project: {summary['total_runs_in_project']}")
    print(f"Runs analyzed (matching state filter): {summary['runs_analyzed']}")
    print(f"Runs with data access errors: {summary['runs_with_errors']}")
    print(f"Runs flagged for deletion: {summary['runs_flagged_for_deletion']}")
    print(
        f"Communities fully flagged: {summary['communities_fully_flagged']} / {summary['total_communities']}"
    )
    print()

    # Print runs flagged for deletion, grouped by community
    if runs_to_delete:
        print("=" * 60)
        print("RUNS FLAGGED FOR DELETION (grouped by community)")
        print("=" * 60)

        # Group by community
        runs_by_community = {}
        for run_info in runs_to_delete:
            cid = run_info["community_id"]
            if cid not in runs_by_community:
                runs_by_community[cid] = []
            runs_by_community[cid].append(run_info)

        for community_id in sorted(runs_by_community.keys()):
            community_run_list = runs_by_community[community_id]
            stats = community_stats[community_id]
            all_flagged = community_id in communities_to_delete

            status_marker = "🗑️  FULL DELETION" if all_flagged else "⚠️  PARTIAL"
            print(
                f"\n{status_marker} Community: {community_id} "
                f"({stats['flagged']}/{stats['total']} runs flagged)"
            )

            for run_info in community_run_list:
                print(f"  • {run_info['run_name']} (ID: {run_info['run_id']})")
                has_metric = run_info.get("has_n_modes_found", "?")
                total_rows = run_info.get("total_history_rows", "?")
                summary_step = run_info.get("summary_step", "N/A")
                summary_runtime = run_info.get("summary_runtime")
                runtime_str = (
                    f"{summary_runtime:.1f}s" if summary_runtime is not None else "N/A"
                )
                print(
                    f"    State: {run_info['state']}, "
                    f"n_modes_found: {run_info['n_datapoints']}, "
                    f"history_rows: {total_rows}"
                )
                print(
                    f"    Run length: _step={summary_step}, "
                    f"_runtime={runtime_str}, "
                    f"has n_modes_found: {has_metric}"
                )
                print(f"    Reason: {run_info['reason']}")
                # Show available columns if n_modes_found is missing
                if not run_info.get("has_n_modes_found", True):
                    cols = run_info.get("available_columns", [])
                    if cols:
                        # Show first 10 columns to avoid spam
                        cols_preview = cols[:10]
                        suffix = f"... (+{len(cols)-10} more)" if len(cols) > 10 else ""
                        print(f"    Available columns: {cols_preview}{suffix}")
                    else:
                        print("    Available columns: (none - empty history)")

        # Print run IDs for easy copy-paste deletion
        print()
        print("=" * 60)
        print("RUN IDs FOR DELETION (copy-paste ready)")
        print("=" * 60)
        run_ids = [r["run_id"] for r in runs_to_delete]
        print(f"Total: {len(run_ids)} runs")
        print()
        print("# Python list format:")
        print(f"run_ids_to_delete = {run_ids}")
        print()
        print("# One per line:")
        for rid in run_ids:
            print(rid)

        # Print communities for full deletion
        if communities_to_delete:
            print()
            print("=" * 60)
            print("COMMUNITIES FOR FULL DELETION")
            print("=" * 60)
            print(f"Total: {len(communities_to_delete)} communities")
            print()
            for cid in sorted(communities_to_delete):
                stats = community_stats[cid]
                print(f"  • {cid} ({stats['total']} runs)")
    else:
        print("No runs flagged for deletion based on current criteria.")

    return {
        "runs_to_delete": runs_to_delete,
        "communities_to_delete": communities_to_delete,
        "community_stats": community_stats,
        "summary": summary,
    }


def collect_community_timeseries_data(  # noqa: C901
    env_community_runs,
    metrics: list[str] | None = None,
    max_steps: int = 1000,
):
    """Collect timeseries data for all agents within each community.

    This function gathers history data from all runs (agents) within each community,
    aligned by step index, to enable variance analysis across agents.

    Args:
        env_community_runs: Dict mapping env_config_id -> community_id -> list of runs
        metrics: List of metric names to collect. Default: ["n_modes_found", "n_new_modes", "novelty_sum"]
        max_steps: Maximum number of steps to collect per run (for memory efficiency)

    Returns:
        Dict with structure:
            env_config_id -> community_id -> {
                'metadata': {'size': int, 'strategy': str},
                'metrics': {
                    metric_name: {
                        'steps': [0, 1, 2, ...],
                        'agent_data': np.array of shape (n_agents, n_steps),
                        'mean': np.array of shape (n_steps,),
                        'std': np.array of shape (n_steps,),
                        'min': np.array of shape (n_steps,),
                        'max': np.array of shape (n_steps,),
                    }
                }
            }
    """
    if metrics is None:
        metrics = ["n_new_modes", "n_modes_found", "novelty_sum"]

    print(f"\n{'='*60}")
    print("COLLECTING COMMUNITY TIMESERIES DATA FOR VARIANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Metrics: {metrics}")
    print(f"Max steps per run: {max_steps}")
    print()

    all_data = {}

    for env_config_id, community_runs in sorted(env_community_runs.items()):
        all_data[env_config_id] = {}

        for community_id, runs in community_runs.items():
            community_size = get_community_size(runs)
            community_strategy = extract_strategy_from_runs(runs)

            community_result = {
                "metadata": {
                    "size": community_size,
                    "strategy": community_strategy,
                },
                "metrics": {},
            }

            # Collect data for each metric
            for metric in metrics:
                agent_series = []  # List of numpy arrays, one per agent
                agent_run_ids = []  # Track which runs contributed data

                for run in runs:
                    try:
                        history = run.history(keys=[metric])
                        if len(history) > 0 and metric in history.columns:
                            values = history[metric].dropna().values
                            if len(values) > 0:
                                # Limit to max_steps
                                values = values[:max_steps]
                                agent_series.append(values)
                                agent_run_ids.append(run.id)
                    except Exception:
                        pass  # Skip runs with errors

                if len(agent_series) >= 2:  # Need at least 2 agents for variance
                    # Find the minimum length across all agents
                    min_len = min(len(s) for s in agent_series)

                    # Truncate all series to the same length
                    aligned_data = np.array([s[:min_len] for s in agent_series])

                    # Calculate statistics across agents at each timestep
                    steps = np.arange(min_len)
                    mean_vals = np.mean(aligned_data, axis=0)
                    std_vals = np.std(aligned_data, axis=0)
                    min_vals = np.min(aligned_data, axis=0)
                    max_vals = np.max(aligned_data, axis=0)

                    community_result["metrics"][metric] = {
                        "steps": steps,
                        "agent_data": aligned_data,
                        "mean": mean_vals,
                        "std": std_vals,
                        "min": min_vals,
                        "max": max_vals,
                        "n_agents": len(agent_series),
                        "agent_run_ids": agent_run_ids,
                    }

            all_data[env_config_id][community_id] = community_result

    # Print summary
    total_communities = sum(len(v) for v in all_data.values())
    communities_with_data = sum(
        1
        for env_data in all_data.values()
        for comm_data in env_data.values()
        if comm_data["metrics"]
    )
    print(f"Total communities processed: {total_communities}")
    print(f"Communities with variance data (≥2 agents): {communities_with_data}")

    return all_data


def plot_community_variance_over_time(  # noqa: C901
    variance_data,
    metric: str = "n_new_modes",
    global_strategy_linestyle_map: dict | None = None,
    global_strategy_marker_map: dict | None = None,
):
    """Plot within-community variance (std dev) over time for a given metric.

    Creates plots showing how agent performance variance evolves within each community,
    allowing comparison across community sizes and strategies.

    Args:
        variance_data: Output from collect_community_timeseries_data()
        metric: Which metric to plot variance for
        global_strategy_linestyle_map: Strategy -> linestyle mapping for consistency
        global_strategy_marker_map: Strategy -> marker mapping for consistency
    """
    print(f"\n{'='*60}")
    print(f"PLOTTING WITHIN-COMMUNITY VARIANCE: {metric}")
    print(f"{'='*60}")

    # Build strategy visual mappings if not provided
    if global_strategy_linestyle_map is None or global_strategy_marker_map is None:
        all_strategies = set()
        for env_data in variance_data.values():
            for comm_data in env_data.values():
                strategy = comm_data["metadata"]["strategy"]
                if strategy:
                    all_strategies.add(strategy)

        available_linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
        available_markers = ["o", "s", "^", "D", "v", "p", "h", "*"]

        sorted_strategies = sorted(
            all_strategies, key=lambda s: STRATEGY_MAPPING.get(s, s) or s
        )
        global_strategy_linestyle_map = {
            s: available_linestyles[i % len(available_linestyles)]
            for i, s in enumerate(sorted_strategies)
        }
        global_strategy_marker_map = {
            s: available_markers[i % len(available_markers)]
            for i, s in enumerate(sorted_strategies)
        }

    for env_config_id, env_data in sorted(variance_data.items()):
        communities_with_metric = {
            cid: cdata for cid, cdata in env_data.items() if metric in cdata["metrics"]
        }

        if len(communities_with_metric) < 1:
            print(f"Environment {env_config_id}: No communities with {metric} data")
            continue

        print(f"\nEnvironment: {env_config_id}")
        print(f"Communities with {metric} data: {len(communities_with_metric)}")

        # Create figure with two subplots: std dev and coefficient of variation
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Collect unique sizes for color mapping
        community_sizes = sorted(
            set(cdata["metadata"]["size"] for cdata in communities_with_metric.values())
        )
        size_colors = cm.tab20(np.linspace(0, 1, max(len(community_sizes), 1)))
        size_color_map = dict(zip(community_sizes, size_colors))

        legend_entries = {}  # (size, strategy) -> handle

        for community_id, comm_data in sorted(communities_with_metric.items()):
            metadata = comm_data["metadata"]
            metric_data = comm_data["metrics"][metric]

            size = metadata["size"]
            strategy = metadata["strategy"]

            steps = metric_data["steps"]
            mean_vals = metric_data["mean"]
            std_vals = metric_data["std"]
            min_vals = metric_data["min"]
            max_vals = metric_data["max"]
            n_agents = metric_data["n_agents"]

            # Calculate coefficient of variation (std / mean), handling division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                cv_vals = np.where(mean_vals > 0, std_vals / mean_vals, 0)

            # Calculate range (max - min)
            range_vals = max_vals - min_vals

            # Get visual styling
            color = size_color_map[size]
            linestyle = global_strategy_linestyle_map.get(strategy, "-")
            marker = global_strategy_marker_map.get(strategy, "o")
            markevery = max(1, len(steps) // 15)

            # Plot 1: Standard Deviation over time
            ax1 = axes[0, 0]
            (line,) = ax1.plot(
                steps,
                std_vals,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=6,
                markevery=markevery,
                linewidth=1.2,
                alpha=0.8,
            )

            # Plot 2: Coefficient of Variation over time
            ax2 = axes[0, 1]
            ax2.plot(
                steps,
                cv_vals,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=6,
                markevery=markevery,
                linewidth=1.2,
                alpha=0.8,
            )

            # Plot 3: Range (max - min) over time
            ax3 = axes[1, 0]
            ax3.plot(
                steps,
                range_vals,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=6,
                markevery=markevery,
                linewidth=1.2,
                alpha=0.8,
            )

            # Plot 4: Mean with std envelope
            ax4 = axes[1, 1]
            ax4.plot(
                steps,
                mean_vals,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=6,
                markevery=markevery,
                linewidth=1.2,
                alpha=0.8,
            )
            ax4.fill_between(
                steps,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.15,
            )

            # Track for legend
            legend_key = (size, strategy)
            if legend_key not in legend_entries:
                legend_entries[legend_key] = line

        # Configure subplot 1: Standard Deviation
        ax1.set_title(
            f"Within-Community Std Dev of {metric}", fontsize=11, fontweight="bold"
        )
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Standard Deviation")
        ax1.grid(True, alpha=0.3)

        # Configure subplot 2: Coefficient of Variation
        ax2.set_title(
            f"Coefficient of Variation (CV) of {metric}", fontsize=11, fontweight="bold"
        )
        ax2.set_xlabel("Step")
        ax2.set_ylabel("CV (std / mean)")
        ax2.grid(True, alpha=0.3)

        # Configure subplot 3: Range
        ax3.set_title(f"Range (max - min) of {metric}", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Range")
        ax3.grid(True, alpha=0.3)

        # Configure subplot 4: Mean with envelope
        ax4.set_title(f"Mean ± Std Dev of {metric}", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Step")
        ax4.set_ylabel(metric)
        ax4.grid(True, alpha=0.3)

        # Create legend
        _add_variance_plot_legend(
            fig,
            legend_entries,
            size_color_map,
            global_strategy_linestyle_map,
            global_strategy_marker_map,
        )

        env_title = env_config_id.replace("_", ", ")
        fig.suptitle(
            f"Agent Variance Analysis: {env_title}\nMetric: {metric}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.show()

        # Print statistics summary
        print(f"\nVariance Statistics for {env_config_id}:")
        for community_id, comm_data in sorted(communities_with_metric.items()):
            metadata = comm_data["metadata"]
            metric_data = comm_data["metrics"][metric]

            size = metadata["size"]
            strategy_short = format_strategy_for_legend(metadata["strategy"])
            n_agents = metric_data["n_agents"]

            # Get final values (last timestep)
            final_std = metric_data["std"][-1]
            final_mean = metric_data["mean"][-1]
            avg_std = np.mean(metric_data["std"])

            print(
                f"  Size {size}, {strategy_short}: "
                f"n_agents={n_agents}, "
                f"final_std={final_std:.2f}, "
                f"final_mean={final_mean:.2f}, "
                f"avg_std={avg_std:.2f}"
            )


def _add_variance_plot_legend(
    fig, legend_entries, size_color_map, strategy_linestyle_map, strategy_marker_map
):
    """Add a combined legend for variance plots."""
    # Create size legend handles
    size_handles = []
    size_labels = []
    for size in sorted(size_color_map.keys()):
        size_handles.append(Line2D([0], [0], color=size_color_map[size], linewidth=3))
        size_labels.append(f"Size {size}")

    # Create strategy legend handles
    strategy_handles = []
    strategy_labels = []
    for strategy in sorted(
        strategy_linestyle_map.keys(), key=lambda s: STRATEGY_MAPPING.get(s, s) or s
    ):
        strategy_handles.append(
            Line2D(
                [0, 0.5, 1],
                [0, 0, 0],
                color="black",
                linestyle=strategy_linestyle_map[strategy],
                marker=strategy_marker_map.get(strategy, "o"),
                markersize=8,
                linewidth=1.5,
            )
        )
        strategy_labels.append(format_strategy_for_legend(strategy))

    # Add legends to figure
    if size_handles:
        size_legend = fig.legend(
            size_handles,
            size_labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            title="Community Size",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
        )
        fig.add_artist(size_legend)

    if strategy_handles:
        fig.legend(
            strategy_handles,
            strategy_labels,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            title="Strategy",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
        )


def plot_variance_comparison_by_size_and_strategy(  # noqa: C901
    variance_data,
    metric: str = "n_new_modes",
):
    """Create summary plots comparing variance metrics across sizes and strategies.

    Creates bar charts and box plots showing:
    - Average std dev by community size
    - Average std dev by strategy
    - Final std dev comparison

    Args:
        variance_data: Output from collect_community_timeseries_data()
        metric: Which metric to analyze
    """
    print(f"\n{'='*60}")
    print(f"VARIANCE COMPARISON BY SIZE AND STRATEGY: {metric}")
    print(f"{'='*60}")

    # Collect summary statistics across all environments
    all_stats = []

    for env_config_id, env_data in variance_data.items():
        for community_id, comm_data in env_data.items():
            if metric not in comm_data["metrics"]:
                continue

            metadata = comm_data["metadata"]
            metric_data = comm_data["metrics"][metric]

            all_stats.append(
                {
                    "env": env_config_id,
                    "community": community_id,
                    "size": metadata["size"],
                    "strategy": metadata["strategy"],
                    "strategy_short": format_strategy_for_legend(metadata["strategy"]),
                    "n_agents": metric_data["n_agents"],
                    "avg_std": np.mean(metric_data["std"]),
                    "final_std": metric_data["std"][-1],
                    "final_mean": metric_data["mean"][-1],
                    "max_std": np.max(metric_data["std"]),
                    "avg_cv": np.mean(
                        np.where(
                            metric_data["mean"] > 0,
                            metric_data["std"] / metric_data["mean"],
                            0,
                        )
                    ),
                }
            )

    if not all_stats:
        print("No variance data available for comparison")
        return

    df = pd.DataFrame(all_stats)

    # Create figure with comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Average Std Dev by Community Size
    ax1 = axes[0, 0]
    size_groups = (
        df.groupby("size")["avg_std"].agg(["mean", "std", "count"]).reset_index()
    )
    bars = ax1.bar(
        range(len(size_groups)),
        size_groups["mean"],
        yerr=size_groups["std"],
        capsize=5,
        alpha=0.7,
        color="steelblue",
    )
    ax1.set_xticks(range(len(size_groups)))
    ax1.set_xticklabels([f"Size {s}" for s in size_groups["size"]])
    ax1.set_ylabel("Average Std Dev")
    ax1.set_title(f"Variance by Community Size\n(metric: {metric})", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, size_groups["count"])):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={int(count)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Average Std Dev by Strategy
    ax2 = axes[0, 1]
    strategy_groups = (
        df.groupby("strategy_short")["avg_std"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    bars = ax2.barh(
        range(len(strategy_groups)),
        strategy_groups["mean"],
        xerr=strategy_groups["std"],
        capsize=5,
        alpha=0.7,
        color="coral",
    )
    ax2.set_yticks(range(len(strategy_groups)))
    ax2.set_yticklabels(strategy_groups["strategy_short"])
    ax2.set_xlabel("Average Std Dev")
    ax2.set_title(f"Variance by Strategy\n(metric: {metric})", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, strategy_groups["count"])):
        ax2.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" n={int(count)}",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Plot 3: Box plot of final std dev by size
    ax3 = axes[1, 0]
    sizes = sorted(df["size"].unique())
    box_data = [df[df["size"] == s]["final_std"].values for s in sizes]
    bp = ax3.boxplot(box_data, labels=[f"Size {s}" for s in sizes], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax3.set_ylabel("Final Std Dev")
    ax3.set_title(
        f"Distribution of Final Std Dev by Size\n(metric: {metric})", fontweight="bold"
    )
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Scatter of avg_std vs size, colored by strategy
    ax4 = axes[1, 1]
    strategies = df["strategy_short"].unique()
    colors = cm.Set2(np.linspace(0, 1, len(strategies)))
    strategy_colors = dict(zip(strategies, colors))

    for strategy in strategies:
        subset = df[df["strategy_short"] == strategy]
        ax4.scatter(
            subset["size"],
            subset["avg_std"],
            c=[strategy_colors[strategy]],
            label=strategy,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    ax4.set_xlabel("Community Size")
    ax4.set_ylabel("Average Std Dev")
    ax4.set_title(f"Variance vs Size by Strategy\n(metric: {metric})", fontweight="bold")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary table
    print(f"\nSummary Statistics for {metric}:")
    print("-" * 80)
    print(
        f"{'Size':<8} {'Strategy':<30} {'n_agents':<10} {'avg_std':<12} {'final_std':<12}"
    )
    print("-" * 80)
    for _, row in df.sort_values(["size", "strategy_short"]).iterrows():
        print(
            f"{row['size']:<8} {row['strategy_short']:<30} "
            f"{row['n_agents']:<10} {row['avg_std']:<12.3f} {row['final_std']:<12.3f}"
        )


def analyze_community_variance(
    env_community_runs,
    metrics: list[str] | None = None,
    global_strategy_linestyle_map: dict | None = None,
    global_strategy_marker_map: dict | None = None,
):
    """Main entry point for community variance analysis.

    Collects timeseries data and generates variance plots for comparing
    agent performance distribution within communities.

    Args:
        env_community_runs: Dict mapping env_config_id -> community_id -> list of runs
        metrics: Metrics to analyze. Default: ["n_modes_found", "n_new_modes", "novelty_sum"]
        global_strategy_linestyle_map: Optional pre-computed strategy -> linestyle mapping
        global_strategy_marker_map: Optional pre-computed strategy -> marker mapping
    """
    if metrics is None:
        metrics = ["n_new_modes", "n_modes_found", "novelty_sum"]

    print("\n" + "=" * 70)
    print("COMMUNITY VARIANCE ANALYSIS")
    print("Analyzing distribution of agent performance within communities")
    print("=" * 70)

    # Step 1: Collect timeseries data
    variance_data = collect_community_timeseries_data(
        env_community_runs, metrics=metrics
    )

    # Step 2: Generate variance plots for each metric
    for metric in metrics:
        plot_community_variance_over_time(
            variance_data,
            metric=metric,
            global_strategy_linestyle_map=global_strategy_linestyle_map,
            global_strategy_marker_map=global_strategy_marker_map,
        )

        plot_variance_comparison_by_size_and_strategy(variance_data, metric=metric)

    return variance_data


def parse_args():
    """Parse command line arguments."""
    description = """
Multinode Scaling Analysis for TorchGFN

Analyze and manage multinode scaling experiments from Weights & Biases.
This tool provides visualization of mode discovery performance across
different community sizes and strategies, plus experiment management utilities.

MODES OF OPERATION:
  1. Full Analysis (default): Generate plots comparing strategies and sizes
  2. Strategy Discovery: List all strategies to update STRATEGY_MAPPING
  3. Deletion Identification: Find incomplete/failed runs to clean up
"""

    epilog = """
EXPERIMENT MANAGEMENT TOOLS:

  Identifying Runs for Deletion
  -----------------------------
  The --identify-deletions mode helps clean up wandb by finding runs that
  terminated early or failed. It uses the _step metric from run.summary,
  which is universally available on ALL ranks (not just rank-0).

  This is important for multinode experiments where:
  - Only rank-0 logs training metrics like n_modes_found
  - Worker ranks still have _step and _runtime in their summary
  - Using n_modes_found would incorrectly flag healthy worker runs

  Output includes:
  - Runs grouped by community (wandb group)
  - Full deletion candidates (all runs in community failed)
  - Partial deletion candidates (some runs failed)
  - Copy-paste ready run IDs for deletion
  - Diagnostic info: _step, _runtime, history columns

EXAMPLES:

  # Full analysis with plots
  python multinode_scaling_analysis.py

  # Quick mode: discover strategies for STRATEGY_MAPPING
  python multinode_scaling_analysis.py --print-strategies-only

  # Variance analysis: compare agent performance distribution within communities
  python multinode_scaling_analysis.py --variance-analysis

  # Variance analysis with specific metrics
  python multinode_scaling_analysis.py --variance-analysis --variance-metrics n_new_modes novelty_sum

  # Find crashed/failed runs with < 100 steps (default)
  python multinode_scaling_analysis.py --identify-deletions

  # Find runs with < 5000 steps
  python multinode_scaling_analysis.py --identify-deletions --min-steps 5000

  # Find ALL runs (including finished) with < 1000 steps
  python multinode_scaling_analysis.py --identify-deletions --min-steps 1000 --deletion-states all

  # Only look at crashed runs
  python multinode_scaling_analysis.py --identify-deletions --deletion-states crashed

STRATEGY MAPPING:

  To customize legend labels, edit STRATEGY_MAPPING at the top of the script:

    STRATEGY_MAPPING = {
        "average_every=100_..._use_selective_averaging=False": "Baseline",
        "average_every=100_..._use_selective_averaging=True": "Selective Avg",
    }

  Use --print-strategies-only to discover all strategy IDs in your wandb project.
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Analysis mode options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--print-strategies-only",
        action="store_true",
        help="Print all strategies found in wandb and exit. "
        "Useful for updating STRATEGY_MAPPING before generating plots.",
    )
    analysis_group.add_argument(
        "--variance-analysis",
        action="store_true",
        help="Run within-community variance analysis. "
        "Plots std dev, CV, and range of agent metrics over time, "
        "comparing across community sizes and strategies.",
    )
    analysis_group.add_argument(
        "--variance-metrics",
        type=str,
        nargs="+",
        default=["n_new_modes", "n_modes_found", "novelty_sum"],
        metavar="METRIC",
        help="Metrics to analyze for variance. (default: n_new_modes n_modes_found novelty_sum)",
    )

    # Experiment management options
    mgmt_group = parser.add_argument_group("Experiment Management")
    mgmt_group.add_argument(
        "--identify-deletions",
        action="store_true",
        help="Identify runs that should be deleted from wandb. "
        "Uses _step (universal across all ranks) to detect early termination.",
    )
    mgmt_group.add_argument(
        "--min-steps",
        type=int,
        default=100,
        metavar="N",
        help="Minimum _step value required (from run.summary). "
        "Runs with fewer steps are flagged for deletion. "
        "Available on ALL ranks, not just rank-0. (default: %(default)s)",
    )
    mgmt_group.add_argument(
        "--deletion-states",
        type=str,
        nargs="+",
        default=["crashed", "failed"],
        metavar="STATE",
        help="Run states to consider for deletion. "
        "Use 'all' to include finished runs too. (default: crashed failed)",
    )

    return parser.parse_args()


def main():
    """Main function to run the complete analysis."""
    args = parse_args()

    # Fetch data
    runs_list = fetch_wandb_runs()

    # Analyze wandb community groups (these are the "communities" of agents)
    run_to_community, community_runs = analyze_groups(runs_list)

    # Handle --identify-deletions mode (early exit)
    if args.identify_deletions:
        identify_runs_for_deletion(
            runs_list,
            run_to_community,
            min_steps=args.min_steps,
            include_states=args.deletion_states,
        )
        return

    # Print available variables (early, so user knows what's available)
    print_available_variables(runs_list, {})

    # Analyze environment configurations
    run_to_env_config, env_config_runs, env_config_details = (
        analyze_environment_configurations(runs_list)
    )

    # Create hierarchical structure: Environment → Communities → Runs
    env_to_communities, community_to_runs, env_community_runs = (
        create_hierarchical_structure(runs_list, run_to_env_config, run_to_community)
    )

    # Compare community vs environment groupings
    compare_community_vs_environment_groupings(run_to_community, run_to_env_config)

    # Print detailed environment configuration info with community breakdown
    print_environment_configurations(
        runs_list,
        env_config_runs,
        env_config_details,
        env_to_communities,
        community_to_runs,
    )

    # Analyze communities within each environment (the main analysis)
    analyze_communities_within_environments(
        env_community_runs, exit_after_printing_strategies=args.print_strategies_only
    )

    if args.print_strategies_only:
        return

    # Run variance analysis if requested
    if args.variance_analysis:
        analyze_community_variance(
            env_community_runs,
            metrics=args.variance_metrics,
        )

    # Print summary of experiment
    print("\n=== EXPERIMENT OVERVIEW ===")
    print(f"Total environments analyzed: {len(env_to_communities)}")
    print(f"Total communities analyzed: {len(community_to_runs)}")
    print(f"Total runs analyzed: {len(runs_list)}")


if __name__ == "__main__":
    main()
