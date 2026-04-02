#!/usr/bin/env python
"""Profile torchgfn Box environment to identify compute bottlenecks.

Usage:
    python benchmark/profile_box.py [--n-iterations 50] [--batch-size 16]
"""

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

# Lazy imports to match benchmark pattern
os_env = __import__("os").environ
os_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Timer:
    """Flat timer for profiling code sections."""

    def __init__(self):
        self.records = defaultdict(list)

    @contextmanager
    def __call__(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.records[name].append(elapsed)

    def summary(self, total_time=None):
        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)

        # Sort by total time descending
        items = []
        for name, times in self.records.items():
            total = sum(times)
            count = len(times)
            mean = total / count
            items.append((name, total, count, mean))

        items.sort(key=lambda x: -x[1])

        if total_time is None:
            total_time = max(x[1] for x in items) if items else 1.0

        print(
            f"{'Section':<55s} {'Total':>8s} {'Count':>6s} "
            f"{'Mean':>8s} {'% Total':>8s}"
        )
        print("-" * 90)
        for name, total, count, mean in items:
            pct = 100 * total / total_time
            print(
                f"{name:<55s} {total*1000:>7.1f}ms {count:>6d} "
                f"{mean*1000:>7.3f}ms {pct:>7.1f}%"
            )
        print("=" * 80)


def profile_iteration(env, sampler, gflownet, optimizer, timer, batch_size):
    """Profile a single training iteration with fine-grained timing."""
    with timer("total"):
        with timer("sample_trajectories"):
            with timer("sampling/setup"):
                from typing import cast

                from gfn.estimators import PolicyEstimatorProtocol

                policy_estimator = cast(PolicyEstimatorProtocol, sampler.estimator)
                states = env.reset(batch_shape=(batch_size,))
                n_trajectories = batch_size
                device = states.device
                dones = states.is_sink_state

                trajectories_states = [states]
                trajectories_actions = [env.actions_from_batch_shape((n_trajectories,))]
                trajectories_terminating_idx = torch.zeros(
                    n_trajectories, dtype=torch.long, device=device
                )
                ctx = policy_estimator.init_context(
                    n_trajectories, device, states.conditions
                )

            step = 0
            while not all(dones):
                with timer("sampling/per_step"):
                    with timer("sampling/per_step/dummy_actions"):
                        actions = env.actions_from_batch_shape((n_trajectories,))

                    step_mask = ~dones

                    with timer("sampling/per_step/compute_dist"):
                        dist, ctx = policy_estimator.compute_dist(
                            states[step_mask],
                            ctx,
                            step_mask,
                            save_estimator_outputs=False,
                        )

                    with timer("sampling/per_step/sample"):
                        with torch.no_grad():
                            valid_actions_tensor = dist.sample()

                    with timer("sampling/per_step/log_probs"):
                        _, ctx = policy_estimator.log_probs(
                            valid_actions_tensor,
                            dist,
                            ctx,
                            step_mask,
                            vectorized=False,
                            save_logprobs=True,
                        )

                    with timer("sampling/per_step/bookkeeping"):
                        valid_actions = env.actions_from_tensor(valid_actions_tensor)
                        actions[step_mask] = valid_actions
                        trajectories_actions.append(actions)
                        new_states = env._step(states, actions)
                        if states.conditions is not None:
                            new_states.conditions = states.conditions

                        step += 1
                        new_dones = new_states.is_sink_state & ~dones
                        trajectories_terminating_idx[new_dones] = step
                        states = new_states
                        dones = dones | new_dones
                        trajectories_states.append(states)

            with timer("sampling/stack"):
                from gfn.containers import Trajectories

                stacked_states = env.States.stack(trajectories_states)
                stacked_actions = env.Actions.stack(trajectories_actions)[1:]
                stacked_logprobs = (
                    torch.stack(ctx.trajectory_log_probs, dim=0)
                    if ctx.trajectory_log_probs
                    else None
                )

                trajectories = Trajectories(
                    env=env,
                    states=stacked_states,
                    actions=stacked_actions,
                    terminating_idx=trajectories_terminating_idx,
                    is_backward=False,
                    log_rewards=None,
                    log_probs=stacked_logprobs,
                    estimator_outputs=None,
                )

        with timer("loss"):
            optimizer.zero_grad()
            loss = gflownet.loss_from_trajectories(
                env,
                trajectories,
                recalculate_all_logprobs=False,
            )

        with timer("backward"):
            loss.backward()

        with timer("optimizer_step"):
            torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 1.0)
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="Profile torchgfn Box environment")
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    from gfn.gflownet import TBGFlowNet
    from gfn.gym import Box
    from gfn.gym.helpers.box_utils import (
        BoxCartesianPBEstimator,
        BoxCartesianPBMLP,
        BoxCartesianPFEstimator,
        BoxCartesianPFMLP,
    )
    from gfn.samplers import Sampler

    device = torch.device("cpu")

    env = Box(delta=0.25, epsilon=1e-10, device=device, debug=False)

    pf_module = BoxCartesianPFMLP(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_layers,
        n_components=args.n_components,
    )
    pb_module = BoxCartesianPBMLP(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_layers,
        n_components=args.n_components,
        trunk=pf_module.trunk,
    )

    pf_estimator = BoxCartesianPFEstimator(
        env,
        pf_module,
        n_components=args.n_components,
        min_concentration=0.1,
        max_concentration=5.1,
    )
    pb_estimator = BoxCartesianPBEstimator(
        env,
        pb_module,
        n_components=args.n_components,
        min_concentration=0.1,
        max_concentration=5.1,
    )

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator).to(device)
    sampler = Sampler(pf_estimator)

    optimizer = torch.optim.Adam(pf_module.parameters(), lr=1e-3)
    optimizer.add_param_group({"params": pb_module.last_layer.parameters(), "lr": 1e-3})
    if "logZ" in dict(gflownet.named_parameters()):
        logZ = dict(gflownet.named_parameters())["logZ"]
        optimizer.add_param_group({"params": [logZ], "lr": 0.1})

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    timer_warmup = Timer()
    for _ in range(args.warmup):
        profile_iteration(
            env, sampler, gflownet, optimizer, timer_warmup, args.batch_size
        )

    # Profile
    print(f"Profiling ({args.n_iterations} iterations, batch_size={args.batch_size})...")
    timer = Timer()
    t0 = time.perf_counter()
    for i in range(args.n_iterations):
        profile_iteration(env, sampler, gflownet, optimizer, timer, args.batch_size)
    total = time.perf_counter() - t0

    print(f"\nTotal wall time: {total*1000:.1f}ms")
    print(f"Mean iter time: {total/args.n_iterations*1000:.2f}ms")
    timer.summary(total_time=total)


if __name__ == "__main__":
    main()
