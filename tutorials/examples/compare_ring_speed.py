"""Compare ring-graph training speed vs recursionpharma/gflownet make_rings.

Usage examples:
  python -m tutorials.examples.compare_ring_speed --n-nodes 6 --batch-size 128 --n-iterations 200
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch

from gfn.containers import ReplayBuffer
from gfn.utils.common import set_seed
from tutorials.examples import train_graph_ring as ring_mod
from gflownet.tasks.make_rings import MakeRingsTrainer
from gflownet.config import Config, init_empty

WARMUP_STEPS = 10


@dataclass
class LocalBenchResult:
    total_seconds: float
    iterations: int
    iterations_per_second: float
    last_loss: float


def benchmark_local(
    n_nodes: int,
    directed: bool,
    use_gnn: bool,
    embedding_dim: int,
    num_conv_layers: int,
    batch_size: int,
    n_iterations: int,
    lr: float,
    lr_Z: float,
    use_buffer: bool,
    device_str: str,
    seed: int,
) -> LocalBenchResult:
    device = torch.device(device_str)
    set_seed(seed)
    env = ring_mod.init_env(n_nodes, directed, device)
    gflownet = ring_mod.init_gflownet(
        n_nodes,
        directed,
        use_gnn,
        embedding_dim,
        num_conv_layers,
        env.num_edge_classes,
        device,
    )

    non_logz_params = [v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"]
    logz_params = [dict(gflownet.named_parameters())["logZ"]] if "logZ" in dict(gflownet.named_parameters()) else []
    params = [
        {"params": non_logz_params, "lr": lr},
        {"params": logz_params, "lr": lr_Z},
    ]
    optimizer = torch.optim.Adam(params)

    replay_buffer: Optional[ReplayBuffer] = None
    if use_buffer:
        replay_buffer = ReplayBuffer(
            env,
            capacity=batch_size,
            prioritized_capacity=True,
            prioritized_sampling=True,
        )

    t0 = time.perf_counter()
    last_loss_val = float("nan")
    for iteration in range(n_iterations):
        trajectories = gflownet.sample_trajectories(
            env,
            n=batch_size,
            save_logprobs=True,
        )
        training_samples = gflownet.to_training_samples(trajectories)

        if use_buffer and replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                if iteration > WARMUP_STEPS:
                    training_samples = training_samples[: batch_size // 2]
                    buffer_samples = replay_buffer.sample(n_samples=batch_size // 2)
                    training_samples.extend(buffer_samples)  # type: ignore[arg-type]

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
        loss.backward()
        optimizer.step()
        last_loss_val = float(loss.item())
    t1 = time.perf_counter()

    total = t1 - t0
    ips = n_iterations / total if total > 0 else float("inf")
    return LocalBenchResult(total_seconds=total, iterations=n_iterations, iterations_per_second=ips, last_loss=last_loss_val)


def benchmark_recursion_inprocess(
    n_nodes: int,
    n_iterations: int,
    batch_size: int,
    device: str,
    use_buffer: bool,
    num_layers: int,
    embedding_dim: int,
) -> float:
    """Import recursionpharma's make_rings and invoke its main (best-effort) in-process.

    Returns the wall-clock seconds for the call. Iterations/sec can be computed
    by the caller using `n_iterations`.
    """
    config = init_empty(Config())
    config.log_dir = "./logs/debug_run_mr4"
    config.overwrite_existing_exp = True
    config.num_workers = 6
    config.num_training_steps = n_iterations
    config.num_validation_gen_steps = 1
    config.algo.max_nodes = n_nodes
    config.algo.num_from_policy = batch_size // 2
    config.algo.tb.do_parameterize_p_b = True
    config.replay.use = use_buffer
    config.replay.capacity = batch_size
    config.replay.warmup = WARMUP_STEPS
    config.replay.num_from_replay = batch_size // 2
    config.model.num_layers = num_layers
    config.model.num_emb = embedding_dim
    config.device = device
    trainer = MakeRingsTrainer(config)
    t0 = time.perf_counter()
    trainer.run()
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ring graph training speed vs recursionpharma/gflownet make_rings")
    # Problem size
    parser.add_argument("--n-nodes", type=int, default=4)
    parser.add_argument("--directed", action="store_true", default=True)
    # Model
    parser.add_argument("--use-gnn", action="store_true", default=True)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-conv-layers", type=int, default=1)
    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-Z", type=float, default=0.1)
    parser.add_argument("--use-buffer", action="store_true", default=True)
    # Misc
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    print("Running local ring training benchmark...")
    local_res = benchmark_local(
        n_nodes=args.n_nodes,
        directed=args.directed,
        use_gnn=args.use_gnn,
        embedding_dim=args.embedding_dim,
        num_conv_layers=args.num_conv_layers,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        lr=args.lr,
        lr_Z=args.lr_Z,
        use_buffer=args.use_buffer,
        device_str=args.device,
        seed=args.seed,
    )
    print(
        "Local (ours): {:.2f}s total, {:.2f} it/s, iterations={}, last_loss={:.4f}".format(
            local_res.total_seconds, local_res.iterations_per_second, local_res.iterations, local_res.last_loss
        )
    )

    wall = benchmark_recursion_inprocess(
        n_nodes=args.n_nodes,
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        device=args.device,
        use_buffer=args.use_buffer,
        num_layers=args.num_conv_layers,
        embedding_dim=args.embedding_dim,
    )
    ips = args.n_iterations / wall if wall > 0 else float("inf")
    print("Recursion: {:.2f}s total, {:.2f} it/s, iterations={}".format(wall, ips, args.n_iterations))


if __name__ == "__main__":
    main()

