r"""
The goal of this script is to reproduce some of the published results on the HyperGrid
environment. Run one of the following commands to reproduce some of the results in
[Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

python train_hypergrid_ddp.py --ndim 4 --height 8 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}
python train_hypergrid_ddp.py --ndim 2 --height 64 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}

And run one of the following to reproduce some of the results in
[Learning GFlowNets from partial episodes for improved convergence and stability](https://arxiv.org/abs/2209.12782)
python train_hypergrid_ddp.py --ndim {2, 4} --height 12 --R0 {1e-3, 1e-4} --tied --loss {TB, DB, SubTB}

This script uses DDP (DistributedDataParallel) for multi-GPU gradient-parallel training.
Launch with torchrun:
  torchrun --nproc_per_node=4 train_hypergrid_ddp.py --loss TB --batch_size 64
Each GPU processes a portion of the batch, and gradients are synchronized via all-reduce
after every backward pass.
"""

import logging
import os
import time
from argparse import ArgumentParser
from math import ceil
from typing import cast

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile
from tqdm import trange
from tutorials.examples.train_hypergrid import (
    ModesReplayBufferManager,
    plot_results,
)

from gfn.containers import NormBasedDiversePrioritizedReplayBuffer, ReplayBuffer
from gfn.containers.replay_buffer_manager import ReplayBufferManager
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.states import DiscreteStates
from gfn.utils.common import Timer, set_seed
from gfn.utils.distributed import DistributedContext
from gfn.utils.modules import MLP, DiscreteUniform, Tabular

logger = logging.getLogger(__name__)


def _make_optimizer_for(gflownet, args) -> torch.optim.Optimizer:
    """Build a fresh AdamW optimizer for a (re)built GFlowNet with logZ group."""
    named = dict(gflownet.named_parameters())
    non_logz = [v for k, v in named.items() if k != "logZ"]
    logz = [named["logZ"]] if "logZ" in named else []

    return torch.optim.AdamW(
        [
            {"params": non_logz, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": logz, "lr": args.lr_Z, "weight_decay": 0.0},
        ]
    )


def set_up_fm_gflownet(args, env, preprocessor):
    """Returns a FM GFlowNet."""
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
    else:
        module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )

    estimator = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    return FMGFlowNet(estimator)


def set_up_pb_pf_estimators(args, env, preprocessor):
    """Returns a pair of estimators for the forward and backward policies."""
    if args.tabular:
        pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        if not args.uniform_pb:
            pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
    else:
        pf_module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            n_noisy_layers=int(args.n_noisy_layers),
            std_init=args.noisy_std_init,
        )
        if not args.uniform_pb:
            pb_module = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk=(
                    pf_module.trunk
                    if args.tied and isinstance(pf_module.trunk, torch.nn.Module)
                    else None
                ),
                n_noisy_layers=int(args.n_noisy_layers),
                std_init=args.noisy_std_init,
            )
    if args.uniform_pb:
        pb_module = DiscreteUniform(env.n_actions - 1)

    assert pb_module is not None
    assert pf_module is not None
    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )

    return (pf_estimator, pb_estimator)


def set_up_logF_estimator(args, env, preprocessor, pf_module):
    """Returns a LogStateFlowEstimator."""
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=1)
    else:
        module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=(
                pf_module.trunk
                if args.tied and isinstance(pf_module.trunk, torch.nn.Module)
                else None
            ),
        )

    return ScalarEstimator(module=module, preprocessor=preprocessor)


def set_up_gflownet(args, env, preprocessor):
    """Returns a GFlowNet complete with the required estimators."""
    if args.loss == "FM":
        return set_up_fm_gflownet(args, env, preprocessor)

    # We need a DiscretePFEstimator and a DiscretePBEstimator.
    pf_estimator, pb_estimator = set_up_pb_pf_estimators(args, env, preprocessor)
    assert pf_estimator is not None
    assert pb_estimator is not None

    if args.loss == "ModifiedDB":
        return ModifiedDBGFlowNet(pf_estimator, pb_estimator)

    elif args.loss == "TB":
        return TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)

    elif args.loss == "ZVar":
        return LogPartitionVarianceGFlowNet(pf=pf_estimator, pb=pb_estimator)

    elif args.loss in ("DB", "SubTB"):
        # We also need a LogStateFlowEstimator.
        logF_estimator = set_up_logF_estimator(
            args,
            env,
            preprocessor,
            pf_estimator,
        )

        if args.loss == "DB":
            return DBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                logF=logF_estimator,
            )
        elif args.loss == "SubTB":
            return SubTBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                logF=logF_estimator,
                weighting=args.subTB_weighting,
                lamda=args.subTB_lambda,
            )


def main(args) -> dict:  # noqa: C901
    """Trains a GFlowNet on the Hypergrid Environment using DDP."""

    if args.half_precision:
        torch.set_default_dtype(torch.bfloat16)

    logger.info("Using default dtype: %s", torch.get_default_dtype())

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Check if plotting is allowed.
    if args.plot:
        if args.wandb_project:
            raise ValueError("plot argument is incompatible with wandb_project")
        if args.ndim != 2:
            raise ValueError("plotting is only supported for 2D environments")

    # Initialize DDP (DistributedDataParallel).
    # Launch with: torchrun --nproc_per_node=N train_hypergrid_ddp.py [...]
    # Ensure RANK/WORLD_SIZE env vars are set (torchrun sets them automatically;
    # for MPI/SLURM launchers, fall back to PMI_RANK/PMI_SIZE or SLURM vars).
    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ.get(
            "PMI_RANK", os.environ.get("SLURM_PROCID", "0")
        )
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ.get(
            "PMI_SIZE", os.environ.get("SLURM_NTASKS", "1")
        )
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ.get(
            "PMI_LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")
        )
    if not dist.is_initialized():
        backend = "nccl" if (torch.cuda.is_available() and not args.no_cuda) else "gloo"
        dist.init_process_group(backend=backend)
    ddp_rank = dist.get_rank()
    ddp_world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    logger.info(
        "DDP initialized: rank=%d, world_size=%d, local_rank=%d, device=%s",
        ddp_rank,
        ddp_world_size,
        local_rank,
        device,
    )

    # Set up the DistributedContext.
    # Dedicate buffer rank(s) if requested.
    num_training_ranks_ddp = ddp_world_size - args.num_remote_buffers
    assert num_training_ranks_ddp > 0, (
        f"Not enough ranks for DDP training: {ddp_world_size} total, "
        f"{args.num_remote_buffers} reserved for buffers"
    )

    # Process group containing only training ranks (used for gradient all-reduce).
    training_ranks_ddp = list(range(num_training_ranks_ddp))
    ddp_train_group = cast(
        dist.ProcessGroup | None, dist.new_group(ranks=training_ranks_ddp)
    )

    # Assign each training rank to a buffer rank.
    assigned_buffer_ddp = None
    assigned_training_ranks_ddp = None
    if args.num_remote_buffers > 0:
        if ddp_rank < num_training_ranks_ddp:
            assigned_buffer_ddp = num_training_ranks_ddp + (
                ddp_rank % args.num_remote_buffers
            )
        else:
            assigned_training_ranks_ddp = [
                r
                for r in range(num_training_ranks_ddp)
                if (r % args.num_remote_buffers) == (ddp_rank - num_training_ranks_ddp)
            ]

    distributed_context = DistributedContext(
        my_rank=ddp_rank,
        world_size=ddp_world_size,
        num_training_ranks=num_training_ranks_ddp,
        agent_group_size=num_training_ranks_ddp,
        train_global_group=ddp_train_group,
        assigned_buffer=assigned_buffer_ddp,
        assigned_training_ranks=assigned_training_ranks_ddp,
    )

    set_seed(args.seed + distributed_context.my_rank)

    # Initialize the environment.
    env = HyperGrid(
        args.ndim,
        args.height,
        device=device,
        reward_fn_str="original",
        reward_fn_kwargs={
            "R0": args.R0,
            "R1": args.R1,
            "R2": args.R2,
        },
        calculate_partition=args.validate_environment,
        store_all_states=args.validate_environment,
        debug=__debug__,
    )
    logger.info("env: %s, buffer_rank: %s", env, distributed_context.is_buffer_rank())

    # If this rank is a buffer rank, run the replay buffer manager and exit.
    if distributed_context.is_buffer_rank():
        if distributed_context.assigned_training_ranks is None:
            num_training_ranks = 0
        else:
            num_training_ranks = len(distributed_context.assigned_training_ranks)

        replay_buffer_manager = ModesReplayBufferManager(
            env=env,
            rank=distributed_context.my_rank,
            num_training_ranks=num_training_ranks,
            diverse_replay_buffer=args.diverse_replay_buffer,
            capacity=args.global_replay_buffer_size,
        )
        replay_buffer_manager.run()
        return {}

    # Initialize WandB.
    use_wandb = args.wandb_project != ""
    if use_wandb:
        if args.wandb_local:
            os.environ["WANDB_MODE"] = "offline"

        import wandb

        # Generate shared group name for wandb across all DDP processes.
        pg = distributed_context.train_global_group
        is_root = distributed_context.my_rank == 0

        if is_root:
            group_name = (
                f"{wandb.util.generate_id()}_{distributed_context.num_training_ranks}"
            )
            group_name_bytes = group_name.encode("utf-8")
            group_name_len_tensor = torch.tensor(
                [len(group_name_bytes)], dtype=torch.long
            )
        else:
            group_name = None
            group_name_bytes = None
            group_name_len_tensor = torch.zeros(1, dtype=torch.long)

        # Broadcast the length
        dist.broadcast(group_name_len_tensor, src=0, group=pg)
        group_name_len = int(group_name_len_tensor.item())

        # Broadcast the payload
        if is_root:
            assert group_name_bytes is not None
            payload = torch.tensor(list(group_name_bytes), dtype=torch.uint8)
        else:
            payload = torch.empty(group_name_len, dtype=torch.uint8)

        dist.broadcast(payload, src=0, group=pg)
        group_name = bytes(payload.tolist()).decode("utf-8")

        wandb.init(
            project=args.wandb_project,
            group=group_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    # Initialize the preprocessor.
    preprocessor = KHotPreprocessor(height=args.height, ndim=args.ndim)

    # Build the initial model and optimizer.
    gflownet = set_up_gflownet(args, env, preprocessor)
    assert gflownet is not None
    gflownet = gflownet.to(device)
    optimizer = _make_optimizer_for(gflownet, args)

    # Create replay buffer if needed.
    replay_buffer = None

    if args.replay_buffer_size > 0:
        if args.diverse_replay_buffer:
            replay_buffer = NormBasedDiversePrioritizedReplayBuffer(
                env,
                capacity=args.replay_buffer_size,
                cutoff_distance=args.cutoff_distance,
                p_norm_distance=args.p_norm_distance,
                remote_manager_rank=distributed_context.assigned_buffer,
                remote_buffer_freq=1,
            )
        else:
            replay_buffer = ReplayBuffer(
                env,
                capacity=args.replay_buffer_size,
                prioritized_capacity=False,
                remote_manager_rank=distributed_context.assigned_buffer,
                remote_buffer_freq=args.remote_buffer_freq,
            )

    gflownet = gflownet.to(device)

    n_iterations = ceil(args.n_trajectories / args.batch_size)
    per_node_batch_size = args.batch_size // distributed_context.num_training_ranks
    modes_found = set()

    is_on_policy = (
        (args.replay_buffer_size == 0)
        and (args.epsilon == 0.0)
        and (args.temperature == 1.0)
        and (int(args.n_noisy_layers) == 0)
    )

    logger.info("n_iterations = %d", n_iterations)
    logger.info("per_node_batch_size = %d", per_node_batch_size)

    # Initialize the profiler.
    if args.profile:
        keep_active = args.trajectories_to_profile // args.batch_size
        prof = profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=keep_active, repeat=1
            ),
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    # Initialize some variables before the training loop.
    timing = {}
    time_start = time.time()
    l1_distances, validation_steps = [], []

    # Used for calculating the L1 distance across all nodes.
    all_visited_terminating_states = env.states_from_batch_shape((0,))

    # Training loop.
    pbar = trange(n_iterations, disable=(distributed_context.my_rank != 0))
    for iteration in pbar:
        iteration_start = time.time()

        # Keep track of visited terminating states on this node.
        with Timer(
            timing, "track_visited_states", enabled=args.timing
        ) as visited_states_timer:
            visited_terminating_states = env.states_from_batch_shape((0,))

            # Profiler.
            if args.profile:
                prof.step()
                if iteration >= 1 + 1 + keep_active:
                    break

        # Sample trajectories.
        with Timer(timing, "generate_samples", enabled=args.timing) as sample_timer:
            trajectories = gflownet.sample_trajectories(
                env,
                n=per_node_batch_size,
                save_logprobs=is_on_policy,
                save_estimator_outputs=not is_on_policy,
                epsilon=args.epsilon,
                temperature=args.temperature,
            )

        # Training objects (incl. possible replay buffer sampling).
        with Timer(
            timing, "to_training_samples", enabled=args.timing
        ) as to_train_samples_timer:
            training_samples = gflownet.to_training_samples(trajectories)

            score_dict = None
            if replay_buffer is not None:
                with torch.no_grad():
                    score_dict = replay_buffer.add(training_samples)
                    training_objects = replay_buffer.sample(
                        n_samples=per_node_batch_size
                    )
            else:
                training_objects = training_samples

        # Loss.
        with Timer(timing, "calculate_loss", enabled=args.timing) as loss_timer:
            optimizer.zero_grad()
            loss = gflownet.loss(
                env,
                training_objects,  # type: ignore
                recalculate_all_logprobs=(not is_on_policy),
                reduction="sum" if args.loss == "SubTB" else "mean",  # type: ignore
            )

        # Backpropagation.
        with Timer(timing, "loss_backward", enabled=args.timing) as loss_backward_timer:
            loss.backward()

        # DDP gradient synchronization: all-reduce gradients across training ranks.
        with Timer(timing, "ddp_grad_sync", enabled=args.timing) as ddp_sync_timer:
            for param in gflownet.parameters():
                if param.grad is not None:
                    dist.all_reduce(
                        param.grad,
                        op=dist.ReduceOp.SUM,
                        group=distributed_context.train_global_group,
                    )
                    param.grad /= distributed_context.num_training_ranks

        # Optimization.
        with Timer(timing, "optimizer", enabled=args.timing) as opt_timer:
            optimizer.step()

        # Calculate how long this iteration took.
        iteration_time = time.time() - iteration_start
        rest_time = iteration_time - sum(
            t
            for t in [
                visited_states_timer.elapsed,
                sample_timer.elapsed,
                to_train_samples_timer.elapsed,
                loss_timer.elapsed,
                loss_backward_timer.elapsed,
                ddp_sync_timer.elapsed,
                opt_timer.elapsed,
            ]
            if t is not None
        )

        log_this_iter = (
            iteration % args.validation_interval == 0
        ) or iteration == n_iterations - 1

        # Keep track of trajectories / states.
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        assert visited_terminating_states is not None
        all_visited_terminating_states.extend(visited_terminating_states)
        to_log = {
            "loss": loss.item(),
            "sample_time": sample_timer.elapsed,
            "to_train_samples_time": to_train_samples_timer.elapsed,
            "loss_time": loss_timer.elapsed,
            "loss_backward_time": loss_backward_timer.elapsed,
            "ddp_sync_time": ddp_sync_timer.elapsed,
            "opt_time": opt_timer.elapsed,
            "rest_time": rest_time,
            "l1_dist": None,  # only logged if validate_environment.
        }
        if score_dict is not None:
            to_log.update(score_dict)

        if log_this_iter:
            if args.validate_environment:
                with Timer(timing, "validation", enabled=args.timing):
                    validation_info, all_visited_terminating_states = env.validate(
                        gflownet,
                        args.validation_samples,
                        all_visited_terminating_states,
                    )
                    assert all_visited_terminating_states is not None
                    to_log.update(validation_info)

            with Timer(timing, "log", enabled=args.timing):
                # Track local modes found on every rank.
                modes_found.update(env.modes_found(all_visited_terminating_states))
                to_log["n_modes_found_local"] = len(modes_found)

                if distributed_context.my_rank == 0:
                    if distributed_context.assigned_buffer is not None:
                        manager_rank = distributed_context.assigned_buffer
                        metadata = ReplayBufferManager.get_metadata(manager_rank)
                        to_log.update(metadata)
                    else:
                        to_log["n_modes_found"] = to_log["n_modes_found_local"]

                    pbar.set_postfix(
                        loss=to_log["loss"],
                        l1_dist=to_log["l1_dist"],
                        n_modes_found=to_log.get("n_modes_found"),
                        n_modes_found_local=to_log["n_modes_found_local"],
                    )

                if use_wandb:
                    wandb.log(to_log, step=iteration)

    logger.info("Finished all iterations")
    total_time = time.time() - time_start
    if args.timing:
        timing["total_rest_time"] = [total_time - sum(sum(v) for k, v in timing.items())]

    timing["total_time"] = [total_time]

    # Log the final timing results.
    if args.timing:
        if distributed_context.my_rank == 0:
            logger.info("\n" + "=" * 80)
            logger.info("\n Timing information:")
            logger.info("=" * 80)

        if distributed_context.my_rank == 0:
            logger.info("%-25s %12s", "Step Name", "Time (s)")
            logger.info("-" * 80)
            for k, v in timing.items():
                logger.info("%-25s %10.4fs", k, sum(v))

    # Stop the profiler if it's active.
    if args.profile:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        prof.export_chrome_trace("trace.json")

    # Plot the results if requested & possible.
    if args.plot:
        plot_results(env, gflownet, l1_distances, validation_steps)

    if distributed_context.my_rank == 0:
        print("Training complete, logs:", to_log)

    # Send termination signal to remote replay buffer manager(s) if used.
    if distributed_context.is_training_rank() and (
        distributed_context.assigned_buffer is not None
    ):
        ReplayBufferManager.send_termination_signal(distributed_context.assigned_buffer)

    # DDP cleanup (barrier on training group only; buffer ranks already exited).
    if dist.is_initialized() and distributed_context.train_global_group is not None:
        dist.barrier(group=distributed_context.train_global_group)

    return to_log


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = ArgumentParser()

    # Machine setting.
    parser.add_argument("--seed", type=int, default=4444, help="Random seed.")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Prevent CUDA usage",
    )

    # DDP settings.
    parser.add_argument(
        "--num_remote_buffers",
        type=int,
        default=1,
        help="Number of remote replay buffer manager ranks (reserves ranks from the end).",
    )
    parser.add_argument(
        "--global_replay_buffer_size",
        type=int,
        default=8192,
        help="Global replay buffer size (only used when num_remote_buffers > 0).",
    )
    parser.add_argument(
        "--remote_buffer_freq",
        type=int,
        default=1,
        help="Frequency (in training iterations) at which training ranks send trajectories to remote replay buffer.",
    )

    # Environment settings.
    parser.add_argument(
        "--ndim",
        type=int,
        default=2,
        help="Number of dimensions in the environment",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=8,
        help="Height of the environment",
    )
    parser.add_argument(
        "--R0",
        type=float,
        default=0.1,
        help="Environment's R0",
    )
    parser.add_argument(
        "--R1",
        type=float,
        default=0.5,
        help="Environment's R1",
    )
    parser.add_argument(
        "--R2",
        type=float,
        default=2.0,
        help="Environment's R2",
    )

    # Training settings.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=2048,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )
    parser.add_argument(
        "--diverse_replay_buffer",
        action="store_true",
        help="Use a diverse replay buffer",
    )
    parser.add_argument(
        "--cutoff_distance",
        type=float,
        default=0.1,
        help="Cutoff distance for diverse replay buffer",
    )
    parser.add_argument(
        "--p_norm_distance",
        type=int,
        default=2,
        help="p-norm distance metric for diverse replay buffer",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["FM", "TB", "DB", "SubTB", "ZVar", "ModifiedDB"],
        default="TB",
        help="Loss function to use",
    )
    parser.add_argument(
        "--subTB_weighting",
        type=str,
        default="geometric_within",
        help="weighting scheme for SubTB",
    )
    parser.add_argument(
        "--subTB_lambda", type=float, default=0.9, help="Lambda parameter for SubTB"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_Z",
        type=float,
        default=0.1,
        help="Specific learning rate for Z (only used for TB loss)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(1e6),
        help=(
            "Total budget of trajectories to train on. "
            "Training iterations = n_trajectories // batch_size"
        ),
    )

    # Policy architecture.
    parser.add_argument(
        "--tabular",
        action="store_true",
        help="Use a lookup table for F, PF, PB instead of an estimator",
    )
    parser.add_argument(
        "--uniform_pb",
        action="store_true",
        help="Use a uniform PB",
    )
    parser.add_argument(
        "--tied",
        action="store_true",
        help="Tie the parameters of PF, PB, and F",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the estimators' neural network modules.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=2,
        help=(
            "Number of hidden layers (of size `hidden_dim`) in the estimators'"
            " neural network modules"
        ),
    )

    # Exploration settings.
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon for epsilon-greedy exploration (default: 0.0, i.e. on-policy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0).",
    )
    parser.add_argument(
        "--n_noisy_layers",
        type=int,
        default=0,
        help="Number of noisy layers in the policy network (default: 0).",
    )
    parser.add_argument(
        "--noisy_std_init",
        type=float,
        default=0.5,
        help="Initial std for noisy layers (default: 0.5).",
    )

    # Validation settings.
    parser.add_argument(
        "--validate_environment",
        action="store_true",
        help="Validate the environment at the end of training",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=100,
        help="How often (in training steps) to validate the gflownet",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    # WandB settings.
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="torchgfn",
        help="Name of the wandb project. If empty, don't use wandb",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="torchgfn",
        help="Name of the wandb entity. If empty, don't use wandb",
    )
    parser.add_argument(
        "--wandb_local",
        action="store_true",
        help="Stores wandb results locally, to be uploaded later.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profiles the execution using PyTorch Profiler.",
    )
    parser.add_argument(
        "--trajectories_to_profile",
        type=int,
        default=2048,
        help=(
            "Number of trajectories to profile using the Pytorch Profiler. "
            "Preferably, a multiple of batch size."
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of true and learned distributions (only works for 2D, incompatible with wandb)",
    )

    parser.add_argument(
        "--timing",
        action="store_true",
        default=True,
        help="Report timing information at the end of training",
    )

    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Use half precision for the model",
    )

    args = parser.parse_args()
    main(args)
