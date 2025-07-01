r"""
The goal of this script is to reproduce some of the published results on the HyperGrid
environment. Run one of the following commands to reproduce some of the results in
[Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

python train_hypergrid.py --ndim 4 --height 8 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}
python train_hypergrid.py --ndim 2 --height 64 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}

And run one of the following to reproduce some of the results in
[Learning GFlowNets from partial episodes for improved convergence and stability](https://arxiv.org/abs/2209.12782)
python train_hypergrid.py --ndim {2, 4} --height 12 --R0 {1e-3, 1e-4} --tied --loss {TB, DB, SubTB}

This script also provides a function `get_exact_P_T` that computes the exact terminating state
distribution for the HyperGrid environment, which is useful for evaluation and visualization.
"""

import datetime
import os
import signal
import sys
import threading
import time
from argparse import ArgumentParser
from math import ceil
from typing import Callable, Optional, cast

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from matplotlib.gridspec import GridSpec
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile
from tqdm import trange

from gfn.containers import NormBasedDiversePrioritizedReplayBuffer, ReplayBuffer
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    GFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator, GFNModule, ScalarEstimator
from gfn.preprocessors import KHotPreprocessor
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform, Tabular
from gfn.utils.training import validate

r"""
Helper class for timing code execution blocks and accumulating elapsed time in a dictionary.

This class is designed to be used as a context manager to measure the execution time of code blocks.
Upon entering the context, it records the start time, and upon exiting, it adds the elapsed time to a
specified key in a provided timing dictionary. This is useful for profiling and tracking the time spent
in different parts of a program, such as during training loops or data processing steps.

    timing_dict (dict): A dictionary where timing results will be accumulated.
    key (str): The key in the timing_dict under which to accumulate elapsed time.

Example:
    for name in ["step1", "step2"]:
        timing[name] = 0

    with Timer(timing, "step1"):
        # Code block to time
        do_something()

    print(f"Elapsed time for step1: {timing['step1']} seconds")
"""
class Timer:
    def __init__(self, timing_dict, key, enabled=True):
        self.timing_dict = timing_dict
        self.key = key
        self.enabled = enabled
        self.elapsed = None

    def __enter__(self):
        if self.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.elapsed = time.perf_counter() - self.start
            if self.key not in self.timing_dict:
                self.timing_dict[self.key] = []
            self.timing_dict[self.key].append(self.elapsed)
        else:
            self.elapsed = 0.0

r"""
Reports load imbalance and timing information from a timing dictionary.
    param all_timing_dict: A list of dictionaries containing timing information for each rank.
        all_timing_dict structure: [rank0_dict, rank1_dict, ...]
        where each rank_dict is: {"step_name": [iter0_time, iter1_time, iter2_time, ...], ...}

    param world_size: The total number of ranks in the distributed setup.
"""
def report_load_imbalance(all_timing_dict, world_size):

    # Header
    print(f"{'Step Name':<25} {'Useful Work':>12} {'Waiting':>12}")
    print("-" * 80)

    for step, times in all_timing_dict[0].items():
        if type(times) is not list:
            times = [times]  # Ensure times is a list

        curr_step_times = {}
        isValidKey = True # Time information for some steps are not present in all ranks. Those are skipped.
        for rank in range(world_size):
            curr_dict = all_timing_dict[rank]
            if step in curr_dict:
                curr_step_times[rank] = curr_dict[step]
            else:
                isValidKey = False
                break
        if not isValidKey:
            print(f"Time for Step - '{step}' not found in all ranks, skipping...")
            continue

        # Calculate the timing profile for the step.
        useful_work = []
        waiting_times = []

        for iteration in range(len(times)):
            rank_times = [curr_step_times[rank][iteration] for rank in curr_step_times]
            max_time = max(rank_times)
            useful_time = sum(rank_times) / len(rank_times)
            waiting_time = max_time - useful_time

            useful_work.append(useful_time)
            waiting_times.append(waiting_time)

        total_useful = sum(useful_work)
        total_waiting = sum(waiting_times)

        print(f"{step:<25} {total_useful:>10.4f}s {total_waiting:>10.4f}s")


def report_time_info(all_timing_dict, world_size):
    overall_timing = {}
    print("Timing information for each rank:")
    for rank in range(world_size):
        print(f"Rank {rank} timing information:")
        for step, times in all_timing_dict[rank].items():
            if type(times) is not list:
                times = [times]  # Ensure times is a list

            avg_time = sum(times) / len(times)
            sum_time = sum(times)
            print(f"  {step}: {avg_time:.4f} seconds (total: {sum_time:.4f} seconds)")

            if overall_timing.get(step) is None:
                overall_timing[step] = [sum_time]
            else:
                overall_timing[step].append(sum_time)

    print("\nMaximum timing information:")
    for step, times in overall_timing.items():
        print(f"  {step}: {max(times):.4f} seconds")

    print("\nAverage timing information:")
    for step, times in overall_timing.items():
        print(f"  {step}: {sum(times) / len(times):.4f} seconds")


def report_timing(all_timing_dict, world_size):
    """Prints the timing information from the timing dictionary."""


    # uncomment if you need rank level timing information.
    # report_time_info(all_timing_dict, world_size)

    # print("Load Imbalance (LI) is as follows:")
    report_load_imbalance(all_timing_dict, world_size)



def average_gradients(model):
    """All-Reduce gradients across all models."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_models(model):
    """Averages model weights across all ranks."""
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        param_tensor = param.data.clone()  # clone to avoid inplace operations
        dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        param.data = param_tensor / world_size


def initialize_distributed_compute(dist_backend: str = "ccl"):
    """Initalizes distributed compute using either ccl or mpi backends."""
    # global my_rank  # TODO: remove globals?
    # global my_size  # TODO: remove globals?

    pmi_size = int(os.environ.get("PMI_SIZE", "0"))  # 0 or 1 default value?
    print("+ Initalizing distributed compute, PMI_SIZE={}".format(pmi_size))

    if pmi_size > 1:
        if dist_backend == "ccl":
            print("+ CCL backend requested...")
            try:
                # Note - intel must be imported before oneccl!
                import oneccl_bindings_for_pytorch  # noqa: F401
            except ImportError as e:
                raise Exception(
                    "import oneccl_bindings_for_pytorch failed, {}".format(e)
                )

        elif dist_backend == "mpi":
            print("+ MPI backend requested...")
            assert torch.distributed.is_mpi_available()
            try:
                import torch_mpi  # noqa: F401
            except ImportError as e:
                raise Exception("import torch_mpi failed, {}".format(e))
        else:
            raise Exception(f"Invalid backend requested: {dist_backend}")

        os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")

        print("+ OMP_NUM_THREADS = ", os.getenv("OMP_NUM_THREADS"))

        world_size = os.environ.get("WORLD_SIZE")
        if world_size is None:
            raise ValueError("WORLD_SIZE is not set")
        rank = os.environ.get("RANK")
        if rank is None:
            raise ValueError("RANK is not set")

        dist.init_process_group(
            backend=dist_backend,
            init_method="env://",
            world_size=int(world_size),
            rank=int(rank),
            timeout=datetime.timedelta(minutes=5),
        )

        my_rank = dist.get_rank()  # Global!
        my_size = dist.get_world_size()  # Global!

        # for now, let us enforce that each agent gets equal number of ranks.
        # TODO: later, we can relax this condition.
        assert my_size % args.num_agent_groups == 0
        agent_group_size = my_size // args.num_agent_groups
        agent_group_rank_list = [
            list(range(i * agent_group_size, (i + 1) * agent_group_size))
            for i in range(args.num_agent_groups)
        ]
        print(agent_group_rank_list)
        agent_group_list = [
            dist.new_group(
                agent_group_rank_list[i],
                backend=dist_backend,
                timeout=datetime.timedelta(minutes=5),
            )
            for i in range(args.num_agent_groups)
        ]

        print(f"+ My rank: {my_rank} size: {my_size}")

        return (my_rank, my_size, agent_group_size, agent_group_list)


class DistributedErrorHandler:
    def __init__(
        self,
        device: torch.device,
        rank: int,
        world_size: int,
        error_check_interval: float = 1.0,
        cleanup_callback: Optional[Callable] = None,
    ):
        """
        Initialize error handler for distributed training.

        Args:
            device: String representing the current device.
            rank: Current process rank
            world_size: Total number of processes
            error_check_interval: How often to check for errors (in seconds)
            cleanup_callback: Optional function to call before shutdown
        """
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.error_check_interval = error_check_interval
        self.cleanup_callback = cleanup_callback
        self.shutdown_flag = threading.Event()
        self.error_tensor = torch.zeros(1, dtype=torch.uint8, device=self.device)

        # Set up error checking thread
        self.checker_thread = threading.Thread(target=self._error_checker, daemon=True)

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def start(self):
        """Start error checking thread"""
        self.checker_thread.start()

    def _signal_handler(self, signum, frame):
        """Handle external signals"""
        print(f"Process {self.rank} received signal {signum}")
        self.shutdown_flag.set()
        self._cleanup()
        sys.exit(1)

    def _error_checker(self):
        """Periodically check for errors across all processes"""
        while not self.shutdown_flag.is_set():
            try:
                # Use all_reduce to check if any process has errored
                error_count = torch.zeros_like(self.error_tensor)
                dist.all_reduce(error_count, op=dist.ReduceOp.SUM)

                if error_count.item() > 0:
                    print(f"Process {self.rank}: Detected error in another process")
                    self.shutdown_flag.set()
                    self._cleanup()
                    sys.exit(1)

            except Exception as e:
                print("Process {}: Error in error checker: {}".format(self.rank, e))
                self.signal_error()
                break

            time.sleep(self.error_check_interval)

    def signal_error(self):
        """Signal that this process has encountered an error"""
        try:
            self.error_tensor.fill_(1)
            dist.all_reduce(self.error_tensor, op=dist.ReduceOp.SUM)
        except Exception as e:
            print(f"Process {self.rank}: Error in signal_error: {str(e)}")
            pass  # If this fails, processes will eventually timeout

        self.shutdown_flag.set()
        self._cleanup()
        sys.exit(1)

    def _cleanup(self):
        """Perform cleanup before shutdown"""
        if self.cleanup_callback:
            try:
                self.cleanup_callback()
            except Exception as e:
                print(f"Process {self.rank}: Error in cleanup: {str(e)}")

        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Process {self.rank}: Error in destroy_process_group: {str(e)}")


def gather_distributed_data(
    local_tensor: torch.Tensor,
    world_size: int | None = None,
    rank: int | None = None,
    verbose: bool = False,
) -> torch.Tensor | None:
    """
    Gather data from all processes in a distributed setting.

    Args:
        local_data: Data from the current process (List or Tensor)
        world_size: Number of processes (optional, will get from env if None)
        rank: Current process rank (optional, will get from env if None)

    Returns:
        On rank 0: Concatenated tensor from all processes
        On other ranks: None
    """
    if verbose:
        print("syncing distributed data")

    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()

    # First gather batch_sizes to allocate correct buffer sizes.
    local_batch_size = torch.tensor(
        [local_tensor.shape[0]], device=local_tensor.device, dtype=local_tensor.dtype
    )
    if rank == 0:
        # Assumes same dimensionality on all ranks!
        batch_size_list = [
            torch.zeros((1,), device=local_tensor.device, dtype=local_tensor.dtype)
            for _ in range(world_size)
        ]
    else:
        batch_size_list = None

    if verbose:
        print("rank={}, batch_size_list={}".format(rank, batch_size_list))
        print(
            "+ gather of local_batch_size={} to batch_size_list".format(local_batch_size)
        )
    dist.gather(local_batch_size, gather_list=batch_size_list, dst=0)
    dist.barrier()  # Add synchronization

    # Pad local tensor to maximum size.
    if verbose:
        print("+ padding local tensor")

    if rank == 0:
        assert batch_size_list is not None
        max_batch_size = max([bs.item() for bs in batch_size_list])
    else:
        max_batch_size = 0

    state_size = local_tensor.shape[1]  # assume states are 1-d, is true for this env.

    # Broadcast max_size to all processes for padding
    max_batch_size_tensor = torch.tensor(max_batch_size, device=local_tensor.device)
    dist.broadcast(max_batch_size_tensor, src=0)

    # Pad local tensor to maximum size.
    if local_tensor.shape[0] < max_batch_size:
        padding = torch.zeros(
            (int(max_batch_size - local_tensor.shape[0]), state_size),
            dtype=local_tensor.dtype,
            device=local_tensor.device,
        )
        local_tensor = torch.cat((local_tensor, padding), dim=0)

    # Gather padded tensors.
    if rank == 0:
        tensor_list = [
            torch.zeros(
                (int(max_batch_size), state_size),
                dtype=local_tensor.dtype,
                device=local_tensor.device,
            )
            for _ in range(world_size)
        ]
    else:
        tensor_list = None

    if verbose:
        print("+ gathering all tensors from world_size={}".format(world_size))
        print("rank={}, tensor_list={}".format(rank, tensor_list))
    dist.gather(local_tensor, gather_list=tensor_list, dst=0)
    dist.barrier()  # Add synchronization

    # Only rank 0 processes the results
    if rank == 0:
        results = []
        assert tensor_list is not None
        assert batch_size_list is not None
        for tensor, batch_size in zip(tensor_list, batch_size_list):
            trimmed_tensor = tensor[: batch_size.item(), ...]
            results.append(trimmed_tensor)

        if verbose:
            print("distributed n_results={}".format(len(results)))

        for r in results:
            print("    {}".format(r.shape))

        return torch.cat(results, dim=0)  # Concatenates along the batch dimension.

    return None  # For all non-zero ranks.


def get_exact_P_T(env: HyperGrid, gflownet: GFlowNet) -> torch.Tensor:
    r"""Evaluates the exact terminating state distribution P_T for HyperGrid.

    For each state s', the terminating state probability is computed as:

    .. math::
        P_T(s') = u(s') P_F(s_f | s')

    where u(s') satisfies the recursion:

    .. math::
        u(s') = \sum_{s \in \text{Par}(s')} u(s) P_F(s' | s)

    with the base case u(s_0) = 1.

    Args:
        env: The HyperGrid environment
        gflownet: The GFlowNet model

    Returns:
        The exact terminating state distribution as a tensor
    """
    if env.ndim != 2:
        raise ValueError("plotting is only supported for 2D environments")

    grid = env.all_states

    # Get the forward policy distribution for all states
    with torch.no_grad():
        # Handle both FM and other GFlowNet types
        policy: GFNModule = cast(
            GFNModule, gflownet.logF if isinstance(gflownet, FMGFlowNet) else gflownet.pf
        )

        estimator_outputs = policy(grid)
        dist = policy.to_probability_distribution(grid, estimator_outputs)
        probabilities = torch.exp(dist.logits)  # Get raw probabilities

    u = torch.ones(grid.batch_shape)

    indices = env.all_indices()
    for index in indices[1:]:
        parents = [
            tuple(list(index[:i]) + [index[i] - 1] + list(index[i + 1 :]) + [i])
            for i in range(len(index))
            if index[i] > 0
        ]
        parents_tensor = torch.tensor(parents)
        parents_indices = parents_tensor[:, :-1].long()  # All but last column for u
        action_indices = parents_tensor[:, -1].long()  # Last column for probabilities

        # Compute u values for parent states.
        parent_u_values = []
        for p in parents_indices:
            grid_idx = torch.all(grid.tensor == p, 1)  # index along flattened grid.
            parent_u_values.append(u[grid_idx])
            # parent_u_values.append(u[tuple(p.tolist())])
            # # torch.all(grid.tensor == p, 1)
        parent_u_values = torch.stack(parent_u_values)
        # parent_u_values = torch.stack([u[tuple(p.tolist())] for p in parents_indices])

        # Compute probabilities for parent transitions.
        parent_probs = []
        for p, a in zip(parents_indices, action_indices):
            grid_idx = torch.all(grid.tensor == p, 1)  # index along flattened grid.
            parent_probs.append(probabilities[grid_idx, a])
        parent_probs = torch.stack(parent_probs)

        u[indices.index(index)] = torch.sum(parent_u_values * parent_probs)

    return (u * probabilities[..., -1]).detach().cpu()


def validate_hypergrid(
    env,
    gflownet,
    n_validation_samples,
    visited_terminating_states: DiscreteStates | None,
    discovered_modes,
):
    # Standard validation shared across envs.
    validation_info, visited_terminating_states = validate(
        env,
        gflownet,
        n_validation_samples,
        visited_terminating_states,
    )

    # validation_info = {}
    # Modes will have a reward greater than 1.
    mode_reward_threshold = 1.0  # Assumes height >= 5. TODO - verify.

    assert isinstance(visited_terminating_states, DiscreteStates)
    modes = visited_terminating_states[
        env.reward(visited_terminating_states) >= mode_reward_threshold
    ].tensor

    # Finds all the unique modes in visited_terminating_states.
    modes_found = set([tuple(s.tolist()) for s in modes])
    discovered_modes.update(modes_found)
    # torch.tensor(list(modes_found)).shape ==[batch_size, 2]
    validation_info["n_modes_found"] = len(discovered_modes)

    # Old way of counting modes -- potentially buggy - to be removed.
    # # Add the mode counting metric.
    # states, scale = visited_terminating_states.tensor, env.scale_factor
    # normalized_states = ((states * scale) - (scale / 2) * (env.height - 1)).abs()

    # modes = torch.all(
    #     (normalized_states > (0.3 * scale) * (env.height - 1))
    #     & (normalized_states <= (0.4 * scale) * (env.height - 1)),
    #     dim=-1,
    # )
    # modes_found = set([tuple(s.tolist()) for s in states[modes.bool()]])
    # discovered_modes.update(modes_found)
    # validation_info["n_modes_found"] = len(discovered_modes)

    return validation_info, visited_terminating_states, discovered_modes


def set_up_fm_gflownet(args, env, preprocessor, agent_group_list, my_agent_group_id):
    """Returns a FM GFlowNet."""
    # We need a LogEdgeFlowEstimator.
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
    else:
        module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )

    if args.distributed:
        module = DDP(module, process_group=agent_group_list[my_agent_group_id])

    estimator = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    return FMGFlowNet(estimator)


def set_up_pb_pf_estimators(
    args, env, preprocessor, agent_group_list, my_agent_group_id
):
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
            )
    if args.uniform_pb:
        pb_module = DiscreteUniform(env.n_actions - 1)

    for v in ["pf_module", "pb_module"]:
        assert locals()[v] is not None, f"{v} is None, Args: {args}"

    if args.distributed:
        pf_module = DDP(pf_module, process_group=agent_group_list[my_agent_group_id])
        pb_module = DDP(pb_module, process_group=agent_group_list[my_agent_group_id])

    assert pf_module is not None
    assert pb_module is not None
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


def set_up_logF_estimator(
    args, env, preprocessor, agent_group_list, my_agent_group_id, pf_module
):
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

    if args.distributed:
        module = DDP(module, process_group=agent_group_list[my_agent_group_id])

    return ScalarEstimator(module=module, preprocessor=preprocessor)


def set_up_gflownet(args, env, preprocessor, agent_group_list, my_agent_group_id):
    """Returns a GFlowNet complete with the required estimators."""
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (forward, backward, logZ/logF) estimators for DB, TB.

    if args.loss == "FM":
        return set_up_fm_gflownet(
            args,
            env,
            preprocessor,
            agent_group_list,
            my_agent_group_id,
        )
    else:
        # We need a DiscretePFEstimator and a DiscretePBEstimator.
        pf_estimator, pb_estimator = set_up_pb_pf_estimators(
            args,
            env,
            preprocessor,
            agent_group_list,
            my_agent_group_id,
        )
        assert pf_estimator is not None
        assert pb_estimator is not None

        if args.loss == "ModifiedDB":
            return ModifiedDBGFlowNet(pf_estimator, pb_estimator)

        elif args.loss == "TB":
            return TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)

        elif args.loss == "ZVar":
            return LogPartitionVarianceGFlowNet(pf=pf_estimator, pb=pb_estimator)

        elif args.loss in ("DB", "SubTB"):
            # We also need a LogStateFlowEstimator.
            logF_estimator = set_up_logF_estimator(
                args,
                env,
                preprocessor,
                agent_group_list,
                my_agent_group_id,
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


def plot_results(env, gflownet, l1_distances, validation_steps):
    # Create figure with 3 subplots with proper spacing
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 0.1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])  # Colorbar axis
    ax3 = fig.add_subplot(gs[3])

    # Get distributions and find global min/max for consistent color scaling
    true_dist = env.true_dist_pmf.reshape(args.height, args.height).cpu().numpy()
    learned_dist = get_exact_P_T(env, gflownet).reshape(args.height, args.height).numpy()

    # Ensure consistent orientation by transposing
    true_dist = true_dist.T
    learned_dist = learned_dist.T

    vmin = min(true_dist.min(), learned_dist.min())
    vmax = max(true_dist.max(), learned_dist.max())

    # True reward distribution
    im1 = ax1.imshow(
        true_dist,
        cmap="viridis",
        interpolation="none",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("True Distribution")

    # Learned reward distribution
    _ = ax2.imshow(
        learned_dist,
        cmap="viridis",
        interpolation="none",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("Learned Distribution")

    # Add colorbar in its own axis
    plt.colorbar(im1, cax=cax)

    # L1 distances over time
    states_per_validation = args.batch_size * args.validation_interval
    validation_states = [i * states_per_validation for i in range(len(l1_distances))]
    ax3.plot(validation_states, l1_distances)
    ax3.set_xlabel("States Visited")
    ax3.set_ylabel("L1 Distance")
    ax3.set_title("L1 Distance Evolution")
    ax3.set_yscale("log")  # Set log scale for y-axis

    plt.tight_layout()
    plt.show()
    plt.close()


def main(args):  # noqa: C901
    """Trains a GFlowNet on the Hypergrid Environment, potentially distributed."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Check if plotting is allowed.
    if args.plot:
        if args.wandb_project:
            raise ValueError("plot argument is incompatible with wandb_project")
        if args.ndim != 2:
            raise ValueError("plotting is only supported for 2D environments")

    # Initialize WandB.
    use_wandb = args.wandb_project != ""
    if use_wandb:

        if args.wandb_local:
            os.environ["WANDB_MODE"] = "offline"

        import wandb

        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    # Initialize distributed compute.
    if args.distributed:
        my_rank, my_size, agent_group_size, agent_group_list = (
            initialize_distributed_compute()
        )
        my_rank = dist.get_rank()
        world_size = torch.distributed.get_world_size()
        my_agent_group_id = my_rank // agent_group_size
        print(f"Running with DDP on rank {my_rank}/{world_size}.")
        print(
            f"agent_group_size, my_agent_group_id = {agent_group_size, my_agent_group_id}"
        )
    else:
        world_size = 1  # Single machine.
        my_rank = 0  # Single machine.
        agent_group_list = my_agent_group_id = None

    set_seed(args.seed + my_rank)

    # Initialize the environment.
    env = HyperGrid(
        args.ndim,
        args.height,
        args.R0,
        args.R1,
        args.R2,
        device=device,
        calculate_partition=args.calculate_partition,
        calculate_all_states=args.calculate_all_states,
    )

    # Initialize the preprocessor.
    preprocessor = KHotPreprocessor(height=args.height, ndim=args.ndim)

    # 2. Create the gflownets: need pairs of modules and estimators.
    gflownet = set_up_gflownet(
        args, env, preprocessor, agent_group_list, my_agent_group_id
    )
    assert gflownet is not None, f"gflownet is None, Args: {args}"

    # Create replay buffer if needed
    replay_buffer = None

    if args.replay_buffer_size > 0:
        if args.diverse_replay_buffer:
            replay_buffer = NormBasedDiversePrioritizedReplayBuffer(
                env,
                capacity=args.replay_buffer_size,
                cutoff_distance=args.cutoff_distance,
                p_norm_distance=args.p_norm_distance,
            )
        else:
            replay_buffer = ReplayBuffer(
                env,
                capacity=args.replay_buffer_size,
                prioritized=False,
            )

    gflownet = gflownet.to(device)

    # 3. Create the optimizer
    non_logz_params = [
        v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
    ]
    if "logZ" in dict(gflownet.named_parameters()):
        logz_params = [dict(gflownet.named_parameters())["logZ"]]
    else:
        logz_params = []

    params = [
        {"params": non_logz_params, "lr": args.lr},
        # Log Z gets dedicated learning rate (typically higher).
        {"params": logz_params, "lr": args.lr_Z},
    ]
    optimizer = torch.optim.Adam(params)

    states_visited = 0
    n_iterations = ceil(args.n_trajectories / args.batch_size)
    per_node_batch_size = args.batch_size // world_size
    validation_info = {"l1_dist": float("inf")}
    discovered_modes = set()
    is_on_policy = args.replay_buffer_size == 0

    print("+ n_iterations = ", n_iterations)
    print("+ per_node_batch_size = ", per_node_batch_size)

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

    if args.distributed:
        # Create and start error handler.
        def cleanup():
            print(f"Process {rank}: Cleaning up...")

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # TODO: remove this or fix it - it's buggy.
        # handler = DistributedErrorHandler(
        #     device_str,
        #     rank,
        #     world_size,
        #     cleanup_callback=cleanup,
        # )
        # handler.start()

    # Initialize some variables before the training loop.
    timing = {}
    time_start = time.time()
    l1_distances, validation_steps = [], []

    # Used for calculating the L1 distance across all nodes.
    all_visited_terminating_states = env.states_from_batch_shape((0,))

    # Barrier for pre-processing. Wait for all processes to reach this point before starting training.
    with Timer(timing, "Pre-processing_barrier", enabled=(args.timing and args.distributed)):
        if args.distributed and args.timing:
            dist.barrier()

    # Training loop.
    pbar = trange(n_iterations)
    for iteration in pbar:
        iteration_start = time.time()

        # Keep track of visited terminating states on this node.
        with Timer(timing, "track_visited_states", enabled=args.timing) as visited_states_timer:
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
                n=args.batch_size,
                save_logprobs=is_on_policy,  # Can be re-used if on-policy.
                save_estimator_outputs=not is_on_policy,  # Only used if off-policy.
            )

        # Training objects (incl. possible replay buffer sampling).
        with Timer(timing, "to_training_samples", enabled=args.timing) as to_train_samples_timer:
            training_samples = gflownet.to_training_samples(trajectories)

            if replay_buffer is not None:
                with torch.no_grad():
                    replay_buffer.add(training_samples)
                    training_objects = replay_buffer.sample(
                        n_trajectories=per_node_batch_size
                    )
            else:
                training_objects = training_samples

        # Loss.
        with Timer(timing, "calculate_loss", enabled=args.timing) as loss_timer:

            optimizer.zero_grad()
            gflownet = cast(GFlowNet, gflownet)
            loss = gflownet.loss(
                env,
                training_objects,  # type: ignore
                recalculate_all_logprobs=args.replay_buffer_size > 0,
                reduction="sum" if args.distributed or args.loss == "SubTB" else "mean",  # type: ignore
            )

            # Normalize the loss by the local batch size if distributed.
            if args.distributed:
                loss = loss / (per_node_batch_size)

        # Barrier.
        with Timer(timing, "barrier 0", enabled=(args.timing and args.distributed)) as bar0_timer:
            if args.distributed and args.timing:
                dist.barrier()

        # Backpropagation.
        with Timer(timing, "loss_backward", enabled=args.timing) as loss_backward_timer:
            loss.backward()

        # Optimization.
        with Timer(timing, "optimizer", enabled=args.timing) as opt_timer:
            optimizer.step()

        # Barrier.
        with Timer(timing, "barrier 1", enabled=(args.timing and args.distributed)) as bar1_timer:
            if args.distributed and args.timing:
                dist.barrier()

        # Model averaging.
        with Timer(timing, "averaging_model", enabled=args.timing) as model_averaging_timer:
            if args.distributed and (iteration % args.average_every == 0):
                print("before averaging model, iteration = ", iteration)
                average_models(gflownet)
                print("after averaging model, iteration = ", iteration)

        # Calculate how long this iteration took.
        iteration_time = time.time() - iteration_start
        rest_time = iteration_time - sum(
            [
                visited_states_timer.elapsed,
                sample_timer.elapsed,
                to_train_samples_timer.elapsed,
                loss_timer.elapsed,
                bar0_timer.elapsed,
                loss_backward_timer.elapsed,
                opt_timer.elapsed,
                bar1_timer.elapsed,
                model_averaging_timer.elapsed,
            ]
        )

        log_this_iter = (
            iteration % args.validation_interval == 0
        ) or iteration == n_iterations - 1

        # Keep track of trajectories / states.
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        with Timer(timing, "gather_visited_states", enabled=args.timing) as gather_timer:
            # If distributed, gather all visited terminating states from all nodes.
            if args.distributed and log_this_iter:
                try:
                    assert visited_terminating_states is not None
                    # Gather all visited terminating states from all nodes.
                    gathered_visited_terminating_states = gather_distributed_data(
                        visited_terminating_states.tensor
                    )
                except Exception as e:
                    print("Process {}: Caught error: {}".format(my_rank, e))
                    # handler.signal_error()
                    sys.exit(1)
            else:
                # Just use the visited terminating states from this node.
                assert visited_terminating_states is not None
                gathered_visited_terminating_states = visited_terminating_states.tensor

        # If we are on the master node, calculate the validation metrics.
        with Timer(timing, "validation", enabled=args.timing) as validation_timer:
            if my_rank == 0:

                # Extend `all_visited_terminating_states` with the gathered data.
                assert gathered_visited_terminating_states is not None
                gathered_visited_terminating_states = cast(
                    DiscreteStates, env.States(gathered_visited_terminating_states)
                )
                states_visited += len(gathered_visited_terminating_states)
                all_visited_terminating_states.extend(gathered_visited_terminating_states)

                to_log = {
                    "loss": loss.item(),
                    "states_visited": states_visited,
                    "sample_time": sample_timer.elapsed,
                    "to_train_samples_time": to_train_samples_timer.elapsed,
                    "loss_time": loss_timer.elapsed,
                    "loss_backward_time": loss_backward_timer.elapsed,
                    "opt_time": opt_timer.elapsed,
                    "model_averaging_time": model_averaging_timer.elapsed,
                    "rest_time": rest_time,
                    "l1_dist": None,  # only logged if calculate_partition.
                    }

                if use_wandb:
                        wandb.log(to_log, step=iteration)

                if log_this_iter:
                    (validation_info, all_visited_terminating_states, discovered_modes) = (
                        validate_hypergrid(
                        env,
                        gflownet,
                        args.validation_samples,
                        all_visited_terminating_states,
                        discovered_modes,
                    )
                )

                    print(
                        "all_visited_terminating_states = ",
                        len(all_visited_terminating_states),
                    )
                    print("visited_terminating_states = ", len(visited_terminating_states))

                    if use_wandb:
                        wandb.log(validation_info, step=iteration)

                    to_log.update(validation_info)

                    pbar.set_postfix(
                        loss=to_log["loss"],
                        l1_dist=to_log["l1_dist"],  # only logged if calculate_partition.
                        n_modes_found=to_log["n_modes_found"],
                    )

        with Timer(timing, "barrier 2", enabled=(args.timing and args.distributed)) as bar2_timer:
            if args.distributed and args.timing:
                dist.barrier()

    total_time = time.time() - time_start
    if args.timing:
        timing["total_rest_time"] = [total_time - sum(
            sum(v)
            for k, v in timing.items())]

    timing["total_time"] = [total_time]

    if args.distributed:
        dist.barrier()

    # Log the final timing results.
    if args.timing:
        print("\n" + "="*80)
        print("\n Timing information:")
        if args.distributed:
            print("-"*80)
            print("The below timing information is averaged across all ranks.")
        print("="*80)

        if args.distributed:
            # Gather timing data from all ranks
            all_timings = [None] * world_size
            dist.all_gather_object(all_timings, timing)

            if my_rank == 0:
                report_timing(all_timings, world_size)
        else:
            # Single machine case
            # Header
            print(f"{'Step Name':<25} {'Time (s)':>12}")
            print("-" * 80)
            for k, v in timing.items():
                print(f"{k:<25} {sum(v):>10.4f}s")

    # Stop the profiler if it's active.
    if args.profile:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        prof.export_chrome_trace("trace.json")

    # Plot the results if requested & possible.
    if args.plot:
        # Create figure with 3 subplots with proper spacing.
        plot_results(env, gflownet, l1_distances, validation_steps)

    try:
        result = validation_info["l1_dist"]
    except KeyError:
        result = validation_info["n_modes_found"]

    if my_rank == 0:
        print("+ Training complete - final_score={:.6f}".format(result))

    return result


if __name__ == "__main__":
    parser = ArgumentParser()

    # Machine setting.
    parser.add_argument("--seed", type=int, default=4444, help="Random seed.")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Prevent CUDA usage",
    )

    # Distributed settings.
    parser.add_argument(
        "--average_every",
        type=int,
        default=20,
        help="Number of epochs after which we average model across all agents",
    )
    parser.add_argument(
        "--num_agent_groups",
        type=int,
        default=1,
        help="Number of agents learning together",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Initializes distributed computation (torch.distributed)",
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
        default=1000,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )
    parser.add_argument(
        "--diverse_replay_buffer",
        action="store_true",
        help="Use a diverse replay buffer",
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

    # Validation settings.
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
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )
    parser.add_argument(
        "--wandb_local",
        action="store_true",
        help="Stores wandb results locally, to be uploaded later.",
    )

    # Settings relevant to the problem size -- toggle off for larger problems.
    parser.add_argument(
        "--calculate_all_states",
        action="store_true",
        default=False,
        help="Disable enumeration of all states.",
    )
    parser.add_argument(
        "--calculate_partition",
        action="store_true",
        default=False,
        help="Disable calculation of the true partition function.",
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
        help="Report timing information at the end of training",
    )

    args = parser.parse_args()
    result = main(args)
