import datetime
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, cast

import torch
import torch.distributed as dist


def report_load_imbalance(
    all_timing_dict: List[Dict[str, List[float]]],
    world_size: int,
) -> None:
    r"""
    Reports load imbalance and timing information from a timing dictionary.
        param all_timing_dict: A list of dictionaries containing timing information for each rank.
            all_timing_dict structure: [rank0_dict, rank1_dict, ...]
            where each rank_dict is: {"step_name": [iter0_time, iter1_time, iter2_time, ...], ...}

        param world_size: The total number of ranks in the distributed setup.
    """
    # Header
    print(f"{'Step Name':<25} {'Useful Work':>12} {'Waiting':>12}")
    print("-" * 80)

    for step, times in all_timing_dict[0].items():
        if type(times) is not list:
            times = [times]  # Ensure times is a list

        curr_step_times = {}
        is_valid_key = True  # Time information for some steps are not present in all ranks. Those are skipped.
        for rank in range(world_size):
            curr_dict = all_timing_dict[rank]
            if step in curr_dict:
                curr_step_times[rank] = curr_dict[step]
            else:
                is_valid_key = False
                break
        if not is_valid_key:
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


def report_time_info(
    all_timing_dict: List[Dict[str, List[float]]],
    world_size: int,
) -> None:
    """
    Reports timing information from a timing dictionary.
        param all_timing_dict: A list of dictionaries containing timing information for each rank.
            all_timing_dict structure: [rank0_dict, rank1_dict, ...]
            where each rank_dict is: {"step_name": [iter0_time, iter1_time, iter2_time, ...], ...}

        param world_size: The total number of ranks in the distributed setup.
    """
    overall_timing = {}
    print("Timing information for each rank:")
    for rank in range(world_size):
        print(f"Rank {rank} timing information:")
        for step, times in all_timing_dict[rank].items():
            if type(times) is not list:
                times = [times]  # Ensure times is a list

            times_tensor = torch.tensor(times)
            avg_time = torch.sum(times_tensor).item() / len(times)
            sum_time = torch.sum(times_tensor).item()
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


def average_gradients(model):
    """All-Reduce gradients across all models."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_models(model, training_group=None):
    """Averages model weights across all ranks."""
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        param_tensor = param.data.clone()  # clone to avoid inplace operations
        dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM, group=training_group)
        param.data = param_tensor / world_size


@dataclass
class DistributedContext:
    """Holds all distributed training/replay buffer groups and ranks."""

    my_rank: int
    world_size: int
    num_training_ranks: int
    agent_group_size: int
    agent_groups: Optional[List[dist.ProcessGroup]] = None
    agent_group_id: Optional[int] = None
    train_global_group: Optional[dist.ProcessGroup] = None
    assigned_buffer: Optional[int] = None
    buffer_group: Optional[dist.ProcessGroup] = None
    assigned_training_ranks: Optional[List[int]] = None

    def is_buffer_rank(self) -> bool:
        """Check if the current rank is part of the buffer group."""
        return self.my_rank >= self.num_training_ranks

    def is_training_rank(self) -> bool:
        """Check if the current rank is part of the training group."""
        return self.my_rank < self.num_training_ranks


def initialize_distributed_compute(
    dist_backend: str,
    num_remote_buffers: int,
    num_agent_groups: int,
) -> DistributedContext:
    """Initalizes distributed compute using either ccl or mpi backends."""
    """
    Initalizes distributed compute using either ccl or mpi backends.

    Args:
        dist_backend: The backend to use for distributed compute.
        num_remote_buffers: The number of remote buffers to use.
    """
    assert dist_backend in [
        "ccl",
        "mpi",
        "gloo",
    ], f"Invalid backend requested: {dist_backend}"

    pmi_size = int(os.environ.get("PMI_SIZE", "0"))  # 0 or 1 default value?
    print("+ Initalizing distributed compute, PMI_SIZE={}".format(pmi_size))

    if pmi_size <= 1:
        print("+ PMI_SIZE <= 1, running in single process mode.")
        return DistributedContext(
            my_rank=0, world_size=1, num_training_ranks=1, agent_group_size=1
        )

    if dist_backend == "ccl":
        print("+ CCL backend requested...")
        try:
            # Note - intel must be imported before oneccl!
            import oneccl_bindings_for_pytorch  # noqa: F401
        except ImportError as e:
            raise Exception("import oneccl_bindings_for_pytorch failed, {}".format(e))

    elif dist_backend == "mpi":
        print("+ MPI backend requested...")
        assert torch.distributed.is_mpi_available()
        try:
            import torch_mpi  # noqa: F401
        except ImportError as e:
            raise Exception("import torch_mpi failed, {}".format(e))

    elif dist_backend == "gloo":
        print("+ Gloo backend requested...")
        assert torch.distributed.is_gloo_available()

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

    dist.barrier()
    print("+ Distributed compute initialized, backend = {}".format(dist_backend))

    my_rank = dist.get_rank()  # Global!
    world_size = dist.get_world_size()  # Global!

    num_training_ranks = world_size - num_remote_buffers

    # make sure that we have atmost 1 remote buffer per training rank.
    assert num_training_ranks >= num_remote_buffers
    print("num_train = ", num_training_ranks)
    print("num_remote_buffers = ", num_remote_buffers)

    # for now, let us enforce that each agent gets equal number of ranks.
    # TODO: later, we can relax this condition.
    assert num_training_ranks % num_agent_groups == 0
    agent_group_size = num_training_ranks // num_agent_groups
    agent_group_rank_list = [
        list(range(i * agent_group_size, (i + 1) * agent_group_size))
        for i in range(num_agent_groups)
    ]
    print(f"Agent group ranks: {agent_group_rank_list}")
    agent_group_list = [
        cast(
            dist.ProcessGroup,
            dist.new_group(
                agent_group_rank_list[i],
                backend=dist_backend,
                timeout=datetime.timedelta(minutes=5),
            ),
        )
        for i in range(num_agent_groups)
    ]

    # all training ranks in one global group
    training_ranks = [
        r for r in range(num_training_ranks)
    ]  # e.g., 0..num_training_ranks-1
    train_global_group = dist.new_group(
        ranks=training_ranks,
        backend=dist_backend,
        timeout=datetime.timedelta(minutes=5),
    )

    buffer_group = None
    assigned_buffer = None
    assigned_training_ranks = {}
    if num_remote_buffers > 0:
        buffer_ranks = list(
            range(num_training_ranks, num_training_ranks + num_remote_buffers)
        )
        buffer_group = dist.new_group(
            buffer_ranks,
            backend=dist_backend,
            timeout=datetime.timedelta(minutes=5),
        )
        print(f"Buffer group ranks: {buffer_ranks}")

        # Each training rank gets assigned to a buffer rank
        if my_rank < (num_training_ranks):
            assigned_buffer = num_training_ranks + (my_rank % num_remote_buffers)
        else:
            assigned_training_ranks[my_rank] = [
                ranks
                for ranks in range(num_training_ranks)
                if (ranks % num_remote_buffers) == (my_rank - num_training_ranks)
            ]

        print(f"+ My rank: {my_rank} size: {world_size}")
        if my_rank < (num_training_ranks):
            print(f"  -> Training group, assigned buffer rank = {assigned_buffer}")
        else:
            print("  -> Buffer group")

    dist.barrier()
    print("+ Distributed compute initialized, rank = ", my_rank)

    return DistributedContext(
        my_rank=my_rank,
        world_size=world_size,
        num_training_ranks=num_training_ranks,
        agent_group_size=agent_group_size,
        agent_groups=agent_group_list,
        agent_group_id=my_rank // agent_group_size,
        train_global_group=train_global_group,
        assigned_buffer=assigned_buffer,
        buffer_group=buffer_group,
        assigned_training_ranks=assigned_training_ranks.get(my_rank, None),
    )


def gather_distributed_data(
    local_tensor: torch.Tensor,
    world_size: int | None = None,
    rank: int | None = None,
    verbose: bool = False,
    training_group=None,
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

    # Add type assertions to help the type checker
    assert isinstance(world_size, int), "world_size must be an integer"
    assert isinstance(rank, int), "rank must be an integer"

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
    dist.gather(
        local_batch_size, gather_list=batch_size_list, dst=0, group=training_group
    )
    dist.barrier(group=training_group)  # Add synchronization

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
    dist.broadcast(max_batch_size_tensor, src=0, group=training_group)

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
    dist.gather(local_tensor, gather_list=tensor_list, dst=0, group=training_group)
    dist.barrier(group=training_group)  # Add synchronization

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
