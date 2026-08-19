from __future__ import annotations

import datetime
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, cast

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from mpi4py import MPI

logger = logging.getLogger(__name__)


def _first_env(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment variable from a list of names."""
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _get_MPI():
    """Lazily import and return the mpi4py MPI module."""
    from mpi4py import MPI

    return MPI


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
    logger.info("%-25s %12s %12s", "Step Name", "Useful Work", "Waiting")
    logger.info("-" * 80)

    for step, times in all_timing_dict[0].items():
        if not isinstance(times, list):
            times = [times]

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
            logger.warning(
                "Time for Step - '%s' not found in all ranks, skipping...", step
            )
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

        logger.info("%-25s %10.4fs %10.4fs", step, total_useful, total_waiting)


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
    logger.info("Timing information for each rank:")
    for rank in range(world_size):
        logger.info("Rank %d timing information:", rank)
        for step, times in all_timing_dict[rank].items():
            if type(times) is not list:
                times = [times]  # Ensure times is a list

            times_tensor = torch.tensor(times)
            avg_time = torch.sum(times_tensor).item() / len(times)
            sum_time = torch.sum(times_tensor).item()
            logger.info(
                "  %s: %.4f seconds (total: %.4f seconds)", step, avg_time, sum_time
            )

            if overall_timing.get(step) is None:
                overall_timing[step] = [sum_time]
            else:
                overall_timing[step].append(sum_time)

    logger.info("\nMaximum timing information:")
    for step, times in overall_timing.items():
        logger.info("  %s: %.4f seconds", step, max(times))

    logger.info("\nAverage timing information:")
    for step, times in overall_timing.items():
        logger.info("  %s: %.4f seconds", step, sum(times) / len(times))


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


def _split_ceil_floor(total: int, n: int) -> List[int]:
    """Split *total* into *n* parts; first (total % n) get ceil, rest get floor."""
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


class RankLayout:
    """Rank layout for distributed training with optional buffer managers.

    Each buffer group is a set of agent ranks plus one manager rank (the last
    rank in the group).  Group members are stored explicitly rather than as
    ``(start, count)``, because :meth:`from_hostnames` can produce groups whose
    ranks are not contiguous.  Use :meth:`build` for arithmetic assignment or
    :meth:`from_hostnames` for hostname-aware node-local assignment.
    """

    def __init__(
        self,
        training_ranks: List[int],
        buffer_ranks: List[int],
        agent_ranks_per_group: List[List[int]],
        num_agent_groups: int,
    ):
        self.training_ranks = training_ranks
        self.buffer_ranks = buffer_ranks
        self.agent_ranks_per_group = agent_ranks_per_group
        self.num_training_ranks = len(training_ranks)
        self.agent_group_size = self.num_training_ranks // num_agent_groups
        self.buffer_rank_set = set(buffer_ranks)
        self.agent_group_rank_list = [
            training_ranks[i * self.agent_group_size : (i + 1) * self.agent_group_size]
            for i in range(num_agent_groups)
        ]
        # Map each rank → its buffer-group index.
        self._rank_to_group: Dict[int, int] = {}
        for g, members in enumerate(agent_ranks_per_group):
            for r in members:
                self._rank_to_group[r] = g
            if g < len(buffer_ranks):
                self._rank_to_group[buffer_ranks[g]] = g

    # --- Per-rank queries ------------------------------------------------

    def assigned_buffer(self, rank: int) -> Optional[int]:
        """Buffer manager rank for a training rank, or ``None``."""
        if not self.buffer_ranks:
            return None
        if rank in self.buffer_rank_set or rank not in self._rank_to_group:
            return None
        return self.buffer_ranks[self._rank_to_group[rank]]

    def assigned_training_ranks(self, rank: int) -> Optional[List[int]]:
        """Training ranks managed by a buffer rank, or ``None``."""
        if rank not in self.buffer_rank_set:
            return None
        return list(self.agent_ranks_per_group[self._rank_to_group[rank]])

    def agent_group_id(self, rank: int) -> int:
        """Agent-group ID for *rank*."""
        if rank not in self._rank_to_group:
            return 0  # coordinator or unassigned
        g = self._rank_to_group[rank]
        members = self.agent_ranks_per_group[g]
        if rank in self.buffer_rank_set:
            # A manager is not itself in an agent group; report the group of
            # the agents it serves so the ID is always a valid group index.
            if not members:
                return 0
            rank = members[0]
        preceding = sum(len(m) for m in self.agent_ranks_per_group[:g])
        return (preceding + members.index(rank)) // self.agent_group_size

    def summary(self, num_nodes: int) -> str:
        """Human-readable layout summary for logging."""
        num_buffers = len(self.buffer_ranks)
        lines = [
            f"Total {num_nodes} node(s), {self.num_training_ranks} training rank(s), "
            f"{num_buffers} buffer manager(s)."
        ]
        if num_buffers == 0:
            for i, members in enumerate(self.agent_ranks_per_group):
                lines.append(f"  Node {i}: Training ranks {members}.")
        else:
            for g in range(num_buffers):
                lines.append(
                    f"  Group {g}: Training ranks {self.agent_ranks_per_group[g]}. "
                    f"Buffer manager rank {self.buffer_ranks[g]}."
                )
        return "\n".join(lines)

    # --- Constructors ----------------------------------------------------

    @classmethod
    def _from_chunks(
        cls,
        chunks: List[List[int]],
        num_agent_groups: int,
        no_buffers: bool = False,
    ) -> "RankLayout":
        """Build from rank chunks (last rank per chunk is manager unless *no_buffers*)."""
        training_ranks: List[int] = []
        buffer_ranks: List[int] = []
        agent_ranks_per_group: List[List[int]] = []
        for chunk in chunks:
            agents = chunk if no_buffers else chunk[:-1]
            agent_ranks_per_group.append(list(agents))
            training_ranks.extend(agents)
            if not no_buffers:
                buffer_ranks.append(chunk[-1])
        num_training = len(training_ranks)
        assert num_training % num_agent_groups == 0, (
            f"num_training_ranks ({num_training}) must be divisible by "
            f"num_agent_groups ({num_agent_groups})"
        )
        return cls(
            training_ranks,
            buffer_ranks,
            agent_ranks_per_group,
            num_agent_groups,
        )

    @classmethod
    def build(
        cls,
        world_size: int,
        num_remote_buffers: int,
        num_nodes: int,
        num_agent_groups: int,
        num_coordinators: int,
    ) -> "RankLayout":
        """Build layout using arithmetic rank assignment."""
        num_training = world_size - num_remote_buffers - num_coordinators
        assert num_training > 0, (
            f"Not enough ranks: world_size={world_size}, buffers={num_remote_buffers}, "
            f"coordinators={num_coordinators}"
        )
        if num_remote_buffers > 0 and num_agent_groups % num_nodes != 0:
            logger.warning(
                "num_agent_groups (%d) is not a multiple of num_nodes (%d): "
                "some selective-averaging groups will span physical node boundaries.",
                num_agent_groups,
                num_nodes,
            )
        if num_remote_buffers == 0:
            sizes = _split_ceil_floor(num_training, num_nodes)
            chunks: List[List[int]] = []
            pos = 0
            for sz in sizes:
                chunks.append(list(range(pos, pos + sz)))
                pos += sz
            return cls._from_chunks(chunks, num_agent_groups, no_buffers=True)

        assert (
            num_training >= num_remote_buffers
        ), f"num_training_ranks ({num_training}) < num_remote_buffers ({num_remote_buffers})"
        sizes = _split_ceil_floor(num_training, num_remote_buffers)
        chunks = []
        pos = 0
        for n_agents in sizes:
            chunks.append(list(range(pos, pos + n_agents + 1)))
            pos += n_agents + 1
        return cls._from_chunks(chunks, num_agent_groups)

    @classmethod
    def from_hostnames(
        cls,
        world_size: int,
        num_remote_buffers: int,
        num_agent_groups: int,
        num_coordinators: int,
        all_hostnames: List[str],
    ) -> "RankLayout":
        """Build layout using actual hostname topology.

        When ``num_remote_buffers`` is a multiple of ``num_nodes``, assigns
        managers to node-local ranks.  Otherwise falls back to :meth:`build`.
        """
        node_to_ranks: Dict[str, List[int]] = {}
        for rank, hostname in enumerate(all_hostnames):
            node_to_ranks.setdefault(hostname, []).append(rank)
        sorted_nodes = sorted(node_to_ranks, key=lambda h: node_to_ranks[h][0])
        num_nodes = len(sorted_nodes)

        if num_remote_buffers % num_nodes != 0:
            logger.warning(
                "num_remote_buffers (%d) is not a multiple of num_nodes (%d): "
                "managers may be placed on different nodes than their agents. "
                "For node-local placement, use num_remote_buffers = k * num_nodes.",
                num_remote_buffers,
                num_nodes,
            )
            return cls.build(
                world_size,
                num_remote_buffers,
                num_nodes,
                num_agent_groups,
                num_coordinators,
            )

        managers_per_node = num_remote_buffers // num_nodes
        logger.info(
            "Hostname-based layout: %d node(s), %d manager(s)/node.",
            num_nodes,
            managers_per_node,
        )

        # Coordinators are the top `num_coordinators` global ranks; drop them by
        # identity, since hostname order does not guarantee they land on the
        # last node.
        coordinator_ranks = set(range(world_size - num_coordinators, world_size))
        per_node = [
            [r for r in sorted(node_to_ranks[h]) if r not in coordinator_ranks]
            for h in sorted_nodes
        ]

        chunks: List[List[int]] = []
        for node_idx, node_ranks in enumerate(per_node):
            if len(node_ranks) < managers_per_node * 2:
                raise ValueError(
                    f"Node {sorted_nodes[node_idx]} has only {len(node_ranks)} rank(s), "
                    f"need at least {managers_per_node * 2} (1 agent + 1 manager per "
                    f"sub-group, {managers_per_node} sub-group(s))."
                )
            sub_sizes = _split_ceil_floor(len(node_ranks), managers_per_node)
            pos = 0
            for sz in sub_sizes:
                chunks.append(node_ranks[pos : pos + sz])
                pos += sz
        layout = cls._from_chunks(chunks, num_agent_groups)
        if any(
            len({all_hostnames[r] for r in group}) > 1
            for group in layout.agent_group_rank_list
        ):
            logger.warning(
                "Some selective-averaging groups span physical node boundaries "
                "(num_agent_groups=%d, num_nodes=%d, uneven ranks per node).",
                num_agent_groups,
                num_nodes,
            )
        return layout


@dataclass
class DistributedContextMPI4Py:
    """Holds all distributed training/replay buffer groups and ranks."""

    my_rank: int
    world_size: int
    num_training_ranks: int
    agent_group_size: int
    agent_groups: Optional[List[MPI.Comm]] = None
    agent_group_id: Optional[int] = None
    train_global_group: Optional[MPI.Comm] = None
    assigned_buffer: Optional[int] = None
    buffer_group: Optional[MPI.Comm] = None
    assigned_training_ranks: Optional[List[int]] = None
    buffer_rank_set: Set[int] = field(default_factory=set)
    coordinator_rank: Optional[int] = None  # Global rank of the coordinator, if any.

    def is_buffer_rank(self) -> bool:
        """Check if the current rank is part of the buffer group."""
        if self.my_rank == self.coordinator_rank:
            return False
        return self.my_rank in self.buffer_rank_set

    def is_training_rank(self) -> bool:
        """Check if the current rank is part of the training group."""
        if self.my_rank == self.coordinator_rank:
            return False
        return self.my_rank not in self.buffer_rank_set


def initialize_distributed_compute_mpi4py(
    num_remote_buffers: int,
    num_agent_groups: int = 1,
    num_coordinators: int = 0,
    layout: Optional[RankLayout] = None,
) -> DistributedContextMPI4Py:
    """Initializes distributed compute using mpi4py.

    Args:
        num_remote_buffers: The number of remote buffers to use.
        num_agent_groups: Number of selective-averaging groups.  Must divide
            ``num_training_ranks``.  For node-local groups, use a multiple of
            the number of physical nodes (e.g. 4 nodes × 4 groups/node = 16
            total). The number of physical nodes is always detected by
            exchanging MPI hostnames across all ranks.
        num_coordinators: Number of coordinator ranks (0 or 1).
        layout: Pre-computed :class:`RankLayout`.  When provided, skips
            hostname gathering and layout computation (avoids a redundant
            allgather when called from :func:`initialize_distributed_compute`).
    """
    MPI = _get_MPI()

    pmi_size = MPI.COMM_WORLD.Get_size()
    logger.info("Initializing distributed compute (mpi4py), PMI_SIZE=%d", pmi_size)

    if pmi_size <= 1:
        logger.info("PMI_SIZE <= 1, running in single process mode.")
        return DistributedContextMPI4Py(
            my_rank=0, world_size=1, num_training_ranks=1, agent_group_size=1
        )

    os.environ["RANK"] = str(MPI.COMM_WORLD.Get_rank())
    os.environ["WORLD_SIZE"] = str(pmi_size)

    logger.info("OMP_NUM_THREADS = %s", os.getenv("OMP_NUM_THREADS"))

    world_size = MPI.COMM_WORLD.Get_size()
    if world_size is None:
        raise ValueError("WORLD_SIZE is not set")
    rank = MPI.COMM_WORLD.Get_rank()
    if rank is None:
        raise ValueError("RANK is not set")

    dist.barrier()
    logger.info("Distributed compute initialized")

    my_rank = rank

    if layout is None:
        # Standalone call — gather hostnames and compute layout.
        comm = MPI.COMM_WORLD
        all_hostnames: List[str] = comm.allgather(MPI.Get_processor_name())
        num_nodes = len(set(all_hostnames))
        logger.info("Detected num_nodes=%d via MPI hostname exchange", num_nodes)

        if num_remote_buffers > 0:
            layout = RankLayout.from_hostnames(
                world_size=world_size,
                num_remote_buffers=num_remote_buffers,
                num_agent_groups=num_agent_groups,
                num_coordinators=num_coordinators,
                all_hostnames=all_hostnames,
            )
        else:
            layout = RankLayout.build(
                world_size=world_size,
                num_remote_buffers=num_remote_buffers,
                num_nodes=num_nodes,
                num_agent_groups=num_agent_groups,
                num_coordinators=num_coordinators,
            )
        logger.info("num_train = %d", layout.num_training_ranks)
        logger.info("num_remote_buffers = %d", num_remote_buffers)
        logger.info("num_nodes = %d", num_nodes)
        logger.info("num_agent_groups = %d", num_agent_groups)
        logger.info("num_coordinators = %d", num_coordinators)
        logger.info("Agent group ranks: %s", layout.agent_group_rank_list)
        if rank == 0:
            logger.info(layout.summary(num_nodes))

    world_group = MPI.COMM_WORLD.Get_group()
    agent_group_list = []
    for ranks in layout.agent_group_rank_list:
        grp = world_group.Incl(ranks)
        agent_group_list.append(MPI.COMM_WORLD.Create(grp))

    grp = world_group.Incl(layout.training_ranks)
    train_global_group = MPI.COMM_WORLD.Create(grp)

    buffer_group = None
    if layout.buffer_ranks:
        grp = world_group.Incl(layout.buffer_ranks)
        buffer_group = MPI.COMM_WORLD.Create(grp)
        logger.info("Buffer group ranks: %s", layout.buffer_ranks)

    logger.info("My rank: %d size: %d", my_rank, world_size)
    if my_rank in layout.buffer_rank_set:
        logger.info(
            "  -> Buffer group, assigned training ranks = %s",
            layout.assigned_training_ranks(my_rank),
        )
    else:
        logger.info(
            "  -> Training group, assigned buffer rank = %s",
            layout.assigned_buffer(my_rank),
        )

    logger.info("Distributed compute initialized (mpi4py), rank = %d", my_rank)

    return DistributedContextMPI4Py(
        my_rank=my_rank,
        world_size=world_size,
        num_training_ranks=layout.num_training_ranks,
        agent_group_size=layout.agent_group_size,
        agent_groups=agent_group_list,
        agent_group_id=layout.agent_group_id(my_rank),
        train_global_group=train_global_group,
        assigned_buffer=layout.assigned_buffer(my_rank),
        buffer_group=buffer_group,
        assigned_training_ranks=layout.assigned_training_ranks(my_rank),
        buffer_rank_set=layout.buffer_rank_set,
        coordinator_rank=(world_size - 1) if num_coordinators > 0 else None,
    )


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
    dc_mpi4py: Optional[DistributedContextMPI4Py] = None
    coordinator_rank: Optional[int] = None  # Global rank of the coordinator, if any.
    buffer_rank_set: Set[int] = field(default_factory=set)
    training_ranks: List[int] = field(
        default_factory=list
    )  # All global training rank IDs.

    def is_buffer_rank(self) -> bool:
        """Check if the current rank is part of the buffer group."""
        if self.coordinator_rank is not None and self.my_rank == self.coordinator_rank:
            return False
        return self.my_rank in self.buffer_rank_set

    def is_training_rank(self) -> bool:
        """Check if the current rank is part of the training group."""
        if self.coordinator_rank is not None and self.my_rank == self.coordinator_rank:
            return False
        return self.my_rank not in self.buffer_rank_set

    def is_coordinator_rank(self) -> bool:
        """Check if the current rank is the coordinator."""
        return (
            self.coordinator_rank is not None and self.my_rank == self.coordinator_rank
        )

    def cleanup(self) -> None:
        """Cleans up the distributed process group."""
        dist.destroy_process_group()
        if self.dc_mpi4py is not None:
            if self.dc_mpi4py.train_global_group is not None:
                self.dc_mpi4py.train_global_group.Free()
            if self.dc_mpi4py.buffer_group is not None:
                self.dc_mpi4py.buffer_group.Free()
            if self.dc_mpi4py.agent_groups is not None:
                for ag in self.dc_mpi4py.agent_groups:
                    ag.Free()
            _get_MPI().Finalize()

    def get_train_group(self, backend: str = "mpi"):
        if backend == "mpi":
            assert self.dc_mpi4py is not None
            return self.dc_mpi4py.train_global_group
        elif backend == "torch":
            return self.train_global_group
        else:
            raise ValueError(f"Unknown backend: {backend}")


def initialize_distributed_compute(
    dist_backend: str,
    num_remote_buffers: int,
    num_agent_groups: int = 1,
    use_coordinator: bool = False,
) -> DistributedContext:
    """Initializes distributed compute using ccl, mpi, or gloo backends.

    Args:
        dist_backend: The backend to use for distributed compute.
        num_remote_buffers: The number of remote buffers to use.
        num_agent_groups: Number of selective-averaging groups.  Must divide
            ``num_training_ranks``.  For node-local groups, use a multiple of
            the number of physical nodes (e.g. 4 nodes × 4 groups/node = 16
            total). The number of physical nodes is always detected by
            exchanging hostnames across all ranks.
            Defaults to 1 (all agents in one group).
        use_coordinator: If True, the last rank becomes a coordinator that
            aggregates mode discoveries across buffer managers.
    """
    assert dist_backend in [
        "ccl",
        "mpi",
        "gloo",
    ], f"Invalid backend requested: {dist_backend}"

    pmi_size = int(
        cast(
            str,
            _first_env(
                "PMI_SIZE",
                "OMPI_COMM_WORLD_SIZE",
                "MV2_COMM_WORLD_SIZE",
                "WORLD_SIZE",
                default="0",
            ),
        )
    )
    logger.info("Initializing distributed compute, detected world_size=%d", pmi_size)

    if pmi_size <= 1:
        logger.info("PMI_SIZE <= 1, running in single process mode.")
        return DistributedContext(
            my_rank=0,
            world_size=1,
            num_training_ranks=1,
            agent_group_size=1,
            training_ranks=[0],
        )

    if dist_backend == "ccl":
        logger.info("CCL backend requested...")
        try:
            # Note - intel must be imported before oneccl!
            import oneccl_bindings_for_pytorch  # noqa: F401  # pyright: ignore[reportUnusedImport]
        except ImportError as e:
            raise Exception("import oneccl_bindings_for_pytorch failed, {}".format(e))

    elif dist_backend == "mpi":
        logger.info("MPI backend requested...")
        assert torch.distributed.is_mpi_available()
        try:
            import torch_mpi  # noqa: F401  # pyright: ignore[reportUnusedImport]
        except ImportError as e:
            raise Exception("import torch_mpi failed, {}".format(e))

    elif dist_backend == "gloo":
        logger.info("Gloo backend requested...")
        assert torch.distributed.is_gloo_available()

    else:
        raise Exception(f"Invalid backend requested: {dist_backend}")

    os.environ["RANK"] = cast(
        str,
        _first_env(
            "PMI_RANK",
            "OMPI_COMM_WORLD_RANK",
            "MV2_COMM_WORLD_RANK",
            "RANK",
            default="0",
        ),
    )
    os.environ["WORLD_SIZE"] = cast(
        str,
        _first_env(
            "PMI_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "MV2_COMM_WORLD_SIZE",
            "WORLD_SIZE",
            default="1",
        ),
    )

    logger.info("OMP_NUM_THREADS = %s", os.getenv("OMP_NUM_THREADS"))

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
    logger.info("Distributed compute initialized, backend = %s", dist_backend)

    my_rank = dist.get_rank()  # Global!
    world_size = dist.get_world_size()  # Global!

    num_coordinators = 1 if use_coordinator else 0

    # Gather hostnames for node-local manager placement (using torch.distributed).
    # This reflects actual process placement, so it is the sole source of
    # truth for num_nodes (no env-var or caller override needed).
    all_hostnames: List[str] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(all_hostnames, socket.gethostname())
    num_nodes = len(set(all_hostnames))
    logger.info("Detected num_nodes=%d via hostname exchange", num_nodes)

    if num_remote_buffers > 0:
        layout = RankLayout.from_hostnames(
            world_size=world_size,
            num_remote_buffers=num_remote_buffers,
            num_agent_groups=num_agent_groups,
            num_coordinators=num_coordinators,
            all_hostnames=all_hostnames,
        )
    else:
        layout = RankLayout.build(
            world_size=world_size,
            num_remote_buffers=num_remote_buffers,
            num_nodes=num_nodes,
            num_agent_groups=num_agent_groups,
            num_coordinators=num_coordinators,
        )
    logger.info("num_train = %d", layout.num_training_ranks)
    logger.info("num_remote_buffers = %d", num_remote_buffers)
    logger.info("num_nodes = %d", num_nodes)
    logger.info("num_agent_groups = %d", num_agent_groups)
    logger.info("num_coordinators = %d", num_coordinators)
    logger.info("Agent group ranks: %s", layout.agent_group_rank_list)
    if my_rank == 0:
        logger.info(layout.summary(num_nodes))

    agent_group_list = [
        cast(
            dist.ProcessGroup,
            dist.new_group(
                layout.agent_group_rank_list[i],
                backend=dist_backend,
                timeout=datetime.timedelta(minutes=5),
            ),
        )
        for i in range(len(layout.agent_group_rank_list))
    ]

    train_global_group = cast(
        dist.ProcessGroup,
        dist.new_group(
            ranks=layout.training_ranks,
            backend=dist_backend,
            timeout=datetime.timedelta(minutes=5),
        ),
    )

    buffer_group = None
    if layout.buffer_ranks:
        buffer_group = cast(
            dist.ProcessGroup,
            dist.new_group(
                layout.buffer_ranks,
                backend=dist_backend,
                timeout=datetime.timedelta(minutes=5),
            ),
        )
        logger.info("Buffer group ranks: %s", layout.buffer_ranks)

    logger.info("My rank: %d size: %d", my_rank, world_size)
    if my_rank in layout.buffer_rank_set:
        logger.info(
            "  -> Buffer group, assigned training ranks = %s",
            layout.assigned_training_ranks(my_rank),
        )
    elif use_coordinator and my_rank == world_size - 1:
        logger.info("  -> Coordinator rank")
    else:
        logger.info(
            "  -> Training group, assigned buffer rank = %s",
            layout.assigned_buffer(my_rank),
        )

    coordinator_rank = (world_size - 1) if use_coordinator else None
    if use_coordinator:
        logger.info("Coordinator rank = %d", coordinator_rank)

    dist.barrier()
    logger.info("Distributed compute initialized, rank = %d", my_rank)

    dc = initialize_distributed_compute_mpi4py(
        num_remote_buffers=num_remote_buffers,
        num_agent_groups=num_agent_groups,
        num_coordinators=num_coordinators,
        layout=layout,
    )

    return DistributedContext(
        my_rank=my_rank,
        world_size=world_size,
        num_training_ranks=layout.num_training_ranks,
        agent_group_size=layout.agent_group_size,
        agent_groups=agent_group_list,
        agent_group_id=layout.agent_group_id(my_rank),
        train_global_group=train_global_group,
        assigned_buffer=layout.assigned_buffer(my_rank),
        buffer_group=buffer_group,
        assigned_training_ranks=layout.assigned_training_ranks(my_rank),
        dc_mpi4py=dc,
        coordinator_rank=coordinator_rank,
        buffer_rank_set=layout.buffer_rank_set,
        training_ranks=layout.training_ranks,
    )


def gather_distributed_data(
    local_tensor: torch.Tensor,
    world_size: int | None = None,
    rank: int | None = None,
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
    logger.debug("Gathering distributed data")

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

    logger.debug("rank=%d, batch_size_list=%s", rank, batch_size_list)
    logger.debug("gather of local_batch_size=%s to batch_size_list", local_batch_size)
    dist.gather(
        local_batch_size, gather_list=batch_size_list, dst=0, group=training_group
    )
    dist.barrier(group=training_group)  # Add synchronization

    # Pad local tensor to maximum size.
    logger.debug("padding local tensor")

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

    logger.debug("gathering all tensors from world_size=%d", world_size)
    logger.debug("rank=%d, tensor_list=%s", rank, tensor_list)
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

        logger.debug("distributed n_results=%d", len(results))
        for r in results:
            logger.debug("    %s", r.shape)

        return torch.cat(results, dim=0)  # Concatenates along the batch dimension.

    return None  # For all non-zero ranks.


default_backend = "mpi"

# Group type that works for both torch ProcessGroup and MPI communicator.
Group = Any


def send(
    data: torch.Tensor,
    dst_rank: int,
    backend: str = default_backend,
    tag: int = 0,
) -> None:
    """Send a byte tensor to ``dst_rank``.

    This is **byte-level transport** — the payload is always sent as raw uint8
    bytes. Both backends guarantee that ``recv`` on the other end will return
    an identical uint8 tensor.

    Protocol differences between backends:

    - **torch**: Uses a length-prefixed two-message protocol. First a 1-element
      int64 tensor containing the payload length is sent (tag=2*tag), then the
      payload itself (tag=2*tag+1). This lets the receiver allocate the right
      buffer size before the data arrives.
    - **mpi**: Sends a single message with the given ``tag``. The receiver uses
      ``MPI.Probe`` to discover the incoming message size before calling
      ``Recv``, so no separate length message is needed.

    Because the wire protocols differ, sender and receiver **must** use the
    same backend.

    Args:
        data: Tensor to send (will be cast to uint8).
        dst_rank: Destination rank (global).
        backend: ``"torch"`` or ``"mpi"``.
        tag: MPI/torch tag for message matching. Use distinct tags to
            multiplex independent message channels on the same rank pair.
    """
    if backend == "torch":
        data = data.to(dtype=torch.uint8).contiguous().cpu()
        length_tensor = torch.tensor([data.numel()], dtype=torch.int64, device="cpu")
        dist.send(tensor=length_tensor, dst=dst_rank, tag=2 * tag)
        dist.send(tensor=data, dst=dst_rank, tag=2 * tag + 1)
    elif backend == "mpi":
        MPI = _get_MPI()
        comm = MPI.COMM_WORLD
        arr = data.detach().cpu().contiguous().numpy()
        comm.Send(arr, dest=dst_rank, tag=tag)
    else:
        raise ValueError(f"Unknown backend: {backend}")


@dataclass
class AsyncSendHandle:
    """Handle for a non-blocking send operation.

    The underlying buffer must remain alive until the send completes.
    Call :meth:`is_complete` to poll or :meth:`wait` to block.
    """

    _request: Any
    _buffer: Any  # numpy array or tensor kept alive until send completes.

    def is_complete(self) -> bool:
        """Check if the send has completed without blocking."""
        if hasattr(self._request, "is_completed"):
            return self._request.is_completed()
        result = self._request.Test()
        # MPI4Py uppercase Isend: Test() returns bool.
        # Torch dist.isend: Test() returns (bool, ...) tuple.
        if isinstance(result, bool):
            return result
        return result[0]

    def wait(self) -> None:
        """Block until the send completes."""
        if hasattr(self._request, "wait"):
            self._request.wait()
        else:
            self._request.Wait()


def isend(
    data: torch.Tensor,
    dst_rank: int,
    backend: str = default_backend,
    tag: int = 0,
) -> AsyncSendHandle:
    """Non-blocking send of a byte tensor to ``dst_rank``.

    Returns an :class:`AsyncSendHandle` whose internal buffer **must** be
    kept alive until :meth:`~AsyncSendHandle.is_complete` returns ``True``
    or :meth:`~AsyncSendHandle.wait` is called.

    Args:
        data: Tensor to send (will be cast to uint8).
        dst_rank: Destination rank (global).
        backend: ``"torch"`` or ``"mpi"``.
        tag: MPI/torch tag for message matching.

    Returns:
        A handle that tracks the outstanding send.
    """
    if backend == "torch":
        data = data.to(dtype=torch.uint8).contiguous().cpu()
        length_tensor = torch.tensor([data.numel()], dtype=torch.int64, device="cpu")
        # Length message is blocking (tiny); payload is non-blocking.
        dist.send(tensor=length_tensor, dst=dst_rank, tag=2 * tag)
        req = dist.isend(tensor=data, dst=dst_rank, tag=2 * tag + 1)
        return AsyncSendHandle(_request=req, _buffer=data)
    elif backend == "mpi":
        MPI = _get_MPI()
        comm = MPI.COMM_WORLD
        arr = data.detach().cpu().contiguous().numpy()
        req = comm.Isend(arr, dest=dst_rank, tag=tag)
        return AsyncSendHandle(_request=req, _buffer=arr)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def recv(
    src_rank: int | None = None,
    backend: str = default_backend,
    tag: int = 0,
) -> tuple[int, torch.Tensor]:
    """Receive a byte tensor from ``src_rank`` (or any rank if ``None``).

    Returns ``(source_rank, data)`` where ``data`` is a uint8 tensor. See
    :func:`send` for protocol details per backend.

    Args:
        src_rank: Source rank to receive from, or ``None`` for any source.
        backend: ``"torch"`` or ``"mpi"``.
        tag: MPI/torch tag for message matching. Must match the tag used
            by the corresponding :func:`send` call.

    Returns:
        Tuple of (source rank, received uint8 tensor).
    """
    if backend == "torch":
        # Step 1: receive the payload length (tag=2*tag).
        length_tensor = torch.zeros(1, dtype=torch.int64, device="cpu")
        if src_rank is None:
            src_rank = dist.recv(tensor=length_tensor, tag=2 * tag)
        else:
            dist.recv(tensor=length_tensor, src=src_rank, tag=2 * tag)

        msg_len = int(length_tensor.item())

        # Step 2: receive the payload (tag=2*tag+1).
        data = torch.empty(msg_len, dtype=torch.uint8, device="cpu")
        dist.recv(tensor=data, src=src_rank, tag=2 * tag + 1)
        return src_rank, data

    elif backend == "mpi":
        MPI = _get_MPI()
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        source = MPI.ANY_SOURCE if src_rank is None else src_rank
        # Probe to discover message size before allocating the receive buffer.
        comm.Probe(source=source, tag=tag, status=status)
        source = status.Get_source()
        count = status.Get_count(MPI.BYTE)
        buf = torch.empty(count, dtype=torch.uint8)
        comm.Recv(buf.numpy(), source=source, tag=tag, status=status)
        return source, buf
    else:
        raise ValueError(f"Unknown backend: {backend}")


def barrier(backend: str = default_backend, group: Group | None = None) -> None:
    """Backend-agnostic barrier synchronization.

    Args:
        backend: ``"torch"`` or ``"mpi"``.
        group: Process group (torch ProcessGroup or MPI communicator).
    """
    if backend == "torch":
        group = dist.group.WORLD if group is None else group
        dist.barrier(group=group)
    elif backend == "mpi":
        MPI = _get_MPI()
        comm = MPI.COMM_WORLD if group is None else group
        comm.Barrier()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_rank(backend: str = default_backend, group: Group | None = None) -> int:
    """Backend-agnostic rank query.

    Args:
        backend: ``"torch"`` or ``"mpi"``.
        group: Process group (torch ProcessGroup or MPI communicator).
    """
    if backend == "torch":
        return dist.get_rank(group)
    elif backend == "mpi":
        MPI = _get_MPI()
        comm = group if group is not None else MPI.COMM_WORLD
        return comm.Get_rank()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_world_size(backend: str = default_backend, group: Group | None = None) -> int:
    """Backend-agnostic world size query.

    Args:
        backend: ``"torch"`` or ``"mpi"``.
        group: Process group (torch ProcessGroup or MPI communicator).
    """
    if backend == "torch":
        return dist.get_world_size(group)
    elif backend == "mpi":
        MPI = _get_MPI()
        comm = group if group is not None else MPI.COMM_WORLD
        return comm.Get_size()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def all_reduce(
    tensor: torch.Tensor,
    op: str = "SUM",
    backend: str = default_backend,
    group: Group | None = None,
) -> None:
    """Backend-agnostic in-place all-reduce.

    The MPI backend round-trips through CPU/numpy and copies the result back
    to the original tensor (preserving its device).

    Args:
        tensor: The tensor to reduce in-place.
        op: Reduction operation. One of ``"SUM"``, ``"MAX"``, ``"MIN"``.
        backend: ``"torch"`` or ``"mpi"``.
        group: Process group (torch ProcessGroup or MPI communicator).
    """
    if backend == "torch":
        torch_ops = {
            "SUM": dist.ReduceOp.SUM,
            "MAX": dist.ReduceOp.MAX,
            "MIN": dist.ReduceOp.MIN,
        }
        dist.all_reduce(tensor, op=torch_ops[op], group=group)
    elif backend == "mpi":
        MPI = _get_MPI()
        comm = group if group is not None else MPI.COMM_WORLD
        mpi_ops = {"SUM": MPI.SUM, "MAX": MPI.MAX, "MIN": MPI.MIN}
        arr = tensor.detach().cpu().numpy().copy()
        comm.Allreduce(MPI.IN_PLACE, arr, op=mpi_ops[op])
        tensor.copy_(torch.from_numpy(arr).to(tensor.device))
    else:
        raise ValueError(f"Unknown backend: {backend}")


def all_gather(
    output_list: list[torch.Tensor],
    tensor: torch.Tensor,
    backend: str = default_backend,
    group: Group | None = None,
) -> None:
    """Backend-agnostic all-gather.

    The MPI backend round-trips through CPU/numpy and copies results back
    to the output tensors (preserving their devices).

    Args:
        output_list: List of pre-allocated tensors (one per rank) to receive
            gathered data into.
        tensor: The local tensor to send.
        backend: ``"torch"`` or ``"mpi"``.
        group: Process group (torch ProcessGroup or MPI communicator).
    """
    if backend == "torch":
        dist.all_gather(output_list, tensor, group=group)
    elif backend == "mpi":
        import numpy as np

        MPI = _get_MPI()
        comm = group if group is not None else MPI.COMM_WORLD
        send_arr = tensor.detach().cpu().numpy().copy()
        recv_arr = np.empty(
            [len(output_list)] + list(send_arr.shape), dtype=send_arr.dtype
        )
        comm.Allgather(send_arr, recv_arr)
        for i, out in enumerate(output_list):
            out.copy_(torch.from_numpy(recv_arr[i]).to(out.device))
    else:
        raise ValueError(f"Unknown backend: {backend}")


def broadcast(
    tensor: torch.Tensor,
    src: int,
    backend: str = default_backend,
    group: Group | None = None,
) -> None:
    """Backend-agnostic broadcast.

    The MPI backend round-trips through CPU/numpy and copies the result back
    to the tensor (preserving its device).

    Args:
        tensor: The tensor to broadcast. On the source rank this is the data
            to send; on other ranks the buffer is overwritten with received data.
        src: Source rank.
        backend: ``"torch"`` or ``"mpi"``.
        group: Process group (torch ProcessGroup or MPI communicator).
    """
    if backend == "torch":
        dist.broadcast(tensor, src=src, group=group)
    elif backend == "mpi":
        MPI = _get_MPI()
        comm = group if group is not None else MPI.COMM_WORLD
        arr = tensor.detach().cpu().numpy().copy()
        comm.Bcast(arr, root=src)
        tensor.copy_(torch.from_numpy(arr).to(tensor.device))
    else:
        raise ValueError(f"Unknown backend: {backend}")
