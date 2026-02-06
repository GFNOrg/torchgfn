from __future__ import annotations

import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, Tuple, cast

import mpi4py.MPI as MPI
import numpy as np
import torch
import torch.distributed as dist

from gfn.gflownet.base import GFlowNet
from gfn.utils.common import Timer

logger = logging.getLogger(__name__)


class SpawnPolicy(ABC):
    def __init__(self, average_every: int) -> None:
        self.average_every = max(int(average_every), 1)

    @abstractmethod
    def __call__(
        self,
        iteration: int,
        model: GFlowNet,
        local_metric: Optional[float] = None,
        group=dist.group.WORLD,
    ) -> Tuple[GFlowNet, torch.optim.Optimizer, dict]:
        """Possibly perform a spawn/averaging step on this iteration."""
        raise NotImplementedError


class AverageAllPolicy(SpawnPolicy):
    """Standard model averaging across all ranks every N iterations."""

    def __init__(self, average_every: int) -> None:
        super().__init__(average_every)

    @torch.no_grad()
    def __call__(
        self,
        iteration: int,
        model: GFlowNet,
        optimizer: torch.optim.Optimizer,
        local_metric: Optional[float] = None,
        group=dist.group.WORLD,
    ) -> Tuple[GFlowNet, torch.optim.Optimizer, dict]:
        if not dist.is_available() or not dist.is_initialized():
            return model, optimizer, {}
        if iteration % self.average_every != 0:
            return model, optimizer, {"averaged_this_iteration": False}

        world_size = float(dist.get_world_size())
        logger.info(
            "Iteration %d: Averaging model across %d ranks", iteration, int(world_size)
        )
        for param in model.parameters():
            param_tensor = param.data.clone()
            dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM, group=group)
            param.data.copy_(param_tensor / world_size)

        return model, optimizer, {"averaged_this_iteration": True}


class AsyncSelectiveAveragingPolicy(SpawnPolicy):
    """Asynchronous selective averaging with background, non-blocking comms.

    Each cadence, ranks send metrics to rank 0 via isend. Rank 0 aggregates
    when it has all metrics, decides a replacement set and instructs donors
    and replacers with point-to-point messages. Donors stream parameters to
    replacers. Replacers aggregate in a background thread; the main thread
    applies the averaged weights at the next safe call, without barriers.
    """

    _OP_NONE = 0
    _OP_ROLE_DONOR = 1
    _OP_ROLE_REPLACER = 2

    _TAG_METRIC = 7001
    _TAG_CONTROL = 7002
    _TAG_PARAM_BASE = 8000

    def __init__(
        self,
        model_builder: Callable[[], Tuple[GFlowNet, torch.optim.Optimizer]],
        average_every: int,
        replacement_ratio: float = 0.2,
        averaging_strategy: str = "mean",
        momentum: float = 0.0,
        poll_interval_s: float = 0.01,
        threshold: Optional[float] = None,
        cooldown: int = 200,
        timing: Optional[dict] = None,  # timing is a dict to capture timing info
    ) -> None:
        super().__init__(average_every)
        self.replacement_ratio = float(replacement_ratio)
        self.averaging_strategy = str(averaging_strategy)
        self.momentum = float(momentum)
        self.poll_interval_s = float(poll_interval_s)
        self.threshold: Optional[float] = threshold
        self.cooldown: int = int(cooldown)
        self._model_builder = model_builder

        self._initialized = False
        self._model: Optional[GFlowNet] = None
        self._shutdown = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._pending_lock = threading.Lock()
        self._last_iter_sent: int = -1
        self._last_trigger_iter: int = -self.cooldown

        # When rebuilding a fresh model + optimizer is desired, we store the
        # averaged parameters and construct the new model at the next safe call.
        self._new_weights: Optional[Dict[str, torch.Tensor]] = None

        # Rank-0 state for metric aggregation
        self._rank0_metric_handles: Dict[int, dist.Work] = {}
        self._rank0_metric_buffers: Dict[int, torch.Tensor] = {}
        self._rank0_buckets: Dict[int, Dict[int, float]] = {}

        # Non-zero ranks control receive state
        self._control_work: Optional[dist.Work] = None
        self._control_role_buf: Optional[torch.Tensor] = None

    def _ensure_initialized(self, model: GFlowNet) -> None:
        if self._initialized:
            return
        if not dist.is_available() or not dist.is_initialized():
            self._initialized = True
            return
        self._validate_params(
            self.replacement_ratio,
            self.averaging_strategy,
            self.momentum,
            self.threshold,
            self.cooldown,
        )
        self._model = model
        self._initialized = True
        rank = dist.get_rank()
        logger.info(
            "Rank %d: Initializing AsyncSelectiveAveragingPolicy "
            "(replacement_ratio=%.2f, strategy=%s, threshold=%s, cooldown=%d)",
            rank,
            self.replacement_ratio,
            self.averaging_strategy,
            self.threshold,
            self.cooldown,
        )
        self._bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._bg_thread.start()
        logger.debug("Rank %d: Background thread started", rank)

    def shutdown(self) -> None:
        logger.debug("Shutting down AsyncSelectiveAveragingPolicy")
        self._shutdown.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1.0)
        logger.debug("AsyncSelectiveAveragingPolicy shutdown complete")

    @torch.no_grad()
    def __call__(
        self,
        iteration: int,
        model: GFlowNet,
        optimizer: torch.optim.Optimizer,
        local_metric: Optional[float] = None,
        group=dist.group.WORLD,
    ) -> Tuple[GFlowNet, torch.optim.Optimizer, dict]:
        self._ensure_initialized(model)
        if not dist.is_available() or not dist.is_initialized():
            return model, optimizer, {}

        # Non-blocking metric send on cadence
        if local_metric is not None and iteration % self.average_every == 0:
            rank = dist.get_rank()
            if iteration != self._last_iter_sent:
                self._last_iter_sent = iteration
                payload = torch.tensor(
                    [float(iteration), float(local_metric)], dtype=torch.float32
                )
                logger.debug(
                    "Rank %d: Sending metric %.4f for iteration %d",
                    rank,
                    local_metric,
                    iteration,
                )
                try:
                    if rank != 0:
                        dist.isend(payload, dst=0, tag=self._TAG_METRIC)
                    else:
                        # Rank 0 includes its own metric directly
                        world_size = dist.get_world_size()
                        self._rank0_record_metric(
                            iteration, rank, float(local_metric), world_size
                        )
                except Exception as e:
                    logger.warning("Rank %d: Failed to send metric: %s", rank, e)

        # If a spawn (full rebuild) has been requested, build a fresh model + optimizer
        # and seed it with the averaged weights received in the background thread.
        averaged_this_iteration = False
        if self._new_weights is not None and self._model_builder is not None:
            new_weights: Optional[Dict[str, torch.Tensor]] = None
            with self._pending_lock:
                new_weights = self._new_weights
                self._new_weights = None
            if new_weights is not None:
                rank = dist.get_rank() if dist.is_initialized() else 0
                logger.info(
                    "Rank %d: Rebuilding model with averaged weights (%d params)",
                    rank,
                    len(new_weights),
                )
                model, optimizer = self._model_builder()
                # Copy averaged params by name into the new model
                for name, param in model.named_parameters():
                    if name in new_weights:
                        param.data.copy_(new_weights[name])
                averaged_this_iteration = True

        return model, optimizer, {"averaged_this_iteration": averaged_this_iteration}

    # ---------------- Rank 0 metric aggregation and dispatch ----------------
    def _rank0_post_metric_recvs(self) -> None:
        world_size = dist.get_world_size()
        for r in range(1, world_size):
            if r not in self._rank0_metric_handles:
                buf = torch.zeros(2, dtype=torch.float32)
                work = dist.irecv(buf, src=r, tag=self._TAG_METRIC)
                assert work is not None
                self._rank0_metric_handles[r] = work
                self._rank0_metric_buffers[r] = buf

    def _rank0_poll_metrics(self) -> None:
        completed = []
        for r, work in list(self._rank0_metric_handles.items()):
            if work.is_completed():
                buf = self._rank0_metric_buffers[r]
                iteration = int(buf[0].item())
                metric = float(buf[1].item())
                world_size = dist.get_world_size()
                self._rank0_record_metric(iteration, r, metric, world_size)
                completed.append(r)
        for r in completed:
            del self._rank0_metric_handles[r]
            del self._rank0_metric_buffers[r]

    def _rank0_record_metric(
        self, iteration: int, rank: int, metric: float, world_size: int
    ) -> None:
        bucket = self._rank0_buckets.get(iteration)
        if bucket is None:
            bucket = {}
            self._rank0_buckets[iteration] = bucket
        bucket[rank] = metric
        logger.debug(
            "Rank 0: Received metric %.4f from rank %d for iteration %d (%d/%d)",
            metric,
            rank,
            iteration,
            len(bucket),
            world_size,
        )
        if len(bucket) == world_size:
            all_metrics = torch.zeros(world_size, dtype=torch.float32)
            for r, m in bucket.items():
                all_metrics[r] = m
            logger.debug(
                "Rank 0: All metrics collected for iteration %d: %s",
                iteration,
                all_metrics.tolist(),
            )
            # Gate dispatch using threshold and cooldown if configured
            should_dispatch = True
            if self.threshold is not None:
                # Cooldown window
                if iteration - self._last_trigger_iter < self.cooldown:
                    logger.debug(
                        "Rank 0: Skipping dispatch (cooldown: %d iters remaining)",
                        self.cooldown - (iteration - self._last_trigger_iter),
                    )
                    should_dispatch = False
                else:
                    # Trigger only if any metric falls below threshold
                    if torch.min(all_metrics).item() >= float(self.threshold):
                        logger.debug(
                            "Rank 0: Skipping dispatch (min metric %.4f >= threshold %.4f)",
                            torch.min(all_metrics).item(),
                            self.threshold,
                        )
                        should_dispatch = False

            if should_dispatch:
                ranks_to_replace, ranks_to_average = self._determine_ranks_for_averaging(
                    all_metrics,
                    world_size,
                    self.replacement_ratio,
                    self.averaging_strategy,
                )
                logger.info(
                    "Rank 0: Dispatching averaging for iteration %d - "
                    "replacing %s with average of %s",
                    iteration,
                    sorted(ranks_to_replace),
                    sorted(ranks_to_average),
                )
                weights = self._compute_averaging_weights(
                    all_metrics, ranks_to_average, self.averaging_strategy
                )
                self._rank0_dispatch_controls(
                    iteration, ranks_to_replace, ranks_to_average, weights
                )
                self._last_trigger_iter = int(iteration)
            del self._rank0_buckets[iteration]

    def _rank0_dispatch_controls(
        self,
        iteration: int,
        ranks_to_replace: Set[int],
        ranks_to_average: Set[int],
        weights: Optional[torch.Tensor],
    ) -> None:
        donors = sorted(list(ranks_to_average))
        replacers = sorted(list(ranks_to_replace))

        # Common header: [iteration, strategy_code]
        if self.averaging_strategy == "mean":
            strategy_code = 0
        elif self.averaging_strategy == "weighted_mean":
            strategy_code = 1
        elif self.averaging_strategy == "best_only":
            strategy_code = 2
        elif self.averaging_strategy == "reset_weights":
            strategy_code = 3
        else:
            strategy_code = 0
        header_common = torch.tensor(
            [int(iteration), int(strategy_code)], dtype=torch.int64
        )

        donors_tensor = torch.tensor(donors, dtype=torch.int64)
        replacers_tensor = torch.tensor(replacers, dtype=torch.int64)

        # Donors: send role, header, len(replacers), replacers list
        for donor in donors:
            try:
                role = torch.tensor([self._OP_ROLE_DONOR], dtype=torch.int64)
                dist.send(role, dst=donor, tag=self._TAG_CONTROL)
                dist.send(header_common, dst=donor, tag=self._TAG_CONTROL)
                size_t = torch.tensor([len(replacers)], dtype=torch.int64)
                dist.send(size_t, dst=donor, tag=self._TAG_CONTROL)
                if len(replacers) > 0:
                    dist.send(replacers_tensor, dst=donor, tag=self._TAG_CONTROL)
            except Exception:
                pass

        # Replacers: send role, header, len(donors), donors list, and optional weights
        for repl in replacers:
            try:
                role = torch.tensor([self._OP_ROLE_REPLACER], dtype=torch.int64)
                dist.send(role, dst=repl, tag=self._TAG_CONTROL)
                dist.send(header_common, dst=repl, tag=self._TAG_CONTROL)
                size_t = torch.tensor([len(donors)], dtype=torch.int64)
                dist.send(size_t, dst=repl, tag=self._TAG_CONTROL)
                if len(donors) > 0:
                    dist.send(donors_tensor, dst=repl, tag=self._TAG_CONTROL)
                if self.averaging_strategy == "weighted_mean" and len(donors) > 0:
                    assert weights is not None
                    wt = weights.to(dtype=torch.float32)
                    dist.send(wt, dst=repl, tag=self._TAG_CONTROL)
            except Exception:
                pass

    # ---------------- Donor / Replacer roles on non-zero ranks ----------------
    def _handle_donor(self, iteration: int, replacers: List[int]) -> None:
        assert self._model is not None
        rank = dist.get_rank()
        logger.info(
            "Rank %d: Acting as DONOR for iteration %d, sending to %s",
            rank,
            iteration,
            replacers,
        )
        param_count = 0
        for name, param in self._model.named_parameters():
            tensor = param.data.detach().clone()
            for dst in replacers:
                try:
                    dist.isend(tensor, dst=dst, tag=self._TAG_PARAM_BASE)
                except Exception as e:
                    logger.warning(
                        "Rank %d: Failed to send param %s to rank %d: %s",
                        rank,
                        name,
                        dst,
                        e,
                    )
            param_count += 1
        logger.debug("Rank %d: Sent %d parameters to replacers", rank, param_count)

    def _handle_replacer(
        self, iteration: int, donors: List[int], weights: Optional[torch.Tensor]
    ) -> None:
        assert self._model is not None
        rank = dist.get_rank()
        logger.info(
            "Rank %d: Acting as REPLACER for iteration %d, receiving from %s (strategy=%s)",
            rank,
            iteration,
            donors,
            self.averaging_strategy,
        )
        avg_state: Dict[str, torch.Tensor] = {}
        named_params = list(self._model.named_parameters())

        donors = donors if self.averaging_strategy != "best_only" else donors[:1]

        for name, param in named_params:
            device = param.device
            param_shape = param.data.shape
            acc = torch.zeros_like(param.data)
            for i, src in enumerate(donors):
                buf = torch.zeros(param_shape, dtype=param.data.dtype, device=device)
                try:
                    dist.recv(buf, src=src, tag=self._TAG_PARAM_BASE)
                except Exception as e:
                    logger.warning(
                        "Rank %d: Failed to receive param %s from rank %d: %s",
                        rank,
                        name,
                        src,
                        e,
                    )
                    continue
                if self.averaging_strategy == "mean":
                    acc.add_(buf)
                elif self.averaging_strategy == "weighted_mean":
                    assert weights is not None and len(weights) == len(donors)
                    acc.add_(weights[i].to(device) * buf)
                elif self.averaging_strategy == "best_only":
                    acc = buf
                    break
                else:
                    raise ValueError(
                        f"Unknown averaging strategy: {self.averaging_strategy}"
                    )
            if self.averaging_strategy == "mean" and len(donors) > 0:
                acc = acc / len(donors)
            avg_state[name] = acc

        logger.debug(
            "Rank %d: Received and averaged %d parameters", rank, len(avg_state)
        )
        with self._pending_lock:
            self._pending_spawn_avg_state = avg_state

    # ---------------- Background thread loop ----------------
    def _background_loop(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        rank = dist.get_rank()
        logger.debug("Rank %d: Background loop started", rank)
        if rank == 0:
            self._rank0_post_metric_recvs()
            while not self._shutdown.is_set():
                try:
                    self._rank0_poll_metrics()
                    self._rank0_post_metric_recvs()
                except Exception as e:
                    logger.warning("Rank 0: Error in background loop: %s", e)
                time.sleep(self.poll_interval_s)
            logger.debug("Rank 0: Background loop exiting")
            return

        # Other ranks: poll for control messages
        while not self._shutdown.is_set():
            try:
                if self._control_work is None:
                    # Post a single persistent irecv for control role
                    self._control_role_buf = torch.zeros(1, dtype=torch.int64)
                    self._control_work = dist.irecv(
                        self._control_role_buf, src=0, tag=self._TAG_CONTROL
                    )  # TODO: this can be recv and merged with below?
                else:
                    # Poll completion
                    if self._control_work.is_completed():
                        role = (
                            int(self._control_role_buf.item())
                            if self._control_role_buf is not None
                            else self._OP_NONE
                        )
                        # Reset for next cycle
                        self._control_work = None
                        self._control_role_buf = None

                        header = torch.zeros(2, dtype=torch.int64)
                        dist.recv(header, src=0, tag=self._TAG_CONTROL)
                        iteration = int(header[0].item())
                        strategy_code = int(header[1].item())

                        logger.debug(
                            "Rank %d: Received control message - role=%d, iteration=%d, strategy=%d",
                            rank,
                            role,
                            iteration,
                            strategy_code,
                        )

                        # list length
                        size_t = torch.zeros(1, dtype=torch.int64)
                        dist.recv(size_t, src=0, tag=self._TAG_CONTROL)
                        list_len = int(size_t.item())
                        ids = []
                        if list_len > 0:
                            ids_buf = torch.zeros(list_len, dtype=torch.int64)
                            dist.recv(ids_buf, src=0, tag=self._TAG_CONTROL)
                            ids = [int(x) for x in ids_buf.tolist()]

                        if role == self._OP_ROLE_DONOR:
                            replacers = [r for r in ids if r != dist.get_rank()]
                            self._handle_donor(iteration, replacers)
                        elif role == self._OP_ROLE_REPLACER:
                            donors = [d for d in ids if d != dist.get_rank()]
                            if strategy_code == 3:  # reset_weights
                                logger.info(
                                    "Rank %d: Reset weights requested for iteration %d",
                                    rank,
                                    iteration,
                                )
                                with self._pending_lock:
                                    # Signal a local rebuild with fresh random weights
                                    self._new_weights = {}
                            else:
                                wbuf = None
                                if (
                                    strategy_code == 1 and len(donors) > 0
                                ):  # weighted_mean
                                    wbuf = torch.zeros(len(donors), dtype=torch.float32)
                                    dist.recv(wbuf, src=0, tag=self._TAG_CONTROL)
                                self._handle_replacer(iteration, donors, wbuf)
                        else:
                            logger.debug(
                                "Rank %d: Unknown role %d, ignoring", rank, role
                            )
                time.sleep(self.poll_interval_s)
            except Exception as e:
                logger.warning("Rank %d: Error in background loop: %s", rank, e)
                time.sleep(self.poll_interval_s)
        logger.debug("Rank %d: Background loop exiting", rank)

    # ---------------- Local helpers (copied from selective policy to avoid lints) ----------------
    @staticmethod
    def _validate_params(
        replacement_ratio: float,
        averaging_strategy: str,
        momentum: float,
        threshold: Optional[float],
        cooldown: int,
    ) -> None:
        if not 0.0 <= replacement_ratio <= 1.0:
            raise ValueError(
                f"replacement_ratio must be between 0 and 1, got {replacement_ratio}"
            )
        if averaging_strategy not in [
            "mean",
            "weighted_mean",
            "best_only",
            "reset_weights",
        ]:
            raise ValueError(f"Unknown averaging_strategy: {averaging_strategy}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be between 0 and 1, got {momentum}")
        if cooldown < 0:
            raise ValueError(f"cooldown must be non-negative, got {cooldown}")

    @staticmethod
    def _determine_ranks_for_averaging(
        all_metrics: torch.Tensor,
        world_size: int,
        replacement_ratio: float,
        averaging_strategy: str,
    ) -> Tuple[Set[int], Set[int]]:
        n_replace = max(1, int(world_size * replacement_ratio))
        sorted_ranks = torch.argsort(all_metrics)
        ranks_to_replace = set(sorted_ranks[:n_replace].tolist())
        ranks_to_average: Set[int] = set(range(world_size)) - ranks_to_replace
        if averaging_strategy == "best_only":
            best_rank = int(sorted_ranks[-1].item())
            ranks_to_average = {best_rank}
        elif averaging_strategy == "reset_weights":
            # No donors when resetting weights locally
            ranks_to_average = set()
        return ranks_to_replace, ranks_to_average

    @staticmethod
    def _compute_averaging_weights(
        all_metrics: torch.Tensor, ranks_to_average: Set[int], averaging_strategy: str
    ) -> Optional[torch.Tensor]:
        if averaging_strategy != "weighted_mean":
            return None
        contributing_metrics = torch.tensor(
            [all_metrics[r] for r in sorted(ranks_to_average)], dtype=torch.float32
        )
        weights = 1.0 / (contributing_metrics + 1e-8)
        return weights / weights.sum()


"""
Selective averaging based on async mpi-3 comms
 Version 1 (fast-general): One buffer with all the params, hence utilizes the network BW much better than general version.
 Hence different model param dtypes can be handled.
 Comments:
      - Need a fix where there is one buffer for all params but in byte format,
      after comms, the buffer can be converted to appropriate dtypes for corr params dtypes.
      - Right now each param has its own buffer and window, which may produce incorrect output.
 Notes:
      - Only mean averaging strategy is implemented here for now.
      - Neighbors are randomly selected from all ranks except self.
      - Agents are killed based on age only for now.
"""


class AsyncSelectiveAveragingPolicympi4pyGeneral(SpawnPolicy):
    r"""
    Asynchronous selective averaging version 2, uses mpi one-sided comms to get the
    selectively averaged parameters from a random set of ranks.
    """

    def __init__(
        self,
        model_builder: Callable[[], Tuple[GFlowNet, torch.optim.Optimizer]],
        model: GFlowNet,
        average_every: int,
        threshold_metric: float = 0.0,
        replacement_ratio: float = 0.2,
        averaging_strategy: str = "mean",
        momentum: float = 0.0,
        poll_interval_s: float = 0.01,
        age_range: Tuple[int, int] = (50, 150),
        group: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(average_every)
        self.myrank = group.Get_rank()
        self.comm_size = group.Get_size()

        self.replacement_ratio = float(replacement_ratio)
        self.averaging_strategy = str(averaging_strategy)
        self.momentum = float(momentum)
        self._model_builder = model_builder

        self._model: Optional[GFlowNet] = None
        self.threshold_metric = float(threshold_metric)

        # timers
        self.timing = {}
        self.stats = {}

        self._model = model
        self.train_comm_group = group
        self._expose = False

        # new agents' stats
        self.agents_killed = 0
        self.averaging_ranks = 0
        self._count = 0

        self.debug_mode = False
        self.age = 0
        self.age_range = age_range
        self.max_age = random.randint(self.age_range[0], self.age_range[1])

        if self.debug_mode:
            self.logfile = f"debug/selective_averaging_rank_{self.myrank}.log"
            with open(self.logfile, "w") as f:
                f.write(f"Selective Averaging Log for Rank {self.myrank}\n")
                f.write("=" * 50 + "\n")

    def shutdown(self) -> None:
        for _, v in self._mpi_tensor_wins.items():
            v[0].Free()

    def print_time(self) -> None:
        print("Selective Averaging timings:", flush=True)
        for k, v in self.timing.items():
            # here v is a list, avg over the list
            # avg = sum(v) / len(v) if len(v) > 0 else 0
            print(f"{k:<35}: {sum(v):>10.4f} seconds")

    def print_stats(self) -> None:
        print("Selective Averaging comms stats:", flush=True)
        avg_donors, num_calls = 0.0, 0
        for k, v in self.stats.items():
            # v is a list, print min, max ,avg, and len of it
            minimum = min(v) if len(v) > 0 else 0
            maximum = max(v) if len(v) > 0 else 0
            avg = sum(v) / len(v) if len(v) > 0 else 0
            length = len(v)
            print(
                f"Rank {self.myrank} - Stat {k:30}: min={minimum}, max={maximum}, avg={avg:.6f}, count={length}"
            )
            if k == "donors":
                avg_donors = avg
                num_calls = length

        _named_params = (
            list(self._model.named_parameters()) if self._model is not None else []
        )
        named_params = [
            (name, param) for name, param in _named_params if param.dim() != 0
        ]
        print(
            f"Rank {self.myrank:<10} -  {'param elements':<15} {'iter':<10} {'total params elements commd':<25}"
        )
        for name, param in named_params:
            param.device
            param_shape = param.data.shape
            print(
                f"Rank {self.myrank:<10} -  {np.prod(param_shape):<15} {avg_donors*num_calls:<10} {np.prod(param_shape)*avg_donors*num_calls:<15}"
            )

    def capture_comm(self, name: str, size: int) -> None:
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(size)

    def _ensure_initialized(self, model: GFlowNet) -> None:
        self._model = model
        self._initialized = True
        # export model parameters to mpi windows (should do that periodically to keep them fresh)
        self._expose_model_parameters(model)

    def reset_age(self) -> None:
        self.max_age = random.randint(self.age_range[0], self.age_range[1])
        self.age = 0

    def is_agent_dying(
        self, local_metric: float, threshold_metric: float, check_agent=0
    ) -> bool:
        if check_agent == 0:  # static theshold
            return local_metric < threshold_metric

        elif check_agent == 1:  # dynamic threshold based on age
            if self.age >= self.max_age:
                print(
                    "+ Agent killed due to age: ",
                    self.age,
                    " max_age: ",
                    self.max_age,
                    flush=True,
                )
                self.reset_age()
                return True

            self.age += 1
            return False

        else:
            raise ValueError(f"Unknown is_agent_dying version: {check_agent}")

    # Execute this function regularly to copy model params to mpi windows
    # for recepotrs to get recent params
    def _copy_model_params_to_buf(
        self,
        model: GFlowNet,
    ) -> None:
        for name, param in model.named_parameters():
            win = self._mpi_tensor_wins[name][0]
            win.Lock(rank=self.myrank, lock_type=MPI.LOCK_EXCLUSIVE)
            self._mpi_tensor_wins[name][1][:] = param.data.cpu().numpy().flatten()
            win.Unlock(rank=self.myrank)

    def _expose_model_parameters(self, model: GFlowNet) -> None:

        # Serialize model parameters to a contiguous numpy array
        param_tensors = {}
        for name, param in model.named_parameters():
            param_tensors[name] = np.zeros_like(param.data.cpu().numpy().flatten())

        # Create MPI windows for each parameter and its shape (2 separate windows set)
        self._mpi_tensor_wins = {}
        self._mpi_shape_wins = {}
        assert isinstance(self.train_comm_group, MPI.Intracomm)
        comm = cast(MPI.Intracomm, self.train_comm_group)
        for name, tensor in param_tensors.items():
            buf = tensor
            win = MPI.Win.Create(buf, comm=comm)
            self._mpi_tensor_wins[name] = (win, buf)

        self._copy_model_params_to_buf(model)

    def _get_donors(self, n, k, d) -> List[int]:
        if k > n - 1:
            raise ValueError("k must be ≤ n-1 when excluding one value")

        # All values from 0..n-1 except d
        candidates = [x for x in range(n) if x != d]
        # Pick k distinct values
        return random.sample(candidates, k)

    def __call__(
        self,
        iteration: int,
        model: GFlowNet,
        optimizer: torch.optim.Optimizer,
        local_metric: float,
        expose_params: bool = True,
        group: MPI.Comm = MPI.COMM_WORLD,
    ) -> Tuple[GFlowNet, torch.optim.Optimizer, dict]:

        if self._expose is False:
            self._expose_model_parameters(model)
            self._expose = True

        self._count += 1
        self._model = model

        check_agent = 1  # version of dying agent check
        # validation info
        layer_name = None
        if self.debug_mode:
            layer_name = "pb.module.last_layer.bias"

        if self.is_agent_dying(local_metric, self.threshold_metric, check_agent):
            with Timer(self.timing, "sa get_params_from_donors"):
                if self.debug_mode:
                    with open(self.logfile, "a") as f:
                        # kill this model and rebuild model with fresh weights
                        num_donors = max(
                            1, int(self.comm_size * 0.5)
                        )  # * self.replacement_ratio))
                        donors = self._get_donors(
                            self.comm_size, num_donors, self.myrank
                        )
                        new_avg_params = self._get_model_params_from_donors(
                            donors, layer_name, f
                        )

                        if layer_name is not None:
                            json.dump(
                                {
                                    self._count: [
                                        self._mpi_tensor_wins[layer_name][1].tolist(),
                                        donors,
                                        new_avg_params[layer_name].tolist(),
                                    ]
                                },
                                f,
                            )
                        f.write("\n")
                else:
                    num_donors = max(
                        1, int(self.comm_size * 0.5)
                    )  # self.replacement_ratio))
                    donors = self._get_donors(self.comm_size, num_donors, self.myrank)
                    new_avg_params = self._get_model_params_from_donors(
                        donors, layer_name, None
                    )

            with Timer(self.timing, "sa new_agent_model_rebuild"):
                model, optimizer = self._model_builder()
                for name, param in model.named_parameters():
                    if name in new_avg_params:
                        param.data.copy_(new_avg_params[name])

        if expose_params is True:
            with Timer(self.timing, "sa copy_params_to_buf"):
                self._copy_model_params_to_buf(model)

        return model, optimizer, {"averaged_this_iteration": True}

    def _get_model_params_from_donors(
        self, donors: List[int], layer_name, f
    ) -> Dict[str, torch.Tensor]:
        avg_state: Dict[str, torch.Tensor] = {}
        _named_params = (
            list(self._model.named_parameters()) if self._model is not None else []
        )
        named_params = [
            (name, param) for name, param in _named_params if param.dim() != 0
        ]
        tot_comm_ele = 0

        self.capture_comm("donors", len(donors))
        for name, param in named_params:
            device = param.device
            param_shape = param.data.shape
            acc = torch.zeros_like(param.data)
            all_donors = []

            for i, src in enumerate(donors):
                tensor_win, tensor_buf = self._mpi_tensor_wins[name]
                tensor_win.Lock(rank=src, lock_type=MPI.LOCK_SHARED)
                flat_size = np.prod(param_shape)
                assert flat_size > 0
                tot_comm_ele += flat_size
                donor_tensor_flat = np.zeros(
                    flat_size, dtype=param.data.cpu().numpy().dtype
                )
                tensor_win.Get([donor_tensor_flat, MPI.FLOAT], target_rank=src)
                tensor_win.Unlock(rank=src)

                donor_tensor = torch.tensor(
                    donor_tensor_flat.reshape(param_shape), device=device
                )
                # Adding all the donor tensors/params
                acc.add_(donor_tensor)

                if self.debug_mode and name == layer_name:
                    all_donors.append(donor_tensor.tolist())
                # Additions: Other averaging strategies can be implemented here

            if self.debug_mode and name == layer_name:
                json.dump({self._count: all_donors}, f)
                f.write("\n")

            # default to mean averaging
            acc = acc / len(donors)
            avg_state[name] = acc

        self.capture_comm("num_param_tensors_received", tot_comm_ele)
        return avg_state

    def _average_received_params(
        self,
    ) -> Dict[str, torch.Tensor]:
        avg_state: Dict[str, torch.Tensor] = {}
        for name, param in self.avg_state.items():
            # device = param.device
            # param_shape = param.data.shape
            acc = torch.zeros_like(param[0].data)

            for i, donor_tensor in enumerate(param):
                # Adding all the donor tensors/params
                acc.add_(donor_tensor)

            # default to mean averaging
            if self.averaging_strategy == "mean":
                acc = acc / len(param)
                avg_state[name] = acc

        return avg_state


"""
Selective averaging based on async mpi-3 comms
 Version 2 (fast): One buffer with all the params, hence utilizes the network BW much better than general version.
 It assumes all the params are of same dtype (float32)
Notes:
      - Only mean averaging strategy is implemented here for now.
      - Neighbors are randomly selected from all ranks except self.
      - Agents are killed based on age only for now.
"""


class AsyncSelectiveAveragingPolicympi4pyFast(SpawnPolicy):
    r"""
    Asynchronous selective averaging version 2, uses mpi one-sided comms to get the
    selectively averaged parameters from a random set of ranks.
    """

    def __init__(
        self,
        model_builder: Callable[[], Tuple[GFlowNet, torch.optim.Optimizer]],
        model: GFlowNet,
        average_every: int,
        threshold_metric: float = 0.0,
        replacement_ratio: float = 0.2,
        averaging_strategy: str = "mean",
        momentum: float = 0.0,
        age_range: Tuple[int, int] = (50, 150),
        group: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(average_every)
        self.myrank = group.Get_rank()
        self.comm_size = group.Get_size()

        self.replacement_ratio = float(replacement_ratio)
        self.averaging_strategy = str(averaging_strategy)
        self.momentum = float(momentum)
        self._model_builder = model_builder

        self._model: Optional[GFlowNet] = None
        self.threshold_metric = float(threshold_metric)
        # timers
        self.timing = {}
        self.stats = {}

        self._model = model
        self.train_comm_group = group
        # self._expose_model_parameters(model)
        self._expose = False

        # **** new agents' stats ****
        self.agents_killed = 0
        self.averaging_ranks = 0
        self._count = 0

        self.total_iterations = 0
        self.num_replacements = 0
        self.debug_mode = False

        self.age = 0
        self.age_range = age_range
        self.max_age = random.randint(self.age_range[0], self.age_range[1])

        # test code, remove it later
        if self.debug_mode:
            self.logfile = f"debug/selective_averaging_rank_{self.myrank}.log"
            with open(self.logfile, "w") as f:
                f.write(f"Selective Averaging Log for Rank {self.myrank}\n")
                f.write("=" * 50 + "\n")

    def shutdown(self) -> None:
        self._mpi_tensor_wins[0].Free()

    def print_time(self) -> None:
        print("Selective Averaging timings:", flush=True)
        for k, v in self.timing.items():
            # here v is a list, avg over the list
            print(f"{k:<35}: {sum(v):>10.4f} seconds")

    def print_stats(self) -> None:
        print("Selective Averaging comms stats:", flush=True)
        print(
            f"Rank {self.myrank} - Agent replaced for {self.num_replacements} out of {self.total_iterations} iterations."
        )
        avg_donors, num_calls = 0.0, 0
        for k, v in self.stats.items():
            # v is a list, print min, max ,avg, and len of it
            minimum = min(v) if len(v) > 0 else 0
            maximum = max(v) if len(v) > 0 else 0
            avg = sum(v) / len(v) if len(v) > 0 else 0
            length = len(v)
            print(
                f"Rank {self.myrank:<10} - Stat {k:30}: min={minimum}, max={maximum}, avg={avg:.6f}, across {length} iters"
            )
            if k == "donors":
                avg_donors = avg
                num_calls = length

        _named_params = (
            list(self._model.named_parameters()) if self._model is not None else []
        )
        named_params = [
            (name, param) for name, param in _named_params if param.dim() != 0
        ]
        print(
            f"Rank {self.myrank:<10} -  {'param elements':<15} {'#comm_iters':<10} {'total params elements communicated':<25}"
        )
        for name, param in named_params:
            param.device
            param_shape = param.data.shape
            print(
                f"Rank {self.myrank:<10} -  {np.prod(param_shape):<15} {avg_donors*num_calls:<10} {np.prod(param_shape)*avg_donors*num_calls:<15}"
            )

    def capture_comm(self, name: str, size: int) -> None:
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(size)

    def _ensure_initialized(self, model: GFlowNet) -> None:
        self._model = model
        self._initialized = True
        # export model parameters to mpi windows (should do that periodically to keep them fresh)
        self._expose_model_parameters(model)

    def reset_age(self) -> None:
        self.max_age = random.randint(self.age_range[0], self.age_range[1])
        self.age = 0

    def is_agent_dying(
        self, local_metric: float, threshold_metric: float, check_policy=0
    ) -> bool:
        if check_policy == 0:  # static theshold
            return local_metric < threshold_metric

        elif check_policy == 1:  # dynamic threshold based on age
            if self.age >= self.max_age:
                print(
                    "+ Agent killed due to age: ",
                    self.age,
                    " max_age: ",
                    self.max_age,
                    flush=True,
                )
                self.reset_age()
                return True

            self.age += 1
            return False

        else:
            raise ValueError(f"Unknown is_agent_dying version: {check_policy}")

    # Execute this function regularly to copy model params to mpi windows
    # for recepotrs to get recent params
    def _copy_model_params_to_buf(
        self,
        model: GFlowNet,
    ) -> None:
        offset = 0
        for name, param in model.named_parameters():
            win = self._mpi_tensor_wins[0]
            win.Lock(rank=self.myrank, lock_type=MPI.LOCK_EXCLUSIVE)
            size = param.data.numel()
            self._mpi_tensor_wins[1][offset : offset + size] = (
                param.data.cpu().numpy().flatten()
            )
            offset += size
            win.Unlock(rank=self.myrank)

    def _expose_model_parameters(self, model: GFlowNet) -> None:
        print("+ Exposing model parameters via MPI windows...", flush=True)
        # Serialize model parameters to a contiguous numpy array
        param_size = 0
        {param.dtype for param in model.parameters()}
        # todo: enable this to work with any dtypes for the model
        # print('model dtypes: ', dtypes)
        param_dtype = np.float32
        th_param_dtype = torch.float32

        # self.param_shapes = {}
        for _, param in model.named_parameters():
            param_size += param.data.numel()
            param_dtype = param.data.cpu().numpy().dtype

        param_tensors_flat = np.zeros(param_size, dtype=param_dtype)
        self.donor_tensor_flat = torch.zeros(param_size, dtype=th_param_dtype)
        self.acc = torch.zeros(param_size, dtype=th_param_dtype)

        # Create MPI windows for the flat parameter tensor
        buf = param_tensors_flat
        assert isinstance(self.train_comm_group, MPI.Intracomm)
        cast(MPI.Intracomm, self.train_comm_group)
        win = MPI.Win.Create(buf, comm=self.train_comm_group)
        # buffer attached to the win, used to copy data in/out of the win
        self._mpi_tensor_wins = (win, buf)

        self._copy_model_params_to_buf(model)

    def _get_donors(self, n, k, d) -> List[int]:
        if k > n - 1:
            raise ValueError("k must be ≤ n-1 when excluding one value")

        # All values from 0..n-1 except d
        candidates = [x for x in range(n) if x != d]
        # Random policy Pick k distinct random values
        return random.sample(candidates, k)

    def __call__(
        self,
        iteration: int,
        model: GFlowNet,
        optimizer: torch.optim.Optimizer,
        local_metric: float,
        expose_params: bool = True,
        group: MPI.Comm = MPI.COMM_WORLD,
    ) -> Tuple[GFlowNet, torch.optim.Optimizer, dict]:

        if self._expose is False:
            self._expose_model_parameters(model)
            self._expose = True

        self._count += 1
        self._model = model
        named_params = list(model.named_parameters())
        for name, param in named_params:
            param.data.shape

        # validation info
        layer_name = None
        if self.debug_mode:
            layer_name = "pb.module.last_layer.bias"

        self.total_iterations += 1
        check_agent = 1  # 0: static thresholding, 1: dynamic based on age
        if self.is_agent_dying(local_metric, self.threshold_metric, check_agent):
            self.num_replacements += 1
            with Timer(self.timing, "sa get_params_from_donors"):
                # kill this model and rebuild model with fresh weights
                num_donors = max(
                    1, int(self.comm_size * 0.5)
                )  # <<<<< parameterize this one
                donors = self._get_donors(self.comm_size, num_donors, self.myrank)
                _new_avg_params = self._get_model_params_from_donors(
                    donors, layer_name, None
                )

            with Timer(self.timing, "sa param_list_to_dict_convert"):
                # conver the flat tensor to model param dict
                new_avg_params: Dict[str, torch.Tensor] = {}
                win_buf: Dict[str, torch.Tensor] = {}

                offset = 0
                for name, param in model.named_parameters():
                    device = param.device
                    flat_size = param.data.numel()
                    assert flat_size == np.prod(param.data.shape)
                    donor_tensor_flat = _new_avg_params[offset : offset + flat_size]
                    donor_tensor = torch.tensor(
                        donor_tensor_flat.reshape(param.data.shape), device=device
                    )
                    if self.debug_mode:
                        buf_tensor_flat = self._mpi_tensor_wins[1][
                            offset : offset + flat_size
                        ]
                        buf_tensor = torch.tensor(
                            buf_tensor_flat.reshape(param.data.shape), device=device
                        )
                        win_buf[name] = buf_tensor

                    new_avg_params[name] = donor_tensor
                    offset += flat_size

            if self.debug_mode:
                with open(self.logfile, "a") as f:
                    if layer_name is not None:
                        json.dump(
                            {
                                self._count: [
                                    win_buf[layer_name].tolist(),
                                    donors,
                                    new_avg_params[layer_name].tolist(),
                                ]
                            },
                            f,
                        )
                        f.write("\n")

            with Timer(self.timing, "sa new_agent_model_rebuild"):
                model, optimizer = self._model_builder()
                for name, param in model.named_parameters():
                    if name in new_avg_params:
                        param.data.copy_(new_avg_params[name])

        if expose_params is True:
            with Timer(self.timing, "sa copy_params_to_buf"):
                self._copy_model_params_to_buf(model)

        return model, optimizer, {"averaged_this_iteration": True}

    def _get_model_params_from_donors(
        self, donors: List[int], layer_name, f
    ) -> torch.Tensor:
        # avg_state: Dict[str, torch.Tensor] = {}
        # _named_params = list(self._model.named_parameters())
        # named_params = [(name, param) for name, param in _named_params if param.dim() != 0]
        tot_comm_ele = 0

        self.capture_comm("donors", len(donors))
        tensor_win, tensor_buf = self._mpi_tensor_wins
        flat_size = tensor_buf.size
        self.donor_tensor_flat.zero_()
        self.acc.zero_()

        if self.averaging_strategy == "mean":
            for i, src in enumerate(donors):
                tensor_win.Lock(rank=src, lock_type=MPI.LOCK_SHARED)
                tensor_win.Get([self.donor_tensor_flat, MPI.FLOAT], target_rank=src)
                tensor_win.Unlock(rank=src)

                # Adding all the donor tensors/params
                self.acc.add_(self.donor_tensor_flat)
                tot_comm_ele = tot_comm_ele + flat_size

            self.acc = self.acc / len(donors)
        else:
            raise ValueError(f"Unknown averaging strategy: {self.averaging_strategy}")

        self.capture_comm("num_param_tensors_received", tot_comm_ele)
        return self.acc


class AverageAllPolicympi4py(SpawnPolicy):
    """Standard model averaging across all ranks every N iterations."""

    def __init__(self, average_every: int) -> None:
        super().__init__(average_every)

    @torch.no_grad()
    def __call__(
        self,
        iteration: int,
        model: GFlowNet,
        optimizer: torch.optim.Optimizer,
        local_metric: Optional[float] = None,
        group: MPI.Comm = MPI.COMM_WORLD,
    ) -> Tuple[GFlowNet, torch.optim.Optimizer, dict]:
        if not dist.is_available() or not dist.is_initialized():
            return model, optimizer, {}
        if iteration % self.average_every != 0:
            return model, optimizer, {"averaged_this_iteration": False}

        # print("AverageAll mpi4py model parameters across all ranks ...", flush=True)
        world_size = group.Get_size()
        for param in model.parameters():
            param_tensor = (
                param.detach().cpu().numpy().copy()
            )  # param.data.clone().numpy()
            # dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM, group=group)
            group.Allreduce(MPI.IN_PLACE, param_tensor, op=MPI.SUM)
            param_tensor /= world_size
            param.data.copy_(torch.from_numpy(param_tensor))

        return model, optimizer, {"averaged_this_iteration": True}
