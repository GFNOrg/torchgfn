from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from gfn.gflownet.base import GFlowNet


class SpawnPolicy(ABC):
    def __init__(self, average_every: int) -> None:
        self.average_every = max(int(average_every), 1)

    @abstractmethod
    def __call__(
        self,
        iteration: int,
        model: GFlowNet,
        local_metric: Optional[float] = None,
    ) -> Tuple[GFlowNet, torch.optim.Optimizer]:
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
    ) -> Tuple[GFlowNet, torch.optim.Optimizer]:
        if not dist.is_available() or not dist.is_initialized():
            return model, optimizer
        if iteration % self.average_every != 0:
            return model, optimizer

        world_size = float(dist.get_world_size())
        for param in model.parameters():
            param_tensor = param.data.clone()
            dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            param.data.copy_(param_tensor / world_size)

        return model, optimizer


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
    ) -> None:
        super().__init__(average_every)
        self.replacement_ratio = float(replacement_ratio)
        self.averaging_strategy = str(averaging_strategy)
        self.momentum = float(momentum)
        self.poll_interval_s = float(poll_interval_s)
        self._model_builder = model_builder

        self._initialized = False
        self._model: Optional[GFlowNet] = None
        self._shutdown = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._pending_lock = threading.Lock()
        self._last_iter_sent: int = -1

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
            self.replacement_ratio, self.averaging_strategy, self.momentum
        )
        self._model = model
        self._initialized = True
        self._bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._bg_thread.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1.0)

    @torch.no_grad()
    def __call__(
        self,
        iteration: int,
        model: GFlowNet,
        optimizer: torch.optim.Optimizer,
        local_metric: Optional[float] = None,
    ) -> Tuple[GFlowNet, torch.optim.Optimizer]:
        self._ensure_initialized(model)
        if not dist.is_available() or not dist.is_initialized():
            return model, optimizer

        # Non-blocking metric send on cadence
        if local_metric is not None and iteration % self.average_every == 0:
            rank = dist.get_rank()
            if iteration != self._last_iter_sent:
                self._last_iter_sent = iteration
                payload = torch.tensor(
                    [float(iteration), float(local_metric)], dtype=torch.float32
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
                except Exception:
                    pass

        # If a spawn (full rebuild) has been requested, build a fresh model + optimizer
        # and seed it with the averaged weights received in the background thread.
        if self._new_weights is not None and self._model_builder is not None:
            new_weights: Optional[Dict[str, torch.Tensor]] = None
            with self._pending_lock:
                new_weights = self._new_weights
                self._new_weights = None
            if new_weights is not None:
                model, optimizer = self._model_builder()
                # Copy averaged params by name into the new model
                for name, param in model.named_parameters():
                    if name in new_weights:
                        param.data.copy_(new_weights[name])

        return model, optimizer

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
        if len(bucket) == world_size:
            all_metrics = torch.zeros(world_size, dtype=torch.float32)
            for r, m in bucket.items():
                all_metrics[r] = m
            ranks_to_replace, ranks_to_average = self._determine_ranks_for_averaging(
                all_metrics, world_size, self.replacement_ratio, self.averaging_strategy
            )
            weights = self._compute_averaging_weights(
                all_metrics, ranks_to_average, self.averaging_strategy
            )
            self._rank0_dispatch_controls(
                iteration, ranks_to_replace, ranks_to_average, weights
            )
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
        for name, param in self._model.named_parameters():
            tensor = param.data.detach().clone()
            for dst in replacers:
                try:
                    dist.isend(tensor, dst=dst, tag=self._TAG_PARAM_BASE)
                except Exception:
                    pass

    def _handle_replacer(
        self, iteration: int, donors: List[int], weights: Optional[torch.Tensor]
    ) -> None:
        assert self._model is not None
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
                except Exception:
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

        with self._pending_lock:
            self._pending_spawn_avg_state = avg_state

    # ---------------- Background thread loop ----------------
    def _background_loop(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        rank = dist.get_rank()
        if rank == 0:
            self._rank0_post_metric_recvs()
            while not self._shutdown.is_set():
                try:
                    self._rank0_poll_metrics()
                    self._rank0_post_metric_recvs()
                except Exception:
                    pass
                time.sleep(self.poll_interval_s)
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
                            pass
                time.sleep(self.poll_interval_s)
            except Exception:
                time.sleep(self.poll_interval_s)

    # ---------------- Local helpers (copied from selective policy to avoid lints) ----------------
    @staticmethod
    def _validate_params(
        replacement_ratio: float, averaging_strategy: str, momentum: float
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
