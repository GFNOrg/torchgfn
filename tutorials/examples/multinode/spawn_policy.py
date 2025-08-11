from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Set

import torch
import torch.distributed as dist


class SpawnPolicy(ABC):
    def __init__(self, average_every: int) -> None:
        self.average_every = max(int(average_every), 1)

    @abstractmethod
    def __call__(self, iteration: int, model: torch.nn.Module, local_metric: Optional[float] = None) -> None:  # noqa: D401
        """Possibly perform a spawn/averaging step on this iteration."""
        raise NotImplementedError


class AverageAllPolicy(SpawnPolicy):
    """Standard model averaging across all ranks every N iterations."""

    def __init__(self, average_every: int) -> None:
        super().__init__(average_every)

    @torch.no_grad()
    def __call__(self, iteration: int, model: torch.nn.Module, local_metric: Optional[float] = None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        if iteration % self.average_every != 0:
            return

        world_size = float(dist.get_world_size())
        for param in model.parameters():
            param_tensor = param.data.clone()
            dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            param.data.copy_(param_tensor / world_size)


class SelectiveAveragingPolicy(SpawnPolicy):
    """Replace worst performing models with averaged weights from better ones.

    Supports strategies: "mean", "weighted_mean", "best_only", with optional momentum.
    """

    def __init__(
        self,
        average_every: int,
        replacement_ratio: float = 0.2,
        averaging_strategy: str = "mean",
        momentum: float = 0.0,
    ) -> None:
        super().__init__(average_every)
        self.replacement_ratio = float(replacement_ratio)
        self.averaging_strategy = str(averaging_strategy)
        self.momentum = float(momentum)

    @torch.no_grad()
    def __call__(self, iteration: int, model: torch.nn.Module, local_metric: Optional[float] = None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        if iteration % self.average_every != 0:
            return
        if local_metric is None:
            return

        world_size = dist.get_world_size()
        if world_size <= 1 or self.replacement_ratio <= 0.0:
            return

        self._validate_params(self.replacement_ratio, self.averaging_strategy, self.momentum)

        # Gather performance metrics from all ranks
        all_metrics = self._gather_performance_metrics(local_metric)

        rank = dist.get_rank()
        # Determine ranks to replace/average on rank 0
        if rank == 0:
            ranks_to_replace, ranks_to_average = self._determine_ranks_for_averaging(
                all_metrics, world_size, self.replacement_ratio, self.averaging_strategy
            )
            averaging_weights = self._compute_averaging_weights(all_metrics, ranks_to_average, self.averaging_strategy)
        else:
            ranks_to_replace, ranks_to_average, averaging_weights = set(), set(), None

        # Broadcast info to all ranks
        (
            ranks_to_replace,
            ranks_to_average_list,
            ranks_to_average,
            averaging_weights,
        ) = self._broadcast_rank_info(ranks_to_replace, ranks_to_average, averaging_weights, self.averaging_strategy)

        dist.barrier()

        # Update weights for ranks that need replacement; all ranks participate in comms
        for _, param in model.named_parameters():
            new_weights = self._compute_averaged_weights(
                param,
                ranks_to_average_list,
                averaging_weights,
                self.averaging_strategy,
                rank,
                ranks_to_replace,
                self.momentum,
            )
            if rank in ranks_to_replace:
                param.data.copy_(new_weights)

        dist.barrier()

    @staticmethod
    def _validate_params(replacement_ratio: float, averaging_strategy: str, momentum: float) -> None:
        if not 0.0 <= replacement_ratio <= 1.0:
            raise ValueError(f"replacement_ratio must be between 0 and 1, got {replacement_ratio}")
        if averaging_strategy not in ["mean", "weighted_mean", "best_only"]:
            raise ValueError(f"Unknown averaging_strategy: {averaging_strategy}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be between 0 and 1, got {momentum}")

    @staticmethod
    def _gather_performance_metrics(local_metric: float) -> torch.Tensor:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        local_tensor = torch.tensor([local_metric], dtype=torch.float32)
        if rank == 0:
            gathered_metrics: List[torch.Tensor] = [torch.zeros(1, dtype=torch.float32) for _ in range(world_size)]
        else:
            gathered_metrics = []
        dist.gather(local_tensor, gather_list=gathered_metrics if rank == 0 else None, dst=0)
        if rank == 0:
            return torch.cat(gathered_metrics)
        else:
            # Return a dummy tensor of the same shape on non-zero ranks
            return torch.empty(0, dtype=torch.float32)

    @staticmethod
    def _determine_ranks_for_averaging(
        all_metrics: torch.Tensor, world_size: int, replacement_ratio: float, averaging_strategy: str
    ) -> Tuple[Set[int], Set[int]]:
        n_replace = max(1, int(world_size * replacement_ratio))
        sorted_ranks = torch.argsort(all_metrics)
        ranks_to_replace = set(sorted_ranks[:n_replace].tolist())
        ranks_to_average: Set[int] = set(range(world_size)) - ranks_to_replace
        if averaging_strategy == "best_only":
            best_rank = int(sorted_ranks[-1].item())
            ranks_to_average = {best_rank}
        return ranks_to_replace, ranks_to_average

    @staticmethod
    def _compute_averaging_weights(
        all_metrics: torch.Tensor, ranks_to_average: Set[int], averaging_strategy: str
    ) -> Optional[torch.Tensor]:
        if averaging_strategy != "weighted_mean":
            return None
        contributing_metrics = torch.tensor([all_metrics[r] for r in sorted(ranks_to_average)], dtype=torch.float32)
        weights = 1.0 / (contributing_metrics + 1e-8)
        return weights / weights.sum()

    @staticmethod
    def _broadcast_rank_info(
        ranks_to_replace: Set[int],
        ranks_to_average: Set[int],
        averaging_weights: Optional[torch.Tensor],
        averaging_strategy: str,
    ) -> Tuple[Set[int], List[int], Set[int], Optional[torch.Tensor]]:
        rank = dist.get_rank()

        ranks_to_replace_tensor = torch.tensor(sorted(list(ranks_to_replace)), dtype=torch.long)
        ranks_to_average_tensor = torch.tensor(sorted(list(ranks_to_average)), dtype=torch.long)

        replace_size = torch.tensor([int(ranks_to_replace_tensor.numel())], dtype=torch.long)
        average_size = torch.tensor([int(ranks_to_average_tensor.numel())], dtype=torch.long)

        dist.broadcast(replace_size, src=0)
        dist.broadcast(average_size, src=0)

        if rank != 0:
            ranks_to_replace_tensor = torch.zeros(int(replace_size.item()), dtype=torch.long)
            ranks_to_average_tensor = torch.zeros(int(average_size.item()), dtype=torch.long)

        dist.broadcast(ranks_to_replace_tensor, src=0)
        dist.broadcast(ranks_to_average_tensor, src=0)

        weights_tensor: Optional[torch.Tensor] = None
        if averaging_strategy == "weighted_mean":
            if rank == 0:
                weights_tensor = averaging_weights
            else:
                weights_tensor = torch.zeros(int(average_size.item()), dtype=torch.float32)
            dist.broadcast(weights_tensor, src=0)

        return (
            set(ranks_to_replace_tensor.tolist()),
            ranks_to_average_tensor.tolist(),
            set(ranks_to_average_tensor.tolist()),
            weights_tensor,
        )

    @staticmethod
    def _compute_averaged_weights(
        param: torch.nn.Parameter,
        ranks_to_average_list: List[int],
        averaging_weights: Optional[torch.Tensor],
        averaging_strategy: str,
        rank: int,
        ranks_to_replace: Set[int],
        momentum: float = 0.0,
    ) -> torch.Tensor:
        param_sum = torch.zeros_like(param.data)
        old_weights = param.data.clone()

        for i, contributing_rank in enumerate(ranks_to_average_list):
            param_copy = param.data.clone() if contributing_rank == rank else torch.zeros_like(param.data)
            dist.broadcast(param_copy, src=contributing_rank)
            if rank in ranks_to_replace:
                if averaging_strategy == "mean":
                    param_sum.add_(param_copy)
                elif averaging_strategy == "weighted_mean":
                    if averaging_weights is None:
                        raise RuntimeError("averaging_weights should not be None for weighted_mean strategy")
                    weight = averaging_weights[i]
                    param_sum.add_(weight * param_copy)
                elif averaging_strategy == "best_only":
                    if i == 0:
                        averaged_weights = param_copy
                        return momentum * old_weights + (1.0 - momentum) * averaged_weights

        if averaging_strategy == "mean":
            averaged_weights = param_sum / max(1, len(ranks_to_average_list))
        elif averaging_strategy == "weighted_mean":
            averaged_weights = param_sum
        elif averaging_strategy == "best_only":
            averaged_weights = param_sum
        else:
            raise ValueError(f"Unknown averaging strategy: {averaging_strategy}")

        return momentum * old_weights + (1.0 - momentum) * averaged_weights
