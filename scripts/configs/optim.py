from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from simple_parsing import subgroups
from simple_parsing.helpers import JsonSerializable

from gfn.losses import Parametrization


@dataclass
class BaseOptimConfig(JsonSerializable, ABC):
    lr: float = 1e-3
    lr_Z: Optional[float] = 0.1

    scheduler_gamma: Optional[float] = None
    scheduler_milestones: Optional[List[int]] = None

    def get_params(
        self, parametrization: Parametrization
    ) -> Tuple[List[Dict[str, List[torch.Tensor]]], Tuple[float, List[int]]]:
        params = [
            {
                "params": [
                    val
                    for key, val in parametrization.parameters.items()
                    if "logZ" not in key
                ],
                "lr": self.lr,
            }
        ]
        if "logZ" in parametrization.parameters:
            params.append(
                {
                    "params": [
                        val
                        for key, val in parametrization.parameters.items()
                        if "logZ" in key
                    ],
                    "lr": self.lr_Z,
                }
            )
        else:
            self.lr_Z = None

        if self.scheduler_gamma is None or self.scheduler_milestones is None:
            scheduler_gamma = 1.0
            scheduler_milestones = [0]
        else:
            scheduler_gamma = self.scheduler_gamma
            scheduler_milestones = self.scheduler_milestones

        return (params, (scheduler_gamma, scheduler_milestones))

    @abstractmethod
    def parse(
        self, parametrization: Parametrization
    ) -> Tuple[torch.optim.Optimizer, Any]:
        pass


@dataclass
class AdamConfig(BaseOptimConfig):
    betas: Tuple[float, float] = (0.9, 0.999)

    def parse(
        self, parametrization: Parametrization
    ) -> Tuple[torch.optim.Optimizer, Any]:
        (params, (scheduler_gamma, scheduler_milestones)) = super().get_params(
            parametrization
        )
        optimizer = torch.optim.Adam(params, betas=self.betas)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
        )
        return (optimizer, scheduler)


@dataclass
class SGDConfig(BaseOptimConfig):
    momentum: float = 0.0

    def parse(
        self, parametrization: Parametrization
    ) -> Tuple[torch.optim.Optimizer, Any]:
        (params, (scheduler_gamma, scheduler_milestones)) = super().get_params(
            parametrization
        )
        optimizer = torch.optim.SGD(params, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
        )
        return (optimizer, scheduler)


@dataclass
class OptimConfig(JsonSerializable):
    optim: BaseOptimConfig = subgroups(
        {"sgd": SGDConfig, "adam": AdamConfig}, default=AdamConfig()
    )

    def parse(
        self, parametrization: Parametrization
    ) -> Tuple[torch.optim.Optimizer, Any]:
        return self.optim.parse(parametrization)
