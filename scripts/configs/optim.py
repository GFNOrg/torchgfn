import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from gfn.losses import Parametrization


@dataclass
class BaseOptimConfig(ABC):
    lr: float = 1e-3
    lr_Z: Optional[float] = 0.1

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
        if any(["logZ" in p for p in parametrization.parameters.keys()]):
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

        return params

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
        params = super().get_params(parametrization)
        optimizer = torch.optim.Adam(params, betas=self.betas)
        return optimizer


@dataclass
class SGDConfig(BaseOptimConfig):
    momentum: float = 0.0

    def parse(
        self, parametrization: Parametrization
    ) -> Tuple[torch.optim.Optimizer, Any]:
        params = super().get_params(parametrization)
        optimizer = torch.optim.SGD(params, momentum=self.momentum)
        return optimizer


def make_optim(config: dict, parametrization: Parametrization) -> torch.optim.Optimizer:
    name = config["optim"]["name"]
    if name.lower() == "sgd".lower():
        optim_class = SGDConfig
    elif name.lower() == "adam".lower():
        optim_class = AdamConfig
    else:
        raise ValueError(f"Invalid optim name: {name}")

    args = inspect.getfullargspec(optim_class.__init__).args
    optim_config = {k: v for k, v in config["optim"].items() if k in args}
    optimizer = optim_class(**optim_config).parse(parametrization)
    print(
        "\nDon't worry if you see a PyTorch warning about duplicate parameters",
        "this is expected for now and will be fixed in a future release of `torchgfn`.\n",
    )
    return optimizer
