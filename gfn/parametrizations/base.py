from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import torch

from ..envs import Env
from ..modules import GFNModule
from ..trajectories import FinalStateDistribution, TrajectoryDistribution


@dataclass
class Parametrization(ABC):
    """
    Abstract Base Class for Flow Parametrizations,
    as defined in Sec. 3 of GFlowNets Foundations.
    All attributes should either have a GFNModule or a nn.Module attribute called `module`,
    or torch.Tensor attribute called `tensor` with requires_grad=True.
    """

    @abstractmethod
    def Pi(self, env: Env, n_samples: int, **kwargs) -> TrajectoryDistribution:
        pass

    def P_T(self, env: Env, n_samples: int, **kwargs) -> FinalStateDistribution:
        return FinalStateDistribution(self.Pi(env, n_samples, **kwargs))

    @property
    def parameters(self) -> dict:
        """
        Return a dictionary of all parameters of the parametrization.
        Note that there might be duplicate parameters (e.g. when two NNs share parameters),
        in which case the optimizer should take as input set(self.parameters.values()).
        """
        parameters_dict = {}
        for field in fields(self.__class__):
            estimator = getattr(self, field.name)
            if hasattr(estimator, "module"):
                assert isinstance(estimator.module, torch.nn.Module) or isinstance(
                    estimator.module, GFNModule
                )
                module_parameters_dict = dict(estimator.module.named_parameters())
                parameters_dict.update(
                    {
                        f"{field.name}_{key}": value
                        for key, value in module_parameters_dict.items()
                    }
                )
            elif hasattr(estimator, "tensor"):
                assert (
                    isinstance(estimator.tensor, torch.Tensor)
                    and estimator.tensor.requires_grad
                )
                parameters_dict[field.name] = estimator.tensor

        return parameters_dict
