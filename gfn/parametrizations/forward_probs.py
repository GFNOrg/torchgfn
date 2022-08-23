from abc import ABC
from dataclasses import dataclass

import torch.nn as nn

from gfn.envs import Env
from gfn.estimators import (
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.parametrizations.base import Parametrization
from gfn.samplers import LogitPFActionSampler, TrajectoriesSampler
from gfn.trajectories.dist import (
    EmpiricalTrajectoryDistribution,
    TrajectoryDistribution,
)


@dataclass
class PFBasedParametrization(Parametrization, ABC):
    r"Base class for parametrizations that explicitly used $P_F$"
    logit_PF: LogitPFEstimator

    def Pi(
        self, env: Env, n_samples: int = 1000, **action_sampler_kwargs
    ) -> TrajectoryDistribution:
        action_sampler = LogitPFActionSampler(self.logit_PF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories(
            n_trajectories=n_samples
        )
        return EmpiricalTrajectoryDistribution(trajectories)

    @property
    def parameters(self) -> dict:
        if not isinstance(self.logit_PF.module, nn.Module):
            return {}
        parameters_dict = dict(self.logit_PF.module.named_parameters())
        return {f"logit_PF_{key}": value for key, value in parameters_dict.items()}


@dataclass
class O_PF(PFBasedParametrization):
    r"""
    $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2$, where
    $\mathcal{O}_1$ is the set of functions from the internal states (no $s_f$)
    to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the non-negativity constraint),
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    Useful for the Detailed Balance Loss. The logit_PB attribute should not be a nn.Module.
    """
    logit_PB: LogitPBEstimator
    logF: LogStateFlowEstimator

    def __post_init__(self):
        assert not isinstance(
            self.logit_PB.module, nn.Module
        ), "logit_PB should not be a nn.Module, use O_PFB instead"

    @property
    def parameters(self) -> dict:
        parameters_dict = super().parameters
        if not isinstance(self.logF.module, nn.Module):
            return parameters_dict
        extra_parameters_dict = dict(self.logF.module.named_parameters())
        extra_parameters_dict = {
            f"logF_{key}": value for key, value in extra_parameters_dict.items()
        }
        return {**parameters_dict, **extra_parameters_dict}


@dataclass
class O_PFB(O_PF):
    r"""
    $\mathcal{O}_{PFB} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1$ is the set of functions from the internal states (no $s_f$)
    to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the non-negativity constraint),
    $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG,
    and $\mathcal{O}_3$ is the set of forward probability functions consistent with the DAG.
    Useful for the Detailed Balance Loss. The logit_PB attribute should be a nn.Module.
    tied: whether logit_PB and logit_PB share the same neural network torso.
    """
    tied: bool = False

    def __post_init__(self):
        assert isinstance(self.logit_PB.module, nn.Module)

    @property
    def parameters(self) -> dict:
        parameters_dict = super().parameters

        if self.tied:
            extra_parameters_dict = dict(
                self.logit_PB.module.last_layer.named_parameters()
            )
        else:
            extra_parameters_dict = dict(self.logit_PB.module.named_parameters())

        extra_parameters_dict = {
            f"logit_PB_{key}": value for key, value in extra_parameters_dict.items()
        }
        return {**parameters_dict, **extra_parameters_dict}


@dataclass
class O_PFZ(PFBasedParametrization):
    r"""
    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    Useful for the Trajectory Balance Loss. The logit_PB attribute should not be a nn.Module.
    """
    logit_PB: LogitPBEstimator
    logZ: LogZEstimator

    def __post_init__(self):
        assert not isinstance(
            self.logit_PB.module, nn.Module
        ), "logit_PB should not be a nn.Module, use O_PFBZ instead"

    @property
    def parameters(self) -> dict:
        parameters_dict = super().parameters
        parameters_dict.update({"logZ": self.logZ.logZ})
        return parameters_dict


@dataclass
class O_PFBZ(O_PFZ):
    r"""
    $\mathcal{O}_{PFBZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG,
    and $\mathcal{O}_3$ is the set of forward probability functions consistent with the DAG.
    Useful for the Trajectory Balance Loss.
    """
    tied: bool = False

    def __post_init__(self):
        assert isinstance(self.logit_PB.module, nn.Module)

    @property
    def parameters(self) -> dict:
        # TODO: can probably be factorized alongside O_PFB.parameters()
        parameters_dict = super().parameters

        if self.tied:
            extra_parameters_dict = dict(
                self.logit_PB.module.last_layer.named_parameters()
            )
        else:
            extra_parameters_dict = dict(self.logit_PB.module.named_parameters())

        extra_parameters_dict = {
            f"logit_PB_{key}": value for key, value in extra_parameters_dict.items()
        }
        return {**parameters_dict, **extra_parameters_dict}
