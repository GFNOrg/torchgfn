from abc import ABC
from dataclasses import dataclass

from gfn.envs import Env
from gfn.estimators import (
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.parametrizations.base import Parametrization
from gfn.samplers import TrajectoriesSampler
from gfn.trajectories.dist import (
    EmpiricalTrajectoryDistribution,
    TrajectoryDistribution,
)

from ..samplers.action_samplers import LogitPFActionSampler


@dataclass
class PFBasedParametrization(Parametrization, ABC):
    r"Base class for parametrizations that explicitly used $P_F$"
    logit_PF: LogitPFEstimator

    def Pi(self, env: Env, **action_sampler_kwargs) -> TrajectoryDistribution:
        action_sampler = LogitPFActionSampler(self.logit_PF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories()
        return EmpiricalTrajectoryDistribution(trajectories)


@dataclass
class ForwardTransitionParametrization(PFBasedParametrization):
    r"""
    $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2$, where
    $\mathcal{O}_1$ is the set of functions from the internal states (no $s_f$)
    to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the nonnegativity constraint),
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    """
    logF: LogStateFlowEstimator


@dataclass
class ForwardTransitionParametrizationWithZ(PFBasedParametrization):
    r"""
    $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    """
    logZ: LogZEstimator


@dataclass
class ForwardBackwardTransitionParametrization(ForwardTransitionParametrization):
    r"""
    $\mathcal{O}_{PFB} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1$ is the set of functions from the internal states (no $s_f$)
    to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the nonnegativity constraint),
    $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG,
    and $\mathcal{O}_3$ is the set of forward probability functions consistent with the DAG.
    """
    logit_PB: LogitPBEstimator


@dataclass
class ForwardBackwardTransitionParametrizationWithZ(
    ForwardTransitionParametrizationWithZ
):
    r"""
    $\mathcal{O}_{PFB} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG,
    and $\mathcal{O}_3$ is the set of forward probability functions consistent with the DAG.
    """
    logit_PB: LogitPBEstimator
