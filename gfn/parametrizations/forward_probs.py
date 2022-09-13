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
from gfn.samplers import LogitPFActionsSampler, TrajectoriesSampler
from gfn.trajectories.dist import (
    EmpiricalTrajectoryDistribution,
    TrajectoryDistribution,
)


@dataclass
class PFBasedParametrization(Parametrization, ABC):
    r"Base class for parametrizations that explicitly used $P_F$"
    logit_PF: LogitPFEstimator
    logit_PB: LogitPBEstimator

    def Pi(
        self, env: Env, n_samples: int = 1000, **actions_sampler_kwargs
    ) -> TrajectoryDistribution:
        actions_sampler = LogitPFActionsSampler(self.logit_PF, **actions_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
        trajectories = trajectories_sampler.sample_trajectories(
            n_trajectories=n_samples
        )
        return EmpiricalTrajectoryDistribution(trajectories)


@dataclass
class DBParametrization(PFBasedParametrization):
    r"""
    Corresponds to $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1$ is the set of functions from the internal states (no $s_f$)
    to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the non-negativity constraint),
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    $\mathcal{O}_3$ is the set of backward probability functions consistent with the DAG, or a singleton
    thereof, if self.logit_PB is a fixed LogitPBEstimator.
    Useful for the Detailed Balance Loss.
    """
    logF: LogStateFlowEstimator


@dataclass
class SubTBParametrization(PFBasedParametrization):
    r"""
    Exactly the same as DBParametrization
    """
    logF: LogStateFlowEstimator


@dataclass
class TBParametrization(PFBasedParametrization):
    r"""
    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    $\mathcal{O}_3$ is the set of backward probability functions consistent with the DAG, or a singleton
    thereof, if self.logit_PB is a fixed LogitPBEstimator.
    Useful for the Trajectory Balance Loss.
    """
    logZ: LogZEstimator
