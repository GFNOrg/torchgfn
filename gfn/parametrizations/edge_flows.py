from dataclasses import dataclass

import torch.nn as nn

from gfn.envs import Env
from gfn.estimators import LogEdgeFlowEstimator
from gfn.parametrizations.base import Parametrization
from gfn.samplers import LogEdgeFlowsActionSampler, TrajectoriesSampler
from gfn.trajectories.dist import (
    EmpiricalTrajectoryDistribution,
    TrajectoryDistribution,
)


@dataclass
class EdgeFlowParametrization(Parametrization):
    r"""
    $\mathcal{O}_{edge}$ is the set of functions from the non-terminating edges
    to $\mathbb{R}^+$. Which is equivalent to the set of functions from the internal nodes
    (i.e. without $s_f$) to $(\mathbb{R})^{n_actions}$, without the exit action (No need for
    positivity if we parametrize log-flows).
    """
    logF: LogEdgeFlowEstimator

    def Pi(
        self, env: Env, n_samples: int = 1000, **action_sampler_kwargs
    ) -> TrajectoryDistribution:
        action_sampler = LogEdgeFlowsActionSampler(self.logF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories(
            n_trajectories=n_samples
        )
        return EmpiricalTrajectoryDistribution(trajectories)

    @property
    def parameters(self) -> dict:
        if not isinstance(self.logF.module, nn.Module):
            return {}
        parameters_dict = dict(self.logF.module.named_parameters())
        return {f"logF_{key}": value for key, value in parameters_dict.items()}
