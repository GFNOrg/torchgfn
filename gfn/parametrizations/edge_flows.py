from gfn.parametrizations.base import Parametrization
from gfn.envs import Env
from gfn.trajectories.dist import TrajectoryDistribution, EmpiricalTrajectoryDistribution
from gfn.samplers import LogEdgeFlowsActionSampler, TrajectoriesSampler
from gfn.estimators import LogEdgeFlowEstimator
from dataclasses import dataclass


@dataclass
class EdgeFlowParametrization(Parametrization):
    r"""
    $\mathcal{O}_{edge}$ is the set of functions from the non-terminating edges
    to $\mathbb{R}^+$. Which is equivalent to the set of functions from the internal nodes
    (i.e. without $s_f$) to $(\mathbb{R})^{n_actions}$, without the exit action (No need for
    positivity if we parametrize log-flows).
    """
    logF: LogEdgeFlowEstimator

    def Pi(self, env: Env, **action_sampler_kwargs) -> TrajectoryDistribution:
        action_sampler = LogEdgeFlowsActionSampler(
            self.logF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories()
        return EmpiricalTrajectoryDistribution(trajectories)


