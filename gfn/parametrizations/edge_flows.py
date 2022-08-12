from gfn.parametrizations.base import O, Parametrization, MarkovianFlow
from gfn.envs import Env
from gfn.trajectories.dist import TrajectoryDistribution, EmpiricalTrajectoryDistribution
from gfn.samplers import LogEdgeFlowsActionSampler, TrajectoriesSampler
from gfn.estimators import LogEdgeFlowEstimator
from typing import Callable
from dataclasses import dataclass


@dataclass
class O_edge(O):
    r"""
    $\mathcal{O}_{edge}$ is the set of functions from the non-terminating edges
    to $\mathbb{R}^+$. Which is equivalent to the set of functions from the internal nodes
    (i.e. without $s_f$) to $(\mathbb{R})^{n_actions}$, without the exit action (No need for
    positivity if we parametrize log-flows).
    """
    name = 'edge'
    logF: LogEdgeFlowEstimator


class EdgeFlowParametrization(Parametrization):
    O_name = 'edge'

    def H(self, flow: MarkovianFlow) -> Callable:
        return flow.log_edge_flow_function()

    def Pi(self, o: O_edge, env: Env, **action_sampler_kwargs) -> TrajectoryDistribution:
        action_sampler = LogEdgeFlowsActionSampler(
            o.logF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories()
        return EmpiricalTrajectoryDistribution(trajectories)
