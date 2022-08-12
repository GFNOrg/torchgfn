from gfn.parametrizations.base import O, Parametrization, MarkovianFlow
from gfn.envs import Env
from ..samplers.action_samplers import LogitPFActionSampler
from gfn.trajectories.dist import TrajectoryDistribution, EmpiricalTrajectoryDistribution
from gfn.samplers import TrajectoriesSampler
from gfn.estimators import LogStateFlowEstimator, LogitPFEstimator
from dataclasses import dataclass


@dataclass
class O_PF(O):
    r"""
    $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2$, where 
    $\mathcal{O}_1$ is the set of functions from the internal states (no $s_f$)
    to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the nonnegativity constraint),
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    """
    name = 'forward_probability'
    logF: LogStateFlowEstimator
    logit_PF: LogitPFEstimator


class ForwardTransitionParametrization(Parametrization):
    O_name = 'forward_probability'

    def H(self, flow: MarkovianFlow) -> O_PF:
        return O_PF(logF=flow.log_state_flow_function(),
                    logit_PF=flow.logit_PF())

    def Pi(self, o: O_PF, env: Env, **action_sampler_kwargs) -> TrajectoryDistribution:
        action_sampler = LogitPFActionSampler(
            o.logit_PF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories()
        return EmpiricalTrajectoryDistribution(trajectories)
