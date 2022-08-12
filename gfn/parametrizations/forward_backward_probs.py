from gfn.parametrizations.base import O, Parametrization, MarkovianFlow
from gfn.envs import Env
from ..samplers.action_samplers import LogitPFActionSampler
from gfn.trajectories.dist import TrajectoryDistribution, EmpiricalTrajectoryDistribution
from gfn.samplers import TrajectoriesSampler
from gfn.estimators import LogStateFlowEstimator, LogitPFEstimator, LogitPBEstimator, LogZEstimator
from dataclasses import dataclass


@dataclass
class O_PFB(O):
    r"""
    $\mathcal{O}_{PFB} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where 
    $\mathcal{O}_1$ is the set of functions from the internal states (no $s_f$)
    to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the nonnegativity constraint),
    $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG,
    and $\mathcal{O}_3$ is the set of forward probability functions consistent with the DAG.
    """
    name = 'forward_backward_probability'
    logF: LogStateFlowEstimator
    logit_PF: LogitPFEstimator
    logit_PB: LogitPBEstimator


class ForwardBackwardTransitionParametrization(Parametrization):
    O_name = 'forward_backward_probability'

    def H(self, flow: MarkovianFlow) -> O_PFB:
        return O_PFB(logF=flow.log_state_flow_function(),
                     logit_PF=flow.logit_PF(),
                     logit_PB=flow.logit_PB())

    def Pi(self, o: O_PFB, env: Env, **action_sampler_kwargs) -> TrajectoryDistribution:
        action_sampler = LogitPFActionSampler(
            o.logit_PF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories()
        return EmpiricalTrajectoryDistribution(trajectories)


@dataclass
class O_PFB_Z(O):
    r"""
    $\mathcal{O}_{PFB} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where 
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG,
    and $\mathcal{O}_3$ is the set of forward probability functions consistent with the DAG.
    """
    name = 'forward_backward_probability_Z'
    logF: LogStateFlowEstimator
    logit_PF: LogitPFEstimator
    logit_PB: LogitPBEstimator
    logZ: LogZEstimator


class ForwardBackwardTransitionParametrizationWithZ(Parametrization):
    O_name = 'forward_backward_probability_Z'

    def H(self, flow: MarkovianFlow) -> O_PFB_Z:
        return O_PFB_Z(logF=flow.log_state_flow_function(),
                       logit_PF=flow.logit_PF(),
                       logit_PB=flow.logit_PB(),
                       logZ=flow.log_Z())

    def Pi(self, o: O_PFB_Z, env: Env, **action_sampler_kwargs) -> TrajectoryDistribution:
        action_sampler = LogitPFActionSampler(
            o.logit_PF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
        trajectories = trajectories_sampler.sample_trajectories()
        return EmpiricalTrajectoryDistribution(trajectories)
