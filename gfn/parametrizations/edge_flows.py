from dataclasses import dataclass

from gfn.envs import Env
from gfn.estimators import LogEdgeFlowEstimator
from gfn.parametrizations.base import Parametrization
from gfn.samplers import LogEdgeFlowsActionSampler, TrajectoriesSampler
from gfn.trajectories.dist import (
    EmpiricalTrajectoryDistribution,
    TrajectoryDistribution,
)


@dataclass
class FMParametrization(Parametrization):
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


if __name__ == "__main__":
    from gfn.envs import HyperGrid
    from gfn.models import NeuralNet
    from gfn.preprocessors import IdentityPreprocessor

    env = HyperGrid()

    preprocessor = IdentityPreprocessor(env)
    module = NeuralNet(
        input_dim=preprocessor.output_dim,
        n_hidden_layers=1,
        hidden_dim=16,
        output_dim=env.n_actions - 1,
    )
    log_F_edge = LogEdgeFlowEstimator(preprocessor, env, module)
    parametrization = FMParametrization(log_F_edge)

    print(parametrization.Pi(env, n_samples=10).sample())
    print(parametrization.parameters.keys())
