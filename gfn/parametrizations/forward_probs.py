from abc import ABC
from dataclasses import dataclass

from gfn.envs import Env
from gfn.estimators import (
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.models import Uniform
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
    logit_PB: LogitPBEstimator

    def Pi(
        self, env: Env, n_samples: int = 1000, **action_sampler_kwargs
    ) -> TrajectoryDistribution:
        action_sampler = LogitPFActionSampler(self.logit_PF, **action_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, action_sampler)
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


if __name__ == "__main__":
    import torch

    from gfn.envs import HyperGrid
    from gfn.estimators import (
        LogitPBEstimator,
        LogitPFEstimator,
        LogStateFlowEstimator,
        LogZEstimator,
    )
    from gfn.models import NeuralNet
    from gfn.parametrizations.base import Parametrization
    from gfn.preprocessors import IdentityPreprocessor
    from gfn.trajectories.dist import TrajectoryDistribution

    env = HyperGrid()

    preprocessor = IdentityPreprocessor(env)

    print("\nTrying the DB parametrization... with learnable logit_PB")

    pb_module = NeuralNet(
        input_dim=preprocessor.output_dim,
        n_hidden_layers=1,
        hidden_dim=16,
        output_dim=env.n_actions - 1,
    )
    pf_module = NeuralNet(
        input_dim=preprocessor.output_dim,
        n_hidden_layers=1,
        hidden_dim=16,
        output_dim=env.n_actions,
    )
    f_module = NeuralNet(
        input_dim=preprocessor.output_dim,
        n_hidden_layers=1,
        hidden_dim=16,
        output_dim=1,
    )
    logit_PF = LogitPFEstimator(preprocessor, pf_module)
    logit_PB = LogitPBEstimator(preprocessor, pb_module)
    logF = LogStateFlowEstimator(preprocessor, f_module)
    logZ = LogZEstimator(torch.tensor(0.0))

    parametrization = DBParametrization(logit_PF, logit_PB, logF)
    print(parametrization.Pi(env, n_samples=10).sample())
    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    print("Now TB loss")
    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    print("\nTrying the DB parametrization... with learnable logit_PB tied to PF")
    pb_module = NeuralNet(output_dim=env.n_actions - 1, torso=logit_PF.module.torso)
    logit_PB = LogitPBEstimator(preprocessor, pb_module)

    parametrization = DBParametrization(logit_PF, logit_PB, logF)
    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    print("Now TB loss")
    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    print("\nTrying the DB parametrization... with uniform PB")
    logit_PB = LogitPBEstimator(
        preprocessor, module=Uniform(output_dim=env.n_actions - 1)
    )
    parametrization = DBParametrization(logit_PF, logit_PB, logF)
    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    print("Now TB loss")
    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))
