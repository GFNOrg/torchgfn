from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
from simple_parsing import choice, subgroups
from simple_parsing.helpers import JsonSerializable

from gfn.envs import Env
from gfn.estimators import (
    LogEdgeFlowEstimator,
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.losses import DetailedBalance, Loss, SubTrajectoryBalance, TrajectoryBalance
from gfn.parametrizations import (
    DBParametrization,
    FMParametrization,
    Parametrization,
    SubTBParametrization,
    TBParametrization,
)


@dataclass
class GFNModuleConfig(JsonSerializable):
    module_name: str = choice(
        "NeuralNet", "Tabular", "Uniform", "Zero", default="NeuralNet"
    )
    n_hidden_layers: int = 2
    hidden_dim: int = 256
    activation_fn: str = "relu"

    def __post_init__(self):
        self.module_kwargs = self.__dict__.copy()


@dataclass
class BaseParametrizationConfig(JsonSerializable, ABC):
    @abstractmethod
    def parse(self, env: Env, **kwargs) -> Tuple[Parametrization, Loss]:
        pass


@dataclass
class FMParametrizationConfig(BaseParametrizationConfig):
    logF_edge: GFNModuleConfig = GFNModuleConfig()

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:

        logF_edge = LogEdgeFlowEstimator(
            env=env,
            **self.logF_edge.module_kwargs,
        )
        _ = FMParametrization(logF_edge)

        # TODO: FlowMatching loss not implemented yet
        raise NotImplementedError("FlowMatching loss not implemented yet")


@dataclass
class PFBasedParametrizationConfig(BaseParametrizationConfig, ABC):
    logit_PF: GFNModuleConfig = GFNModuleConfig()
    logit_PB: GFNModuleConfig = GFNModuleConfig()
    tied: bool = True

    def get_estimators(
        self,
        env: Env,
    ) -> Tuple[LogitPFEstimator, LogitPBEstimator]:

        logit_PF = LogitPFEstimator(env=env, **self.logit_PF.module_kwargs)
        logit_PB_kwargs = self.logit_PB.module_kwargs
        if (
            self.tied
            and self.logit_PF.module_name
            and self.logit_PB.module_name == "NeuralNet"
        ):
            torso = logit_PF.module.torso
        else:
            torso = None
        logit_PB_kwargs["torso"] = torso
        logit_PB = LogitPBEstimator(env=env, **logit_PB_kwargs)

        return (logit_PF, logit_PB)


@dataclass
class StateFlowBasedParametrizationConfig(PFBasedParametrizationConfig, ABC):
    logF_state: GFNModuleConfig = GFNModuleConfig()

    def get_estimators(
        self,
        env: Env,
    ) -> Tuple[LogitPFEstimator, LogitPBEstimator, LogStateFlowEstimator]:

        logit_PF, logit_PB = super().get_estimators(env)
        logF_state_kwargs = self.logF_state.module_kwargs
        if (
            self.tied
            and self.logit_PF.module_name == "NeuralNet"
            and self.logF_state.module_name == "NeuralNet"
        ):
            torso = logit_PF.module.torso
        else:
            torso = None
        logF_state_kwargs["torso"] = torso
        logF_state = LogStateFlowEstimator(env=env, **self.logF_state.module_kwargs)

        return (logit_PF, logit_PB, logF_state)


@dataclass
class DBParametrizationConfig(StateFlowBasedParametrizationConfig):
    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB, logF_state = self.get_estimators(env)

        parametrization = DBParametrization(logit_PF, logit_PB, logF_state)
        loss = DetailedBalance(parametrization)
        return (parametrization, loss)


@dataclass
class SubTBParametrizationConfig(StateFlowBasedParametrizationConfig):
    # TODO: Should be merged with DBParametrizationConfig
    weighing: str = choice(
        "equal", "geometric", "TB", "DB", "geometric2", default="geometric"
    )
    lamda: float = 0.9

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB, logF_state = self.get_estimators(env)

        parametrization = SubTBParametrization(logit_PF, logit_PB, logF_state)
        loss = SubTrajectoryBalance(
            parametrization, weighing=self.weighing, lamda=self.lamda
        )
        return (parametrization, loss)


@dataclass
class TBParametrizationConfig(PFBasedParametrizationConfig):
    logZ_init: float = 0.0
    reward_clip_min: float = 1e-5

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB = self.get_estimators(env)
        logZ_tensor = torch.tensor(self.logZ_init, dtype=torch.float)
        logZ = LogZEstimator(tensor=logZ_tensor)
        parametrization = TBParametrization(logit_PF, logit_PB, logZ)
        loss = TrajectoryBalance(parametrization, self.reward_clip_min)
        return (parametrization, loss)


@dataclass
class ParametrizationConfig(JsonSerializable):
    parametrization: BaseParametrizationConfig = subgroups(
        {
            "FM": FMParametrizationConfig,
            "DB": DBParametrizationConfig,
            "TB": TBParametrizationConfig,
            "SubTB": SubTBParametrizationConfig,
        },
        default=TBParametrizationConfig(),
    )

    def parse(self, env: Env) -> Tuple[Parametrization, Loss]:
        return self.parametrization.parse(env=env)
