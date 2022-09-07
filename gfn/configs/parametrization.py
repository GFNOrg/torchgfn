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
from gfn.losses import DetailedBalance, Loss, TrajectoryBalance
from gfn.parametrizations import (
    DBParametrization,
    FMParametrization,
    Parametrization,
    TBParametrization,
)
from gfn.preprocessors import (
    EnumPreprocessor,
    IdentityPreprocessor,
    KHotPreprocessor,
    OneHotPreprocessor,
    Preprocessor,
)

from .module import GFNModuleConfig, NeuralNetConfig, TabularConfig, UniformConfig


@dataclass
class BaseParametrizationConfig(JsonSerializable, ABC):
    @staticmethod
    def adjust_module_config(
        module_config: GFNModuleConfig, preprocessor: Preprocessor, output_dim: int
    ) -> None:
        module_config.input_dim = preprocessor.output_dim
        module_config.output_dim = output_dim

    @abstractmethod
    def parse(self, env: Env, **kwargs) -> Tuple[Parametrization, Loss]:
        pass


@dataclass
class FMParametrizationConfig(BaseParametrizationConfig):
    def parse(
        self,
        env: Env,
        preprocessor: Preprocessor,
        logF_edge_config: GFNModuleConfig,
        **kwargs,
    ) -> Tuple[Parametrization, Loss]:
        del kwargs
        self.adjust_module_config(logF_edge_config, preprocessor, env.n_actions - 1)

        logF_module = logF_edge_config.parse(env=env)
        logF_edge = LogEdgeFlowEstimator(preprocessor=preprocessor, module=logF_module)
        _ = FMParametrization(logF_edge)

        # TODO: FlowMatching loss not implemented yet
        raise NotImplementedError("FlowMatching loss not implemented yet")


@dataclass
class PFBasedParametrizationConfig(BaseParametrizationConfig, ABC):
    tied: bool = True

    def get_estimators(
        self,
        env: Env,
        preprocessor: Preprocessor,
        logit_PF_config: GFNModuleConfig,
        logit_PB_config: GFNModuleConfig,
        **kwargs,
    ) -> Tuple[Preprocessor, LogitPFEstimator, LogitPBEstimator]:
        del kwargs
        if self.tied and not (
            isinstance(logit_PF_config, NeuralNetConfig)
            and isinstance(logit_PB_config, NeuralNetConfig)
        ):
            print("Setting back tied to False")
            self.tied = False
        self.adjust_module_config(logit_PF_config, preprocessor, env.n_actions)
        self.adjust_module_config(logit_PB_config, preprocessor, env.n_actions - 1)

        logit_PF_module = logit_PF_config.parse(env=env)
        logit_PF = LogitPFEstimator(preprocessor=preprocessor, module=logit_PF_module)
        logit_PB_module = logit_PB_config.parse(
            env=env,
            tied_to=logit_PF_module if self.tied else None,
            tied_to_name="logit_PF" if self.tied else None,
        )
        logit_PB = LogitPBEstimator(preprocessor=preprocessor, module=logit_PB_module)

        return (logit_PF, logit_PB)


@dataclass
class DBParametrizationConfig(PFBasedParametrizationConfig):
    def parse(
        self,
        env: Env,
        preprocessor: Preprocessor,
        logit_PF_config: GFNModuleConfig,
        logit_PB_config: GFNModuleConfig,
        logF_state_config: GFNModuleConfig,
        **kwargs,
    ) -> Tuple[Parametrization, Loss]:
        del kwargs
        logit_PF, logit_PB = super().get_estimators(
            env,
            preprocessor,
            logit_PF_config=logit_PF_config,
            logit_PB_config=logit_PB_config,
        )
        self.adjust_module_config(logF_state_config, preprocessor, 1)

        logF_module = logF_state_config.parse(
            env=env,
            tied_to=logit_PF.module if self.tied else None,
            tied_to_name="logit_PF" if self.tied else None,
        )
        logF_state = LogStateFlowEstimator(
            preprocessor=preprocessor, module=logF_module
        )
        parametrization = DBParametrization(logit_PF, logit_PB, logF_state)
        loss = DetailedBalance(parametrization)
        return (parametrization, loss)


@dataclass
class TBParametrizationConfig(PFBasedParametrizationConfig):
    logZ_init: float = 0.0
    reward_clip_min: float = 1e-5

    def parse(
        self,
        env: Env,
        preprocessor: Preprocessor,
        logit_PF_config: GFNModuleConfig,
        logit_PB_config: GFNModuleConfig,
        **kwargs,
    ) -> Tuple[Parametrization, Loss]:
        del kwargs
        logit_PF, logit_PB = super().get_estimators(
            env,
            preprocessor,
            logit_PF_config=logit_PF_config,
            logit_PB_config=logit_PB_config,
        )
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
        },
        default=TBParametrizationConfig(),
    )
    preprocessor: str = choice("Identity", "OneHot", "KHot", "Enum", default="KHot")

    logF_edge: GFNModuleConfig = subgroups(
        {
            "tabular": TabularConfig,
            "uniform": UniformConfig,
            "neural_net": NeuralNetConfig,
        },
        default=NeuralNetConfig(),
    )
    logF_state: GFNModuleConfig = subgroups(
        {
            "tabular": TabularConfig,
            "uniform": UniformConfig,
            "neural_net": NeuralNetConfig,
        },
        default=NeuralNetConfig(),
    )
    logit_PF: GFNModuleConfig = subgroups(
        {
            "tabular": TabularConfig,
            "uniform": UniformConfig,
            "neural_net": NeuralNetConfig,
        },
        default=NeuralNetConfig(),
    )
    logit_PB: GFNModuleConfig = subgroups(
        {
            "tabular": TabularConfig,
            "uniform": UniformConfig,
            "neural_net": NeuralNetConfig,
        },
        default=NeuralNetConfig(),
    )

    def __post_init__(self):
        if isinstance(self.parametrization, FMParametrizationConfig):
            self.logit_PF = None
            self.logit_PB = None
            self.logF_state = None
        else:
            self.logF_edge = None
            if isinstance(self.parametrization, TBParametrizationConfig):
                self.logF_state = None

    def get_preprocessor(self, env) -> Preprocessor:
        if self.preprocessor == "Identity":
            preprocessor = IdentityPreprocessor(env)
        elif self.preprocessor == "OneHot":
            preprocessor = OneHotPreprocessor(env)
        elif self.preprocessor == "KHot":
            preprocessor = KHotPreprocessor(env)
        else:
            preprocessor = EnumPreprocessor(env)
        return preprocessor

    def parse(self, env: Env) -> Tuple[Parametrization, Loss]:
        preprocessor = self.get_preprocessor(env)
        return self.parametrization.parse(
            env=env,
            preprocessor=preprocessor,
            logF_edge_config=self.logF_edge,
            logF_state_config=self.logF_state,
            logit_PF_config=self.logit_PF,
            logit_PB_config=self.logit_PB,
        )
