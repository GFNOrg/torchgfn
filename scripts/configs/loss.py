from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Literal
import inspect


import torch

from gfn.envs import Env
from gfn.estimators import LogEdgeFlowEstimator, LogStateFlowEstimator, LogZEstimator
from gfn.losses import (
    DBParametrization,
    DetailedBalance,
    FlowMatching,
    FMParametrization,
    LogPartitionVarianceLoss,
    Loss,
    Parametrization,
    PFBasedParametrization,
    SubTBParametrization,
    SubTrajectoryBalance,
    TBParametrization,
    TrajectoryBalance,
)
from gfn.utils import DiscretePBEstimator, DiscretePFEstimator


@dataclass
class GFNModuleConfig:
    module_name: Literal["NeuralNet", "Tabular", "Uniform", "Zero"] = "NeuralNet"
    n_hidden_layers: int = 2
    hidden_dim: int = 256
    activation_fn: str = "relu"

    def __post_init__(self):
        self.nn_kwargs = self.__dict__.copy()


@dataclass
class BaseLossConfig(ABC):
    @abstractmethod
    def parse(self, env: Env, **kwargs) -> Tuple[Parametrization, Loss]:
        pass


@dataclass
class FMLossConfig(BaseLossConfig):
    logF_edge: GFNModuleConfig = field(default_factory=GFNModuleConfig)
    alpha: float = 1.0

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logF_edge = LogEdgeFlowEstimator(
            env=env,
            **self.logF_edge.nn_kwargs,
        )
        parametrization = FMParametrization(logF_edge)

        loss = FlowMatching(parametrization, alpha=self.alpha)

        return parametrization, loss


@dataclass
class PFBasedLossConfig(BaseLossConfig, ABC):
    logit_PF: GFNModuleConfig = field(default_factory=GFNModuleConfig)
    logit_PB: GFNModuleConfig = field(default_factory=GFNModuleConfig)
    tied: bool = True

    def get_estimators(
        self,
        env: Env,
    ) -> Tuple[DiscretePFEstimator, DiscretePBEstimator]:
        logit_PF = DiscretePFEstimator(env=env, **self.logit_PF.nn_kwargs)
        logit_PB_kwargs = self.logit_PB.nn_kwargs
        if (
            self.tied
            and self.logit_PF.module_name
            and self.logit_PB.module_name == "NeuralNet"
        ):
            torso = logit_PF.module.torso
        else:
            torso = None
        logit_PB_kwargs["torso"] = torso
        logit_PB = DiscretePBEstimator(env=env, **logit_PB_kwargs)

        return (logit_PF, logit_PB)


@dataclass
class StateFlowBasedLossConfig(PFBasedLossConfig, ABC):
    logF_state: GFNModuleConfig = field(default_factory=GFNModuleConfig)

    def get_estimators(
        self,
        env: Env,
        forward_looking: bool = False,
    ) -> Tuple[DiscretePFEstimator, DiscretePBEstimator, LogStateFlowEstimator]:
        logit_PF, logit_PB = super().get_estimators(env)
        logF_state_kwargs = self.logF_state.nn_kwargs
        if (
            self.tied
            and self.logit_PF.module_name == "NeuralNet"
            and self.logF_state.module_name == "NeuralNet"
        ):
            torso = logit_PF.module.torso
        else:
            torso = None
        logF_state_kwargs["torso"] = torso
        logF_state = LogStateFlowEstimator(
            env=env, forward_looking=forward_looking, **self.logF_state.nn_kwargs
        )

        return (logit_PF, logit_PB, logF_state)


@dataclass
class DBLossConfig(StateFlowBasedLossConfig):
    forward_looking: bool = False

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB, logF_state = self.get_estimators(env, self.forward_looking)

        parametrization = DBParametrization(logit_PF, logit_PB, logF_state)
        loss = DetailedBalance(parametrization)
        return (parametrization, loss)


@dataclass
class SubTBLossConfig(StateFlowBasedLossConfig):
    weighing: Literal[
        "equal",
        "equal_within",
        "geometric",
        "TB",
        "DB",
        "ModifiedDB",
        "geometric_within",
    ] = "geometric_within"
    forward_looking: bool = False
    lamda: float = 0.9

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB, logF_state = self.get_estimators(env, self.forward_looking)

        parametrization = SubTBParametrization(logit_PF, logit_PB, logF_state)
        loss = SubTrajectoryBalance(
            parametrization, weighing=self.weighing, lamda=self.lamda
        )
        return (parametrization, loss)


@dataclass
class TBLossConfig(PFBasedLossConfig):
    logZ_init: float = 0.0
    log_reward_clip_min: float = -12

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB = self.get_estimators(env)
        logZ_tensor = torch.tensor(self.logZ_init, dtype=torch.float)
        logZ = LogZEstimator(tensor=logZ_tensor)
        parametrization = TBParametrization(logit_PF, logit_PB, logZ)
        loss = TrajectoryBalance(parametrization, self.log_reward_clip_min)
        return (parametrization, loss)


@dataclass
class LogPartitionVarianceLossConfig(PFBasedLossConfig):
    def parse(self, env: Env) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB = self.get_estimators(env)
        parametrization = PFBasedParametrization(logit_PF, logit_PB)
        loss = LogPartitionVarianceLoss(parametrization)
        return (parametrization, loss)


def make_loss(config: dict, env: Env) -> Tuple[Parametrization, Loss]:
    name = config["loss"]["name"]
    if name.lower() == "flowmatching".lower():
        loss_class = FMLossConfig
    elif name.lower() == "detailed-balance".lower():
        loss_class = DBLossConfig
    elif name.lower() == "trajectory-balance".lower():
        loss_class = TBLossConfig
    elif name.lower() == "sub-tb".lower():
        loss_class = SubTBLossConfig
    else:
        raise ValueError("Invalid loss name: {}".format(name))

    args = inspect.getfullargspec(loss_class.__init__).args
    loss_config = {k: v for k, v in config["loss"].items() if k in args}
    return loss_class(**loss_config).parse(env)
