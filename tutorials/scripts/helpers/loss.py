import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Tuple

import torch

from gfn.env import Env
from gfn.losses import (
    DBGFlowNet,
    DetailedBalance,
    FlowMatching,
    FMGFlowNet,
    LogPartitionVarianceLoss,
    Loss,
    GFlowNet,
    PFBasedGFlowNet,
    SubTBGFlowNet,
    SubTrajectoryBalance,
    TBGFlowNet,
    TrajectoryBalance,
)
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator


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
    def parse(self, env: Env, **kwargs) -> Tuple[GFlowNet, Loss]:
        pass


@dataclass
class FMLossConfig(BaseLossConfig):
    logF_edge: GFNModuleConfig = field(default_factory=GFNModuleConfig)
    alpha: float = 1.0

    def parse(
        self,
        env: Env,
    ) -> Tuple[GFlowNet, Loss]:
        logF_edge = ScalarEstimator(
            env=env,
            **self.logF_edge.nn_kwargs,
        )
        parametrization = FMGFlowNet(logF_edge)

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
    ) -> DiscretePolicyEstimator:
        logit_PF = DiscretePolicyEstimator(
            env=env,
            forward=True,
            **self.logit_PF.nn_kwargs,
        )
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
        logit_PB = DiscretePolicyEstimator(env=env, forward=False, **logit_PB_kwargs)

        return (logit_PF, logit_PB)


@dataclass
class StateFlowBasedLossConfig(PFBasedLossConfig, ABC):
    logF_state: GFNModuleConfig = field(default_factory=GFNModuleConfig)

    def get_estimators(
        self,
        env: Env,
        forward_looking: bool = False,
    ) -> DiscretePolicyEstimator:
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
        # TODO: I need to verify this.
        logF_state = DiscretePolicyEstimator(
            env=env,
            forward=True,
            forward_looking=forward_looking,
            **self.logF_state.nn_kwargs,
        )

        return (logit_PF, logit_PB, logF_state)


@dataclass
class DBLossConfig(StateFlowBasedLossConfig):
    forward_looking: bool = False

    def parse(
        self,
        env: Env,
    ) -> Tuple[GFlowNet, Loss]:
        logit_PF, logit_PB, logF_state = self.get_estimators(env, self.forward_looking)

        parametrization = DBGFlowNet(logit_PF, logit_PB, logF_state)
        loss = DetailedBalance(parametrization)
        return (parametrization, loss)


@dataclass
class SubTBLossConfig(StateFlowBasedLossConfig):
    weighting: Literal[
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
    ) -> Tuple[GFlowNet, Loss]:
        logit_PF, logit_PB, logF_state = self.get_estimators(env, self.forward_looking)

        parametrization = SubTBGFlowNet(logit_PF, logit_PB, logF_state)
        loss = SubTrajectoryBalance(
            parametrization, weighting=self.weighting, lamda=self.lamda
        )
        return (parametrization, loss)


@dataclass
class TBLossConfig(PFBasedLossConfig):
    logZ_init: float = 0.0
    log_reward_clip_min: float = -12

    def parse(
        self,
        env: Env,
    ) -> Tuple[GFlowNet, Loss]:
        logit_PF, logit_PB = self.get_estimators(env)
        parametrization = TBGFlowNet(logit_PF, logit_PB, logZ=logZ_init)
        loss = TrajectoryBalance(parametrization, self.log_reward_clip_min)
        return (parametrization, loss)


@dataclass
class LogPartitionVarianceLossConfig(PFBasedLossConfig):
    def parse(self, env: Env) -> Tuple[GFlowNet, Loss]:
        logit_PF, logit_PB = self.get_estimators(env)
        parametrization = PFBasedGFlowNet(logit_PF, logit_PB)
        loss = LogPartitionVarianceLoss(parametrization)
        return (parametrization, loss)


def make_loss(config: dict, env: Env) -> Tuple[GFlowNet, Loss]:
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
