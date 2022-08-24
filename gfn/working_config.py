from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from simple_parsing import choice, subgroups
from simple_parsing.helpers import JsonSerializable

from gfn.envs import Env, HyperGrid
from gfn.estimators import (
    GFNModule,
    LogEdgeFlowEstimator,
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.losses import DetailedBalance, Loss, TrajectoryBalance
from gfn.models import NeuralNet, Tabular, Uniform
from gfn.parametrizations import (
    DBParametrization,
    FMParametrization,
    Parametrization,
    PFBasedParametrization,
    TBParametrization,
)
from gfn.preprocessors import (
    IdentityPreprocessor,
    KHotPreprocessor,
    OneHotPreprocessor,
    Preprocessor,
)
from gfn.samplers import (
    ActionSampler,
    LogEdgeFlowsActionSampler,
    LogitPFActionSampler,
    TrainingSampler,
    TrajectoriesSampler,
    TransitionsSampler,
)


@dataclass
class EnvConfig(JsonSerializable):
    ...


@dataclass
class HyperGridConfig(EnvConfig):
    # TODO: maybe move this inside HyperGrid and factorize code there (and add other params),
    # and do the same for the other configs in this file
    ndim: int = 2
    height: int = 4

    def parse(self) -> Env:
        assert self.ndim > 0 and self.height > 0
        return HyperGrid(ndim=self.ndim, height=self.height)


@dataclass
class GFNModuleConfig(JsonSerializable, ABC):
    input_dim: Optional[int] = None
    output_dim: int = 1

    @abstractmethod
    def parse(**kwargs) -> GFNModule:
        pass


@dataclass
class TabularConfig(GFNModuleConfig):
    def parse(self, env: Env, **kwargs) -> Tabular:
        return Tabular(env=env, output_dim=self.output_dim)


@dataclass
class UniformConfig(GFNModuleConfig):
    def parse(self, env: Env, **kwargs) -> Uniform:
        return Uniform(env=env, output_dim=self.output_dim)


@dataclass
class NeuralNetConfig(GFNModuleConfig):
    n_hidden_layers: int = 2
    hidden_dim: int = 256
    activation_fn: str = "relu"

    def parse(self, tied_to: Optional[GFNModule] = None, **kwargs) -> NeuralNet:
        return NeuralNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_hidden_layers=self.n_hidden_layers,
            hidden_dim=self.hidden_dim,
            activation_fn=self.activation_fn,
            torso=tied_to.torso if tied_to is not None else None,
        )


@dataclass
class ParametrizationConfig(JsonSerializable, ABC):
    preprocessor: str = choice("Identity", "OneHot", "KHot", default="Identity")

    def get_preprocessor(self, env) -> Preprocessor:
        if self.preprocessor == "Identity":
            preprocessor = IdentityPreprocessor(env)
        elif self.preprocessor == "OneHot":
            preprocessor = OneHotPreprocessor(env)
        else:
            preprocessor = KHotPreprocessor(env)
        return preprocessor

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
class FMParametrizationConfig(ParametrizationConfig):
    def parse(
        self, env: Env, logF_edge_config: GFNModuleConfig, **kwargs
    ) -> Tuple[Parametrization, Loss]:
        preprocessor = super().get_preprocessor(env)
        self.adjust_module_config(logF_edge_config, preprocessor, env.n_actions - 1)

        logF_module = logF_edge_config.parse(env=env)
        logF_edge = LogEdgeFlowEstimator(
            preprocessor=preprocessor, env=env, module=logF_module
        )
        _ = FMParametrization(logF_edge)

        # TODO: FlowMatching loss not implemented yet
        raise NotImplementedError("FlowMatching loss not implemented yet")


@dataclass
class PFBasedParametrizationConfig(ParametrizationConfig, ABC):
    tied: bool = False

    def get_preprocessor_and_estimators(
        self,
        env: Env,
        logit_PF_config: GFNModuleConfig,
        logit_PB_config: GFNModuleConfig,
        **kwargs,
    ) -> Tuple[Preprocessor, LogitPFEstimator, LogitPBEstimator]:
        if self.tied and not (
            isinstance(logit_PF_config, NeuralNetConfig)
            and isinstance(logit_PB_config, NeuralNetConfig)
        ):
            print("Setting back tied to False")
            self.tied = False
        preprocessor = super().get_preprocessor(env)
        self.adjust_module_config(logit_PF_config, preprocessor, env.n_actions)
        self.adjust_module_config(logit_PB_config, preprocessor, env.n_actions - 1)

        logit_PF_module = logit_PF_config.parse(env=env)
        logit_PF = LogitPFEstimator(preprocessor=preprocessor, module=logit_PF_module)
        logit_PB_module = logit_PB_config.parse(
            env=env, tied_to=logit_PF_module if self.tied else None
        )
        logit_PB = LogitPBEstimator(preprocessor=preprocessor, module=logit_PB_module)

        return (preprocessor, logit_PF, logit_PB)


@dataclass
class DBParametrizationConfig(PFBasedParametrizationConfig):
    def parse(
        self,
        env: Env,
        logit_PF_config: GFNModuleConfig,
        logit_PB_config: GFNModuleConfig,
        logF_state_config: GFNModuleConfig,
        **kwargs,
    ) -> Tuple[Parametrization, Loss]:
        preprocessor, logit_PF, logit_PB = super().get_preprocessor_and_estimators(
            env, logit_PF_config=logit_PF_config, logit_PB_config=logit_PB_config
        )
        self.adjust_module_config(logF_state_config, preprocessor, 1)

        logF_module = logF_state_config.parse(env=env)
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
        logit_PF_config: GFNModuleConfig,
        logit_PB_config: GFNModuleConfig,
        **kwargs,
    ) -> Tuple[Parametrization, Loss]:
        _, logit_PF, logit_PB = super().get_preprocessor_and_estimators(
            env, logit_PF_config=logit_PF_config, logit_PB_config=logit_PB_config
        )
        logZ_tensor = torch.tensor(self.logZ_init, dtype=torch.float)
        logZ = LogZEstimator(tensor=logZ_tensor)
        parametrization = TBParametrization(logit_PF, logit_PB, logZ)
        loss = TrajectoryBalance(parametrization, self.reward_clip_min)
        return (parametrization, loss)


@dataclass
class OptimConfig(JsonSerializable, ABC):
    lr: float = 1e-3
    lr_Z: Optional[float] = 0.1

    scheduler_gamma: Optional[float] = None
    scheduler_milestones: Optional[List[int]] = None

    def get_params(
        self, parametrization: Parametrization
    ) -> Tuple[List[Dict[str, List[torch.Tensor]]], Tuple[float, List[int]]]:
        params = [
            {
                "params": [
                    val
                    for key, val in parametrization.parameters.items()
                    if key != "logZ"
                ],
                "lr": self.lr,
            }
        ]
        if "logZ" in parametrization.parameters:
            params.append(
                {"params": [parametrization.parameters["logZ"]], "lr": self.lr_Z}
            )

        if self.scheduler_gamma is None or self.scheduler_milestones is None:
            scheduler_gamma = 1.0
            scheduler_milestones = [0]
        else:
            scheduler_gamma = self.scheduler_gamma
            scheduler_milestones = self.scheduler_milestones

        return (params, (scheduler_gamma, scheduler_milestones))

    @abstractmethod
    def parse(
        self, parametrization: Parametrization
    ) -> Tuple[torch.optim.Optimizer, Any]:
        pass


@dataclass
class AdamConfig(OptimConfig):
    betas: Tuple[float, float] = (0.9, 0.999)

    def parse(
        self, parametrization: Parametrization
    ) -> Tuple[torch.optim.Optimizer, Any]:
        (params, (scheduler_gamma, scheduler_milestones)) = super().get_params(
            parametrization
        )
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=self.betas)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
        )
        return (optimizer, scheduler)


@dataclass
class SGDConfig(OptimConfig):
    momentum: float = 0.0

    def parse(
        self, parametrization: Parametrization
    ) -> Tuple[torch.optim.Optimizer, Any]:
        (params, (scheduler_gamma, scheduler_milestones)) = super().get_params(
            parametrization
        )
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
        )
        return (optimizer, scheduler)


@dataclass
class ActionSamplerConfig(JsonSerializable):
    temperature: float = 1.0
    sf_temperature: float = 0.0
    scheduler_gamma: Optional[float] = None
    scheduler_milestones: Optional[List[int]] = None

    def parse(self, parametrization: Parametrization) -> ActionSampler:
        if isinstance(parametrization, FMParametrization):
            action_sampler_cls = LogEdgeFlowsActionSampler
            estimator = parametrization.logF
        elif isinstance(parametrization, PFBasedParametrization):
            action_sampler_cls = LogitPFActionSampler
            estimator = parametrization.logit_PF
        else:
            raise ValueError(f"Unknown parametrization {parametrization}")
        return action_sampler_cls(
            estimator,
            temperature=self.temperature,
            sf_temperature=self.sf_temperature,
            scheduler_gamma=self.scheduler_gamma,
            scheduler_milestones=self.scheduler_milestones,
        )


def new_GFNModuleConfig():
    return subgroups(
        {
            "tabular": TabularConfig,
            "uniform": UniformConfig,
            "neural_net": NeuralNetConfig,
        },
        default=NeuralNetConfig(),
    )


@dataclass
class GFlowNetConfig(JsonSerializable):
    """Configuration for the GFlowNet trainer."""

    env: EnvConfig = subgroups(
        {"HyperGrid": HyperGridConfig}, default=HyperGridConfig()
    )
    logF_edge: GFNModuleConfig = new_GFNModuleConfig()
    logF_state: GFNModuleConfig = new_GFNModuleConfig()
    logit_PF: GFNModuleConfig = new_GFNModuleConfig()
    logit_PB: GFNModuleConfig = new_GFNModuleConfig()

    parametrization: ParametrizationConfig = subgroups(
        {
            "FM": FMParametrizationConfig,
            "DB": DBParametrizationConfig,
            "TB": TBParametrizationConfig,
        },
        default=TBParametrizationConfig(),
    )
    optim: OptimConfig = subgroups(
        {"Adam": AdamConfig, "SGD": SGDConfig}, default=AdamConfig()
    )
    action_sampler: ActionSamplerConfig = ActionSamplerConfig()
    batch_size: int = 16
    n_iterations: int = 10000
    device: Optional[str] = None

    def parse(self):
        # Setting useless attributes to None for better logging
        if isinstance(self.parametrization, PFBasedParametrizationConfig):
            self.logF_edge = None
        else:
            self.logit_PF = None
            self.logit_PB = None
        if not isinstance(self.parametrization, DBParametrizationConfig):
            self.logF_state = None
        if not isinstance(self.parametrization, TBParametrizationConfig):
            self.optim.lr_Z = None

        env = self.env.parse()
        setattr(self, "device", str(env.device))
        device = env.device
        parametrization, loss_fn = self.parametrization.parse(
            env=env,
            logF_edge_config=self.logF_edge,
            logF_state_config=self.logF_state,
            logit_PF_config=self.logit_PF,
            logit_PB_config=self.logit_PB,
        )
        action_sampler = self.action_sampler.parse(parametrization)

        # The following action sampler will be used for sampling evaluation trajectories
        raw_action_sampler = ActionSamplerConfig().parse(parametrization)

        if isinstance(self.parametrization, DBParametrizationConfig):
            training_sampler_cls = TransitionsSampler
        elif isinstance(self.parametrization, TBParametrizationConfig):
            training_sampler_cls = TrajectoriesSampler
        else:
            raise NotImplementedError("Unimplemented parametrization")
        training_sampler: TrainingSampler = training_sampler_cls(
            env=env, action_sampler=action_sampler
        )
        validation_trajectories_sampler = TrajectoriesSampler(
            env=env, action_sampler=raw_action_sampler
        )
        optimizer, scheduler = self.optim.parse(parametrization)
        return (
            env,
            parametrization,
            loss_fn,
            optimizer,
            scheduler,
            training_sampler,
            validation_trajectories_sampler,
            self.batch_size,
            self.n_iterations,
            device,
        )
