from dataclasses import dataclass
from typing import Literal, Optional

import torch
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
from gfn.models import NeuralNet, Tabular, Uniform
from gfn.parametrizations import O_PF, O_PFB, O_PFBZ, O_PFZ, EdgeFlowParametrization
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor, OneHotPreprocessor


@dataclass
class EnvConfig(JsonSerializable):
    name: Literal["HyperGrid"] = "HyperGrid"
    ndim: int = 2
    height: int = 4

    def parse(self) -> Env:
        assert self.ndim > 0 and self.height > 0
        return HyperGrid(ndim=self.ndim, height=self.height)


@dataclass
class GFNModuleConfig(JsonSerializable):
    name: Literal["NeuralNet", "Tabular", "Uniform"] = "NeuralNet"
    input_dim: Optional[int] = None
    output_dim: int = 1
    n_hidden_layers: Optional[int] = 2
    hidden_dim: Optional[int] = 256
    activation_fn: Optional[Literal["relu", "tanh"]] = "relu"

    def __post_init__(self):
        if self.name != "NeuralNet":
            self.n_hidden_layers = None
            self.hidden_dim = None
            self.activation_fn = None
        else:
            assert self.input_dim is not None

    def parse(self, env: Env, tied_to: Optional[GFNModule]) -> GFNModule:
        if self.name == "NeuralNet":
            return NeuralNet(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                n_hidden_layers=self.n_hidden_layers,
                hidden_dim=self.hidden_dim,
                activation_fn=self.activation_fn,
                torso=tied_to.torso if tied_to is not None else None,
            )
        elif self.name == "Tabular":
            return Tabular(env=env, output_dim=self.output_dim)
        else:
            return Uniform(output_dim=self.output_dim)


@dataclass
class ParametrizationConfig(JsonSerializable):
    name: Literal["edge", "PF", "PFZ", "PFB", "PFBZ"] = "PFZ"
    PF: Optional[GFNModuleConfig] = GFNModuleConfig()
    PB: Optional[GFNModuleConfig] = GFNModuleConfig("Uniform")
    tied: Optional[bool] = False
    preprocessor: Literal["OneHot", "KHot", "Identity"] = "KHot"
    F_edge: Optional[GFNModuleConfig] = None
    F_state: Optional[GFNModuleConfig] = None
    logZ_init: Optional[float] = 0.0

    def __post_init__(self):
        if self.name != "edge":
            assert self.PF is not None and self.PB is not None
            if self.PF.name == "Uniform":
                raise ValueError("Uniform is not a valid value for pf")
            if self.PF.name == "Tabular" and self.preprocessor != "Identity":
                self.preprocessor = "Identity"
                print("Warning: preprocessor set to Identity for Tabular PF")

            if self.name == "PFZ" and self.PB.name != "Uniform":
                self.PB = GFNModuleConfig(name="Uniform")
                print(
                    "Warning: PB is set to Uniform for PFZ - It's the only non-learnable pb implemented"
                )
            if self.name == "PF" or self.name == "PFB":
                assert self.F_state is not None and self.F_state.name != "Uniform"

            if self.name != "PFZ" and self.name != "PFBZ":
                self.logZ_init = None

            if self.tied:
                if self.PF.name != "NeuralNet" or self.PB.name != "NeuralNet":
                    raise ValueError(
                        "Tied parameterization is only possible for NeuralNet pf and pb"
                    )
                self.PB.hidden_dim = self.PF.hidden_dim
                self.PB.n_hidden_layers = self.PF.n_hidden_layers
                self.PB.activation_fn = self.PF.activation_fn

            if self.PF.name != "NeuralNet" or self.PB.name != "NeuralNet":
                self.tied = None

        else:
            assert self.F_edge is not None
            self.PF = None
            self.PB = None
            self.tied = None
            self.F_state = None

    def parse(self, env: Env):
        if self.preprocessor == "Identity":
            preprocessor = IdentityPreprocessor(env)
        elif self.preprocessor == "OneHot":
            preprocessor = OneHotPreprocessor(env)
        else:
            preprocessor = KHotPreprocessor(env)

        if self.name == "edge":
            assert self.F_edge is not None
            self.F_edge.output_dim = env.n_actions - 1
            self.F_edge.input_dim = preprocessor.output_dim
            logF_edge_module = self.F_edge.parse(env)
            logF_edge = LogEdgeFlowEstimator(
                preprocessor=preprocessor, env=env, module=logF_edge_module
            )
            return EdgeFlowParametrization(logF_edge)
        else:
            assert self.PF is not None and self.PB is not None
            self.PF.input_dim = preprocessor.output_dim
            self.PB.input_dim = preprocessor.output_dim
            self.PF.output_dim = env.n_actions
            self.PB.output_dim = env.n_actions - 1
            logit_PF_module = self.PF.parse(env)
            logit_PB_module = self.PB.parse(
                env, tied_to=logit_PF_module if self.tied else None
            )
            logit_PF = LogitPFEstimator(
                preprocessor=preprocessor, module=logit_PF_module
            )
            logit_PB = LogitPBEstimator(
                preprocessor=preprocessor, module=logit_PB_module
            )
            if self.name == "PF" or self.name == "PFB":
                assert self.F_state is not None
                self.F_state.output_dim = 1
                self.F_state.input_dim = preprocessor.output_dim
                logF_state_module = self.F_state.parse(env)
                logF_state = LogStateFlowEstimator(
                    preprocessor=preprocessor, module=logF_state_module
                )
                if self.name == "PF":
                    return O_PF(logit_PF, logit_PB, logF_state)
                else:
                    return O_PFB(logit_PF, logit_PB, logF_state, tied=self.tied)
            elif self.name == "PFZ" or self.name == "PFBZ":
                assert self.logZ_init is not None
                logZ_tensor = torch.tensor(self.logZ_init, dtype=torch.float)
                logZ = LogZEstimator(logZ_tensor)
                if self.name == "PFZ":
                    return O_PFZ(logit_PF, logit_PB, logZ)
                else:
                    return O_PFBZ(logit_PF, logit_PB, logZ, tied=self.tied)


@dataclass
class OptimConfig(JsonSerializable):
    ...


@dataclass
class LossConfig(JsonSerializable):
    ...


@dataclass
class GFlowNetConfig(JsonSerializable):
    """Configuration for the GFlowNet trainer."""

    env: EnvConfig = EnvConfig()
    parametrization: ParametrizationConfig = ParametrizationConfig()
    optim: OptimConfig = OptimConfig()
    loss: LossConfig = LossConfig()

    def parse(self):
        ...
