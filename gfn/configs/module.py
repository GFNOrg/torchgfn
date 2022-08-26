from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from simple_parsing.helpers import JsonSerializable

from gfn.envs import Env
from gfn.modules import GFNModule, NeuralNet, Tabular, Uniform


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
    tied_to: Optional[str] = None

    def parse(
        self,
        tied_to: Optional[GFNModule] = None,
        tied_to_name: Optional[str] = None,
        **kwargs
    ) -> NeuralNet:
        del kwargs
        if tied_to is not None:
            self.n_hidden_layers = None
            self.hidden_dim = None
            self.activation_fn = None
            self.tied_to = tied_to_name

        return NeuralNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_hidden_layers=self.n_hidden_layers,
            hidden_dim=self.hidden_dim,
            activation_fn=self.activation_fn,
            torso=tied_to.torso if tied_to is not None else None,
        )
