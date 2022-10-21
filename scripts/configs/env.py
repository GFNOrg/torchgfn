from dataclasses import dataclass
from typing import Literal

from simple_parsing import choice, subgroups
from simple_parsing.helpers import JsonSerializable

from gfn.envs import Env, HyperGrid


@dataclass
class BaseEnvConfig(JsonSerializable):
    ...


@dataclass
class HyperGridConfig(BaseEnvConfig):
    ndim: int = 2
    height: int = 4
    R0: float = 0.1
    R1: float = 0.5
    R2: float = 2.0
    reward_cos: bool = False
    preprocessor_name: str = choice("KHot", "OneHot", "Identity", default="KHot")

    def parse(self, device_str: Literal["cpu", "cuda"]) -> Env:
        return HyperGrid(
            ndim=self.ndim,
            height=self.height,
            R0=self.R0,
            R1=self.R1,
            R2=self.R2,
            reward_cos=self.reward_cos,
            device_str=device_str,
            preprocessor_name=self.preprocessor_name,
        )


@dataclass
class MoleculesConfig(BaseEnvConfig):
    def parse(self, device_str: Literal["cpu", "cuda"]) -> Env:
        raise NotImplementedError("Not implemented yet")


@dataclass
class EnvConfig(JsonSerializable):
    env: BaseEnvConfig = subgroups(
        {"HyperGrid": HyperGridConfig, "Molecules": MoleculesConfig},
        default=HyperGridConfig(),
    )

    def parse(self, device: Literal["cpu", "cuda"]) -> Env:
        return self.env.parse(device)
