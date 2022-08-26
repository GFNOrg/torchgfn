from dataclasses import dataclass
from typing import Literal

from simple_parsing import subgroups
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

    def parse(self, device: Literal["cpu", "cuda"]) -> Env:
        return HyperGrid(
            ndim=self.ndim,
            height=self.height,
            R0=self.R0,
            R1=self.R1,
            R2=self.R2,
            reward_cos=self.reward_cos,
            device=device,
        )


@dataclass
class BitSequenceConfig(BaseEnvConfig):
    def parse(self, device: Literal["cpu", "cuda"]) -> Env:
        raise NotImplementedError("Not implemented yet")


@dataclass
class EnvConfig(JsonSerializable):
    env: BaseEnvConfig = subgroups(
        {"HyperGrid": HyperGridConfig, "BitSequence": BitSequenceConfig},
        default=HyperGridConfig(),
    )

    def parse(self, device: Literal["cpu", "cuda"]) -> Env:
        return self.env.parse(device)
