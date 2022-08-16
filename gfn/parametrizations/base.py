from abc import ABC, abstractmethod
from dataclasses import dataclass

from gfn.envs import Env
from gfn.trajectories import FinalStateDistribution, TrajectoryDistribution


@dataclass
class Parametrization(ABC):
    """
    Abstract Base Class for Flow Parametrizations,
    as defined in Sec. 3 of GFlowNets Foundations
    """

    @abstractmethod
    def Pi(self, env: Env, **kwargs) -> TrajectoryDistribution:
        pass

    def P_T(self, env: Env) -> FinalStateDistribution:
        return FinalStateDistribution(self.Pi(env))
