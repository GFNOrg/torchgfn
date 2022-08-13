from abc import ABC, abstractmethod
from dataclasses import dataclass
from gfn.trajectories import TrajectoryDistribution, FinalStateDistribution
from gfn.envs import Env


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
