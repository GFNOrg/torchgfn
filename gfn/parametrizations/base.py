from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from gfn.trajectories import TrajectoryDistribution, FinalStateDistribution
from gfn.estimators import LogEdgeFlowEstimator, LogStateFlowEstimator, LogitPFEstimator, LogitPBEstimator, LogZEstimator
from gfn.envs import Env


class MarkovianFlow(ABC):
    @abstractmethod
    def log_edge_flow_function(self) -> LogEdgeFlowEstimator:
        pass

    @abstractmethod
    def log_state_flow_function(self) -> LogStateFlowEstimator:
        pass

    @abstractmethod
    def logit_PF(self) -> LogitPFEstimator:
        pass

    @abstractmethod
    def logit_PB(self) -> LogitPBEstimator:
        pass

    @abstractmethod
    def log_Z(self) -> LogZEstimator:
        pass


class O(ABC):
    r"""Abstract Base Class for the set $\mathcal{O}$,
    as defined in Sec. 3 of GFlowNets Foundations"""
    name: str


@dataclass
class Parametrization(ABC):
    """ 
    Abstract Base Class for Flow Parametrizations,
    as defined in Sec. 3 of GFlowNets Foundations
    """
    O_name: str

    @abstractmethod
    def Pi(self, o: O, env: Env, **kwargs) -> TrajectoryDistribution:
        pass

    @abstractmethod
    def H(self, flow: MarkovianFlow) -> O:
        pass

    def P_T(self, o: O, env: Env) -> FinalStateDistribution:
        return FinalStateDistribution(self.Pi(o, env))
