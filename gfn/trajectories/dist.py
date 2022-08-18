from abc import ABC, abstractmethod

from gfn.containers import States, Trajectories


class TrajectoryDistribution(ABC):
    """
    Represents an abstract distribution over trajectories.
    """

    @abstractmethod
    def sample(self, n_trajectories: int) -> Trajectories:
        """
        Sample a batch of trajectories.
        """
        pass


class EmpiricalTrajectoryDistribution(TrajectoryDistribution):
    """
    Represents an empirical distribution over trajectories.
    """

    def __init__(self, trajectories: Trajectories):
        self.trajectories = trajectories

    def sample(self, n_trajectories: int) -> Trajectories:
        return self.trajectories.sample(n_trajectories)


class FinalStateDistribution:
    """
    Represents a distribution over final states.
    """

    def __init__(self, trajectory_distribution: TrajectoryDistribution) -> None:
        self.trajectory_distribution = trajectory_distribution

    def sample(self, n_final_states: int) -> States:
        """
        Sample a batch of final states.
        """
        trajectories = self.trajectory_distribution.sample(n_final_states)
        return trajectories.last_states
