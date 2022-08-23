from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional

import torch
from torchtyping import TensorType

from gfn.containers import States, Trajectories

# Typing
TensorPmf = TensorType["n_states", float]


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

    def sample(self, n_trajectories: Optional[int] = None) -> Trajectories:
        if n_trajectories is None:
            return self.trajectories
        return self.trajectories.sample(n_trajectories)


class FinalStateDistribution:
    """
    Represents a distribution over final states.
    """

    def __init__(
        self, trajectory_distribution: EmpiricalTrajectoryDistribution
    ) -> None:
        self.trajectory_distribution = trajectory_distribution
        self.states_to_indices = (
            self.trajectory_distribution.trajectories.env.get_states_indices
        )
        self.n_states = self.trajectory_distribution.trajectories.env.n_states

    def sample(self, n_final_states: Optional[int] = None) -> States:
        """
        Sample a batch of final states.
        """
        trajectories = self.trajectory_distribution.sample(n_final_states)
        return trajectories.last_states

    def pmf(self) -> TensorPmf:
        """
        Compute the probability mass function of the distribution.
        """
        samples = self.sample()
        samples_indices = self.states_to_indices(samples).numpy().tolist()
        counter = Counter(samples_indices)
        counter_list = [counter[state_idx] for state_idx in range(self.n_states)]
        return torch.tensor(counter_list, dtype=torch.float) / len(samples_indices)
