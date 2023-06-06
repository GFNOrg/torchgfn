from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories
from gfn.envs import Env
from gfn.states import States

# Typing
TensorPmf = TensorType["n_states", float]


class TrajectoryDistribution(ABC):
    """Represents an abstract distribution over trajectories."""

    @abstractmethod
    def sample(self, n_trajectories: int) -> Trajectories:
        """Sample a batch of trajectories."""
        pass


class TerminatingStatesDistribution(ABC):
    """Represents an abstract distribution over terminating states."""

    @abstractmethod
    def pmf(self) -> TensorPmf:
        """Compute the probability mass function of the distribution."""
        pass


class EmpiricalTrajectoryDistribution(TrajectoryDistribution):
    """Represents an empirical distribution over trajectories.

    Attributes:
        trajectories: a batch of trajectories.
    """
    def __init__(self, trajectories: Trajectories):
        """Initialize distribution from an empirical batch of trajectories.

        Args:
            trajectories: a batch of trajectories.

        """
        self.trajectories = trajectories

    def sample(self, n_trajectories: Optional[int] = None) -> Trajectories:
        if n_trajectories is None:
            return self.trajectories
        return self.trajectories.sample(n_trajectories)


class EmpiricalTerminatingStatesDistribution(TerminatingStatesDistribution):
    """Represents an empirical distribution over terminating states.

    Attributes:
        states: the States instance.
        n_states: number of states in the trajectory.
        states_to_indices: a tuple of terminating states indices.
        env_n_terminating_states: the number of terminating states.
    """

    def __init__(self, env: Env, states: States) -> None:
        """Initialize the distiribution from an environment and batch of states.
        Args:
            env: the Environment.
            states: a linear batch of states.
        """
        assert len(states.batch_shape) == 1, "States should be a linear batch of states"
        self.states = states
        self.n_states = states.batch_shape[0]
        self.states_to_indices = env.get_terminating_states_indices
        self.env_n_terminating_states = env.n_terminating_states

    def pmf(self) -> TensorPmf:
        states_indices = self.states_to_indices(self.states).cpu().numpy().tolist()
        counter = Counter(states_indices)
        counter_list = [
            counter[state_idx] if state_idx in counter else 0
            for state_idx in range(self.env_n_terminating_states)
        ]
        return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)


class TrajectoryBasedTerminatingStateDistribution(TerminatingStatesDistribution):
    """Represents a distribution over final states.

    Attributes:
        trajectory_distribution: A distribution over trajectories observed.
        states_to_indices: A tuple of all terminating state indices for the
            trajectories.
        env_n_terminating_states: Tuple containing the number of terminating states for
            these trajectories.
    """

    def __init__(
        self, trajectory_distribution: EmpiricalTrajectoryDistribution
    ) -> None:
        """Initalize the distribution from a batch of trajectories.

        Args:
            trajectory_distribution: A distribution over trajectories observed.
        """
        self.trajectory_distribution = trajectory_distribution
        # TODO: See if we can remove env from living inside trajectories.
        self.states_to_indices = (
            self.trajectory_distribution.trajectories.env.get_terminating_states_indices
        )
        self.env_n_terminating_states = (
            self.trajectory_distribution.trajectories.env.n_terminating_states
        )

    # TODO: change variable name to n ?
    def sample(self, n_final_states: Optional[int] = None) -> States:
        """Sample a batch of n final states."""
        trajectories = self.trajectory_distribution.sample(n_final_states)
        return trajectories.last_states

    def pmf(self) -> TensorPmf:
        """Compute the probability mass function of the distribution."""
        samples = self.sample()
        samples_indices = self.states_to_indices(samples).cpu().numpy().tolist()
        counter = Counter(samples_indices)
        counter_list = [
            counter[state_idx] if state_idx in counter else 0
            for state_idx in range(self.env_n_terminating_states)
        ]
        return torch.tensor(counter_list, dtype=torch.float) / len(samples_indices)
