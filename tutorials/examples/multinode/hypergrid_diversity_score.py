"""
L1 Distance Scorer for ReplayBufferManager

This module provides a scoring function that computes the L1 (Manhattan) distance
of trajectories' final states, which can be used as a diversity metric for the
ReplayBufferManager in torchgfn.
"""

import torch

from gfn.containers.replay_buffer import ContainerUnion
from gfn.containers.states_container import StatesContainer
from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions


class HypergridDiversityScore:

    def __init__(self, ndim: int, height: int):
        self.ndim = ndim
        self.height = height
        self._values_set = torch.zeros(ndim, height, dtype=torch.bool)

    def __call__(self, container: ContainerUnion) -> float:
        """
        Computes a diversity scores, by computing the average of new values across the grid dimension.

        Args:
            container: A container object (Trajectories, Transitions, or StatesContainer)
                        containing states to evaluate.

        Returns:
            float: The score function (between 0 and 1), as average of the score function of the states.
        """

        if len(container) == 0:
            raise ValueError("Container is empty. Cannot compute L1 distance.")

        if isinstance(container, Trajectories):
            states = container.final_states
        elif isinstance(container, Transitions):
            states = container.next_states
        elif isinstance(container, StatesContainer):
            states = container
        else:
            raise ValueError(f"Unsupported container type: {type(container)}")

        # Stack states into a single tensor for vectorized operations
        state_tensor = states.tensor  # (batch, ndim)
        score = 1 - self._values_set[state_tensor].mean()

        self._values_set[state_tensor] = True
        return score.item()
