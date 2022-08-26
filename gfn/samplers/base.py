from abc import ABC, abstractmethod
from typing import Any

from gfn.envs import Env

from .actions_samplers import ActionsSampler, BackwardActionsSampler


class TrainingSampler(ABC):
    """
    Abstract base class for samplers of objects (states, trajectories, or transitions) that are used during training
    """

    def __init__(self, env: Env, actions_sampler: ActionsSampler, **kwargs):
        self.env = env
        self.actions_sampler = actions_sampler
        self.is_backward = isinstance(actions_sampler, BackwardActionsSampler)

    @abstractmethod
    def sample(self, n_objects: int) -> Any:
        """
        Sample a batch of objects.
        """
        pass
