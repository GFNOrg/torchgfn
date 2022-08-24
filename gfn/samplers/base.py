from abc import ABC, abstractmethod


class TrainingSampler(ABC):
    """
    Abstract base class for samplers of objects (states, trajectories, or transitions) that are used during training
    """

    @abstractmethod
    def sample(self, n_objects: int) -> object:
        """
        Sample a batch of objects.
        """
        pass
