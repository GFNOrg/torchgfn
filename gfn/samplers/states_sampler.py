from gfn.containers import States

from .base import TrainingSampler


class StatesSampler(TrainingSampler):
    def sample(self, n_states: int) -> States:
        del n_states
        raise NotImplementedError("StatesSampler.sample is not implemented")
