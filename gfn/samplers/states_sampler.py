from gfn.containers import States
from gfn.containers.trajectories import Trajectories
from gfn.envs.env import Env
from gfn.samplers.actions_samplers import ActionsSampler
from gfn.samplers.trajectories_sampler import TrajectoriesSampler

from .base import TrainingSampler


class StatesSampler(TrainingSampler):
    def __init__(self, env: Env, actions_sampler: ActionsSampler, **kwargs):
        super().__init__(env, actions_sampler, **kwargs)
        self.trajectories_sampler = TrajectoriesSampler(env, actions_sampler, **kwargs)

        class TrainingStates(env.States):
            """This subclasses the base States class. It adds an extra attribute to a batch of States,
            `last_states`, which is a batch of States representing the last states of the trajectories"""

            @classmethod
            def from_trajectories(cls, trajectories: Trajectories):
                states = trajectories.states.flatten()
                states.last_states = trajectories.last_states
                return states

        self.TrainingStates = TrainingStates

    def sample(self, n_objects: int) -> States:
        trajectories = self.trajectories_sampler.sample(n_objects=n_objects)
        return self.TrainingStates.from_trajectories(trajectories)
