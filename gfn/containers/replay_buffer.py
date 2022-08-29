from typing import Literal

import torch

from ..envs import Env
from .trajectories import Trajectories
from .transitions import Transitions


class ReplayBuffer:
    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
        objects: Literal["transitions", "trajectories"] = "trajectories",
    ):
        self.env = env
        self.capacity = capacity
        self.type = objects
        if objects == "transitions":
            self.training_objects = Transitions(env)
        else:
            self.training_objects = Trajectories(env)

        self._is_full = False
        self._index = 0

    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} {self.type})"

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def add(self, training_objects: Transitions | Trajectories):
        to_add = len(training_objects)

        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity

        self.training_objects.extend(training_objects)
        self.training_objects = self.training_objects[-self.capacity :]

    def sample(self, n_objects: int):
        return self.training_objects.sample(n_objects)


if __name__ == "__main__":
    from gfn.envs import HyperGrid
    from gfn.envs.utils import OneHotPreprocessor
    from gfn.gfn_models import PF
    from gfn.utils import evaluate_trajectories, sample_trajectories

    ndim = 3
    H = 8
    max_length = 6
    temperature = 2

    env = HyperGrid(ndim, H)
    preprocessor = OneHotPreprocessor(ndim, H)
    print(
        "Sampling 5 trajectories starting from the origin with a random P_F network, max_length {}".format(
            max_length
        )
    )
    pf = PF(input_dim=H**ndim, n_actions=ndim + 1, preprocessor=preprocessor, h=32)
    start_states = torch.zeros(5, ndim).float()
    trajectories, actions, dones = sample_trajectories(
        env, pf, start_states, max_length, temperature
    )
    rewards = evaluate_trajectories(env, trajectories, actions, dones)
    print("Number of done trajectories amongst samples: ", dones.sum().item())

    print("Initializing a buffer of capacity 10...")
    buffer = ReplayBuffer(capacity=10, max_length=max_length, state_dim=ndim)
    print("Storing the done trajectories in the buffer")
    buffer.add(trajectories, actions, rewards, dones)
    print(f"There are {len(buffer)} trajectories in the buffer")

    print("Resampling 7 trajectories and adding the done ones to the same buffer")
    start_states = torch.zeros(7, ndim).float()
    trajectories, actions, dones = sample_trajectories(
        env, pf, start_states, max_length, temperature
    )
    rewards = evaluate_trajectories(env, trajectories, actions, dones)
    print("Number of done trajectories amongst samples: ", dones.sum().item())
    buffer.add(trajectories, actions, rewards, dones)
    print(f"There are {len(buffer)} trajectories in the buffer")

    print("Sampling 2 trajectories: ")
    trajectories, actions, rewards = buffer.sample(2)
    print(trajectories, actions, rewards)
