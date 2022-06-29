import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, max_length, state_dim):
        self.capacity = capacity
        self.max_length = max_length

        self._trajectories = torch.zeros((capacity, max_length + 1, state_dim))
        self._actions = torch.zeros((capacity, max_length)).long()
        self._rewards = torch.zeros((capacity, ))

        self._index = 0
        self._is_full = False

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def add(self, trajectories, actions, rewards, dones):
        to_add = trajectories[dones].shape[0]
        indices = torch.arange(self._index, self._index + to_add) % self.capacity

        self._is_full |= (self._index + to_add >= self.capacity)
        self._index = (self._index + to_add) % self.capacity

        self._trajectories[indices] = trajectories[dones]
        self._actions[indices] = actions[dones]
        self._rewards[indices] = rewards[dones]

    def sample(self, batch_size, rng=np.random.default_rng()):
        indices = rng.choice(len(self), size=batch_size, replace=False)

        return self._trajectories[indices], self._actions[indices], self._rewards[indices]

    
if __name__ == '__main__':
    from gflownet_playground.envs.hypergrid.hypergrid_env import HyperGrid
    from gflownet_playground.envs.hypergrid.utils import OneHotPreprocessor
    from gflownet_playground.gfn_models import PF
    from gflownet_playground.utils import sample_trajectories, evaluate_trajectories


    ndim = 3
    H = 8
    max_length = 6
    temperature = 2

    env = HyperGrid(ndim, H)
    preprocessor = OneHotPreprocessor(ndim, H)
    print('Sampling 5 trajectories starting from the origin with a random P_F network, max_length {}'.format(max_length))
    pf = PF(input_dim=H ** ndim, n_actions=ndim + 1, preprocessor=preprocessor, h=32)
    start_states = torch.zeros(5, ndim).float()
    trajectories, actions, dones = sample_trajectories(env, pf, start_states, max_length, temperature)
    rewards = evaluate_trajectories(env, trajectories, actions, dones)
    print('Number of done trajectories amongst samples: ', dones.sum().item())

    print('Initializing a buffer of capacity 10...')
    buffer = ReplayBuffer(capacity=10, max_length=max_length, state_dim=ndim)
    print('Storing the done trajectories in the buffer')
    buffer.add(trajectories, actions, rewards, dones)
    print('There are {} trajectories in the buffer'.format(len(buffer)))

    print('Resampling 7 trajectories and adding the done ones to the same buffer')
    start_states = torch.zeros(7, ndim).float()
    trajectories, actions, dones = sample_trajectories(env, pf, start_states, max_length, temperature)
    rewards = evaluate_trajectories(env, trajectories, actions, dones)
    print('Number of done trajectories amongst samples: ', dones.sum().item())
    buffer.add(trajectories, actions, rewards, dones)
    print('There are {} trajectories in the buffer'.format(len(buffer)))

    print('Sampling 2 trajectories: ')
    trajectories, actions, rewards = buffer.sample(2)
    print(trajectories, actions, rewards)



