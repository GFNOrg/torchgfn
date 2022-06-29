import torch
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from copy import deepcopy

class Preprocessor(ABC):
    """
    Base class for Preprocessors. The goal is to transform tensors representing raw states
    to tensors that can be used as input to neural networks.
    """
    @abstractmethod
    def preprocess(self, states):
        """
        :param states: Tensor of shape (k x state_dim)
        :outputs: Tensor of shape (k x dim_in) where dim_in is the input dimension of the neural network
        """
        pass


class IdentityPreprocessor(Preprocessor): 
    def __init__(self, input_dim):
        self.output_dim = input_dim

    def preprocess(self, states):
        assert states.shape[-1] == self.output_dim
        return states


def sample_trajectories(env, pf, start_states, max_length, temperature=1.):
    """
    Function to roll-out trajectories starting from start_states using pf
    :param env: object of type gflownet_playground.envs.env.Env
    :param pf: nn.Module representing forward transition probabilities (e.g. gflownet_playground.gfn_models.PF)
    :param start_states: start_states to start with. tensor of size k x state_dim
    :param max_length: int, maximum length of trajectories
    :param temperature: float, temperature to trade off between raw P_F and uniform
    """
    rewards = torch.full((start_states.shape[0],), - float('inf'))
    n_trajectories = start_states.shape[0]
    all_trajectories = torch.ones(
        (n_trajectories, max_length + 1, start_states.shape[1])) * (- float('inf'))
    all_actions = - torch.ones((n_trajectories, max_length)).to(torch.long)
    with torch.no_grad():
        dones = torch.zeros(n_trajectories).bool()
        states = start_states
        all_trajectories[:, 0, :] = states
        step = 0
        while dones.sum() < n_trajectories and step < max_length:
            old_dones = deepcopy(dones)
            masks = env.mask_maker(states[~dones])
            logits = pf(states[~dones], masks)
            probs = torch.softmax(logits / temperature, 1)
            dist = Categorical(probs)
            actions = dist.sample()
            all_actions[~dones, step] = actions#[~dones]
            states[~dones], dones[~dones] = env.step(states[~dones], actions)
            step += 1
            all_trajectories[~old_dones, step, :] = states[~old_dones]
        rewards[dones] = env.reward(states[dones])
    last_states = states[dones]
    return all_trajectories, last_states, all_actions.long(), dones, rewards


if __name__ == '__main__':
    from gflownet_playground.envs.hypergrid.hypergrid_env import HyperGrid
    from gflownet_playground.envs.hypergrid.utils import OneHotPreprocessor
    from gflownet_playground.gfn_models import PF, UniformPB, UniformPF


    ndim = 3
    H = 8
    max_length = 150
    temperature = 5

    env = HyperGrid(ndim, H)
    preprocessor = OneHotPreprocessor(ndim, H)
    print('Initializing a random P_F netowork...')
    pf = PF(input_dim=H ** ndim, n_actions=ndim + 1, preprocessor=preprocessor, h=32)
    print('Starting with 5 random start states:')
    start_states = torch.randint(0, H, (5, ndim)).float()
    print(start_states)
    print('Rolling-out trajectories of max_length {}...'.format(max_length))
    trajectories, actions, dones, rewards = sample_trajectories(env, pf, start_states, max_length, temperature)
    print('Trajectories: {}'.format(trajectories))
    print('Actions: {}'.format(actions))
    print('Dones: {}'.format(dones))
    print('Rewards: {}'.format(rewards))
