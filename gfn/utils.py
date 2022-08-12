import torch
from torch import Tensor
from torch import nn
from torch.distributions import Categorical
from copy import deepcopy


from gfn.envs.env import Env


def sample_trajectories(env: Env, pf: nn.Module, start_states: Tensor, max_length: int, temperature: float = 1.):
    """
    Function to roll-out trajectories starting from start_states using pf
    :param env: object of type gfn.envs.env.Env
    :param pf: nn.Module representing forward transition probabilities (e.g. gfn.gfn_models.PF)
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
            all_actions[~dones, step] = actions
            states[~dones], dones[~dones] = env.step(states[~dones], actions)
            step += 1
            all_trajectories[~old_dones, step, :] = states[~old_dones]
        rewards[dones] = env.reward(states[dones])
    last_states = states[dones]
    return all_trajectories, last_states, all_actions.long(), dones, rewards


def uniform_sample_trajectories(env, start_states):
    """
    Function to generate trajectories from a uniform distribution starting from start_states
    :param env: object of type gfn.envs.env.Env
    :param start_states: start_states to start with. tensor of size k x state_dim
    """
    # TODO: uniform sample final state and use uniform PB until getting to start_state
    import numpy as np

    k = start_states.shape[0]
    trajectories = []
    actionss = []
    rewards = []
    for i in range(k):
        trajectory = []
        actions = []
        state = start_states[[i]]
        trajectory.append(state[0])
        done = False
        while not done:
            mask = env.mask_maker(state)
            action = np.random.choice(
                [i for i, j in enumerate(mask[0]) if not j])
            actions.append(action)
            action = torch.tensor([action])
            # action = torch.randint(0, env.n_actions, (1,))
            next_state, done = env.step(state, action)
            trajectory.append(next_state[0])
            state = next_state
        trajectories.append(torch.stack(trajectory))
        actionss.append(torch.tensor(actions))
        rewards.append(env.reward(state))

    return trajectories, actionss, rewards


if __name__ == '__main__':
    from gfn.envs.hypergrid_env import HyperGrid
    from gfn.envs.utils import OneHotPreprocessor
    from gfn.gfn_models import PF, UniformPB, UniformPF

    ndim = 3
    H = 8
    max_length = 150
    temperature = 5

    env = HyperGrid(ndim, H)
    preprocessor = OneHotPreprocessor(ndim, H)
    print('Initializing a random P_F netowork...')
    pf = PF(input_dim=H ** ndim, n_actions=ndim +
            1, preprocessor=preprocessor, h=32)
    print('Starting with 5 random start states:')
    start_states = torch.randint(0, H, (5, ndim)).float()
    print(start_states)
    print('Rolling-out trajectories of max_length {}...'.format(max_length))
    trajectories, last_states, actions, dones, rewards = sample_trajectories(
        env, pf, start_states, max_length, temperature)
    print('Trajectories: {}'.format(trajectories))
    print('Last states: {}'.format(last_states))
    print('Actions: {}'.format(actions))
    print('Dones: {}'.format(dones))
    print('Rewards: {}'.format(rewards))

    print('Sampling 10 trajectories from a uniform distribution...')
    start_states = torch.zeros((10, ndim)).float()
    trajectories, actionss, rewards = uniform_sample_trajectories(
        env, start_states)
    print('Trajectories: {}'.format(trajectories))
    print('Actions: {}'.format(actionss))
    print('Rewards: {}'.format(rewards))
