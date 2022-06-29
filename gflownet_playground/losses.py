import torch
import torch.nn as nn
from torch.distributions import Categorical


def trajectory_balance_loss(env, pf, pb, logZ, traj, actions, reward, MIN_REW=1e-5):
    """
    :param env: instantiation of envs.env.Env with a self.mask_maker function
    :param pf: nn.Module representing P_F (can be an instantiation of gfn_models.PF e.g.)
    :param pb: nn.Module representing P_B (can be an instantiation of gfn_models.UniformPB e.g.)
    :param logZ: scalar tensor 
    :param traj: trajectory to evaluate the loss on, as a tensor of size n x state_dim
    :param actions: actions compatible with the trajectory, as a tensor of size n, the last of which is n_actions - 1 (i.e. done action)
    :param MIN_REW: minimum reward value to replace lower rewards (including -infinity) with
    :param reward: reward at the end of the trajectory
    :return: loss(traj)
    needs the GFN to be paramertrized with logZ
    """
    assert traj.shape[0] == len(actions) and all(actions[:-1] < actions[-1])
    trajectory_logits = pf(traj, env.mask_maker(traj))
    denominators = torch.logsumexp(trajectory_logits, 1)
    forward_logprobs = torch.gather(trajectory_logits, 1, actions.unsqueeze(1)).squeeze(1)
    forward_logprobs = forward_logprobs - denominators

    backward_logits = pb(traj[1:], env.backward_mask_maker(traj[1:]))
    log_PB_all = backward_logits.log_softmax(1)
    log_PB_actions = torch.gather(log_PB_all, 1, actions[:-1].unsqueeze(1)).squeeze(1)

    log_pb = log_PB_actions

    pred = forward_logprobs.sum() - log_pb.sum() + logZ
    target = torch.log(reward.clamp_min(MIN_REW).to(pred.dtype))
    loss = nn.MSELoss()(pred, target)

    trajectory_logprob = forward_logprobs.sum()

    return loss, trajectory_logprob

def online_TB_loss(env, pf, pb, logZ, start_states, loss_fn=nn.MSELoss(),
 temperature=1., device=None):
    """
    Function to roll-out a batch of trajectories starting from start_states using pf, then evaluate the average TB loss on the trajectories
    :param env: object of type gflownet_playground.envs.env.Env
    :param pf: nn.Module representing forward transition probabilities (e.g. gflownet_playground.gfn_models.PF)
    :param pb: nn.Module representing backward transition probabilities (e.g. gflownet_playground.gfn_models.UniformPB)
    :param logZ: scalar tensor
    :param start_states: start_states to start with. tensor of size k x state_dim
    :param temperature: float, temperature to trade off between raw P_F and uniform
    :return: final_states: tensor of shape (k x state_dim), average TB loss on the trajectories as a scalar tensor, rewards as a tensor of shape (k)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_trajectories = start_states.shape[0]
    trajectories_logprobs = torch.zeros(n_trajectories).to(device)

    dones = torch.zeros(n_trajectories).bool()
    states = start_states
    actions = None 
    log_PB_actions = None
    log_PF_actions = None
    while torch.any(~dones):
        non_terminal_states = states[~dones]
        if actions is not None:
            backward_masks = env.backward_mask_maker(states[~dones])
            pb_logits = pb(non_terminal_states, backward_masks)
            log_PB_all = pb_logits.log_softmax(1)
            log_PB_actions = torch.gather(log_PB_all, 1, actions[~over].unsqueeze(1)).squeeze(1)
            trajectories_logprobs[~dones] -= log_PB_actions
        
        masks = env.mask_maker(non_terminal_states)
        logits = pf(non_terminal_states, masks)
        probs = Categorical(logits=logits / temperature)
        actions = probs.sample()
        next_states, over = env.step(non_terminal_states, actions) 
        log_PF_all = logits.log_softmax(1)   
        log_PF_actions = torch.gather(log_PF_all, 1, actions.unsqueeze(1)).squeeze(1)
        trajectories_logprobs[~dones] += log_PF_actions
        states[~dones] = next_states
        dones[~dones] = over

    rewards = env.reward(states)
    pred = trajectories_logprobs + logZ
    loss = loss_fn(pred, rewards)
        
    return states, loss, rewards

if __name__ == '__main__':
    from gflownet_playground.envs.hypergrid.hypergrid_env import HyperGrid
    from gflownet_playground.envs.hypergrid.utils import OneHotPreprocessor, uniform_backwards_prob
    from gflownet_playground.gfn_models import PF
    from gflownet_playground.utils import sample_trajectories, evaluate_trajectories
    from gflownet_playground.replay_buffer import ReplayBuffer

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

    print('Sampling 1 trajectory from the buffer: ')
    trajectories, actionss, rewards = buffer.sample(1)
    trajectory = trajectories[0]
    actions = actionss[0]
    reward = rewards[0]
    print(trajectory, actions, reward)
    print(trajectory_balance_loss(env, pf, logZ=torch.FloatTensor([0]),
     traj=trajectory[:-1][actions != -1], actions=actions[actions != -1], reward=reward, pb_fn=uniform_backwards_prob))
