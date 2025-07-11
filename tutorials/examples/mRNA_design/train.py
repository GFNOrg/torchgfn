import numpy as np
from tqdm import tqdm
from utils import compute_reward


def train(args, env, gflownet, sampler, optimizer, scheduler, device):

    loss_history = []
    reward_history = []
    reward_components_history = []

    unique_sequences = set()

    for it in tqdm(range(args.n_iterations), dynamic_ncols=True):

        weights = np.random.dirichlet([1, 1, 1])

        env.set_weights(weights)

        trajectories = sampler.sample_trajectories(
            env, args.batch_size, save_logprobs=True, epsilon=args.epsilon
        )

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        final_states = trajectories.terminating_states.tensor.to(device)
        rewards, components = [], []

        for state in final_states:

            state = state.to(device)

            r, c = compute_reward(state, env.codon_gc_counts, env.weights)
            rewards.append(r)
            components.append(c)

            seq = "".join([env.idx_to_codon[i.item()] for i in state])
            unique_sequences.add(seq)

        avg_reward = sum(rewards) / len(rewards)
        reward_history.append(avg_reward)
        reward_components_history.extend(components)
        loss_history.append(loss.item())

    return loss_history, reward_history, reward_components_history, unique_sequences
