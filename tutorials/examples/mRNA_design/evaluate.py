from utils import compute_reward


def evaluate(env, sampler, weights, n_samples=100):

    env.set_weights(weights)
    eval_trajectories = sampler.sample_trajectories(env, n=n_samples)
    final_states = eval_trajectories.terminating_states.tensor
    samples = {}

    gc_list = []
    mfe_list = []
    cai_list = []

    for state in final_states:

        reward, components = compute_reward(state, env.codon_gc_counts, weights)
        seq = "".join([env.idx_to_codon[i.item()] for i in state])
        samples[seq] = [reward, components]

        gc_list.append(components[0])
        mfe_list.append(components[1])
        cai_list.append(components[2])

    return samples, gc_list, mfe_list, cai_list
