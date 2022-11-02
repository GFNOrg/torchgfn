"""This implements a simple dynamic programming algorithm for computing edge flows given a reward function and
a backward transition probability function. Currently it's implemented for uniform P_B, but can
trivially be extended to other P_B manually specified. Only discrete environments are handled.

DO NOT USE FOR LARGE ENVIRONMENTS !
"""

import torch
from configs import EnvConfig
from simple_parsing import ArgumentParser

from gfn.containers.states import correct_cast
from gfn.estimators import LogEdgeFlowEstimator
from gfn.losses import FMParametrization
from gfn.modules import Tabular, Uniform
from gfn.utils import validate

parser = ArgumentParser()

parser.add_arguments(EnvConfig, dest="env_config")

args = parser.parse_args()

env_config: EnvConfig = args.env_config


env = env_config.parse(device="cpu")

F_edge = torch.zeros(env.n_states, env.n_actions)

logit_PB = Uniform(output_dim=env.n_actions - 1)


all_states = env.all_states
terminating_states = env.terminating_states

all_states_indices = env.get_states_indices(all_states)
terminating_states_indices = env.get_states_indices(terminating_states)

# Zeroth step: Define the necessary containers
Y = set()  # Contains the state indices that do not need more visits

# The following represents a queue of indices of the states that need to be visited,
# and their final state flow
U = []

# First step: Fill the terminating flows with the rewards and initialize the state flows
F_edge[terminating_states_indices, -1] = env.reward(terminating_states)
F_state = torch.zeros(env.n_states)
F_state[terminating_states_indices] = env.reward(terminating_states)

# Second step: Store the states that have no children besides s_f
all_states.forward_masks, _ = correct_cast(
    all_states.forward_masks, all_states.backward_masks
)
for index in all_states_indices[all_states.forward_masks.long().sum(1) == 1].numpy():
    U.append((index, F_edge[index, -1].item()))


print("Calculating edge flows...")

# Third Step: Iterate over the states in U and update the flows
while len(U) > 0:
    s_prime_index, F_s_prime = U.pop(0)
    Y.add(s_prime_index)
    state_prime = all_states[[s_prime_index]]
    _, state_prime.backward_masks = correct_cast(
        state_prime.forward_masks, state_prime.backward_masks
    )

    backward_mask = state_prime.backward_masks[0]
    pb_logits = logit_PB(env.get_states_indices(state_prime))
    pb_logits[~backward_mask] = -float("inf")
    pb = torch.softmax(pb_logits, dim=0)
    for i in range(env.n_actions - 1):
        if backward_mask[i]:
            state = env.backward_step(state_prime, torch.tensor([i]))
            state.forward_masks, _ = correct_cast(
                state.forward_masks, state.backward_masks
            )
            s_index = env.get_states_indices(state)[0].item()
            pb_logits = logit_PB(env.get_states_indices(state_prime))
            F_edge[s_index, i] = F_s_prime * pb[i].item()
            F_state[s_index] = F_state[s_index] + F_edge[s_index, i]
            if all(
                [
                    env.get_states_indices(env.step(state, torch.tensor([j])))[0].item()
                    in Y
                    for j in range(env.n_actions - 1)
                    if state.forward_masks[0, j]
                ]
            ):
                U.append((s_index, F_state[s_index].item()))

print(F_edge)

print("Validating...")

# Sanity check - should get the right pmf
logF_edge = torch.log(F_edge + 1e-10)
logF_edge_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
logF_edge_module.logits = logF_edge
logF_edge_estimator = LogEdgeFlowEstimator(
    env=env, module_name="Tabular", module=logF_edge_module
)
parametrization = FMParametrization(logF=logF_edge_estimator)
print(validate(env, parametrization, n_validation_samples=100000))
