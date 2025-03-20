"""
The goal of this script is to reproduce the results of DAG-GFlowNet for
Bayesian structure learning (Deleu et al., 2022) using the GraphEnv.

Specifically, we consider a randomly generated (under the Erdős-Rényi model) linear-Gaussian
Bayesian network over `n_nodes` nodes. We generate 100 datapoints from it, and use them to
calculate the BGe score. The GFlowNet is learned to generate directed acyclic graphs (DAGs)
proportionally to their BGe score, using the modified DB loss.

Key components:
- BayesianStructure: Environment for Bayesian structure learning
- LinearTransformerPolicyModule: Linear transformer policy module
- ModifiedDBGFlowNet: GFlowNet with modified detailed balance loss
"""

from copy import deepcopy
from typing import cast

import torch

from gfn.actions import Actions
from gfn.gym.bayesian_structure import BayesianStructure
from gfn.states import GraphStates
from gfn.utils.common import set_seed

DEFAULT_SEED = 4444


def get_random_action(env: BayesianStructure, states: GraphStates) -> Actions:
    """Perform a random action in the environment."""
    assert isinstance(states, env.States)

    action_masks = cast(torch.Tensor, states.forward_masks)
    # shape: (batch_size, n_actions) where n_actions = n_nodes^2 + 1

    action_probs = torch.rand(action_masks.shape, device=action_masks.device)
    action_probs = action_probs * action_masks
    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
    action_tensor = torch.multinomial(action_probs, num_samples=1)

    return env.Actions(action_tensor)  # type: ignore


def main(args):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Create the environment
    env = BayesianStructure(
        n_nodes=args.n_nodes,
        state_evaluator=lambda x: torch.zeros(x.batch_size, device=x.device),  # TODO
        device=device,
    )

    states = env.reset(args.batch_size)
    dummy_actions = env.actions_from_batch_shape((args.batch_size,))
    dummy_logprobs = torch.full(
        (args.batch_size,), fill_value=0, dtype=torch.float32, device=device
    )
    trajectories_states: list[GraphStates] = [deepcopy(states)]
    trajectories_actions: list[Actions] = [dummy_actions]  # type: ignore
    trajectories_logprobs: list[torch.Tensor] = [dummy_logprobs]
    trajectories_terminating_idx = torch.zeros(
        args.batch_size, dtype=torch.long, device=device
    )
    trajectories_log_rewards = torch.zeros(
        args.batch_size, dtype=torch.float32, device=device
    )

    dones = states.is_sink_state
    step = 0
    while not dones.all():
        actions: Actions = deepcopy(dummy_actions)

        valid_actions: Actions = get_random_action(env, states[~dones])

        actions[~dones] = valid_actions
        trajectories_actions.append(actions)
        trajectories_logprobs.append(dummy_logprobs)

        next_states = env._step(states, actions)
        step += 1

        new_dones = next_states.is_sink_state & ~dones
        trajectories_terminating_idx[new_dones] = step
        trajectories_log_rewards[new_dones] = torch.ones_like(
            trajectories_log_rewards[new_dones]
        )
        dones = dones | new_dones

        states = next_states
        trajectories_states.append(deepcopy(states))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    main(args)
