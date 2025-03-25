"""
The goal of this script is to reproduce the results of DAG-GFlowNet for
Bayesian structure learning (Deleu et al., 2022) using the GraphEnv.

Specifically, we consider a randomly generated (under the Erdős-Rényi model) linear-Gaussian
Bayesian network over `num_nodes` nodes. We generate 100 datapoints from it, and use them to
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
from gfn.gym.helpers.bayesian_structure.factories import get_scorer
from gfn.states import GraphStates
from gfn.utils.common import set_seed

DEFAULT_SEED = 4444


def get_random_action(env: BayesianStructure, states: GraphStates) -> Actions:
    """Perform a random action in the environment."""
    assert isinstance(states, env.States)

    action_masks = cast(torch.Tensor, states.forward_masks)
    # shape: (batch_size, n_actions) where n_actions = num_nodes^2 + 1

    action_probs = torch.rand(action_masks.shape, device=action_masks.device)
    action_probs = action_probs * action_masks
    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
    action_tensor = torch.multinomial(action_probs, num_samples=1)

    return env.Actions(action_tensor)  # type: ignore


def main(args):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    rng = torch.Generator(device="cpu")  # The device should be cpu
    rng.manual_seed(seed)

    # Create the scorer
    scorer, _, _ = get_scorer(
        args.graph_name,
        args.prior_name,
        args.num_nodes,
        args.num_edges,
        args.num_samples,
        args.node_names,
        rng=rng,
    )

    # Create the environment
    env = BayesianStructure(
        num_nodes=args.num_nodes,
        state_evaluator=scorer.state_evaluator,
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
        # Note: next_states[new_dones] are the dummy states, so we should pass
        # states[new_dones], which are the terminating states to get the log rewards.
        trajectories_log_rewards[new_dones] = env.log_reward(states[new_dones])
        dones = dones | new_dones

        states = next_states
        trajectories_states.append(deepcopy(states))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Environment parameters
    parser.add_argument("--num_nodes", type=int, default=5)
    parser.add_argument(
        "--num_edges",
        type=int,
        default=5,
        help="Number of edges in the sampled erdos renyi graph",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples."
    )
    parser.add_argument("--graph_name", type=str, default="erdos_renyi_lingauss")
    parser.add_argument("--prior_name", type=str, default="uniform")
    parser.add_argument(
        "--node_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of node names.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    main(args)
