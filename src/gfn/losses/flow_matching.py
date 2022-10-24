from dataclasses import dataclass
from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.containers.states import States, correct_cast
from gfn.distributions import EmpiricalTrajectoryDistribution, TrajectoryDistribution
from gfn.envs import Env
from gfn.estimators import LogEdgeFlowEstimator
from gfn.losses.base import Parametrization, StateDecomposableLoss
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler

# Typing
ScoresTensor = TensorType["n_states", float]
LossTensor = TensorType[0, float]


@dataclass
class FMParametrization(Parametrization):
    r"""
    $\mathcal{O}_{edge}$ is the set of functions from the non-terminating edges
    to $\mathbb{R}^+$. Which is equivalent to the set of functions from the internal nodes
    (i.e. without $s_f$) to $(\mathbb{R})^{n_actions}$, without the exit action (No need for
    positivity if we parametrize log-flows).
    """
    logF: LogEdgeFlowEstimator

    def Pi(
        self, env: Env, n_samples: int = 1000, **actions_sampler_kwargs
    ) -> TrajectoryDistribution:
        actions_sampler = DiscreteActionsSampler(self.logF, **actions_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
        trajectories = trajectories_sampler.sample_trajectories(
            n_trajectories=n_samples
        )
        return EmpiricalTrajectoryDistribution(trajectories)


class FlowMatching(StateDecomposableLoss):
    def __init__(self, parametrization: FMParametrization) -> None:
        self.parametrization = parametrization
        self.env = parametrization.logF.env

    def get_scores(self, states_tuple: Tuple[States, States]) -> ScoresTensor:
        """
        Compute the scores for the given states, defined as the log-sum incoming flows minus log-sum outgoing flows.
        The first element of the tuple is the set of intermediary states in a trajectory that are not s0.
        The second element of the tuple is the set of terminal states in each trajectory that is not s0 -> sf.
        The reward function is queried only for the second element of the tuple (even if there are terminal states in the first element).
        The batch_shape of each states object in the tuple should be (n_states,).

        As of now, only discrete environments are handled.
        """
        intermediary_states, terminating_states = states_tuple
        terminating_states_mask = torch.zeros(
            intermediary_states.batch_shape,
            dtype=torch.bool,
            device=intermediary_states.states_tensor.device,
        )
        terminating_states_mask = torch.cat(
            (
                terminating_states_mask,
                torch.ones(
                    terminating_states.batch_shape,
                    dtype=torch.bool,
                    device=intermediary_states.states_tensor.device,
                ),
            )
        )
        intermediary_states.extend(terminating_states)

        states = intermediary_states
        assert len(states.batch_shape) == 1

        states.forward_masks, states.backward_masks = correct_cast(
            states.forward_masks, states.backward_masks
        )

        terminating_states_rewards = self.env.reward(terminating_states)

        incoming_flows = torch.full_like(
            states.backward_masks, -float("inf"), dtype=torch.float
        )
        outgoing_flows = torch.full_like(
            states.forward_masks, -float("inf"), dtype=torch.float
        )
        outgoing_flows[terminating_states_mask, -1] = torch.log(
            terminating_states_rewards
        )
        for action_idx in range(self.env.n_actions - 1):
            valid_backward_mask = states.backward_masks[:, action_idx]
            valid_forward_mask = states.forward_masks[:, action_idx]
            valid_backward_states = states[valid_backward_mask]
            valid_forward_states = states[valid_forward_mask]
            _, valid_backward_states.backward_masks = correct_cast(
                valid_backward_states.forward_masks,
                valid_backward_states.backward_masks,
            )
            backward_actions = torch.full_like(
                valid_backward_states.backward_masks[:, 0], action_idx, dtype=torch.long
            )

            valid_backward_states_parents = self.env.backward_step(
                valid_backward_states, backward_actions
            )

            incoming_flows[valid_backward_mask, action_idx] = self.parametrization.logF(
                valid_backward_states_parents
            )[:, action_idx]
            outgoing_flows[valid_forward_mask, action_idx] = self.parametrization.logF(
                valid_forward_states
            )[:, action_idx]
        incoming_flows = torch.logsumexp(incoming_flows, dim=1)
        outgoing_flows = torch.logsumexp(outgoing_flows, dim=1)

        return incoming_flows - outgoing_flows

    def __call__(self, states_tuple: Tuple[States, States]) -> LossTensor:
        scores = self.get_scores(states_tuple)
        return scores.pow(2).mean()
