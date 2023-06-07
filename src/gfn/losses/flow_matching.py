from dataclasses import dataclass
from typing import Tuple

import torch
from torchtyping import TensorType as TT

from gfn.casting import correct_cast
from gfn.distributions import EmpiricalTrajectoryDistribution, TrajectoryDistribution
from gfn.envs import Env
from gfn.estimators import LogEdgeFlowEstimator
from gfn.losses.base import Parametrization, StateDecomposableLoss
from gfn.samplers import ActionsSampler, TrajectoriesSampler
from gfn.states import States


@dataclass
class FMParametrization(Parametrization):
    r"""Flow Matching Parameterization dataclass, with edge flow extimator.

    $\mathcal{O}_{edge}$ is the set of functions from the non-terminating edges
    to $\mathbb{R}^+$. Which is equivalent to the set of functions from the internal nodes
    (i.e. without $s_f$) to $(\mathbb{R})^{n_actions}$, without the exit action (No need for
    positivity if we parametrize log-flows).
    """
    logF: LogEdgeFlowEstimator

    def Pi(
        self, env: Env, n_samples: int = 1000, **actions_sampler_kwargs
    ) -> TrajectoryDistribution:
        actions_sampler = ActionsSampler(self.logF, **actions_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
        trajectories = trajectories_sampler.sample_trajectories(
            n_trajectories=n_samples
        )
        return EmpiricalTrajectoryDistribution(trajectories)


# TODO: Should this loss live within the Parameterization, as a method?
# TODO: Should this be called FlowMatchingLoss?
class FlowMatching(StateDecomposableLoss):
    """Loss object for the Flow Matching objective.

    This method is described in section
    3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).

    Attributes:
        parameterization: a FMParameterization instance.
        env: the environment from within the parameterization.
        alpha: weight for the reward matching loss.
    """
    def __init__(self, parametrization: FMParametrization, alpha=1.0) -> None:
        """Instantiate the FlowMatching Loss object.

        Args:
            parameterization: a FMParameterization instance.
            alpha: weight for the reward matching loss.
        """
        self.parametrization = parametrization
        self.env = parametrization.logF.env
        self.alpha = alpha

    def flow_matching_loss(self, states: States) -> TT["n_trajectories", torch.float]:
        """Computes the FM for the provided states.

        The Flow Matching loss is defined as the log-sum incoming flows minus log-sum
        outgoing flows. The states should not include $s_0$. The batch shape should be
        `(n_states,)`. As of now, only discrete environments are handled.

        Raises:
            AssertionError: If the batch shape is not linear.
            AssertionError: If any state is at $s_0$.
        """

        assert len(states.batch_shape) == 1
        assert not torch.any(states.is_initial_state)

        states.forward_masks, states.backward_masks = correct_cast(
            states.forward_masks, states.backward_masks
        )

        incoming_log_flows = torch.full_like(
            states.backward_masks, -float("inf"), dtype=torch.float
        )
        outgoing_log_flows = torch.full_like(
            states.forward_masks, -float("inf"), dtype=torch.float
        )

        for action_idx in range(self.env.n_actions - 1):
            # TODO: can this be done in a vectorized way? Maybe by "repeating" the
            # states and creating a big actions tensor?
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

            incoming_log_flows[
                valid_backward_mask, action_idx
            ] = self.parametrization.logF(valid_backward_states_parents)[:, action_idx]
            outgoing_log_flows[
                valid_forward_mask, action_idx
            ] = self.parametrization.logF(valid_forward_states)[:, action_idx]

        # Now the exit action
        valid_forward_mask = states.forward_masks[:, -1]
        outgoing_log_flows[valid_forward_mask, -1] = self.parametrization.logF(
            states[valid_forward_mask]
        )[:, -1]

        log_incoming_flows = torch.logsumexp(incoming_log_flows, dim=-1)
        log_outgoing_flows = torch.logsumexp(outgoing_log_flows, dim=-1)

        return (log_incoming_flows - log_outgoing_flows).pow(2).mean()

    def reward_matching_loss(self, terminating_states: States) -> TT[0, float]:
        """Calculates the reward matching loss from the terminating states."""
        assert terminating_states.log_rewards is not None
        log_edge_flows = self.parametrization.logF(terminating_states)
        terminating_log_edge_flows = log_edge_flows[:, -1]
        log_rewards = terminating_states.log_rewards
        return (terminating_log_edge_flows - log_rewards).pow(2).mean()

    # TODO: should intermeidary_states and terminating_states be two input args?
    def __call__(self, states_tuple: Tuple[States, States]) -> TT[0, float]:
        """Calculates flow matching loss from intermediate and terminating states."""
        intermediary_states, terminating_states = states_tuple
        fm_loss = self.flow_matching_loss(intermediary_states)
        rm_loss = self.reward_matching_loss(terminating_states)
        return fm_loss + self.alpha * rm_loss
