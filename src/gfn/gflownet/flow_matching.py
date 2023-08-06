from typing import Tuple

import torch
from torchtyping import TensorType as TT

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.base import GFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.states import DiscreteStates


class FMGFlowNet(GFlowNet):
    r"""Flow Matching GFlowNet, with edge flow estimator.

    $\mathcal{O}_{edge}$ is the set of functions from the non-terminating edges
    to $\mathbb{R}^+$. Which is equivalent to the set of functions from the internal nodes
    (i.e. without $s_f$) to $(\mathbb{R})^{n_actions}$, without the exit action (No need for
    positivity if we parametrize log-flows).

    The loss is described in section
    3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).

    Attributes:
        logF: LogEdgeFlowEstimator
        alpha: weight for the reward matching loss.
    """

    def __init__(self, logF: DiscretePolicyEstimator, alpha: float = 1.0):
        super().__init__()

        self.logF = logF
        self.alpha = alpha

    def sample_trajectories(self, env: Env, n_samples: int = 1000) -> Trajectories:
        if not env.is_discrete:
            raise NotImplementedError(
                "Flow Matching GFlowNet only supports discrete environments for now."
            )
        sampler = Sampler(estimator=self.logF)
        trajectories = sampler.sample_trajectories(env, n_trajectories=n_samples)
        return trajectories

    def flow_matching_loss(
        self, env: Env, states: DiscreteStates
    ) -> TT["n_trajectories", torch.float]:
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

        incoming_log_flows = torch.full_like(
            states.backward_masks, -float("inf"), dtype=torch.float
        )
        outgoing_log_flows = torch.full_like(
            states.forward_masks, -float("inf"), dtype=torch.float
        )

        for action_idx in range(env.n_actions - 1):
            valid_backward_mask = states.backward_masks[:, action_idx]
            valid_forward_mask = states.forward_masks[:, action_idx]
            valid_backward_states = states[valid_backward_mask]
            valid_forward_states = states[valid_forward_mask]

            backward_actions = torch.full_like(
                valid_backward_states.backward_masks[:, 0], action_idx, dtype=torch.long
            ).unsqueeze(-1)
            backward_actions = env.Actions(backward_actions)

            valid_backward_states_parents = env.backward_step(
                valid_backward_states, backward_actions
            )

            incoming_log_flows[valid_backward_mask, action_idx] = self.logF(
                valid_backward_states_parents
            )[:, action_idx]

            outgoing_log_flows[valid_forward_mask, action_idx] = self.logF(
                valid_forward_states
            )[:, action_idx]

        # Now the exit action
        valid_forward_mask = states.forward_masks[:, -1]
        outgoing_log_flows[valid_forward_mask, -1] = self.logF(
            states[valid_forward_mask]
        )[:, -1]

        log_incoming_flows = torch.logsumexp(incoming_log_flows, dim=-1)
        log_outgoing_flows = torch.logsumexp(outgoing_log_flows, dim=-1)

        return (log_incoming_flows - log_outgoing_flows).pow(2).mean()

    def reward_matching_loss(
        self, env: Env, terminating_states: DiscreteStates
    ) -> TT[0, float]:
        """Calculates the reward matching loss from the terminating states."""
        del env  # Unused
        assert terminating_states.log_rewards is not None
        log_edge_flows = self.logF(terminating_states)

        # Handle the boundary condition (for all x, F(X->S_f) = R(x)).
        terminating_log_edge_flows = log_edge_flows[:, -1]
        log_rewards = terminating_states.log_rewards
        return (terminating_log_edge_flows - log_rewards).pow(2).mean()

    def loss(
        self, env: Env, states_tuple: Tuple[DiscreteStates, DiscreteStates]
    ) -> TT[0, float]:
        """Given a batch of non-terminal and terminal states, compute a loss.

        Unlike the GFlowNets Foundations paper, we allow more flexibility by passing a
        tuple of states, the first one being the internal states of the trajectories
        (i.e. non-terminal states), and the second one being the terminal states of the
        trajectories."""
        intermediary_states, terminating_states = states_tuple
        fm_loss = self.flow_matching_loss(env, intermediary_states)
        rm_loss = self.reward_matching_loss(env, terminating_states)
        return fm_loss + self.alpha * rm_loss

    def to_training_samples(
        self, trajectories: Trajectories
    ) -> tuple[DiscreteStates, DiscreteStates]:
        """Converts a batch of trajectories into a batch of training samples."""
        return trajectories.to_non_initial_intermediary_and_terminating_states()
