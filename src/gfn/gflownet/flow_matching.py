from typing import Any, Tuple, Union

import torch

from gfn.containers import Trajectories
from gfn.env import DiscreteEnv
from gfn.gflownet.base import GFlowNet
from gfn.modules import ConditionalDiscretePolicyEstimator, DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)


class FMGFlowNet(GFlowNet[Tuple[DiscreteStates, DiscreteStates]]):
    r"""Flow Matching GFlowNet, with edge flow estimator.

    $\mathcal{O}_{edge}$ is the set of functions from the non-terminating edges
    to $\mathbb{R}^+$. Which is equivalent to the set of functions from the internal nodes
    (i.e. without $s_f$) to $(\mathbb{R})^{n_actions}$, without the exit action (No need for
    positivity if we parametrize log-flows).

    The loss is described in section
    3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).

    Attributes:
        logF: an estimator of log edge flows.
        alpha: weight for the reward matching loss.
    """

    def __init__(self, logF: DiscretePolicyEstimator, alpha: float = 1.0):
        super().__init__()

        assert isinstance(  # TODO: need a more flexible type check.
            logF,
            DiscretePolicyEstimator | ConditionalDiscretePolicyEstimator,
        ), "logF must be a DiscretePolicyEstimator or ConditionalDiscretePolicyEstimator"
        self.logF = logF
        self.alpha = alpha

    def sample_trajectories(
        self,
        env: DiscreteEnv,
        n: int,
        conditioning: torch.Tensor | None = None,
        save_logprobs: bool = False,
        save_estimator_outputs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Sample trajectory with optional kwargs controling the policy."""
        if not env.is_discrete:
            raise NotImplementedError(
                "Flow Matching GFlowNet only supports discrete environments for now."
            )
        sampler = Sampler(estimator=self.logF)
        trajectories = sampler.sample_trajectories(
            env,
            n=n,
            conditioning=conditioning,
            save_logprobs=save_logprobs,
            save_estimator_outputs=save_estimator_outputs,
            **policy_kwargs,
        )
        return trajectories

    def flow_matching_loss(
        self,
        env: DiscreteEnv,
        states: DiscreteStates,
        conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
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

        # TODO: Need to vectorize this loop.
        for action_idx in range(env.n_actions - 1):
            valid_backward_mask = states.backward_masks[:, action_idx]
            valid_forward_mask = states.forward_masks[:, action_idx]
            valid_backward_states = states[valid_backward_mask]
            valid_forward_states = states[valid_forward_mask]

            backward_actions = torch.full_like(
                valid_backward_states.backward_masks[:, 0], action_idx, dtype=torch.long
            ).unsqueeze(-1)
            backward_actions = env.actions_from_tensor(backward_actions)

            valid_backward_states_parents = env._backward_step(
                valid_backward_states, backward_actions
            )

            if conditioning is not None:
                # Mask out only valid conditioning elements.
                valid_backward_conditioning = conditioning[valid_backward_mask]
                valid_forward_conditioning = conditioning[valid_forward_mask]

                with has_conditioning_exception_handler("logF", self.logF):
                    incoming_log_flows[valid_backward_mask, action_idx] = self.logF(
                        valid_backward_states_parents,
                        valid_backward_conditioning,
                    )[:, action_idx]

                    outgoing_log_flows[valid_forward_mask, action_idx] = self.logF(
                        valid_forward_states,
                        valid_forward_conditioning,
                    )[:, action_idx]

            else:
                with no_conditioning_exception_handler("logF", self.logF):
                    incoming_log_flows[valid_backward_mask, action_idx] = self.logF(
                        valid_backward_states_parents,
                    )[:, action_idx]

                    outgoing_log_flows[valid_forward_mask, action_idx] = self.logF(
                        valid_forward_states,
                    )[:, action_idx]

        # Now the exit action.
        valid_forward_mask = states.forward_masks[:, -1]
        if conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                outgoing_log_flows[valid_forward_mask, -1] = self.logF(
                    states[valid_forward_mask],
                    conditioning[valid_forward_mask],
                )[:, -1]
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                outgoing_log_flows[valid_forward_mask, -1] = self.logF(
                    states[valid_forward_mask],
                )[:, -1]

        log_incoming_flows = torch.logsumexp(incoming_log_flows, dim=-1)
        log_outgoing_flows = torch.logsumexp(outgoing_log_flows, dim=-1)

        return (log_incoming_flows - log_outgoing_flows).pow(2).mean()

    def reward_matching_loss(
        self,
        env: DiscreteEnv,
        terminating_states: DiscreteStates,
        conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        """Calculates the reward matching loss from the terminating states."""
        del env  # Unused
        assert terminating_states.log_rewards is not None

        if conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                log_edge_flows = self.logF(terminating_states, conditioning)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                log_edge_flows = self.logF(terminating_states)

        # Handle the boundary condition (for all x, F(X->S_f) = R(x)).
        terminating_log_edge_flows = log_edge_flows[:, -1]
        log_rewards = terminating_states.log_rewards
        return (terminating_log_edge_flows - log_rewards).pow(2).mean()

    def loss(
        self,
        env: DiscreteEnv,
        states_tuple: Union[
            Tuple[DiscreteStates, DiscreteStates, torch.Tensor, torch.Tensor],
            Tuple[DiscreteStates, DiscreteStates, None, None],
        ],
    ) -> torch.Tensor:
        """Given a batch of non-terminal and terminal states, compute a loss.

        Unlike the GFlowNets Foundations paper, we allow more flexibility by passing a
        tuple of states, the first one being the internal states of the trajectories
        (i.e. non-terminal states), and the second one being the terminal states of the
        trajectories."""
        (
            intermediary_states,
            terminating_states,
            intermediary_conditioning,
            terminating_conditioning,
        ) = states_tuple
        fm_loss = self.flow_matching_loss(
            env, intermediary_states, intermediary_conditioning
        )
        rm_loss = self.reward_matching_loss(
            env, terminating_states, terminating_conditioning
        )
        return fm_loss + self.alpha * rm_loss

    def to_training_samples(
        self, trajectories: Trajectories
    ) -> Tuple[States, States, torch.Tensor | None, torch.Tensor | None]:
        """Converts a batch of trajectories into a batch of training samples."""
        return trajectories.to_non_initial_intermediary_and_terminating_states()
