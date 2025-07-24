import warnings
from typing import Any

import torch

from gfn.containers import StatesContainer, Trajectories
from gfn.env import DiscreteEnv
from gfn.gflownet.base import GFlowNet, loss_reduce
from gfn.modules import ConditionalDiscretePolicyEstimator, DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.states import DiscreteStates
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)

warnings.filterwarnings("once", message="recalculate_all_logprobs is not used for FM.*")


class FMGFlowNet(GFlowNet[StatesContainer[DiscreteStates]]):
    r"""GFlowNet for the Flow Matching loss with an edge flow estimator.

    $\mathcal{O}_{edge}$ is the set of functions from the non-terminating edges
    to $\mathbb{R}^+$. Which is equivalent to the set of functions from the internal nodes
    (i.e. without $s_f$) to $(\mathbb{R})^{n_actions}$, without the exit action (No need
    for positivity if we parametrize log-flows).

    The flow matching loss is described in section
    3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).

    Attributes:
        logF: A DiscretePolicyEstimator or ConditionalDiscretePolicyEstimator for
            estimating the log flow of the edges (states -> next_states).
        alpha: A scalar weight for the reward matching loss.
    """

    def __init__(self, logF: DiscretePolicyEstimator, alpha: float = 1.0):
        """Initializes a FMGFlowNet instance.

        Args:
            logF: A DiscretePolicyEstimator or ConditionalDiscretePolicyEstimator for
                estimating the log flow of the edges (states -> next_states).
            alpha: A scalar weight for the reward matching loss.
        """
        super().__init__()

        assert isinstance(
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
        """Samples trajectories using the edge flow estimator.

        Args:
            env: The discrete environment to sample trajectories from.
            n: Number of trajectories to sample.
            conditioning: Optional conditioning tensor for conditional environments.
            save_logprobs: Whether to save the log-probabilities of the actions.
            save_estimator_outputs: Whether to save the estimator outputs.
            **policy_kwargs: Additional keyword arguments for the sampler.

        Returns:
            A Trajectories object containing the sampled trajectories.
        """
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
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the flow matching loss for the (non-initial) states.

        The Flow Matching loss is defined as the log-sum incoming flows minus log-sum
        outgoing flows. The states should not include $s_0$. The batch shape should be
        `(n_states,)`. As of now, only discrete environments are handled.

        Args:
            env: The discrete environment where the states are sampled from.
            states: The DiscreteStates object to evaluate (should not include $s_0$).
            conditioning: Optional conditioning tensor for conditional environments.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed flow matching loss as a tensor. The shape depends on the
            reduction method.
        """
        if len(states) == 0:
            return torch.tensor(0.0, device=states.device)

        assert len(states.batch_shape) == 1
        assert not torch.any(states.is_initial_state)

        incoming_log_flows = torch.full_like(
            states.backward_masks, -float("inf"), dtype=torch.get_default_dtype()
        )
        outgoing_log_flows = torch.full_like(
            states.forward_masks, -float("inf"), dtype=torch.get_default_dtype()
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
        scores = (log_incoming_flows - log_outgoing_flows).pow(2)

        return loss_reduce(scores, reduction)

    def reward_matching_loss(
        self,
        env: DiscreteEnv,
        terminating_states: DiscreteStates,
        conditioning: torch.Tensor | None,
        log_rewards: torch.Tensor | None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the reward matching loss for the terminating states.

        Args:
            env: The discrete environment where the states are sampled from (unused).
            terminating_states: The DiscreteStates object containing terminating states.
            conditioning: Optional conditioning tensor for conditional environments.
            log_rewards: The log rewards for the terminating states.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed reward matching loss as a tensor. The shape depends on the
            reduction method.
        """
        if len(terminating_states) == 0:
            return torch.tensor(0.0, device=terminating_states.device)
        del env  # Unused
        if conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                log_edge_flows = self.logF(terminating_states, conditioning)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                log_edge_flows = self.logF(terminating_states)

        # Handle the boundary condition (for all x, F(X->S_f) = R(x)).
        terminating_log_edge_flows = log_edge_flows[:, -1]
        scores = (terminating_log_edge_flows - log_rewards).pow(2)

        return loss_reduce(scores, reduction)

    def loss(
        self,
        env: DiscreteEnv,
        states_container: StatesContainer[DiscreteStates],
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the flow matching loss for a batch of states.

        The flow matching loss is described in section
        3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).
        Unlike the original implementation, we allow more flexibility by treating the
        intermediary and terminating states separately.

        Args:
            env: The discrete environment where the states are sampled from.
            states_container: The StatesContainer object containing both intermediary and
                terminating states.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs (unused for FM).
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed flow matching loss as a tensor. The shape depends on the
            reduction method.
        """
        assert isinstance(states_container.intermediary_states, DiscreteStates)
        assert isinstance(states_container.terminating_states, DiscreteStates)
        if recalculate_all_logprobs:
            warnings.warn(
                "recalculate_all_logprobs is not used for FM. Ignoring the argument."
            )
        del recalculate_all_logprobs  # Unused for FM.
        fm_loss = self.flow_matching_loss(
            env,
            states_container.intermediary_states,
            states_container.intermediary_conditioning,
            reduction=reduction,
        )
        rm_loss = self.reward_matching_loss(
            env,
            states_container.terminating_states,
            states_container.terminating_conditioning,
            states_container.terminating_log_rewards,
            reduction=reduction,
        )
        return fm_loss + self.alpha * rm_loss

    def to_training_samples(
        self, trajectories: Trajectories
    ) -> StatesContainer[DiscreteStates]:
        """Converts trajectories to a StatesContainer for flow matching loss.

        Args:
            trajectories: The Trajectories object to convert.

        Returns:
            A StatesContainer object containing all states from the trajectories.
        """
        return trajectories.to_states_container()
