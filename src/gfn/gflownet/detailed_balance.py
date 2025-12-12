import math
from typing import Tuple

import torch

from gfn.actions import Actions
from gfn.containers import Trajectories, Transitions
from gfn.env import Env
from gfn.estimators import ConditionalScalarEstimator, Estimator, ScalarEstimator
from gfn.gflownet.base import PFBasedGFlowNet, loss_reduce
from gfn.states import States
from gfn.utils.handlers import (
    has_conditions_exception_handler,
    no_conditions_exception_handler,
    warn_about_recalculating_logprobs,
)
from gfn.utils.prob_calculations import get_transition_pfs_and_pbs


def check_compatibility(
    states: States, actions: Actions, transitions: Transitions
) -> None:
    """Checks compatibility between states and actions in transitions.

    Args:
        states: The states in the transitions.
        actions: The actions in the transitions.
        transitions: The transitions object.

    Raises:
        TypeError: If transitions is not of type Transitions.
        ValueError: If there is a mismatch between states and actions batch shapes.
    """
    if states.batch_shape != tuple(actions.batch_shape):
        if type(transitions) is not Transitions:
            raise TypeError(
                "`transitions` is type={}, not Transitions".format(type(transitions))
            )
        else:
            raise ValueError(" wrong happening with log_pf evaluations")


class DBGFlowNet(PFBasedGFlowNet[Transitions]):
    r"""GFlowNet for the Detailed Balance loss.

    Corresponds to $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2 \times
    \mathcal{O}_3$, where $\mathcal{O}_1$ is the set of functions from the internal
    states (no $s_f$) to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the
    non-negativity constraint), and $\mathcal{O}_2$ is the set of forward probability
    functions consistent with the DAG. $\mathcal{O}_3$ is the set of backward
    probability functions consistent with the DAG, or a singleton thereof, if
    `self.pb` is a fixed `DiscretePBEstimator`.

    The detailed balance loss is described in section
    3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator.
        logF: A ScalarEstimator or ConditionalScalarEstimator for estimating the log
            flow of the states.
        forward_looking: Whether to use the forward-looking GFN loss. When True,
            rewards must be defined over edges; this implementation treats the edge
            reward as the difference between the successor and current state rewards,
            so only valid if the environment follows that assumption.
        constant_pb: Whether to ignore the backward policy estimator, e.g., if the
            gflownet DAG is a tree, and pb is therefore always 1.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        logF: ScalarEstimator | ConditionalScalarEstimator,
        forward_looking: bool = False,
        constant_pb: bool = False,
        log_reward_clip_min: float = -float("inf"),
        debug: bool = False,
    ) -> None:
        """Initializes a DBGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
                pb is therefore always 1.
            logF: A ScalarEstimator or ConditionalScalarEstimator for estimating the log
                flow of the states.
            forward_looking: Whether to use the forward-looking GFN loss. When True,
                rewards should be defined over edges; this implementation treats the
                edge reward as the difference between the successor and current state
                rewards, so only valid if the environment follows that assumption.
            constant_pb: Whether to ignore the backward policy estimator, e.g., if the
                gflownet DAG is a tree, and pb is therefore always 1. Must be set
                explicitly by user to ensure that pb is an Estimator except under this
                special case.
            log_reward_clip_min: If finite, clips log rewards to this value.
            debug: If True, keep runtime safety checks active; disable in compiled runs.

        """
        super().__init__(
            pf,
            pb,
            constant_pb=constant_pb,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
        )

        # Disallow recurrent PF for transition-based DB
        from gfn.estimators import RecurrentDiscretePolicyEstimator

        if isinstance(self.pf, RecurrentDiscretePolicyEstimator):
            raise TypeError(
                "DBGFlowNet does not support recurrent PF estimators (transitions path cannot propagate carry)."
            )

        assert any(
            isinstance(logF, cls)
            for cls in [ScalarEstimator, ConditionalScalarEstimator]
        ), "logF must be a ScalarEstimator or derived"

        self.logF = logF
        self.forward_looking = forward_looking

    def logF_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns a dictionary of named parameters containing 'logF' in their name.

        Returns:
            A dictionary of named parameters containing 'logF' in their name.
        """
        return {k: v for k, v in self.named_parameters() if "logF" in k}

    def logF_parameters(self) -> list[torch.Tensor]:
        """Returns a list of parameters containing 'logF' in their name.

        Returns:
            A list of parameters containing 'logF' in their name.
        """
        return [v for k, v in self.named_parameters() if "logF" in k]

    def get_pfs_and_pbs(
        self, transitions: Transitions, recalculate_all_logprobs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluates forward and backward logprobs for each transition in the batch.

        More specifically, it evaluates $\log P_F(s' \mid s)$ and $\log P_B(s \mid s')$
        for each transition in the batch.

        If recalculate_all_logprobs=True, we re-evaluate the logprobs of the transitions
        using the current self.pf. Otherwise, the following applies:
            - If transitions have log_probs attribute, use them - this is usually for
                on-policy learning.
            - Else (transitions have none of them), re-evaluate the logprobs using
                the current self.pf - this is usually for off-policy learning with
                replay buffer.

        Args:
            transitions: The Transitions object to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns:
            A tuple of tensors of shape (n_transitions,) containing the log_pf and
            log_pb for each transition.
        """
        return get_transition_pfs_and_pbs(
            self.pf,
            self.pb,
            transitions,
            recalculate_all_logprobs,
        )

    def get_scores(
        self,
        env: Env,
        transitions: Transitions,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        r"""Calculates the scores for a batch of transitions.

        The scores for each transition are defined as:
        $\log \left( \frac{F(s)P_F(s' \mid s)}{F(s') P_B(s \mid s')} \right)$.

        Args:
            env: The environment where the transitions are sampled from.
            transitions: The Transitions object to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns:
            A tensor of shape (n_transitions,) representing the scores for each
            transition.
        """
        # Guard bad inputs under debug to avoid graph breaks in torch.compile.
        if self.debug and transitions.is_backward:
            raise ValueError("Backward transitions are not supported")

        states = transitions.states
        actions = transitions.actions

        if len(states) == 0:
            return torch.tensor(0.0, device=transitions.device)

        if self.debug:
            check_compatibility(states, actions, transitions)
            assert (
                not transitions.states.is_sink_state.any()
            ), "Transition from sink state is not allowed. This is a bug."

        ### Compute log_pf and log_pb
        log_pf, log_pb = self.get_pfs_and_pbs(transitions, recalculate_all_logprobs)

        ### Compute log_F_s
        # LogF is potentially a conditional computation.
        if transitions.conditions is not None:
            with has_conditions_exception_handler("logF", self.logF):
                log_F_s = self.logF(states, transitions.conditions).squeeze(-1)
        else:
            with no_conditions_exception_handler("logF", self.logF):
                log_F_s = self.logF(states).squeeze(-1)

        ### Compute log_F_s_next
        log_F_s_next = torch.zeros_like(log_F_s)
        is_terminating = transitions.is_terminating
        is_intermediate = ~is_terminating

        # Assign log_F_s_next for intermediate next states
        interm_next_states = transitions.next_states[is_intermediate]
        # log_F is potentially a conditional computation.
        if transitions.conditions is not None:
            with has_conditions_exception_handler("logF", self.logF):
                log_F_s_next[is_intermediate] = self.logF(
                    interm_next_states,
                    transitions.conditions[is_intermediate],
                ).squeeze(-1)
        else:
            with no_conditions_exception_handler("logF", self.logF):
                log_F_s_next[is_intermediate] = self.logF(interm_next_states).squeeze(-1)

        # Apply forward-looking if applicable
        if self.forward_looking:
            # Keep explanatory warning only in debug to avoid compile-time graph breaks.
            if self.debug:
                import warnings

                warnings.warn(
                    "Rewards should be defined over edges in forward-looking settings. "
                    "The current implementation is a special case of this, where the edge "
                    "reward is defined as the difference between the reward of two states "
                    "that the edge connects. If your environment is not the case, "
                    "forward-looking may be inappropriate."
                )

            # Reward calculation can also be conditional.
            if transitions.conditions is not None:
                log_rewards_state = env.log_reward(states, transitions.conditions)  # type: ignore
                log_rewards_next = env.log_reward(
                    interm_next_states, transitions.conditions[is_intermediate]  # type: ignore
                )
            else:
                log_rewards_state = env.log_reward(states)
                log_rewards_next = env.log_reward(interm_next_states)
            if math.isfinite(self.log_reward_clip_min):
                log_rewards_state = log_rewards_state.clamp_min(self.log_reward_clip_min)
                log_rewards_next = log_rewards_next.clamp_min(self.log_reward_clip_min)

            log_F_s = log_F_s + log_rewards_state
            log_F_s_next[is_intermediate] = (
                log_F_s_next[is_intermediate] + log_rewards_next
            )

        # Assign log_F_s_next for terminating transitions as log_rewards
        log_rewards = transitions.log_rewards
        assert log_rewards is not None
        if math.isfinite(self.log_reward_clip_min):
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)
        log_F_s_next[is_terminating] = log_rewards[is_terminating]

        ### Compute scores
        preds = log_pf + log_F_s
        targets = log_pb + log_F_s_next
        scores = preds - targets
        assert scores.shape == (transitions.n_transitions,)
        return scores

    def loss(
        self,
        env: Env,
        transitions: Transitions,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the detailed balance loss.

        The detailed balance loss is described in section
        3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).

        Args:
            env: The environment where the transitions are sampled from.
            transitions: The Transitions object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').
                Run with self.debug=False for improved performance.

        Returns:
            The computed detailed balance loss as a tensor. The shape depends on the
            reduction method.
        """
        if self.debug:
            warn_about_recalculating_logprobs(transitions, recalculate_all_logprobs)
        scores = self.get_scores(
            env,
            transitions,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )
        scores = scores**2
        loss = loss_reduce(scores, reduction)

        if self.debug and torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        """Converts trajectories to transitions for detailed balance loss.

        Args:
            trajectories: The Trajectories object to convert.

        Returns:
            A Transitions object containing all transitions from the trajectories.
        """
        return trajectories.to_transitions()


class ModifiedDBGFlowNet(PFBasedGFlowNet[Transitions]):
    r"""The Modified Detailed Balance GFlowNet.

    Only applicable to environments where all states are terminating. See section 3.2
    of [Bayesian Structure Learning with Generative Flow Networks](https://arxiv.org/abs/2202.13903)
    for more details.

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
            pb is therefore always 1.
        constant_pb: Whether to ignore the backward policy estimator, e.g., if the
            gflownet DAG is a tree, and pb is therefore always 1. Must be set explicitly
            by user to ensure that pb is an Estimator except under this special case.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        constant_pb: bool = False,
        debug: bool = False,
    ) -> None:
        """Initializes a ModifiedDBGFlowNet instance.

        Args:
            pf: Forward policy estimator.
            pb: Backward policy estimator or None.
            constant_pb: See base class.
            debug: If True, keep runtime safety checks active; disable in compiled runs.

        """
        super().__init__(pf, pb, constant_pb=constant_pb, debug=debug)

    def get_scores(
        self,
        transitions: Transitions,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        """Calculates DAG-GFN-style modified detailed balance scores.

        Note that this method is only applicable to environments where all states are
        terminating, i.e., the sink state is reachable from all states.

        If recalculate_all_logprobs=True, we re-evaluate the logprobs of the transitions
        using the current self.pf. Otherwise, the following applies:
            - If transitions have log_probs attribute, use them - this is usually for
                on-policy learning.
            - Else, re-evaluate the log_probs using the current self.pf - this is usually
                for off-policy learning with replay buffer.

        Args:
            transitions: The Transitions object to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns:
            A tensor of shape (n_transitions,) containing the scores for each transition.
        """
        if self.debug and transitions.is_backward:
            raise ValueError("Backward transitions are not supported")

        if len(transitions) == 0:
            return torch.tensor(0.0, device=transitions.device)

        mask = ~transitions.next_states.is_sink_state
        states = transitions.states[mask]
        valid_next_states = transitions.next_states[mask]
        actions = transitions.actions[mask]
        all_log_rewards = transitions.all_log_rewards[mask]

        if self.debug:
            check_compatibility(states, actions, transitions)

        if transitions.conditions is not None:
            with has_conditions_exception_handler("pf", self.pf):
                module_output = self.pf(states, transitions.conditions[mask])
        else:
            with no_conditions_exception_handler("pf", self.pf):
                module_output = self.pf(states)

        if len(states) == 0:
            return torch.tensor(0.0, device=transitions.device)

        pf_dist = self.pf.to_probability_distribution(states, module_output)

        if transitions.has_log_probs and not recalculate_all_logprobs:
            valid_log_pf_actions = transitions[mask].log_probs
            assert valid_log_pf_actions is not None
        else:
            # Evaluate the log PF of the actions sampled off policy.
            valid_log_pf_actions = pf_dist.log_prob(actions.tensor)
        # Avoid .item() in hot path to stay compile-friendly; broadcast exit_action tensor.
        exit_action_tensor = actions.__class__.exit_action.to(
            actions.tensor.device, dtype=actions.tensor.dtype
        ).expand_as(actions.tensor)
        valid_log_pf_s_exit = pf_dist.log_prob(exit_action_tensor)

        # The following two lines are slightly inefficient, given that most
        # next_states are also states, for which we already did a forward pass.
        if transitions.conditions is not None:
            with has_conditions_exception_handler("pf", self.pf):
                module_output = self.pf(valid_next_states, transitions.conditions[mask])
        else:
            with no_conditions_exception_handler("pf", self.pf):
                module_output = self.pf(valid_next_states)

        valid_log_pf_s_prime_exit = self.pf.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(exit_action_tensor[: len(valid_next_states)])

        non_exit_actions = actions[~actions.is_exit]

        if self.pb is not None:
            if transitions.conditions is not None:
                with has_conditions_exception_handler("pb", self.pb):
                    module_output = self.pb(
                        valid_next_states, transitions.conditions[mask]
                    )
            else:
                with no_conditions_exception_handler("pb", self.pb):
                    module_output = self.pb(valid_next_states)

            valid_log_pb_actions = self.pb.to_probability_distribution(
                valid_next_states, module_output
            ).log_prob(non_exit_actions.tensor)
        else:
            # If pb is None, we assume that the gflownet DAG is a tree, and therefore
            # the backward policy probability is always 1 (log probs are 0).
            valid_log_pb_actions = torch.zeros_like(valid_log_pf_s_exit)

        preds = all_log_rewards[:, 0] + valid_log_pf_actions + valid_log_pf_s_prime_exit
        targets = all_log_rewards[:, 1] + valid_log_pb_actions + valid_log_pf_s_exit

        scores = preds - targets
        if self.debug and torch.any(torch.isinf(scores)):
            raise ValueError("scores contains inf")

        return scores

    def loss(
        self,
        env: Env,
        transitions: Transitions,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the modified detailed balance loss.

        Args:
            env: The environment where the transitions are sampled from (unused).
            transitions: The Transitions object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed modified detailed balance loss as a tensor. The shape depends
            on the reduction method.
        """
        del env
        warn_about_recalculating_logprobs(transitions, recalculate_all_logprobs)
        scores = self.get_scores(
            transitions,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )
        scores = scores**2
        return loss_reduce(scores, reduction)

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        """Converts trajectories to transitions for modified detailed balance loss.

        Args:
            trajectories: The Trajectories object to convert.

        Returns:
            A Transitions object containing all transitions from the trajectories.
        """
        return trajectories.to_transitions()
