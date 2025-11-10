import math
from typing import Tuple

import torch

from gfn.actions import Actions
from gfn.containers import Trajectories, Transitions
from gfn.env import ConditionalEnv, Env
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
        forward_looking: Whether to use the forward-looking GFN loss.
        log_reward_clip_min: If finite, clips log rewards to this value.
        safe_log_prob_min: If True, uses -1e10 as the minimum log probability value
            to avoid numerical instability, otherwise uses -1e38.
        constant_pb: Whether to ignore the backward policy estimator, e.g., if the
            gflownet DAG is a tree, and pb is therefore always 1.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        logF: ScalarEstimator | ConditionalScalarEstimator,
        forward_looking: bool = False,
        log_reward_clip_min: float = -float("inf"),
        safe_log_prob_min: bool = True,
        constant_pb: bool = False,
    ) -> None:
        """Initializes a DBGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
                pb is therefore always 1.
            logF: A ScalarEstimator or ConditionalScalarEstimator for estimating the log
                flow of the states.
            forward_looking: Whether to use the forward-looking GFN loss.
            log_reward_clip_min: If finite, clips log rewards to this value.
            safe_log_prob_min: If True, uses -1e10 as the minimum log probability value
                to avoid numerical instability, otherwise uses -1e38.
            constant_pb: Whether to ignore the backward policy estimator, e.g., if the
                gflownet DAG is a tree, and pb is therefore always 1. Must be set
                explicitly by user to ensure that pb is an Estimator except under this
                special case.

        """
        super().__init__(pf, pb, constant_pb=constant_pb)

        # Disallow recurrent PF for transition-based DB
        from gfn.estimators import RecurrentDiscretePolicyEstimator  # type: ignore

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
        self.log_reward_clip_min = log_reward_clip_min
        if safe_log_prob_min:
            self.log_prob_min = -1e10
        else:
            self.log_prob_min = -1e38

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
        self, env: Env, transitions: Transitions, recalculate_all_logprobs: bool = True
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
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")

        states = transitions.states
        actions = transitions.actions

        if len(states) == 0:
            return torch.tensor(0.0, device=transitions.device)

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)
        check_compatibility(states, actions, transitions)

        log_pf_actions, log_pb_actions = self.get_pfs_and_pbs(
            transitions, recalculate_all_logprobs
        )

        # LogF is potentially a conditional computation.
        if transitions.conditions is not None:
            with has_conditions_exception_handler("logF", self.logF):
                log_F_s = self.logF(states, transitions.conditions).squeeze(-1)
        else:
            with no_conditions_exception_handler("logF", self.logF):
                log_F_s = self.logF(states).squeeze(-1)

        if self.forward_looking:
            if isinstance(env, ConditionalEnv):
                assert transitions.conditions is not None
                log_rewards = env.log_reward(states, transitions.conditions).squeeze(-1)
            else:
                log_rewards = env.log_reward(states).squeeze(-1)
            if math.isfinite(self.log_reward_clip_min):
                log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)
            log_F_s = log_F_s + log_rewards

        preds = log_pf_actions + log_F_s

        # uncomment next line for debugging
        # assert transitions.next_states.is_sink_state.equal(transitions.is_terminating)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_terminating]
        valid_transitions_is_terminating = transitions.is_terminating[
            ~transitions.states.is_sink_state
        ]

        if len(valid_next_states) == 0:
            return torch.tensor(0.0, device=transitions.device)

        # LogF is potentially a conditional computation.
        if transitions.conditions is not None:
            with has_conditions_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(
                    valid_next_states,
                    transitions.conditions[~transitions.is_terminating],
                ).squeeze(-1)
        else:
            with no_conditions_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(valid_next_states).squeeze(-1)

        log_F_s_next = torch.zeros_like(log_pb_actions)
        log_F_s_next[~valid_transitions_is_terminating] = valid_log_F_s_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        log_F_s_next[valid_transitions_is_terminating] = valid_transitions_log_rewards[
            valid_transitions_is_terminating
        ]
        targets = log_pb_actions + log_F_s_next

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

        Returns:
            The computed detailed balance loss as a tensor. The shape depends on the
            reduction method.
        """
        warn_about_recalculating_logprobs(transitions, recalculate_all_logprobs)
        scores = self.get_scores(env, transitions, recalculate_all_logprobs)
        scores = scores**2
        loss = loss_reduce(scores, reduction)

        if torch.isnan(loss).any():
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
    ) -> None:
        """Initializes a ModifiedDBGFlowNet instance.

        Args:
            pf: Forward policy estimator.
            pb: Backward policy estimator or None.
            constant_pb: See base class.

        """
        super().__init__(pf, pb, constant_pb=constant_pb)

    def get_scores(
        self, transitions: Transitions, recalculate_all_logprobs: bool = True
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
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")

        if len(transitions) == 0:
            return torch.tensor(0.0, device=transitions.device)

        mask = ~transitions.next_states.is_sink_state
        states = transitions.states[mask]
        valid_next_states = transitions.next_states[mask]
        actions = transitions.actions[mask]
        all_log_rewards = transitions.all_log_rewards[mask]

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
        valid_log_pf_s_exit = pf_dist.log_prob(
            torch.full_like(actions.tensor, actions.__class__.exit_action[0].item())
        )

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
        ).log_prob(
            torch.full_like(actions.tensor, actions.__class__.exit_action[0].item())
        )

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
        if torch.any(torch.isinf(scores)):
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
            transitions, recalculate_all_logprobs=recalculate_all_logprobs
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
