import math
from typing import Tuple

import torch

from gfn.containers import Trajectories, Transitions
from gfn.env import Env
from gfn.gflownet.base import PFBasedGFlowNet, loss_reduce
from gfn.modules import ConditionalScalarEstimator, GFNModule, ScalarEstimator
from gfn.utils.handlers import (has_conditioning_exception_handler,
                                no_conditioning_exception_handler,
                                warn_about_recalculating_logprobs)
from gfn.utils.prob_calculations import get_transition_pfs_and_pbs


def check_compatibility(states, actions, transitions):
    if states.batch_shape != tuple(actions.batch_shape):
        if type(transitions) is not Transitions:
            raise TypeError(
                "`transitions` is type={}, not Transitions".format(type(transitions))
            )
        else:
            raise ValueError(" wrong happening with log_pf evaluations")


class DBGFlowNet(PFBasedGFlowNet[Transitions]):
    r"""The Detailed Balance GFlowNet.

    Corresponds to $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2 \times
    \mathcal{O}_3$, where $\mathcal{O}_1$ is the set of functions from the internal
    states (no $s_f$) to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the
    non-negativity constraint), and $\mathcal{O}_2$ is the set of forward probability
    functions consistent with the DAG. $\mathcal{O}_3$ is the set of backward
    probability functions consistent with the DAG, or a singleton thereof, if
    `self.logit_PB` is a fixed `DiscretePBEstimator`.

    Attributes:
        logF: a ScalarEstimator instance.
        forward_looking: whether to implement the forward looking GFN loss.
        log_reward_clip_min: If finite, clips log rewards to this value.
        safe_log_prob_min: If True, uses a -1e10 as the minimum log probability value
            to avoid numerical instability, otherwise uses 1e-38.
    """

    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        logF: ScalarEstimator | ConditionalScalarEstimator,
        forward_looking: bool = False,
        log_reward_clip_min: float = -float("inf"),
        safe_log_prob_min: bool = True,
    ):
        super().__init__(pf, pb)
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

    def logF_named_parameters(self):
        try:
            return {k: v for k, v in self.named_parameters() if "logF" in k}
        except KeyError as e:
            print(
                "logF not found in self.named_parameters. Are the weights tied with PF? {}".format(
                    e
                )
            )

    def logF_parameters(self):
        try:
            return [v for k, v in self.named_parameters() if "logF" in k]
        except KeyError as e:
            print(
                "logF not found in self.named_parameters. Are the weights tied with PF? {}".format(
                    e
                )
            )

    def get_pfs_and_pbs(
        self, transitions: Transitions, recalculate_all_logprobs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_transition_pfs_and_pbs(
            self.pf, self.pb, transitions, recalculate_all_logprobs
        )

    def get_scores(
        self, env: Env, transitions: Transitions, recalculate_all_logprobs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a batch of transitions, calculate the scores.

        Args:
            transitions: a batch of transitions.

        Unless recalculate_all_logprobs=True, in which case we re-evaluate the logprobs of the transitions with
        the current self.pf. The following applies:
            - If transitions have log_probs attribute, use them - this is usually for on-policy learning
            - Else, re-evaluate the log_probs using the current self.pf - this is usually for
              off-policy learning with replay buffer

        Returns: A tuple of three tensors of shapes (n_transitions,), representing the
            log probabilities of the actions, the log probabilities of the backward actions, and th scores.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")

        states = transitions.states
        actions = transitions.actions

        if len(states) == 0:
            return (
                torch.tensor(self.log_prob_min, device=transitions.device),
                torch.tensor(self.log_prob_min, device=transitions.device),
                torch.tensor(0.0, device=transitions.device),
            )

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)
        check_compatibility(states, actions, transitions)

        log_pf_actions, log_pb_actions = self.get_pfs_and_pbs(
            transitions, recalculate_all_logprobs
        )

        # LogF is potentially a conditional computation.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                log_F_s = self.logF(states, transitions.conditioning).squeeze(-1)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                log_F_s = self.logF(states).squeeze(-1)

        if self.forward_looking:
            log_rewards = env.log_reward(states)
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
            return (
                torch.tensor(self.log_prob_min, device=transitions.device),
                torch.tensor(self.log_prob_min, device=transitions.device),
                torch.tensor(0.0, device=transitions.device),
            )

        # LogF is potentially a conditional computation.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(
                    valid_next_states,
                    transitions.conditioning[~transitions.is_terminating],
                ).squeeze(-1)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
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
        return (log_pf_actions, log_pb_actions, scores)

    def loss(
        self,
        env: Env,
        transitions: Transitions,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Detailed balance loss.

        The detailed balance loss is described in section
        3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).
        """
        warn_about_recalculating_logprobs(transitions, recalculate_all_logprobs)
        _, _, scores = self.get_scores(env, transitions, recalculate_all_logprobs)
        scores = scores**2
        loss = loss_reduce(scores, reduction)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()


class ModifiedDBGFlowNet(PFBasedGFlowNet[Transitions]):
    r"""The Modified Detailed Balance GFlowNet. Only applicable to environments where
    all states are terminating.

    See Bayesian Structure Learning with Generative Flow Networks
    https://arxiv.org/abs/2202.13903 for more details.
    """

    def get_scores(
        self, transitions: Transitions, recalculate_all_logprobs: bool = True
    ) -> torch.Tensor:
        """DAG-GFN-style detailed balance, when all states are connected to the sink.

        Unless recalculate_all_logprobs=True, in which case we re-evaluate the logprobs of the transitions with
        the current self.pf. The following applies:
            - If transitions have log_probs attribute, use them - this is usually for on-policy learning
            - Else, re-evaluate the log_probs using the current self.pf - this is usually for
              off-policy learning with replay buffer

        Raises:
            ValueError: when backward transitions are supplied (not supported).
            ValueError: when the computed scores contain `inf`.
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

        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(states, transitions.conditioning[mask])
        else:
            with no_conditioning_exception_handler("pf", self.pf):
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
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(
                    valid_next_states, transitions.conditioning[mask]
                )
        else:
            with no_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(valid_next_states)

        valid_log_pf_s_prime_exit = self.pf.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(
            torch.full_like(actions.tensor, actions.__class__.exit_action[0].item())
        )

        non_exit_actions = actions[~actions.is_exit]

        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pb", self.pb):
                module_output = self.pb(
                    valid_next_states, transitions.conditioning[mask]
                )
        else:
            with no_conditioning_exception_handler("pb", self.pb):
                module_output = self.pb(valid_next_states)

        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

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
        """Calculates the modified detailed balance loss."""
        del env
        warn_about_recalculating_logprobs(transitions, recalculate_all_logprobs)
        scores = self.get_scores(
            transitions, recalculate_all_logprobs=recalculate_all_logprobs
        )
        scores = scores**2
        return loss_reduce(scores, reduction)

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()
