from dataclasses import dataclass

import torch
from torchtyping import TensorType

from gfn.containers import Transitions
from gfn.estimators import LogStateFlowEstimator
from gfn.losses.base import EdgeDecomposableLoss, PFBasedParametrization
from gfn.samplers.actions_samplers import (
    BackwardDiscreteActionsSampler,
    DiscreteActionsSampler,
)

# Typing
ScoresTensor = TensorType["n_transitions", float]
LossTensor = TensorType[0, float]


@dataclass
class DBParametrization(PFBasedParametrization):
    r"""
    Corresponds to $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1$ is the set of functions from the internal states (no $s_f$)
    to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the non-negativity constraint),
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    $\mathcal{O}_3$ is the set of backward probability functions consistent with the DAG, or a singleton
    thereof, if self.logit_PB is a fixed LogitPBEstimator.
    Useful for the Detailed Balance Loss.
    """
    logF: LogStateFlowEstimator


class DetailedBalance(EdgeDecomposableLoss):
    def __init__(self, parametrization: DBParametrization, on_policy: bool = False):
        "If on_policy is True, the log probs stored in the transitions are used."
        self.parametrization = parametrization
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )
        self.on_policy = on_policy

    def get_scores(self, transitions: Transitions):
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions == -1)

        if states.batch_shape != tuple(actions.shape):
            raise ValueError("Something wrong happening with log_pf evaluations")
        if self.on_policy:
            valid_log_pf_actions = transitions.log_probs
        else:
            valid_pf_logits = self.actions_sampler.get_logits(states)
            valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
            valid_log_pf_actions = torch.gather(
                valid_log_pf_all, dim=-1, index=actions.unsqueeze(-1)
            ).squeeze(-1)

        valid_log_F_s = self.parametrization.logF(states).squeeze(-1)

        preds = valid_log_pf_actions + valid_log_F_s

        targets = torch.zeros_like(preds)

        # uncomment next line for debugging
        # assert transitions.next_states.is_sink_state.equal(transitions.is_done)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_done]
        non_exit_actions = actions[actions != transitions.env.n_actions - 1]
        valid_pb_logits = self.backward_actions_sampler.get_logits(valid_next_states)
        valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)
        valid_log_pb_actions = torch.gather(
            valid_log_pb_all, dim=-1, index=non_exit_actions.unsqueeze(-1)
        ).squeeze(-1)

        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        valid_log_F_s_next = self.parametrization.logF(valid_next_states).squeeze(-1)
        targets[~valid_transitions_is_done] = valid_log_pb_actions
        log_pb_actions = targets.clone()
        targets[~valid_transitions_is_done] += valid_log_F_s_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        targets[valid_transitions_is_done] = valid_transitions_log_rewards[
            valid_transitions_is_done
        ]

        scores = preds - targets

        return (valid_log_pf_actions, log_pb_actions, scores)

    def __call__(self, transitions: Transitions) -> LossTensor:
        _, _, scores = self.get_scores(transitions)
        loss = torch.mean(scores**2)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def get_modified_scores(self, transitions: Transitions) -> ScoresTensor:
        "DAG-GFN-style detailed balance, for when all states are connected to the sink"
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        mask = ~transitions.next_states.is_sink_state
        states = transitions.states[mask]
        valid_next_states = transitions.next_states[mask]
        actions = transitions.actions[mask]
        all_log_rewards = transitions.all_log_rewards[mask]

        valid_pf_logits = self.actions_sampler.get_logits(states)
        valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
        valid_log_pf_actions = torch.gather(
            valid_log_pf_all, dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        valid_log_pf_s_exit = valid_log_pf_all[:, -1]

        # The following two lines are slightly inefficient, given that most
        # next_states are also states, for which we already did a forward pass.
        valid_log_pf_s_prime_all = self.actions_sampler.get_logits(
            valid_next_states
        ).log_softmax(dim=-1)
        valid_log_pf_s_prime_exit = valid_log_pf_s_prime_all[:, -1]

        valid_pb_logits = self.backward_actions_sampler.get_logits(valid_next_states)
        valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)
        valid_log_pb_actions = torch.gather(
            valid_log_pb_all, dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        preds = all_log_rewards[:, 0] + valid_log_pf_actions + valid_log_pf_s_prime_exit
        targets = all_log_rewards[:, 1] + valid_log_pb_actions + valid_log_pf_s_exit

        scores = preds - targets
        if torch.any(torch.isinf(scores)):
            raise ValueError("scores contains inf")

        return scores
