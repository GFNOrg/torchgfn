import torch
from torchtyping import TensorType

from gfn.containers import Transitions
from gfn.losses.base import EdgeDecomposableLoss
from gfn.parametrizations import DBParametrization, TBParametrization
from gfn.samplers.actions_samplers import LogitPBActionsSampler, LogitPFActionsSampler

# Typing
ScoresTensor = TensorType["n_transitions", float]
LossTensor = TensorType[0, float]


class DetailedBalance(EdgeDecomposableLoss):
    def __init__(self, parametrization: DBParametrization | TBParametrization):
        self.parametrization = parametrization
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)

    def get_scores(self, transitions: Transitions):
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        valid_states = transitions.states[~transitions.states.is_sink_state]
        valid_actions = transitions.actions[transitions.actions != -1]

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions == -1)

        if valid_states.batch_shape != tuple(valid_actions.shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        valid_pf_logits = self.actions_sampler.get_logits(valid_states)
        valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
        valid_log_pf_actions = torch.gather(
            valid_log_pf_all, dim=-1, index=valid_actions.unsqueeze(-1)
        ).squeeze(-1)

        valid_log_F_s = self.parametrization.logF(valid_states).squeeze(-1)

        preds = valid_log_pf_actions + valid_log_F_s

        targets = torch.zeros_like(preds)

        # uncomment next line for debugging
        # assert transitions.next_states.is_sink_state.equal(transitions.is_done)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_done]
        non_exit_valid_actions = valid_actions[
            valid_actions != transitions.env.n_actions - 1
        ]
        valid_pb_logits = self.backward_actions_sampler.get_logits(valid_next_states)
        valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)
        valid_log_pb_actions = torch.gather(
            valid_log_pb_all, dim=-1, index=non_exit_valid_actions.unsqueeze(-1)
        ).squeeze(-1)

        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        valid_log_F_s_next = self.parametrization.logF(valid_next_states).squeeze(-1)
        targets[~valid_transitions_is_done] = valid_log_pb_actions
        log_pb_actions = targets.clone()
        targets[~valid_transitions_is_done] += valid_log_F_s_next
        assert transitions.rewards is not None
        valid_transitions_rewards = transitions.rewards[
            ~transitions.states.is_sink_state
        ]
        targets[valid_transitions_is_done] = torch.log(
            valid_transitions_rewards[valid_transitions_is_done]
        )

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
        valid_states = transitions.states[mask]
        valid_next_states = transitions.next_states[mask]
        valid_actions = transitions.actions[mask]
        all_rewards = transitions.all_rewards[mask]

        valid_pf_logits = self.actions_sampler.get_logits(valid_states)
        valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
        valid_log_pf_actions = torch.gather(
            valid_log_pf_all, dim=-1, index=valid_actions.unsqueeze(-1)
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
            valid_log_pb_all, dim=-1, index=valid_actions.unsqueeze(-1)
        ).squeeze(-1)

        preds = (
            torch.log(all_rewards[:, 0])
            + valid_log_pf_actions
            + valid_log_pf_s_prime_exit
        )
        targets = (
            torch.log(all_rewards[:, 1]) + valid_log_pb_actions + valid_log_pf_s_exit
        )

        scores = preds - targets
        if torch.any(torch.isinf(scores)):
            raise ValueError("scores contains inf")

        return scores
