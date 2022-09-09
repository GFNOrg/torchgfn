import torch
from torchtyping import TensorType

from gfn.containers import Transitions
from gfn.losses.base import EdgeDecomposableLoss
from gfn.parametrizations import DBParametrization
from gfn.samplers.actions_samplers import LogitPBActionsSampler, LogitPFActionsSampler


class DetailedBalance(EdgeDecomposableLoss):
    def __init__(self, parametrization: DBParametrization):
        self.parametrization = parametrization
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)

    def __call__(self, transitions: Transitions) -> TensorType[0]:
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
        targets[~valid_transitions_is_done] = valid_log_pb_actions + valid_log_F_s_next
        assert transitions.rewards is not None
        valid_transitions_rewards = transitions.rewards[
            ~transitions.states.is_sink_state
        ]
        targets[valid_transitions_is_done] = torch.log(
            valid_transitions_rewards[valid_transitions_is_done]
        )

        loss = torch.nn.MSELoss()(preds, targets)
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss
