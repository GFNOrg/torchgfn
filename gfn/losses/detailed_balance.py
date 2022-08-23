import torch
from torchtyping import TensorType

from gfn.containers import Transitions
from gfn.losses.base import EdgeDecomposableLoss
from gfn.parametrizations import O_PFB
from gfn.samplers.action_samplers import LogitPBActionSampler, LogitPFActionSampler


class DetailedBalance(EdgeDecomposableLoss):
    def __init__(self, o: O_PFB, delta: float = 1e-5):
        self.o = o
        self.delta = delta
        self.action_sampler = LogitPFActionSampler(o.logit_PF)
        self.backward_action_sampler = LogitPBActionSampler(o.logit_PB)

    def __call__(self, transitions: Transitions) -> TensorType[0]:
        if transitions.is_backwards:
            raise ValueError("Backwards transitions are not supported")
        valid_states = transitions.states[~transitions.states.is_sink_state]
        valid_actions = transitions.actions[transitions.actions != -1]

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions == -1)

        if valid_states.batch_shape != tuple(valid_actions.shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        valid_pf_logits = self.action_sampler.get_logits(valid_states)
        valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
        valid_log_pf_actions = torch.gather(
            valid_log_pf_all, dim=-1, index=valid_actions.unsqueeze(-1)
        ).squeeze(-1)

        valid_log_F_s = self.o.logF(valid_states).squeeze(-1)

        preds = valid_log_pf_actions + valid_log_F_s

        targets = torch.zeros_like(preds)

        # uncomment next line for debugging
        assert transitions.next_states.is_sink_state.equal(transitions.is_done)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_done]
        non_exit_valid_actions = valid_actions[
            valid_actions != transitions.env.n_actions - 1
        ]
        valid_pb_logits = self.backward_action_sampler.get_logits(valid_next_states)
        valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)
        valid_log_pb_actions = torch.gather(
            valid_log_pb_all, dim=-1, index=non_exit_valid_actions.unsqueeze(-1)
        ).squeeze(-1)

        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        valid_log_F_s_next = self.o.logF(valid_next_states).squeeze(-1)
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


if __name__ == "__main__":
    from gfn.envs import HyperGrid
    from gfn.estimators import LogitPBEstimator, LogitPFEstimator, LogStateFlowEstimator
    from gfn.models import Uniform, ZeroGFNModule
    from gfn.preprocessors import KHotPreprocessor
    from gfn.samplers import FixedActions, TransitionsSampler

    height = 4
    ndim = 2
    env = HyperGrid(height=height, ndim=ndim)

    print("Evaluating the loss on 5 trajectories with manually chosen actions")
    action_sampler = FixedActions(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 2, 2, 1]]
        )
    )

    transitions_sampler = TransitionsSampler(env, action_sampler)
    transitions = transitions_sampler.sample_transitions(n_transitions=5)
    print(transitions)

    preprocessor = KHotPreprocessor(env)
    pf = Uniform(env.n_actions)
    pb = Uniform(env.n_actions - 1)

    logit_PF = LogitPFEstimator(preprocessor, pf)
    logit_PB = LogitPBEstimator(preprocessor, pb)
    zero_module = ZeroGFNModule()
    logF = LogStateFlowEstimator(preprocessor, zero_module)

    o = O_PFB(logit_PF=logit_PF, logF=logF, logit_PB=logit_PB)

    loss = DetailedBalance(o)
    print(loss(transitions))

    transitions = transitions_sampler.sample_transitions(states=transitions.next_states)
    print(transitions)
    print(loss(transitions))
