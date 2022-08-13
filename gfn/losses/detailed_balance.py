import torch
from torchtyping import TensorType
from gfn.containers import Transitions
from gfn.parametrizations import O_PFB
from gfn.losses.base import EdgeDecomposableLoss
from gfn.samplers.action_samplers import LogitPFActionSampler, LogitPBActionSampler


class DetailedBalance(EdgeDecomposableLoss):
    def __init__(self, o: O_PFB, delta: float = 1e-5):
        self.o = o
        self.delta = delta
        self.action_sampler = LogitPFActionSampler(o.logit_PF)
        self.backward_action_sampler = LogitPBActionSampler(o.logit_PB)

    def __call__(self, transitions: Transitions) -> TensorType[0]:
        pf_logits = self.action_sampler.get_logits(transitions.states)
        log_pf_all = pf_logits.log_softmax(dim=-1)
        log_pf_actions = torch.gather(
            log_pf_all, dim=-1, index=transitions.actions.unsqueeze(-1)).squeeze(-1)  # should be 1D

        pb_logits = self.backward_action_sampler.get_logits(
            transitions.next_states)
        log_pb_all = pb_logits.log_softmax(dim=-1)
        log_pb_actions = torch.full((transitions.n_transitions, ), -10.,  # The value -10 doesn't matter
                                    device=transitions.env.device)
        log_pb_actions[~transitions.is_done] = torch.gather(
            log_pb_all[~transitions.is_done], dim=-1,
            index=transitions.actions[~transitions.is_done].unsqueeze(-1)).squeeze(-1)  # should be 1D

        log_F_s = self.o.logF(transitions.states).squeeze(-1)
        log_F_s_next = self.o.logF(transitions.next_states).squeeze(-1)

        log_rewards = transitions.rewards

        preds = log_pf_actions + log_F_s
        targets = transitions.is_done.float() * log_rewards + \
            (1 - transitions.is_done.float()) * (log_pb_actions + log_F_s_next)

        loss = torch.nn.MSELoss()(preds, targets)
        if torch.isnan(loss):
            raise ValueError('loss is nan')

        return loss


if __name__ == '__main__':
    from gfn.envs import HyperGrid
    from gfn.preprocessors import KHotPreprocessor
    from gfn.samplers import FixedActions, LogitPFActionSampler, TransitionsSampler
    from gfn.models import ZeroGFNModule, Uniform
    from gfn.estimators import LogitPBEstimator, LogitPFEstimator, LogStateFlowEstimator

    n_envs = 5
    height = 4
    ndim = 2
    env = HyperGrid(n_envs=n_envs, height=height, ndim=ndim)

    print('Evaluating the loss on 5 trajectories with manually chosen actions')
    action_sampler = FixedActions(torch.tensor([[0, 1, 2, 0],
                                                [1, 1, 1, 2],
                                                [2, 2, 2, 2],
                                                [1, 0, 1, 2],
                                                [1, 2, 2, 1]]))

    transitions_sampler = TransitionsSampler(env, action_sampler)
    transitions = transitions_sampler.sample_transitions()
    print(transitions)

    preprocessor = KHotPreprocessor(env)
    pf = Uniform(env.n_actions)
    pb = Uniform(env.n_actions - 1)

    logit_PF = LogitPFEstimator(preprocessor, env, pf)
    logit_PB = LogitPBEstimator(preprocessor, env, pb)
    zero_module = ZeroGFNModule()
    logF = LogStateFlowEstimator(preprocessor, zero_module)

    o = O_PFB(logF, logit_PF, logit_PB)

    loss = DetailedBalance(o)
    print(loss(transitions))
