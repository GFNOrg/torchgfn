from typing import Union

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.parametrizations import O_PFBZ, O_PFZ
from gfn.samplers.action_samplers import LogitPBActionSampler, LogitPFActionSampler


class TrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        o: Union[O_PFBZ, O_PFZ],
        reward_clip_min: float = 1e-5,
    ):
        self.o = o
        self.reward_clip_min = reward_clip_min
        self.action_sampler = LogitPFActionSampler(o.logit_PF)
        self.backward_action_sampler = LogitPBActionSampler(o.logit_PB)

    def __call__(self, trajectories: Trajectories) -> TensorType[0]:
        if trajectories.is_backwards:
            raise ValueError("Backwards trajectories are not supported")

        valid_states = trajectories.states[~trajectories.states.is_sink_state]
        valid_actions = trajectories.actions[trajectories.actions != -1]

        # uncomment next line for debugging
        # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions == -1)

        if valid_states.batch_shape != tuple(valid_actions.shape):
            raise ValueError("Something wrong happening with log_pf evaluations")
        valid_pf_logits = self.action_sampler.get_logits(valid_states)
        valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
        valid_log_pf_actions = torch.gather(
            valid_log_pf_all, dim=-1, index=valid_actions.unsqueeze(-1)
        ).squeeze(-1)
        log_pf_trajectories = torch.zeros_like(trajectories.actions, dtype=torch.float)
        log_pf_trajectories[trajectories.actions != -1] = valid_log_pf_actions

        log_pf_trajectories = log_pf_trajectories.sum(dim=0)

        valid_pb_logits = self.backward_action_sampler.get_logits(
            valid_states[~valid_states.is_initial_state]
        )
        valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)
        non_exit_valid_actions = valid_actions[
            valid_actions != trajectories.env.n_actions - 1
        ]
        valid_log_pb_actions = torch.gather(
            valid_log_pb_all, dim=-1, index=non_exit_valid_actions.unsqueeze(-1)
        ).squeeze(-1)
        log_pb_trajectories = torch.zeros_like(trajectories.actions, dtype=torch.float)
        log_pb_trajectories_slice = torch.zeros_like(valid_actions, dtype=torch.float)
        log_pb_trajectories_slice[
            valid_actions != trajectories.env.n_actions - 1
        ] = valid_log_pb_actions
        log_pb_trajectories[trajectories.actions != -1] = log_pb_trajectories_slice

        log_pb_trajectories = log_pb_trajectories.sum(dim=0)

        preds = log_pf_trajectories - log_pb_trajectories + self.o.logZ.logZ

        rewards = trajectories.rewards
        assert rewards is not None

        targets = torch.log(rewards.clamp_min(self.reward_clip_min))

        loss = torch.nn.MSELoss()(preds, targets)
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss


if __name__ == "__main__":
    from gfn.envs import HyperGrid
    from gfn.estimators import LogitPBEstimator, LogitPFEstimator, LogZEstimator
    from gfn.models import NeuralNet, Uniform
    from gfn.preprocessors import KHotPreprocessor
    from gfn.samplers.action_samplers import FixedActions
    from gfn.samplers.trajectories_sampler import TrajectoriesSampler

    height = 4
    ndim = 2
    env = HyperGrid(height=height, ndim=ndim)

    print("Evaluating the loss on 5 trajectories with manually chosen actions")
    action_sampler = FixedActions(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 2, 2, 1]]
        )
    )
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    print(trajectories)

    preprocessor = KHotPreprocessor(env)
    pf = Uniform(env.n_actions)
    pb = Uniform(env.n_actions - 1)

    logZ = torch.tensor(0.0)
    logit_PF = LogitPFEstimator(preprocessor, pf)
    logit_PB = LogitPBEstimator(preprocessor, pb)
    logZ = LogZEstimator(logZ)

    o = O_PFZ(logit_PF, logit_PB, logZ)

    loss = TrajectoryBalance(o)
    print(loss(trajectories))
    # sanity check, by hand, we should get the following loss
    pbs = torch.tensor([0.5, 1, 1, 0.25, 1.0])
    pfs = torch.tensor(
        [1.0 / (3**3), 1.0 / (3**3) * 0.5, 1.0 / 3, 1.0 / (3**4), 1.0 / (3**2)]
    )
    true_losses_exp = torch.exp(logZ.logZ) * pfs / (pbs * trajectories.rewards)
    true_loss = torch.log(true_losses_exp).pow(2).mean()

    if true_loss == loss(trajectories):
        print("OK - TB LOSS PROBABLY OK")
    else:
        raise ValueError("TB LOSS NOT PROPERLY CALCULATED")

    n_trajectories = 25
    print(
        f"\nEvaluating the loss on {n_trajectories} trajectories with randomly chosen actions (P_F and P_B as modules)"
    )

    preprocessor = KHotPreprocessor(env)
    pf = NeuralNet(
        input_dim=preprocessor.output_dim, output_dim=env.n_actions, hidden_dim=16
    )

    pb = NeuralNet(
        input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1, hidden_dim=16
    )

    print("Now with LogitPFActionSampler")
    logit_PF = LogitPFEstimator(preprocessor, pf)
    action_sampler = LogitPFActionSampler(logit_PF=logit_PF)
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories(
        n_trajectories=n_trajectories
    )
    print(trajectories)

    logZ = torch.tensor(-5.0)

    logit_PB = LogitPBEstimator(preprocessor, pb)
    logZ = LogZEstimator(logZ)

    o = O_PFBZ(logit_PF, logit_PB, logZ)

    loss = TrajectoryBalance(o)
    print(loss(trajectories))
