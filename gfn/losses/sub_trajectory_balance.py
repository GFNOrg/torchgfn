# TODO: merge with trajectory_balance.py

from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.containers import SubTrajectories, Trajectories
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.parametrizations import DBParametrization
from gfn.samplers.actions_samplers import LogitPBActionsSampler, LogitPFActionsSampler

# Typing
ScoresTensor = TensorType["n_trajectories", float]
LossTensor = TensorType[0, float]


class SubTrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: DBParametrization,
        reward_clip_min: float = 1e-5,
    ):
        self.parametrization = parametrization
        self.reward_clip_min = reward_clip_min
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)

    def get_sub_trajectories(self, trajectories: Trajectories) -> Trajectories:
        pass

    def get_scores(
        self, sub_trajectories: SubTrajectories
    ) -> Tuple[ScoresTensor, ScoresTensor, ScoresTensor]:
        # TODO: this can be improved. Many things used here can and should be in the sub_trajectories class
        valid_states = sub_trajectories.states[:-1][
            ~sub_trajectories.states[:-1].is_sink_state
        ]
        valid_actions = sub_trajectories.actions[sub_trajectories.actions != -1]

        # valid_subtrajs mean subtrajectories not starting in s_f, for which no score is computed
        valid_subtrajs = sub_trajectories[sub_trajectories.actions[0, :] != -1]
        actions_for_valid_subtrajs = valid_subtrajs.actions

        assert valid_states.batch_shape == tuple(valid_actions.shape)

        valid_pf_logits = self.actions_sampler.get_logits(valid_states)
        valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
        valid_log_pf_actions = torch.gather(
            valid_log_pf_all, dim=-1, index=valid_actions.unsqueeze(-1)
        ).squeeze(-1)

        log_pf_trajectories = torch.zeros_like(
            actions_for_valid_subtrajs,
            dtype=torch.float,
        )
        log_pf_trajectories[actions_for_valid_subtrajs != -1] = valid_log_pf_actions

        log_pf_trajectories = log_pf_trajectories.sum(dim=0)

        start_states = sub_trajectories.states[:1][
            ~sub_trajectories.states[:1].is_sink_state
        ]

        pf_scores = log_pf_trajectories + self.parametrization.logF(
            start_states
        ).squeeze(-1)

        valid_states = sub_trajectories.states[1:][
            ~sub_trajectories.states[1:].is_sink_state
        ]
        non_exit_valid_actions = valid_actions[
            valid_actions != sub_trajectories.env.n_actions - 1
        ]

        assert valid_states.batch_shape == tuple(non_exit_valid_actions.shape)

        valid_pb_logits = self.backward_actions_sampler.get_logits(valid_states)
        valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)

        valid_log_pb_actions = torch.gather(
            valid_log_pb_all, dim=-1, index=non_exit_valid_actions.unsqueeze(-1)
        ).squeeze(-1)

        log_pb_trajectories = torch.zeros_like(
            actions_for_valid_subtrajs, dtype=torch.float
        )
        log_pb_trajectories_slice = torch.zeros_like(valid_actions, dtype=torch.float)
        log_pb_trajectories_slice[
            valid_actions != sub_trajectories.env.n_actions - 1
        ] = valid_log_pb_actions

        log_pb_trajectories[
            actions_for_valid_subtrajs != -1
        ] = log_pb_trajectories_slice

        log_pb_trajectories = log_pb_trajectories.sum(dim=0)

        end_states_non_terminal = sub_trajectories.states[-1:][
            ~sub_trajectories.states[-1:].is_sink_state
        ]

        logF = torch.zeros_like(log_pb_trajectories, dtype=torch.float)
        logF[
            sub_trajectories.when_is_done[sub_trajectories.actions[0, :] != -1] == -1
        ] = self.parametrization.logF(end_states_non_terminal).squeeze(-1)

        end_states_terminal = sub_trajectories[
            sub_trajectories.when_is_done > 0
        ].states[
            sub_trajectories.when_is_done[sub_trajectories.when_is_done > 0] - 1,
            torch.arange((sub_trajectories.when_is_done > 0).sum()),
        ]

        logF[valid_subtrajs.when_is_done > 0] = (
            sub_trajectories.env.reward(end_states_terminal).squeeze(-1).log()
        )

        pb_scores = log_pb_trajectories + logF

        return log_pf_trajectories, log_pb_trajectories, pf_scores - pb_scores

    def __call__(self, sub_trajectories: SubTrajectories) -> LossTensor:
        _, _, scores = self.get_scores(sub_trajectories)
        loss = scores.pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        return loss


if __name__ == "__main__":

    from gfn.containers import Trajectories, Transitions
    from gfn.containers.sub_trajectories import SubTrajectories
    from gfn.envs import HyperGrid
    from gfn.estimators import (
        LogitPBEstimator,
        LogitPFEstimator,
        LogStateFlowEstimator,
        LogZEstimator,
    )
    from gfn.losses import DetailedBalance, TrajectoryBalance
    from gfn.modules import Tabular, Uniform
    from gfn.parametrizations import DBParametrization, TBParametrization
    from gfn.preprocessors import EnumPreprocessor
    from gfn.samplers import TrajectoriesSampler, UniformActionsSampler

    env = HyperGrid(ndim=2, height=6)
    sampler = UniformActionsSampler()
    trajectories_sampler = TrajectoriesSampler(env, sampler)

    trajs = trajectories_sampler.sample(n_objects=100)

    start_idx = 0
    end_idx = 100
    sub_trajs = SubTrajectories(
        env,
        states=trajs.states[start_idx:end_idx],
        actions=trajs.actions[start_idx : end_idx - 1],
        when_is_done=(trajs.when_is_done - start_idx) * (trajs.when_is_done < (end_idx))
        + torch.full(size=(len(trajs),), fill_value=-1, dtype=torch.long)
        * (trajs.when_is_done >= (end_idx)),
    )

    preprocessor = EnumPreprocessor(env)
    logit_PF = Uniform(output_dim=env.n_actions)
    logit_PF = LogitPFEstimator(preprocessor, logit_PF)
    logit_PB = Uniform(output_dim=env.n_actions - 1)
    logit_PB = LogitPBEstimator(preprocessor, logit_PB)
    logF = Tabular(env, 1)
    logF = LogStateFlowEstimator(preprocessor, logF)

    parametrization = DBParametrization(logit_PF, logit_PB, logF)
    sub_tb = SubTrajectoryBalance(parametrization)

    scores = sub_tb.get_scores(sub_trajs)

    loss = sub_tb(sub_trajs)

    print(scores)
    if end_idx > trajs.max_length and start_idx == 0:
        print("Comparing to TB")
        logZ = torch.tensor(0.0)
        logZ = LogZEstimator(logZ)

        parametrization_2 = TBParametrization(logit_PF, logit_PB, logZ)

        tb = TrajectoryBalance(parametrization_2)

        scores_tb = tb.get_scores(trajs)

        assert torch.all(scores[-1] == scores_tb[-1])
        print("OK")
    if end_idx == start_idx + 2:
        print("Comparing to DB")
        db = DetailedBalance(parametrization)

        transitions = Transitions.from_trajectories(sub_trajs)

        loss_db = db(transitions)

        assert loss == loss_db
        print("OK")
