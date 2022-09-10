from typing import List, Tuple

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories, States, SubTrajectories
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.modules import NeuralNet
from gfn.parametrizations import DBParametrization
from gfn.samplers.actions_samplers import LogitPBActionsSampler, LogitPFActionsSampler

# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]


# class SubTrajectoryBalance(TrajectoryDecomposableLoss):
#     def __init__(
#         self,
#         parametrization: DBParametrization,
#         reward_clip_min: float = 1e-5,
#     ):
#         self.parametrization = parametrization
#         self.reward_clip_min = reward_clip_min
#         self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
#         self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)

#     def get_scores(self, trajectories: Trajectories) -> Trajectories:
#         # Note that this counts sub-trajectories that terminate early, multiple times
#         log_pfs_all = []
#         log_pbs_all = []
#         scores_all = []
#         max_length = trajectories.max_length
#         for i in range(max_length - 1):
#             for j in range(i + 2, max_length + 2):
#                 sub_trajectories = SubTrajectories.from_trajectories_fixed_length(
#                     trajectories, i, j
#                 )
#                 log_pf, log_pb, scores = self.get_scores_fixed_length(sub_trajectories)
#                 log_pbs_all.append(log_pb * (j - i - 1))
#                 log_pfs_all.append(log_pf * (j - i - 1))
#                 scores_all.append(scores * (j - i - 1))
#         log_pf_all = torch.cat(log_pfs_all, dim=0)
#         log_pb_all = torch.cat(log_pbs_all, dim=0)
#         scores_all = torch.cat(scores_all, dim=0)
#         return log_pf_all, log_pb_all, scores_all

#     def get_scores_fixed_length(
#         self, sub_trajectories: SubTrajectories
#     ) -> Tuple[ScoresTensor, ScoresTensor, ScoresTensor]:
#         # TODO: this can be improved. Many things used here can and should be in the sub_trajectories class
#         if sub_trajectories.max_length == 0:
#             return torch.zeros(0), torch.zeros(0), torch.zeros(0)
#         valid_states = sub_trajectories.states[:-1][
#             ~sub_trajectories.states[:-1].is_sink_state
#         ]
#         valid_actions = sub_trajectories.actions[sub_trajectories.actions != -1]

#         # valid_subtrajs mean subtrajectories not starting in s_f, for which no score is computed
#         valid_subtrajs = sub_trajectories[sub_trajectories.actions[0, :] != -1]
#         actions_for_valid_subtrajs = valid_subtrajs.actions

#         assert valid_states.batch_shape == tuple(valid_actions.shape)

#         valid_pf_logits = self.actions_sampler.get_logits(valid_states)
#         valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
#         valid_log_pf_actions = torch.gather(
#             valid_log_pf_all, dim=-1, index=valid_actions.unsqueeze(-1)
#         ).squeeze(-1)

#         log_pf_trajectories = torch.zeros_like(
#             actions_for_valid_subtrajs,
#             dtype=torch.float,
#         )
#         log_pf_trajectories[actions_for_valid_subtrajs != -1] = valid_log_pf_actions

#         log_pf_trajectories = log_pf_trajectories.sum(dim=0)

#         start_states = sub_trajectories.states[:1][
#             ~sub_trajectories.states[:1].is_sink_state
#         ]

#         pf_scores = log_pf_trajectories + self.parametrization.logF(
#             start_states
#         ).squeeze(-1)

#         valid_states = sub_trajectories.states[1:][
#             ~sub_trajectories.states[1:].is_sink_state
#         ]
#         non_exit_valid_actions = valid_actions[
#             valid_actions != sub_trajectories.env.n_actions - 1
#         ]

#         assert valid_states.batch_shape == tuple(non_exit_valid_actions.shape)

#         valid_pb_logits = self.backward_actions_sampler.get_logits(valid_states)
#         valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)

#         valid_log_pb_actions = torch.gather(
#             valid_log_pb_all, dim=-1, index=non_exit_valid_actions.unsqueeze(-1)
#         ).squeeze(-1)

#         log_pb_trajectories = torch.zeros_like(
#             actions_for_valid_subtrajs, dtype=torch.float
#         )
#         log_pb_trajectories_slice = torch.zeros_like(valid_actions, dtype=torch.float)
#         log_pb_trajectories_slice[
#             valid_actions != sub_trajectories.env.n_actions - 1
#         ] = valid_log_pb_actions

#         log_pb_trajectories[
#             actions_for_valid_subtrajs != -1
#         ] = log_pb_trajectories_slice

#         log_pb_trajectories = log_pb_trajectories.sum(dim=0)

#         end_states_non_terminal = sub_trajectories.states[-1:][
#             ~sub_trajectories.states[-1:].is_sink_state
#         ]

#         logF = torch.zeros_like(log_pb_trajectories, dtype=torch.float)
#         logF[
#             sub_trajectories.when_is_done[sub_trajectories.actions[0, :] != -1] == -1
#         ] = self.parametrization.logF(end_states_non_terminal).squeeze(-1)

#         end_states_terminal = sub_trajectories[
#             sub_trajectories.when_is_done > 0
#         ].states[
#             sub_trajectories.when_is_done[sub_trajectories.when_is_done > 0] - 1,
#             torch.arange((sub_trajectories.when_is_done > 0).sum()),
#         ]

#         logF[valid_subtrajs.when_is_done > 0] = (
#             sub_trajectories.env.reward(end_states_terminal).squeeze(-1).log()
#         )

#         pb_scores = log_pb_trajectories + logF

#         return log_pf_trajectories, log_pb_trajectories, pf_scores - pb_scores

#     def __call__(self, trajectories: Trajectories) -> LossTensor:
#         _, _, scores = self.get_scores(trajectories)
#         loss = scores.pow(2).mean()
#         if torch.isnan(loss):
#             raise ValueError("Loss is NaN")

#         return loss


class SubTrajectoryBalance2(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: DBParametrization,
        reward_clip_min: float = 1e-5,
        lamda: float = 0.1,
    ):
        # Lamda is a discount factor for longer trajectories. The part of the loss
        # corresponding to sub-trajectories of length i is multiplied by lamda^i
        # where an edge is of length 1. As lamda approaches 1, each loss becomes equally weighted.
        self.parametrization = parametrization
        self.reward_clip_min = reward_clip_min
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)
        self.lamda = lamda

    def get_scores(
        self, trajectories: Trajectories
    ) -> Tuple[List[ScoresTensor], List[ScoresTensor]]:
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, fill_value=-float("inf")
        )
        log_pf_trajectories_cum = log_pf_trajectories.cumsum(dim=0)
        log_pf_trajectories_cum = torch.cat(
            (torch.zeros(1, trajectories.n_trajectories), log_pf_trajectories_cum),
            dim=0,
        )
        log_pb_trajectories_cum = log_pb_trajectories.cumsum(dim=0)
        log_pb_trajectories_cum = torch.cat(
            (torch.zeros(1, trajectories.n_trajectories), log_pb_trajectories_cum),
            dim=0,
        )
        states = trajectories.states
        log_state_flows = torch.full_like(log_pf_trajectories, fill_value=-float("inf"))
        mask = ~states.is_sink_state
        valid_states = states[mask]
        log_state_flows[mask[:-1]] = self.parametrization.logF(valid_states).squeeze(-1)

        sink_states_mask = log_state_flows == -float("inf")
        is_terminal_mask = trajectories.actions == trajectories.env.n_actions - 1
        full_mask = sink_states_mask | is_terminal_mask

        all_preds = []
        all_targets = []
        for i in range(1, 1 + trajectories.max_length):
            current_log_state_flows = (
                log_state_flows if i == 1 else log_state_flows[: -(i - 1)]
            )
            preds = (
                log_pf_trajectories_cum[i:]
                - log_pf_trajectories_cum[:-i]
                + current_log_state_flows
            )
            targets = torch.full_like(preds, fill_value=-float("inf"))
            targets.T[is_terminal_mask[i - 1 :].T] = torch.log(
                trajectories.rewards[trajectories.when_is_done >= i]
            )
            if i > 1:
                targets[is_terminal_mask[i - 1 :]] += (
                    log_pb_trajectories_cum[i - 1 :] - log_pb_trajectories_cum[: -i + 1]
                )[:-1][is_terminal_mask[i - 1 :]]
            targets[~full_mask[i - 1 :]] = (
                log_pb_trajectories_cum[i:] - log_pb_trajectories_cum[:-i]
            )[:-1][~full_mask[i - 1 : -1]] + log_state_flows[i:][~sink_states_mask[i:]]

            flattening_mask = trajectories.when_is_done.lt(
                torch.arange(i, trajectories.max_length + 1).unsqueeze(-1)
            )
            flat_preds = preds[~flattening_mask]
            flat_targets = targets[~flattening_mask]
            if torch.any(torch.isnan(flat_preds)):
                raise ValueError("NaN in preds")
            if torch.any(torch.isnan(flat_targets)):
                raise ValueError("NaN in targets")

            all_preds.append(flat_preds)
            all_targets.append(flat_targets)

        return (all_preds, all_targets)

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        all_preds, all_targets = self.get_scores(trajectories)
        losses = [(p - t).pow(2).mean() for p, t in zip(all_preds, all_targets)]
        max_l = len(losses)
        losses = torch.stack(losses)
        weights = self.lamda ** torch.arange(1, max_l + 1)
        weights = weights / weights.sum()
        return torch.sum(weights * losses)


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
    from gfn.modules import Tabular, Uniform, NeuralNet
    from gfn.parametrizations import DBParametrization, TBParametrization
    from gfn.preprocessors import EnumPreprocessor, OneHotPreprocessor
    from gfn.samplers import TrajectoriesSampler

    env = HyperGrid(ndim=2, height=6)
    preprocessor = OneHotPreprocessor(env)
    logit_PF = Uniform(output_dim=env.n_actions)
    logit_PF = LogitPFEstimator(preprocessor, logit_PF)
    sampler = LogitPFActionsSampler(logit_PF, sf_temperature=2)
    trajectories_sampler = TrajectoriesSampler(env, sampler)

    logit_PB = Uniform(output_dim=env.n_actions - 1)
    logit_PB = LogitPBEstimator(preprocessor, logit_PB)
    logF = NeuralNet(input_dim=preprocessor.output_dim, output_dim=1, hidden_dim=32)
    logF = LogStateFlowEstimator(preprocessor, logF)

    parametrization = DBParametrization(logit_PF, logit_PB, logF)

    trajs = trajectories_sampler.sample(n_objects=5)
    sub_tb = SubTrajectoryBalance2(parametrization)
    all_preds, all_targets = sub_tb.get_scores(trajs)

    loss = sub_tb(trajs)
    assert False
