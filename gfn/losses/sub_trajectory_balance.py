import torch
from torchtyping import TensorType
from typing import List, Tuple

from gfn.constants import ScoresTensor, LossTensor  # For typing
from gfn.containers import Trajectories, Transitions, States, SubTrajectories

# TODO(@saleml): This is a namespace collision with gfn.containers.SubTrajectories,
#  so I take it we can delete the gfn.containers.subTrajectories?
from gfn.containers.sub_trajectories import SubTrajectories
from gfn.envs import HyperGrid
from gfn.estimators import (
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.parametrizations import DBParametrization, TBParametrization
from gfn.preprocessors import EnumPreprocessor, OneHotPreprocessor
from gfn.samplers.actions_samplers import LogitPBActionsSampler, LogitPFActionsSampler
from gfn.losses import DetailedBalance, TrajectoryBalance
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.modules import Tabular, Uniform, NeuralNet
from gfn.samplers import TrajectoriesSampler


class SubTrajectoryBalance2(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: DBParametrization,
        reward_clip_min: float = 1e-5,
        lamda: float = 0.1,
    ):
        """
        :param parametrization: parametrization of the reward function
        :param reward_clip_min: minimum value of the reward function
        :param lamda: discount factor
        :return: None
        """
        self.parametrization = parametrization
        self.reward_clip_min = reward_clip_min
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)
        self.lamda = lamda

    def transform(
        self,
        trajectories: Trajectories,
        log_p_trajectories: LogPTrajectoriesTensor,
    ) -> LogPTrajectoriesTensor:
        """
        TODO(@saleml): Add details here and update to better fn name

        :param trajectories: trajectories
        :param log_p_trajectories: log probabilities of trajectories
        :return: transformed log probabilities
        """
        return torch.cat(
            (
                torch.zeros(1, trajectories.n_trajectories),
                log_p_trajectories.cumsum(dim=0),
            ),
            dim=0,
        )

    def get_scores(
        self,
        trajectories: Trajectories,
    ) -> Tuple[List[ScoresTensor], List[ScoresTensor]]:
        """
        TODO(@saleml): Add details here

        :param trajectories: trajectories
        :return: list of scores for each trajectory
        """
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, fill_value=-float("inf")
        )

        log_pf_trajectories_cum = transform(trajectories, log_pf_trajectories)
        log_pb_trajectories_cum = transform(trajectories, log_pb_trajectories)

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

            # TODO(@saleml): Break complex expression into multiple lines
            # TODO(@saleml): Add comments describing what is happening here
            if i > 1:
                targets[is_terminal_mask[i - 1 :]] += (
                    log_pb_trajectories_cum[i - 1 :] - log_pb_trajectories_cum[: -i + 1]
                )[:-1][is_terminal_mask[i - 1 :]]

            # TODO(@saleml): Break complex expression into multiple lines
            # TODO(@saleml): Add comments describing what is happening here
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
        """
        TODO(@saleml): Add details here

        :param trajectories: trajectories
        :return: loss
        """
        all_preds, all_targets = self.get_scores(trajectories)
        losses = [(p - t).pow(2).mean() for p, t in zip(all_preds, all_targets)]
        max_l = len(losses)
        losses = torch.stack(losses)
        weights = self.lamda ** torch.arange(1, max_l + 1)
        weights = weights / weights.sum()
        return torch.sum(weights * losses)


def main():
    """
    TODO(@saleml): Add details here
    """
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

    # TODO(@saleml): These variables appear unused, and not sure what the assert False is for? Is this script meant
    #  to be run interactive? Or could you codify the test case you expect?
    all_preds, all_targets = sub_tb.get_scores(trajs)
    loss = sub_tb(trajs)
    assert False


if __name__ == "__main__":
    main()
