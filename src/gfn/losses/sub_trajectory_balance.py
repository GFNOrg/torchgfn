from dataclasses import dataclass
from typing import List, Literal, Tuple

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories
from gfn.estimators import LogStateFlowEstimator
from gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss
from gfn.samplers.actions_samplers import (
    BackwardDiscreteActionsSampler,
    DiscreteActionsSampler,
)

# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]


@dataclass
class SubTBParametrization(PFBasedParametrization):
    r"""
    Exactly the same as DBParametrization
    """
    logF: LogStateFlowEstimator


class SubTrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: SubTBParametrization,
        log_reward_clip_min: float = -12,
        weighing: Literal[
            "DB",
            "ModifiedDB",
            "TB",
            "geometric",
            "equal",
            "geometric_within",
            "equal_within",
        ] = "geometric",
        lamda: float = 0.9,
        on_policy: bool = False,
    ):
        """
        :param parametrization: parametrization of the model
        :param log_reward_clip_min: minimum value of the log-reward. Log-Rewards lower than this value will be clipped to this value. Defaults to -12 (roughly log(1e-5)).
        :param weighing: how to weigh the different sub-trajectories of each trajectory.
            - "DB": Considers all one-step transitions of each trajectory in the batch and weighs them equally (regardless of the length of trajectory).
                    Should be equivalent to DetailedBalance loss.
            - "ModifiedDB": Considers all one-step transitions of each trajectory in the batch and weighs them inversely proportional to the trajectory length.
                    This ensures that the loss is not dominated by long trajectories. Each trajectory contributes equally to the loss.
            - "TB": Considers only the full trajectory. Should be equivalent to TrajectoryBalance loss.
            - "equal_within": Each sub-trajectory of each trajectory is weighed equally within the trajectory. Then each trajectory is weighed equally within the batch.
            - "equal": Each sub-trajectory of each trajectory is weighed equally within the set of all sub-trajectories.
            - "geometric_within": Each sub-trajectory of each trajectory is weighed proportionally to (lamda ** len(sub_trajectory)), within each trajectory.
            - "geometric": Each sub-trajectory of each trajectory is weighed proportionally to (lamda ** len(sub_trajectory)), within the set of all sub-trajectories.
        :param lamda: parameter for geometric weighing
        :param on_policy: whether the loss is computed on-policy (in which case the log probs stored in the trajectories are used) or off-policy
        """
        # Lamda is a discount factor for longer trajectories. The part of the loss
        # corresponding to sub-trajectories of length i is multiplied by lamda^i
        # where an edge is of length 1. As lamda approaches 1, each loss becomes equally weighted.
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )
        self.weighing = weighing
        self.lamda = lamda
        self.on_policy = on_policy

    def cumulative_logprobs(
        self,
        trajectories: Trajectories,
        log_p_trajectories: LogPTrajectoriesTensor,
    ) -> LogPTrajectoriesTensor:
        """
        :param trajectories: trajectories
        :param log_p_trajectories: log probabilities of each transition in each trajectory
        :return: cumulative sum of log probabilities of each trajectory
        """
        return torch.cat(
            (
                torch.zeros(
                    1, trajectories.n_trajectories, device=log_p_trajectories.device
                ),
                log_p_trajectories.cumsum(dim=0),
            ),
            dim=0,
        )

    def get_scores(
        self, trajectories: Trajectories
    ) -> Tuple[List[ScoresTensor], List[ScoresTensor]]:
        """
        Returns two elements:
        - A list of tensors, each of which representing the scores of all sub-trajectories of length k, for k in [1, ..., trajectories.max_length].
            where the score of a sub-trajectory tau is log P_F(tau) + log F(tau_0) - log P_B(tau) - log F(tau_{-1}). The shape of the k-th tensor
            is (trajectories.max_length - k + 1, trajectories.n_trajectories), k starting from 1.
        - A list of tensors representing what should be masked out in the each element of the first list, given that not all sub-trajectories
            of length k exist for each trajectory. The entries of those tensors are True if the corresponding sub-trajectory does not exist.
        """
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, fill_value=-float("inf"), no_pf=self.on_policy
        )
        if self.on_policy:
            log_pf_trajectories = trajectories.log_probs

        log_pf_trajectories_cum = self.cumulative_logprobs(
            trajectories, log_pf_trajectories
        )
        log_pb_trajectories_cum = self.cumulative_logprobs(
            trajectories, log_pb_trajectories
        )

        states = trajectories.states
        log_state_flows = torch.full_like(log_pf_trajectories, fill_value=-float("inf"))
        mask = ~states.is_sink_state
        valid_states = states[mask]
        log_state_flows[mask[:-1]] = self.parametrization.logF(valid_states).squeeze(-1)

        sink_states_mask = log_state_flows == -float("inf")
        is_terminal_mask = trajectories.actions == trajectories.env.n_actions - 1
        full_mask = sink_states_mask | is_terminal_mask

        flattening_masks = []
        scores = []
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
            targets.T[is_terminal_mask[i - 1 :].T] = trajectories.log_rewards[trajectories.when_is_done >= i].clamp_min(self.log_reward_clip_min)  # type: ignore

            # For now, the targets contain the log-rewards of the ending sub trajectories
            # We need to add to that the log-probabilities of the backward actions up-to
            # the sub-trajectory's terminating state
            if i > 1:
                targets[is_terminal_mask[i - 1 :]] += (
                    log_pb_trajectories_cum[i - 1 :] - log_pb_trajectories_cum[: -i + 1]
                )[:-1][is_terminal_mask[i - 1 :]]

            # The following creates the targets for the non-finishing sub-trajectories
            targets[~full_mask[i - 1 :]] = (
                log_pb_trajectories_cum[i:] - log_pb_trajectories_cum[:-i]
            )[:-1][~full_mask[i - 1 : -1]] + log_state_flows[i:][~sink_states_mask[i:]]

            flattening_mask = trajectories.when_is_done.lt(
                torch.arange(
                    i,
                    trajectories.max_length + 1,
                    device=trajectories.when_is_done.device,
                ).unsqueeze(-1)
            )
            flat_preds = preds[~flattening_mask]
            flat_targets = targets[~flattening_mask]

            if torch.any(torch.isnan(flat_preds)):
                raise ValueError("NaN in preds")

            if torch.any(torch.isnan(flat_targets)):
                raise ValueError("NaN in targets")

            flattening_masks.append(flattening_mask)
            scores.append(preds - targets)

        return (
            scores,
            flattening_masks,
        )

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        (
            scores,
            flattening_masks,
        ) = self.get_scores(trajectories)

        flattening_mask = torch.cat(flattening_masks)
        all_scores = torch.cat(scores, 0)

        # all_scores is a tensor of shape (max_length * (max_length + 1) / 2, n_trajectories)
        n_rows = int(trajectories.max_length * (1 + trajectories.max_length) / 2)

        if self.weighing == "equal_within":
            # the following tensor represents the inverse of how many sub-trajectories there are in each trajectory
            contributions = 2.0 / (
                trajectories.when_is_done * (trajectories.when_is_done + 1)
            )
            contributions = contributions / len(trajectories)
            # if we repeat the previous tensor, we get a tensor of shape (max_length * (max_length + 1) / 2, n_trajectories)
            # that we can multiply with all_scores to get a loss where each sub-trajectory is weighted equally within each trajectory
            contributions = contributions.repeat(n_rows, 1)
        elif self.weighing == "equal":
            n_sub_trajectories = int(
                (trajectories.when_is_done * (trajectories.when_is_done + 1) / 2)
                .sum()
                .item()
            )
            contributions = torch.ones(n_rows, len(trajectories)) / n_sub_trajectories
        elif self.weighing == "TB":
            # Each trajectory contributes one element to the loss, equally weighted
            contributions = torch.zeros_like(all_scores)
            indices = (
                trajectories.max_length * (trajectories.when_is_done - 1)
                - (trajectories.when_is_done - 1) * (trajectories.when_is_done - 2) / 2
            ).long()
            contributions.scatter_(0, indices.unsqueeze(0), 1)
            contributions = contributions / len(trajectories)
        elif self.weighing == "DB":
            # Longer trajectories contribute more to the loss
            return scores[0][~flattening_masks[0]].pow(2).mean()
        elif self.weighing == "ModifiedDB":
            # The following tensor represents the inverse of how many transitions there are in each trajectory
            contributions = 1.0 / trajectories.when_is_done
            contributions = contributions / len(trajectories)
            contributions = contributions.repeat(trajectories.max_length, 1)
            contributions = torch.cat(
                (
                    contributions,
                    torch.zeros(
                        (n_rows - trajectories.max_length, len(trajectories)),
                        device=contributions.device,
                    ),
                ),
                0,
            )

        elif self.weighing == "geometric_within":
            # the following tensor represents the weights given to each possible sub-trajectory length
            contributions = (
                self.lamda ** torch.arange(trajectories.max_length).double()
            ).float()
            contributions = contributions.unsqueeze(-1).repeat(1, len(trajectories))

            contributions = contributions.repeat_interleave(
                torch.arange(trajectories.max_length, 0, -1),
                dim=0,
                output_size=int(
                    trajectories.max_length * (trajectories.max_length + 1) / 2
                ),
            )
            r"""
            Now we need to divide each column by n + (n-1) lambda +...+ 1*lambda^{n-1}
            where n is the length of the trajectory corresponding to that column
            We can do it the ugly way, or using the cool identity:
            https://www.wolframalpha.com/input?i=sum%28%28n-i%29+*+lambda+%5Ei%2C+i%3D0..n%29
            """
            per_trajectory_denominator = (
                1.0
                / (1 - self.lamda) ** 2
                * (
                    self.lamda * (self.lamda ** trajectories.when_is_done.double() - 1)
                    + (1 - self.lamda) * trajectories.when_is_done.double()
                )
            ).float()
            contributions = contributions / per_trajectory_denominator
            contributions = contributions / len(trajectories)

        elif self.weighing == "geometric":
            # The position i of the following 1D tensor represents the number of sub-trajectories of length i in the batch
            # n_sub_trajectories = torch.maximum(
            #     trajectories.when_is_done - torch.arange(3).unsqueeze(-1),
            #     torch.tensor(0),
            # ).sum(1)

            # The following tensor's k-th entry represents the mean of all losses of sub-trajectories of length k
            per_length_losses = torch.stack(
                [
                    scores[~flattening_mask].pow(2).mean()
                    for scores, flattening_mask in zip(scores, flattening_masks)
                ]
            )
            ld = self.lamda
            weights = (
                (1 - ld)
                / (1 - ld**trajectories.max_length)
                * (
                    ld
                    ** torch.arange(
                        trajectories.max_length, device=per_length_losses.device
                    )
                )
            )
            assert (weights.sum() - 1.0).abs() < 1e-5, f"{weights.sum()}"
            return (per_length_losses * weights).sum()
        else:
            raise ValueError(f"Unknown weighing method {self.weighing}")

        flat_contributions = contributions[~flattening_mask]
        assert (
            flat_contributions.sum() - 1.0
        ).abs() < 1e-5, f"{flat_contributions.sum()}"
        losses = flat_contributions * all_scores[~flattening_mask].pow(2)
        return losses.sum()
