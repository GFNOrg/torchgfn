from typing import List, Literal, Tuple

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.parametrizations import DBParametrization
from gfn.samplers.actions_samplers import LogitPBActionsSampler, LogitPFActionsSampler

# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]


class SubTrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: DBParametrization,
        reward_clip_min: float = 1e-5,
        weighing: Literal["DB", "TB", "geometric", "equal"] = "geometric",
        lamda: float = 0.9,
    ):
        # Lamda is a discount factor for longer trajectories. The part of the loss
        # corresponding to sub-trajectories of length i is multiplied by lamda^i
        # where an edge is of length 1. As lamda approaches 1, each loss becomes equally weighted.
        self.parametrization = parametrization
        self.reward_clip_min = reward_clip_min
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)
        self.weighing = weighing
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
        all_unflat_preds = []
        all_unflat_targets = []
        flattening_masks = []
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
                trajectories.rewards[trajectories.when_is_done >= i]  # type: ignore
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

            all_unflat_preds.append(preds)
            all_unflat_targets.append(targets)

            flattening_masks.append(flattening_mask)

        return (
            all_preds,
            all_targets,
            all_unflat_preds,
            all_unflat_targets,
            flattening_masks,
        )

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        (
            all_preds,
            all_targets,
            all_unflat_preds,
            all_unflat_targets,
            flattening_masks,
        ) = self.get_scores(trajectories)

        flattening_mask = torch.cat(flattening_masks)
        pre_losses = torch.cat(all_unflat_preds, 0) - torch.cat(all_unflat_targets, 0)

        if self.weighing == "equal":
            # the following tensor represents the contributions of each loss element to the final loss
            contributions = (
                2.0 / (trajectories.when_is_done * (trajectories.when_is_done + 1))
            ).repeat(
                int(trajectories.max_length * (1 + trajectories.max_length) / 2), 1
            )
        elif self.weighing == "TB":
            # Each trajectory contributes one element to the loss
            contributions = torch.zeros_like(pre_losses)
            indices = (
                trajectories.max_length * (trajectories.when_is_done - 1)
                - (trajectories.when_is_done - 1) * (trajectories.when_is_done - 2) / 2
            ).long()
            contributions.scatter_(0, indices.unsqueeze(0), 1)
        elif self.weighing == "DB":
            # Longer trajectories contribute more to the loss
            per_length_losses = torch.stack(
                [(p - t).pow(2).mean() for p, t in zip(all_preds, all_targets)]
            )
            return per_length_losses[0]
        elif self.weighing == "geometric":
            # the following tensor represents the weights given to each possible sub-trajectory length
            contributions = self.lamda ** torch.arange(trajectories.max_length)
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

        elif self.weighing == "geometric2":
            # The position i of the following 1D tensor represents the number of sub-trajectories of length i in the batch
            # n_sub_trajectories = torch.maximum(
            #     trajectories.when_is_done - torch.arange(3).unsqueeze(-1),
            #     torch.tensor(0),
            # ).sum(1)
            per_length_losses = torch.stack(
                [(p - t).pow(2).mean() for p, t in zip(all_preds, all_targets)]
            )
            ld = self.lamda
            weights = (
                (1 - ld)
                / (1 - ld**trajectories.max_length)
                * (ld ** torch.arange(trajectories.max_length))
            )
            assert (weights.sum() - 1.0).abs() < 1e-5, f"{weights.sum()}"
            return (per_length_losses * weights).sum()
        else:
            raise ValueError(f"Unknown weighing method {self.weighing}")

        flat_contributions = contributions[~flattening_mask]
        flat_contributions = flat_contributions / len(trajectories)
        assert (
            flat_contributions.sum() - 1.0
        ).abs() < 1e-5, f"{flat_contributions.sum()}"
        losses = flat_contributions * pre_losses[~flattening_mask].pow(2)
        return losses.sum()
