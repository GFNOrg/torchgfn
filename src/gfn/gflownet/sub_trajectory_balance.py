import math
from typing import List, Literal, Tuple

import torch
from torchtyping import TensorType as TT

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.base import TrajectoryBasedGFlowNet
from gfn.modules import GFNModule, ScalarEstimator

ContributionsTensor = TT["max_len * (1 + max_len) / 2", "n_trajectories"]
CumulativeLogProbsTensor = TT["max_length + 1", "n_trajectories"]
LogStateFlowsTensor = TT["max_length", "n_trajectories"]
LogTrajectoriesTensor = TT["max_length", "n_trajectories", torch.float]
MaskTensor = TT["max_length", "n_trajectories"]
PredictionsTensor = TT["max_length + 1 - i", "n_trajectories"]
TargetsTensor = TT["max_length + 1 - i", "n_trajectories"]


class SubTBGFlowNet(TrajectoryBasedGFlowNet):
    r"""GFlowNet for the Sub Trajectory Balance Loss.

    This method is described in [Learning GFlowNets from partial episodes
    for improved convergence and stability](https://arxiv.org/abs/2209.12782).

    Attributes:
        logF: a LogStateFlowEstimator instance.
        weighting: sub-trajectories weighting scheme.
            - "DB": Considers all one-step transitions of each trajectory in the
                batch and weighs them equally (regardless of the length of
                trajectory). Should be equivalent to DetailedBalance loss.
            - "ModifiedDB": Considers all one-step transitions of each trajectory
                in the batch and weighs them inversely proportional to the
                trajectory length. This ensures that the loss is not dominated by
                long trajectories. Each trajectory contributes equally to the loss.
            - "TB": Considers only the full trajectory. Should be equivalent to
                TrajectoryBalance loss.
            - "equal_within": Each sub-trajectory of each trajectory is weighed
                equally within the trajectory. Then each trajectory is weighed
                equally within the batch.
            - "equal": Each sub-trajectory of each trajectory is weighed equally
                within the set of all sub-trajectories.
            - "geometric_within": Each sub-trajectory of each trajectory is weighed
                proportionally to (lamda ** len(sub_trajectory)), within each
                trajectory. THIS CORRESPONDS TO THE ONE IN THE PAPER.
            - "geometric": Each sub-trajectory of each trajectory is weighed
                proportionally to (lamda ** len(sub_trajectory)), within the set of
                all sub-trajectories.
        lamda: discount factor for longer trajectories.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        logF: ScalarEstimator,
        weighting: Literal[
            "DB",
            "ModifiedDB",
            "TB",
            "geometric",
            "equal",
            "geometric_within",
            "equal_within",
        ] = "geometric_within",
        lamda: float = 0.9,
        log_reward_clip_min: float = -float("inf"),
        forward_looking: bool = False,
    ):
        super().__init__(pf, pb)
        self.logF = logF
        self.weighting = weighting
        self.lamda = lamda
        self.log_reward_clip_min = log_reward_clip_min
        self.forward_looking = forward_looking

    def cumulative_logprobs(
        self,
        trajectories: Trajectories,
        log_p_trajectories: LogTrajectoriesTensor,
    ) -> CumulativeLogProbsTensor:
        """Calculates the cumulative log probabilities for all trajectories.

        Args:
            trajectories: a batch of trajectories.
            log_p_trajectories: log probabilities of each transition in each trajectory.

        Returns: cumulative sum of log probabilities of each trajectory.
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

    def calculate_preds(
        self,
        log_pf_trajectories_cum: CumulativeLogProbsTensor,
        log_state_flows: LogStateFlowsTensor,
        i: int,
    ) -> PredictionsTensor:
        """
        Calculate the predictions tensor for the current sub-trajectory length.
        """
        current_log_state_flows = (
            log_state_flows if i == 1 else log_state_flows[: -(i - 1)]
        )

        preds = (
            log_pf_trajectories_cum[i:]
            - log_pf_trajectories_cum[:-i]
            + current_log_state_flows
        )

        return preds

    def calculate_targets(
        self,
        trajectories: Trajectories,
        preds: PredictionsTensor,
        log_pb_trajectories_cum: CumulativeLogProbsTensor,
        log_state_flows: LogStateFlowsTensor,
        is_terminal_mask: MaskTensor,
        sink_states_mask: MaskTensor,
        full_mask: MaskTensor,
        i: int,
    ) -> TargetsTensor:
        """
        Calculate the targets tensor for the current sub-trajectory length.
        """
        targets = torch.full_like(preds, fill_value=-float("inf"))
        assert trajectories.log_rewards is not None
        log_rewards = trajectories.log_rewards[trajectories.when_is_done >= i]

        if math.isfinite(self.log_reward_clip_min):
            log_rewards.clamp_min(self.log_reward_clip_min)

        targets.T[is_terminal_mask[i - 1 :].T] = log_rewards

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

        return targets

    def calculate_log_state_flows(
        self,
        env: Env,
        trajectories: Trajectories,
        log_pf_trajectories: LogTrajectoriesTensor,
    ) -> LogStateFlowsTensor:
        """
        Calculate log state flows and masks for sink and terminal states.

        Args:
            trajectories: The trajectories data.
            env: The environment object.

        Returns:
            log_state_flows: Log state flows.
            full_mask: A boolean tensor representing full states.
        """
        states = trajectories.states
        log_state_flows = torch.full_like(log_pf_trajectories, fill_value=-float("inf"))
        mask = ~states.is_sink_state
        valid_states = states[mask]

        log_F = self.logF(valid_states).squeeze(-1)
        if self.forward_looking:
            log_rewards = env.log_reward(states).unsqueeze(-1)
            log_F = log_F + log_rewards

        log_state_flows[mask[:-1]] = log_F
        return log_state_flows

    def calculate_masks(
        self,
        log_state_flows: LogStateFlowsTensor,
        trajectories: Trajectories,
    ) -> Tuple[MaskTensor, MaskTensor, MaskTensor]:
        """
        Calculate masks for sink and terminal states.
        """
        sink_states_mask = log_state_flows == -float("inf")
        is_terminal_mask = trajectories.actions.is_exit
        full_mask = sink_states_mask | is_terminal_mask

        return full_mask, sink_states_mask, is_terminal_mask

    def get_scores(
        self, env: Env, trajectories: Trajectories
    ) -> Tuple[List[TT[0, float]], List[TT[0, float]]]:
        """Scores all submitted trajectories.

        Returns:
            - A list of tensors, each of which representing the scores of all
                sub-trajectories of length k, for k in `[1, ...,
                trajectories.max_length]`, where the score of a sub-trajectory tau is
                $log P_F(tau) + log F(tau_0) - log P_B(tau) - log F(tau_{-1})$. The
                shape of the k-th tensor is `(trajectories.max_length - k + 1,
                trajectories.n_trajectories)`, k starting from 1.
            - A list of tensors representing what should be masked out in the each
                element of the first list, given that not all sub-trajectories
                of length k exist for each trajectory. The entries of those tensors are
                True if the corresponding sub-trajectory does not exist.
        """
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, fill_value=-float("inf")
        )

        log_pf_trajectories_cum = self.cumulative_logprobs(
            trajectories, log_pf_trajectories
        )
        log_pb_trajectories_cum = self.cumulative_logprobs(
            trajectories, log_pb_trajectories
        )

        log_state_flows = self.calculate_log_state_flows(
            env, trajectories, log_pf_trajectories
        )
        full_mask, sink_states_mask, is_terminal_mask = self.calculate_masks(
            log_state_flows, trajectories
        )

        flattening_masks = []
        scores = []
        for i in range(1, 1 + trajectories.max_length):
            preds = self.calculate_preds(log_pf_trajectories_cum, log_state_flows, i)
            targets = self.calculate_targets(
                trajectories,
                preds,
                log_pb_trajectories_cum,
                log_state_flows,
                is_terminal_mask,
                sink_states_mask,
                full_mask,
                i,
            )

            flattening_mask = trajectories.when_is_done.lt(
                torch.arange(
                    i,
                    trajectories.max_length + 1,
                    device=trajectories.when_is_done.device,
                ).unsqueeze(-1)
            )

            flat_preds = preds[~flattening_mask]
            if torch.any(torch.isnan(flat_preds)):
                raise ValueError("NaN in preds")

            flat_targets = targets[~flattening_mask]
            if torch.any(torch.isnan(flat_targets)):
                raise ValueError("NaN in targets")

            flattening_masks.append(flattening_mask)
            scores.append(preds - targets)

        return (scores, flattening_masks)

    def get_equal_within_contributions(
        self, trajectories: Trajectories
    ) -> ContributionsTensor:
        """
        Calculates contributions for the 'equal_within' weighting method.
        """
        is_done = trajectories.when_is_done
        max_len = trajectories.max_length
        n_rows = int(max_len * (1 + max_len) / 2)

        # the following tensor represents the inverse of how many sub-trajectories there are in each trajectory
        contributions = 2.0 / (is_done * (is_done + 1)) / len(trajectories)

        # if we repeat the previous tensor, we get a tensor of shape
        # (max_len * (max_len + 1) / 2, n_trajectories) that we can multiply with
        # all_scores to get a loss where each sub-trajectory is weighted equally
        # within each trajectory.
        contributions = contributions.repeat(n_rows, 1)

        return contributions

    def get_equal_contributions(
        self, trajectories: Trajectories
    ) -> ContributionsTensor:
        """
        Calculates contributions for the 'equal' weighting method.
        """
        is_done = trajectories.when_is_done
        max_len = trajectories.max_length
        n_rows = int(max_len * (1 + max_len) / 2)
        n_sub_trajectories = int((is_done * (is_done + 1) / 2).sum().item())
        contributions = torch.ones(n_rows, len(trajectories)) / n_sub_trajectories
        return contributions

    def get_tb_contributions(
        self, trajectories: Trajectories, all_scores: TT
    ) -> ContributionsTensor:
        """
        Calculates contributions for the 'TB' weighting method.
        """
        max_len = trajectories.max_length
        is_done = trajectories.when_is_done

        # Each trajectory contributes one element to the loss, equally weighted
        contributions = torch.zeros_like(all_scores)
        indices = (max_len * (is_done - 1) - (is_done - 1) * (is_done - 2) / 2).long()
        contributions.scatter_(0, indices.unsqueeze(0), 1)
        contributions = contributions / len(trajectories)

        return contributions

    def get_modified_db_contributions(
        self, trajectories: Trajectories
    ) -> ContributionsTensor:
        """
        Calculates contributions for the 'ModifiedDB' weighting method.
        """
        is_done = trajectories.when_is_done
        max_len = trajectories.max_length
        n_rows = int(max_len * (1 + max_len) / 2)

        # The following tensor represents the inverse of how many transitions
        # there are in each trajectory.
        contributions = (1.0 / is_done / len(trajectories)).repeat(max_len, 1)
        contributions = torch.cat(
            (
                contributions,
                torch.zeros(
                    (n_rows - max_len, len(trajectories)),
                    device=contributions.device,
                ),
            ),
            0,
        )
        return contributions

    def get_geometric_within_contributions(
        self, trajectories: Trajectories
    ) -> ContributionsTensor:
        """
        Calculates contributions for the 'geometric_within' weighting method.
        """
        L = self.lamda
        max_len = trajectories.max_length
        is_done = trajectories.when_is_done

        # The following tensor represents the weights given to each possible
        # sub-trajectory length.
        contributions = (L ** torch.arange(max_len).double()).float()
        contributions = contributions.unsqueeze(-1).repeat(1, len(trajectories))
        contributions = contributions.repeat_interleave(
            torch.arange(max_len, 0, -1),
            dim=0,
            output_size=int(max_len * (max_len + 1) / 2),
        )

        # Now we need to divide each column by n + (n-1) lambda +...+ 1*lambda^{n-1}
        # where n is the length of the trajectory corresponding to that column
        # We can do it the ugly way, or using the cool identity:
        # https://www.wolframalpha.com/input?i=sum%28%28n-i%29+*+lambda+%5Ei%2C+i%3D0..n%29
        per_trajectory_denom = (
            1.0
            / (1 - L) ** 2
            * (L * (L ** is_done.double() - 1) + (1 - L) * is_done.double())
        ).float()
        contributions = contributions / per_trajectory_denom / len(trajectories)

        return contributions

    def loss(self, env: Env, trajectories: Trajectories) -> TT[0, float]:
        # Get all scores and masks from the trajectories.
        scores, flattening_masks = self.get_scores(env, trajectories)
        flattening_mask = torch.cat(flattening_masks)
        all_scores = torch.cat(scores, 0)

        if self.weighting == "DB":
            # Longer trajectories contribute more to the loss
            return scores[0][~flattening_masks[0]].pow(2).mean()

        elif self.weighting == "geometric":
            # The position i of the following 1D tensor represents the number of sub-
            # trajectories of length i in the batch.
            # n_sub_trajectories = torch.maximum(
            #     trajectories.when_is_done - torch.arange(3).unsqueeze(-1),
            #     torch.tensor(0),
            # ).sum(1)

            # The following tensor's k-th entry represents the mean of all losses of
            # sub-trajectories of length k.
            per_length_losses = torch.stack(
                [
                    scores[~flattening_mask].pow(2).mean()
                    for scores, flattening_mask in zip(scores, flattening_masks)
                ]
            )
            max_len = trajectories.max_length
            L = self.lamda
            ratio = (1 - L) / (1 - L**max_len)
            weights = ratio * (
                L ** torch.arange(max_len, device=per_length_losses.device)
            )
            assert (weights.sum() - 1.0).abs() < 1e-5, f"{weights.sum()}"
            return (per_length_losses * weights).sum()

        elif self.weighting == "equal_within":
            contributions = self.get_equal_within_contributions(trajectories)

        elif self.weighting == "equal":
            contributions = self.get_equal_contributions(trajectories)

        elif self.weighting == "TB":
            contributions = self.get_tb_contributions(trajectories, all_scores)

        elif self.weighting == "ModifiedDB":
            contributions = self.get_modified_db_contributions(trajectories)

        elif self.weighting == "geometric_within":
            contributions = self.get_geometric_within_contributions(trajectories)

        else:
            raise ValueError(f"Unknown weighting method {self.weighting}")

        flat_contributions = contributions[~flattening_mask]
        assert (
            flat_contributions.sum() - 1.0
        ).abs() < 1e-5, f"{flat_contributions.sum()}"
        losses = flat_contributions * all_scores[~flattening_mask].pow(2)
        return losses.sum()
