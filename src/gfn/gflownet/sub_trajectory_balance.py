import math
import warnings
from typing import List, Literal, Tuple, TypeAlias

import torch

from gfn.containers import Trajectories
from gfn.env import ConditionalEnv, Env
from gfn.estimators import ConditionalScalarEstimator, Estimator, ScalarEstimator
from gfn.gflownet.base import TrajectoryBasedGFlowNet, loss_reduce
from gfn.utils.handlers import (
    has_conditions_exception_handler,
    no_conditions_exception_handler,
    warn_about_recalculating_logprobs,
)

ContributionsTensor: TypeAlias = (
    torch.Tensor
)  # shape: [maxlen * (1 + maxlen) / 2, n_traj]
CumulativeLogProbsTensor: TypeAlias = torch.Tensor  # shape: [maxlen + 1, n_traj]
LogStateFlowsTensor: TypeAlias = torch.Tensor  # shape: [maxlen, n_traj]
LogTrajectoriesTensor: TypeAlias = torch.Tensor  # shape: [maxlen, n_traj]
MaskTensor: TypeAlias = torch.Tensor  # shape: [maxlen, n_traj]
PredictionsTensor: TypeAlias = torch.Tensor  # shape: [maxlen + 1 - i, n_traj]
TargetsTensor: TypeAlias = torch.Tensor  # shape: [maxlen + 1 - i, n_traj]


class SubTBGFlowNet(TrajectoryBasedGFlowNet):
    r"""GFlowNet for the Sub-Trajectory Balance loss.

    An implementation of the sub-trajectory balance loss as described in
    [Learning GFlowNets from partial episodes for improved convergence and stability](https://arxiv.org/abs/2209.12782).

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
            pb is therefore always 1.
        logF: A ScalarEstimator or ConditionalScalarEstimator for estimating the log flow
            of the states.
        weighting: The sub-trajectories weighting scheme.
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
        lamda: Discount factor for longer trajectories (used in geometric weighting).
        log_reward_clip_min: If finite, clips log rewards to this value.
        forward_looking: Whether to use the forward-looking GFN loss.
        constant_pb: Whether to ignore the backward policy estimator, e.g., if the
            gflownet DAG is a tree, and pb is therefore always 1.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        logF: ScalarEstimator | ConditionalScalarEstimator,
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
        constant_pb: bool = False,
    ):
        """Initializes a SubTBGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator.
            logF: A ScalarEstimator or ConditionalScalarEstimator for estimating the
                log flow of the states.
            weighting: The sub-trajectory weighting scheme (see class docstring for
                details).
            lamda: Discount factor for longer trajectories (used in geometric weighting).
            log_reward_clip_min: If finite, clips log rewards to this value.
            forward_looking: Whether to use the forward-looking GFN loss.
            constant_pb: Whether to ignore the backward policy estimator, e.g., if the
                gflownet DAG is a tree, and pb is therefore always 1. Must be set
                explicitly by user to ensure that pb is an Estimator except under this
                special case.

        """
        super().__init__(pf, pb, constant_pb=constant_pb)
        assert any(
            isinstance(logF, cls)
            for cls in [ScalarEstimator, ConditionalScalarEstimator]
        ), "logF must be a ScalarEstimator or derived"
        self.logF = logF
        self.weighting = weighting
        self.lamda = lamda
        self.log_reward_clip_min = log_reward_clip_min
        self.forward_looking = forward_looking

    def logF_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns a dictionary of named parameters containing 'logF' in their name.

        Returns:
            A dictionary of named parameters containing 'logF' in their name.
        """
        return {k: v for k, v in self.named_parameters() if "logF" in k}

    def logF_parameters(self) -> list[torch.Tensor]:
        """Returns a list of parameters containing 'logF' in their name.

        Returns:
            A list of parameters containing 'logF' in their name.
        """
        return [v for k, v in self.named_parameters() if "logF" in k]

    def cumulative_logprobs(
        self,
        trajectories: Trajectories,
        log_p_trajectories: LogTrajectoriesTensor,
    ) -> CumulativeLogProbsTensor:
        """Calculates the cumulative logprobs for all trajectories.

        Args:
            trajectories: The batch of trajectories.
            log_p_trajectories: Tensor of shape (max_length, n_trajectories)
                containing the logprobs of the forward or backward actions of the
                trajectories.

        Returns:
            A tensor of shape (max_length + 1, n_trajectories) containing the
            cumulative sum of logprobs for each trajectory.
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
        log_pf_traj_cum: CumulativeLogProbsTensor,
        log_state_flows: LogStateFlowsTensor,
        i: int,
    ) -> PredictionsTensor:
        """Calculates the predictions tensor for the current sub-trajectory length.

        Args:
            log_pf_traj_cum: Tensor of shape (max_length + 1, n_trajectories)
                containing the cumulative sum of logprobs of the forward actions for
                each trajectory.
            log_state_flows: Tensor of shape (max_length, n_trajectories) containing
                the estimated log flow of the states.
            i: The sub-trajectory length.

        Returns:
            The predictions tensor of shape (max_length + 1 - i, n_trajectories).
        """
        current_log_state_flows = (
            log_state_flows if i == 1 else log_state_flows[: -(i - 1)]
        )

        preds = log_pf_traj_cum[i:] - log_pf_traj_cum[:-i] + current_log_state_flows

        return preds

    def calculate_targets(
        self,
        trajectories: Trajectories,
        preds: PredictionsTensor,
        log_pb_traj_cum: CumulativeLogProbsTensor,
        log_state_flows: LogStateFlowsTensor,
        is_terminal_mask: MaskTensor,
        sink_states_mask: MaskTensor,
        i: int,
    ) -> TargetsTensor:
        """Calculates the targets tensor for the current sub-trajectory length.

        Args:
            trajectories: The batch of trajectories.
            preds: Tensor of shape (max_length + 1 - i, n_trajectories) containing
                the predictions for the current sub-trajectory length.
            log_pb_traj_cum: Tensor of shape (max_length + 1, n_trajectories)
                containing the cumulative sum of logprobs of the backward actions for
                each trajectory.
            log_state_flows: Tensor of shape (max_length, n_trajectories) containing
                the estimated log flow of the states.
            is_terminal_mask: A mask of shape (max_length, n_trajectories) indicating
                whether the state is terminal.
            sink_states_mask: A mask of shape (max_length, n_trajectories) indicating
                whether the state is a sink state.
            i: The sub-trajectory length.

        Returns:
            The targets tensor of shape (max_length + 1 - i, n_trajectories).
        """
        targets = torch.full_like(preds, fill_value=-float("inf"))
        assert trajectories.log_rewards is not None
        log_rewards = trajectories.log_rewards[trajectories.terminating_idx >= i]

        if math.isfinite(self.log_reward_clip_min):
            log_rewards.clamp_min(self.log_reward_clip_min)

        targets.T[is_terminal_mask[i - 1 :].T] = log_rewards

        # For now, the targets contain the log-rewards of the ending sub trajectories
        # We need to add to that the log-probabilities of the backward actions up-to
        # the sub-trajectory's terminating state
        if i > 1:
            delta_pb = (log_pb_traj_cum[i - 1 :] - log_pb_traj_cum[: -i + 1])[:-1]
            targets[is_terminal_mask[i - 1 :]] += delta_pb[is_terminal_mask[i - 1 :]]

        # The following creates the targets for the non-finishing sub-trajectories
        full_mask = sink_states_mask | is_terminal_mask
        delta_pb2 = (log_pb_traj_cum[i:] - log_pb_traj_cum[:-i])[:-1]
        rhs_mask = ~full_mask[i - 1 : -1]
        targets[~full_mask[i - 1 :]] = (
            delta_pb2[rhs_mask] + log_state_flows[i:][~sink_states_mask[i:]]
        )

        return targets

    def calculate_log_state_flows(
        self,
        env: Env,
        trajectories: Trajectories,
        log_pf_trajectories: LogTrajectoriesTensor,
    ) -> LogStateFlowsTensor:
        """Calculates log flows of each state in the trajectories.

        Args:
            env: The environment object.
            trajectories: The batch of trajectories.
            log_pf_trajectories: Tensor of shape (max_length, n_trajectories) containing
                the logprobs of the forward actions of the trajectories.

        Returns:
            A tensor of shape (max_length, n_trajectories) containing the log flows of
            each state in the trajectories.
        """
        states = trajectories.states
        log_state_flows = torch.full_like(log_pf_trajectories, fill_value=-float("inf"))
        mask = ~states.is_sink_state
        valid_states = states[mask]

        if trajectories.conditions is not None:
            # Compute the condition matrix broadcast to match valid_states.
            # The conditions tensor has shape (n_trajectories, condition_vector_dim)
            # We need to repeat it to match the batch shape of the states
            conditions = trajectories.conditions.repeat(states.batch_shape[0], 1, 1)
            assert conditions.shape[:2] == states.batch_shape
            conditions = conditions[mask]
            with has_conditions_exception_handler("logF", self.logF):
                log_F = self.logF(valid_states, conditions).squeeze(-1)

            if self.forward_looking:
                assert isinstance(env, ConditionalEnv)
                log_F = log_F + env.log_reward(valid_states, conditions)

        else:
            with no_conditions_exception_handler("logF", self.logF):
                log_F = self.logF(valid_states).squeeze(-1)

            if self.forward_looking:
                log_F = log_F + env.log_reward(valid_states)

        log_state_flows[mask[:-1]] = log_F
        return log_state_flows

    def calculate_masks(
        self,
        log_state_flows: LogStateFlowsTensor,
        trajectories: Trajectories,
    ) -> Tuple[MaskTensor, MaskTensor]:
        """Calculates masks indicating sink and terminal states.

        Args:
            log_state_flows: Tensor of shape (max_length, n_trajectories) containing
                the log flows of the states.
            trajectories: The batch of trajectories.

        Returns:
            A tuple of two mask tensors (sink_states_mask, is_terminal_mask), each of
            shape (max_length, n_trajectories).
        """
        sink_states_mask = log_state_flows == -float("inf")
        is_terminal_mask = trajectories.actions.is_exit

        return sink_states_mask, is_terminal_mask

    def get_scores(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""Computes sub-trajectory balance scores for all submitted trajectories.

        Args:
            env: The environment where the trajectories are sampled from.
            trajectories: The batch of trajectories to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns:
            A tuple (scores, flattening_masks):
                - scores: A list of tensors, each representing the scores of all
                    sub-trajectories of length k, for k in [1, ..., max_length], where
                    the score of a sub-trajectory $\tau_{n:n+k} = (s_n, ..., s_{n+k})$ is
                    $\log P_F(\tau_{n:n+k}) + \log F(s_n) - \log P_B(\tau_{n:n+k}) - \log F(s_{n+k})$.
                    The shape of each score from k-length sub-trajectory is
                    (max_length - k + 1, n_trajectories).
                - flattening_masks: A list of tensors indicating what should be masked out
                    from the each element of the first list (scores), given that not all
                    sub-trajectories of length k exist for each trajectory. The entries of
                    those tensors are True if the corresponding sub-trajectory does not
                    exist.
        """
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories,
            recalculate_all_logprobs=recalculate_all_logprobs,
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
        sink_states_mask, is_terminal_mask = self.calculate_masks(
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
                i,
            )

            flattening_mask = trajectories.terminating_idx.lt(
                torch.arange(
                    i,
                    trajectories.max_length + 1,
                    device=trajectories.terminating_idx.device,
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

        return scores, flattening_masks

    def get_equal_within_contributions(
        self, trajectories: Trajectories
    ) -> ContributionsTensor:
        """Calculates contributions for the 'equal_within' weighting method.

        Args:
            trajectories: The batch of trajectories.

        Returns:
            The contributions tensor of shape (max_len * (max_len+1) / 2, n_trajectories).
        """
        terminating_idx = trajectories.terminating_idx
        max_len = trajectories.max_length
        n_rows = int(max_len * (1 + max_len) / 2)

        # the following tensor represents the inverse of how many sub-trajectories there are in each trajectory
        contributions = (
            2.0 / (terminating_idx * (terminating_idx + 1)) / len(trajectories)
        )

        # if we repeat the previous tensor, we get a tensor of shape
        # (max_len * (max_len + 1) / 2, n_trajectories) that we can multiply with
        # all_scores to get a loss where each sub-trajectory is weighted equally
        # within each trajectory.
        contributions = contributions.repeat(n_rows, 1)

        return contributions

    def get_equal_contributions(self, trajectories: Trajectories) -> ContributionsTensor:
        """Calculates contributions for the 'equal' weighting method.

        Args:
            trajectories: The batch of trajectories.

        Returns:
            The contributions tensor of shape (max_len * (max_len+1) / 2, n_trajectories).
        """
        terminating_idx = trajectories.terminating_idx
        max_len = trajectories.max_length
        n_rows = int(max_len * (1 + max_len) / 2)
        n_sub_trajectories = int(
            (terminating_idx * (terminating_idx + 1) / 2).sum().item()
        )
        contributions = torch.ones(n_rows, len(trajectories)) / n_sub_trajectories
        return contributions

    def get_tb_contributions(self, trajectories: Trajectories) -> ContributionsTensor:
        """Calculates contributions for the 'TB' weighting method.

        Args:
            trajectories: The batch of trajectories.

        Returns:
            The contributions tensor of shape (max_len * (max_len+1) / 2, n_trajectories).
        """
        max_len = trajectories.max_length
        n_rows = int(max_len * (1 + max_len) / 2)
        contributions = torch.zeros(n_rows, len(trajectories))

        # Each trajectory contributes one element to the loss, equally weighted
        t_idx = trajectories.terminating_idx
        indices = (max_len * (t_idx - 1) - (t_idx - 1) * (t_idx - 2) / 2).long()
        contributions.scatter_(0, indices.unsqueeze(0), 1)
        contributions = contributions / len(trajectories)

        return contributions

    def get_modified_db_contributions(
        self, trajectories: Trajectories
    ) -> ContributionsTensor:
        """Calculates contributions for the 'ModifiedDB' weighting method.

        Args:
            trajectories: The batch of trajectories.

        Returns:
            The contributions tensor of shape (max_len * (max_len+1) / 2, n_trajectories).
        """
        terminating_idx = trajectories.terminating_idx
        max_len = trajectories.max_length
        n_rows = int(max_len * (1 + max_len) / 2)

        # The following tensor represents the inverse of how many transitions
        # there are in each trajectory.
        contributions = (1.0 / terminating_idx / len(trajectories)).repeat(max_len, 1)
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
        """Calculates contributions for the 'geometric_within' weighting method.

        Args:
            trajectories: The batch of trajectories.

        Returns:
            The contributions tensor of shape (max_len * (max_len+1) / 2, n_trajectories).
        """
        L = self.lamda
        max_len = trajectories.max_length
        t_idx = trajectories.terminating_idx

        # The following tensor represents the weights given to each possible
        # sub-trajectory length.
        contributions = (L ** torch.arange(max_len, device=t_idx.device).double()).to(
            torch.get_default_dtype()
        )
        contributions = contributions.unsqueeze(-1).repeat(1, len(trajectories))
        contributions = contributions.repeat_interleave(
            torch.arange(max_len, 0, -1, device=t_idx.device),
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
            * (L * (L ** t_idx.double() - 1) + (1 - L) * t_idx.double())
        ).to(torch.get_default_dtype())
        contributions = contributions / per_trajectory_denom / len(trajectories)

        return contributions

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the sub-trajectory balance loss.

        Args:
            env: The environment where the trajectories are sampled from.
            trajectories: The batch of trajectories to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed sub-trajectory balance loss as a tensor. The shape depends on
            the reduction method.
        """
        warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        # Get all scores and masks from the trajectories.
        scores, flattening_masks = self.get_scores(
            env, trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        flattening_mask = torch.cat(flattening_masks)
        all_scores = torch.cat(scores, 0)

        if self.weighting == "DB":
            # Longer trajectories contribute more to the loss.
            # TODO: is this correct with `loss_reduce`?
            final_scores = scores[0][~flattening_masks[0]].pow(2)
            return loss_reduce(final_scores, reduction)

        elif self.weighting == "geometric":
            # The position i of the following 1D tensor represents the number of sub-
            # trajectories of length i in the batch.
            # n_sub_trajectories = torch.maximum(
            #     trajectories.terminating_idx - torch.arange(3).unsqueeze(-1),
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

        # TODO: we need to know what reductions are valid for each weighting method.
        weight_functions = {
            "equal_within": self.get_equal_within_contributions,
            "equal": self.get_equal_contributions,
            "TB": self.get_tb_contributions,
            "ModifiedDB": self.get_modified_db_contributions,
            "geometric_within": self.get_geometric_within_contributions,
        }
        try:
            contributions = weight_functions[self.weighting](trajectories)
        except KeyError:
            raise ValueError(f"Unknown weighting method {self.weighting}")

        flat_contributions = contributions[~flattening_mask]
        assert (
            flat_contributions.sum() - 1.0
        ).abs() < 1e-5, f"{flat_contributions.sum()}"
        final_scores = flat_contributions * all_scores[~flattening_mask].pow(2)

        # TODO: default was sum, should we allow mean?
        if reduction == "mean":
            warnings.warn(
                "Mean reduction is not supported for SubTBGFlowNet with geometric "
                "weighting, using sum instead."
            )
            reduction = "sum"

        return loss_reduce(final_scores, reduction)
