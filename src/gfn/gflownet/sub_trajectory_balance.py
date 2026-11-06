import math
import warnings
from typing import List, Literal, Tuple, TypeAlias, cast

import torch

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.estimators import ConditionalScalarEstimator, Estimator, ScalarEstimator
from gfn.gflownet.base import TrajectoryBasedGFlowNet, loss_reduce
from gfn.gflownet.losses import RegressionLoss
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


SubTBWeighting = Literal[
    "DB",
    "ModifiedDB",
    "TB",
    "geometric",
    "equal",
    "geometric_within",
    "equal_within",
]


def _geometric_length_weights(
    lamda: float | torch.Tensor,
    n_lengths: torch.Tensor,
    max_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Returns geometric sub-trajectory length weights, rescaled to at most one.

    Entry ``[k, ...]`` weighs sub-trajectories of length ``k + 1`` among the
    ``n_lengths`` shortest lengths. It is ``lamda ** k`` divided by the largest of
    those weights (``1`` when ``lamda <= 1``, ``lamda ** (n_lengths - 1)``
    otherwise), and 0 for ``k >= n_lengths``. Every weight is thus a non-negative
    integer power of ``min(lamda, 1 / lamda)`` in [0, 1], so none overflows for any
    finite ``lamda >= 0`` and any trajectory length. The rescaling cancels once
    callers normalize, and because the normalizer is built from these same values,
    normalized weights sum to one up to rounding.

    Args:
        lamda: The geometric base, finite and non-negative.
        n_lengths: Integer tensor with the number of lengths to weight, e.g. one
            trajectory length per batch element.
        max_len: The number of lengths in the returned tensor.
        dtype: The floating-point dtype of the returned tensor.

    Returns:
        A tensor of shape (max_len, *n_lengths.shape).
    """
    offsets = torch.arange(max_len, device=n_lengths.device)
    offsets = offsets.view(-1, *([1] * n_lengths.dim()))
    lam = torch.as_tensor(lamda, device=n_lengths.device, dtype=dtype)
    flip = lam > 1
    base = torch.where(flip, lam.reciprocal(), lam)
    exponents = torch.where(flip, n_lengths - 1 - offsets, offsets).clamp(min=0)
    return base.pow(exponents).masked_fill(offsets >= n_lengths, 0.0)


def _subtrajectory_counts(terminating_idx: torch.Tensor, max_len: int) -> torch.Tensor:
    """Returns the number of sub-trajectories of each length in each trajectory.

    Args:
        terminating_idx: The trajectory lengths, of shape (batch_size,).
        max_len: The number of lengths to count.

    Returns:
        A tensor of shape (max_len, batch_size) whose entry ``[k, b]`` is the number
        of sub-trajectories of length ``k + 1`` in trajectory ``b``.
    """
    offsets = torch.arange(max_len, device=terminating_idx.device).unsqueeze(-1)
    return (terminating_idx - offsets).clamp(min=0)


def _repeat_per_start(per_length: torch.Tensor, max_len: int) -> ContributionsTensor:
    """Repeats each length's row once per start position, as laid out by get_scores.

    Args:
        per_length: A tensor whose row ``k`` holds values for length ``k + 1``.
        max_len: The maximum trajectory length.

    Returns:
        A tensor with max_len * (max_len + 1) / 2 rows, where row ``k`` of
        ``per_length`` is repeated ``max_len - k`` times.
    """
    return per_length.repeat_interleave(
        torch.arange(max_len, 0, -1, device=per_length.device),
        dim=0,
        output_size=max_len * (max_len + 1) // 2,
    )


def _assert_normalized(weights: torch.Tensor, name: str) -> None:
    """Checks that ``weights`` are finite, non-negative, and sum to one.

    The sum-to-one tolerance scales with ``weights.dtype``. torch accumulates
    reduced-precision sums in float32, so their error comes from rounding: each
    weight costs at most half an epsilon relative, or half the smallest subnormal
    absolute if it underflows, and the normalization and the total cost about one
    more epsilon. The 1e-5 floor covers float32 summation error over large batches
    and keeps the float32 and float64 checks unchanged.

    Args:
        weights: The weights to check.
        name: A name for the weights, used in error messages.

    Raises:
        AssertionError: If any check fails.
    """
    assert torch.isfinite(weights).all(), f"{name} contain non-finite values."
    assert (weights >= 0).all(), f"{name} contain negative values."
    info = torch.finfo(weights.dtype)
    tol = max(1e-5, info.eps * (2 + weights.numel() * info.tiny))
    total = weights.sum()
    assert (total - 1.0).abs() <= tol, f"{name} sum to {total.item()}, not 1."


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
            - "geometric": Sub-trajectories are grouped by length across the whole
                batch. Each length gets a total weight proportional to
                (lamda ** length), normalized over the lengths present in the
                batch, which is split equally among the sub-trajectories of that
                length. A length's total weight does not depend on how many
                sub-trajectories have it, so the few longest sub-trajectories can
                weigh as much as all one-step transitions. With lamda=1, every
                length gets the same total weight.
        lamda: Base of the geometric weighting over sub-trajectory lengths (used by
            the "geometric" and "geometric_within" weightings). A sub-trajectory of
            length k is weighed proportionally to lamda ** k: values below 1 favor
            short sub-trajectories, 1 weighs all lengths equally, and values above
            1 favor long ones.
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
        weighting: SubTBWeighting = "geometric_within",
        lamda: float = 0.9,
        log_reward_clip_min: float = -float("inf"),
        forward_looking: bool = False,
        constant_pb: bool = False,
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
    ):
        """Initializes a SubTBGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator.
            logF: A ScalarEstimator or ConditionalScalarEstimator for estimating the
                log flow of the states.
            weighting: The sub-trajectory weighting scheme (see class docstring for
                details).
            lamda: Base of the geometric weighting over sub-trajectory lengths (see
                class docstring). Must be finite and non-negative.
            log_reward_clip_min: If finite, clips log rewards to this value.
            forward_looking: Whether to use the forward-looking GFN loss.
            constant_pb: Whether to ignore the backward policy estimator, e.g., if the
                gflownet DAG is a tree, and pb is therefore always 1. Must be set
                explicitly by user to ensure that pb is an Estimator except under this
                special case.
            debug: If True, keep runtime safety checks active; disable in compiled runs.
            loss_fn: Regression loss applied to balance residuals.
                Defaults to :class:`~gfn.gflownet.losses.SquaredLoss`.

        Raises:
            TypeError: If logF is not a ScalarEstimator or ConditionalScalarEstimator.
            ValueError: If lamda is negative or not finite.
        """
        super().__init__(
            pf,
            pb,
            constant_pb=constant_pb,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
        )
        if not isinstance(logF, (ScalarEstimator, ConditionalScalarEstimator)):
            raise TypeError(
                f"logF must be a ScalarEstimator or ConditionalScalarEstimator, "
                f"got {type(logF).__name__}"
            )
        # Negative bases give negative weights, which make the objective unbounded
        # below; NaN or infinite bases give non-finite weights.
        if not (math.isfinite(lamda) and lamda >= 0):
            raise ValueError(f"lamda must be finite and non-negative, got {lamda}.")
        self.logF = logF
        self.weighting = weighting
        self.lamda = lamda
        self.forward_looking = forward_looking

        if debug and hasattr(logF, "debug"):
            logF.debug = True

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
            log_p_trajectories: Tensor of shape (max_length, batch_size)
                containing the logprobs of the forward or backward actions of the
                trajectories.

        Returns:
            A tensor of shape (max_length + 1, batch_size) containing the
            cumulative sum of logprobs for each trajectory.
        """
        return torch.cat(
            (
                torch.zeros(
                    1, trajectories.batch_size, device=log_p_trajectories.device
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
            log_pf_traj_cum: Tensor of shape (max_length + 1, batch_size)
                containing the cumulative sum of logprobs of the forward actions for
                each trajectory.
            log_state_flows: Tensor of shape (max_length, batch_size) containing
                the estimated log flow of the states.
            i: The sub-trajectory length.

        Returns:
            The predictions tensor of shape (max_length + 1 - i, batch_size).
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
        log_rewards: torch.Tensor | None = None,
    ) -> TargetsTensor:
        """Calculates the targets tensor for the current sub-trajectory length.

        Args:
            trajectories: The batch of trajectories.
            preds: Tensor of shape (max_length + 1 - i, batch_size) containing
                the predictions for the current sub-trajectory length.
            log_pb_traj_cum: Tensor of shape (max_length + 1, batch_size)
                containing the cumulative sum of logprobs of the backward actions for
                each trajectory.
            log_state_flows: Tensor of shape (max_length, batch_size) containing
                the estimated log flow of the states.
            is_terminal_mask: A mask of shape (max_length, batch_size) indicating
                whether the state is terminal.
            sink_states_mask: A mask of shape (max_length, batch_size) indicating
                whether the state is a sink state.
            i: The sub-trajectory length.
            log_rewards: Optional custom log rewards tensor of shape
                (n_trajectories,). When None, uses the environment rewards.
                Useful for intrinsic rewards (see
                "Towards Improving Exploration through Sibling Augmented
                GFlowNets", Madan et al., ICLR 2025).

        Returns:
            The targets tensor of shape (max_length + 1 - i, batch_size).
        """
        targets = torch.full_like(preds, fill_value=-float("inf"))
        if log_rewards is None:
            # Guard behind debug: log_rewards is always set for terminating trajectories;
            # bare assert would cause a graph break in torch.compile.
            if self.debug:
                assert trajectories.log_rewards is not None
            # cast: log_rewards is always set for terminating trajectories.
            log_rewards = cast(torch.Tensor, trajectories.log_rewards)
        if self.debug:
            assert log_rewards.shape == (trajectories.batch_size,)
        log_rewards = log_rewards[trajectories.terminating_idx >= i]

        if math.isfinite(self.log_reward_clip_min):
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)

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

    def _resolve_logprobs(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool,
        log_pf_trajectories: torch.Tensor | None,
        log_pb_trajectories: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns caller-supplied logprobs, or evaluates them from the policies.

        Args:
            trajectories: The batch of trajectories.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            log_pf_trajectories: Optional precomputed forward logprobs.
            log_pb_trajectories: Optional precomputed backward logprobs.

        Returns:
            A tuple ``(log_pf, log_pb)``, each of shape (max_length, n_trajectories).

        Raises:
            ValueError: If only one of the two tensors is supplied, or if a supplied
                tensor has the wrong shape.
        """
        if log_pf_trajectories is None and log_pb_trajectories is None:
            return self.get_pfs_and_pbs(
                trajectories,
                recalculate_all_logprobs=recalculate_all_logprobs,
            )
        if log_pf_trajectories is None or log_pb_trajectories is None:
            raise ValueError(
                "log_pf_trajectories and log_pb_trajectories must be supplied together."
            )
        expected = (trajectories.max_length, trajectories.batch_size)
        for name, tensor in (
            ("log_pf_trajectories", log_pf_trajectories),
            ("log_pb_trajectories", log_pb_trajectories),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)}, expected {expected}."
                )
        return log_pf_trajectories, log_pb_trajectories

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
            log_pf_trajectories: Tensor of shape (max_length, batch_size) containing
                the logprobs of the forward actions of the trajectories.

        Returns:
            A tensor of shape (max_length, batch_size) containing the log flows of
            each state in the trajectories.
        """
        states = trajectories.states
        log_state_flows = torch.full_like(log_pf_trajectories, fill_value=-float("inf"))
        mask = ~states.is_sink_state
        valid_states = states[mask]

        if trajectories.states.conditions is not None:
            # Compute the condition matrix broadcast to match valid_states.
            # The conditions tensor has shape (batch_size, condition_dim)
            # The states have batch shape (max_length, batch_size)
            # We need to repeat the conditions to match the batch shape of the states.
            conditions = trajectories.states.conditions
            # (max_length, batch_size, condition_dim)
            # Guard behind debug: shape assertion causes a graph break in torch.compile;
            # conditions shape is structurally guaranteed by States construction.
            if self.debug:
                assert conditions.shape[:2] == states.batch_shape
            conditions = conditions[mask]
            with has_conditions_exception_handler("logF", self.logF):
                log_F = self.logF(valid_states, conditions).squeeze(-1)

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
            log_state_flows: Tensor of shape (max_length, batch_size) containing
                the log flows of the states.
            trajectories: The batch of trajectories.

        Returns:
            A tuple of two mask tensors (sink_states_mask, is_terminal_mask), each of
            shape (max_length, batch_size).
        """
        sink_states_mask = log_state_flows == -float("inf")
        is_terminal_mask = trajectories.actions.is_exit

        return sink_states_mask, is_terminal_mask

    def get_scores(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        env: Env | None = None,
        *,
        log_rewards: torch.Tensor | None = None,
        log_pf_trajectories: torch.Tensor | None = None,
        log_pb_trajectories: torch.Tensor | None = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""Computes sub-trajectory balance scores for all submitted trajectories.

        Args:
            trajectories: The batch of trajectories to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            env: The environment where the trajectories are sampled from.
            log_rewards: Optional custom log rewards tensor of shape
                (n_trajectories,). When None, uses the environment rewards.
                Useful for intrinsic rewards (see
                "Towards Improving Exploration through Sibling Augmented
                GFlowNets", Madan et al., ICLR 2025).
            log_pf_trajectories: Optional precomputed forward logprobs of shape
                (max_length, n_trajectories). When given, the forward policy is not
                re-evaluated. Passing detached logprobs freezes the policy so only
                ``logF`` receives gradient — this is what the Sub-EB critic of
                :class:`~gfn.gflownet.policy_gradient.PolicyGradientGFlowNet` needs.
                Both ``log_pf_trajectories`` and ``log_pb_trajectories`` must be
                supplied together.
            log_pb_trajectories: Optional precomputed backward logprobs, same shape
                and semantics as ``log_pf_trajectories``.

        Returns:
            A tuple (scores, flattening_masks):
                - scores: A list of tensors, each representing the scores of all
                    sub-trajectories of length k, for k in [1, ..., max_length], where
                    the score of a sub-trajectory $\tau_{n:n+k} = (s_n, ..., s_{n+k})$ is
                    $\log P_F(\tau_{n:n+k}) + \log F(s_n) - \log P_B(\tau_{n:n+k}) - \log F(s_{n+k})$.
                    The shape of each score from k-length sub-trajectory is
                    (max_length - k + 1, batch_size).
                - flattening_masks: A list of tensors indicating what should be masked out
                    from the each element of the first list (scores), given that not all
                    sub-trajectories of length k exist for each trajectory. The entries of
                    those tensors are True if the corresponding sub-trajectory does not
                    exist.
        """
        log_pf_trajectories, log_pb_trajectories = self._resolve_logprobs(
            trajectories,
            recalculate_all_logprobs,
            log_pf_trajectories,
            log_pb_trajectories,
        )

        log_pf_trajectories_cum = self.cumulative_logprobs(
            trajectories, log_pf_trajectories
        )
        log_pb_trajectories_cum = self.cumulative_logprobs(
            trajectories, log_pb_trajectories
        )

        assert env is not None, "SubTBGFlowNet.get_scores requires env"
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
                log_rewards=log_rewards,
            )

            flattening_mask = trajectories.terminating_idx.lt(
                torch.arange(
                    i,
                    trajectories.max_length + 1,
                    device=trajectories.terminating_idx.device,
                ).unsqueeze(-1)
            )

            if self.debug:
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
            The contributions tensor of shape (max_len * (max_len+1) / 2, batch_size).
        """
        terminating_idx = trajectories.terminating_idx
        max_len = trajectories.max_length
        n_rows = int(max_len * (1 + max_len) / 2)

        # the following tensor represents the inverse of how many sub-trajectories there are in each trajectory
        contributions = (
            2.0 / (terminating_idx * (terminating_idx + 1)) / len(trajectories)
        )

        # if we repeat the previous tensor, we get a tensor of shape
        # (max_len * (max_len + 1) / 2, batch_size) that we can multiply with
        # all_scores to get a loss where each sub-trajectory is weighted equally
        # within each trajectory.
        contributions = contributions.repeat(n_rows, 1)

        return contributions

    def get_equal_contributions(self, trajectories: Trajectories) -> ContributionsTensor:
        """Calculates contributions for the 'equal' weighting method.

        Args:
            trajectories: The batch of trajectories.

        Returns:
            The contributions tensor of shape (max_len * (max_len+1) / 2, batch_size).
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
            The contributions tensor of shape (max_len * (max_len+1) / 2, batch_size).
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
            The contributions tensor of shape (max_len * (max_len+1) / 2, batch_size).
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
            The contributions tensor of shape (max_len * (max_len+1) / 2, batch_size).
        """
        t_idx = trajectories.terminating_idx
        max_len = trajectories.max_length
        # weights[k, b] and counts[k, b] are the weight and the number of
        # length-(k + 1) sub-trajectories in trajectory b.
        weights = _geometric_length_weights(
            self.lamda, t_idx, max_len, torch.get_default_dtype()
        )
        counts = _subtrajectory_counts(t_idx, max_len)

        # Normalize within each trajectory, then weigh trajectories equally.
        per_length = weights / (counts * weights).sum(0) / len(trajectories)
        return _repeat_per_start(per_length, max_len)

    def get_geometric_contributions(
        self, trajectories: Trajectories
    ) -> ContributionsTensor:
        """Calculates contributions for the 'geometric' weighting method.

        Each length gets a total weight proportional to lamda ** length, normalized
        over the lengths present in the batch, and split equally among the
        sub-trajectories of that length. The weighted sum of the per-sub-trajectory
        losses is therefore a weighted mean of the per-length mean losses.

        Args:
            trajectories: The batch of trajectories.

        Returns:
            The contributions tensor of shape (max_len * (max_len+1) / 2, batch_size).
        """
        t_idx = trajectories.terminating_idx
        max_len = trajectories.max_length
        # Every length up to the longest trajectory's occurs in the batch. Longer
        # lengths only exist in a container padded beyond its longest trajectory;
        # they get zero weight, so the present lengths' weights still sum to one.
        weights = _geometric_length_weights(
            self.lamda, t_idx.max(), max_len, torch.get_default_dtype()
        )
        counts = _subtrajectory_counts(t_idx, max_len).sum(-1)

        per_length = weights / weights.sum() / counts.clamp(min=1)
        contributions = _repeat_per_start(per_length, max_len)
        return contributions.unsqueeze(-1).expand(-1, len(trajectories))

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
        *,
        log_rewards: torch.Tensor | None = None,
        log_pf_trajectories: torch.Tensor | None = None,
        log_pb_trajectories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the sub-trajectory balance loss.

        Args:
            env: The environment where the trajectories are sampled from.
            trajectories: The batch of trajectories to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').
                With 'DB' and 'TB' weighting, the per-transition or per-trajectory
                losses are reduced as requested. Every other weighting scales each
                sub-trajectory's loss by its normalized weight, so the loss is their
                sum: 'mean' is treated as 'sum' (a warning is emitted when
                debug=True), and 'none' returns the weighted per-sub-trajectory
                losses, which sum to the loss.
            log_rewards: Optional custom log rewards tensor of shape
                (n_trajectories,). When None, uses the environment rewards.
                When provided, this overrides the terminal reward term used by
                the loss. In particular, for ``forward_looking=True``, the
                state-flow computation may still depend on ``env.log_reward(...)``,
                so custom ``log_rewards`` do not fully replace environment
                rewards in that mode. Useful for intrinsic rewards affecting the
                terminal boundary term (see "Towards Improving Exploration
                through Sibling Augmented GFlowNets", Madan et al., ICLR 2025).
            log_pf_trajectories: Optional precomputed forward logprobs of shape
                (max_length, n_trajectories); see :meth:`get_scores`. Passing detached
                logprobs freezes the policies so only ``logF`` receives gradient.
            log_pb_trajectories: Optional precomputed backward logprobs, same shape
                and semantics as ``log_pf_trajectories``.

        Returns:
            The computed sub-trajectory balance loss as a tensor. The shape depends on
            the reduction method.
        """
        if self.debug:
            warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)

        if self.weighting == "TB":
            # TB weighting is mathematically equivalent to TBGFlowNet (with logZ
            # replaced by logF(s0)).  Computing via cumsum (used in get_scores) can
            # give different float32 results than TBGFlowNet's direct sum(dim=0),
            # because cumsum is strictly sequential while sum(dim=0) may use SIMD
            # vector reduction. For continuous environments with large log-prob
            # magnitudes (e.g. Box), this difference can exceed typical tolerances.
            # Bypass get_scores entirely and mirror TBGFlowNet's exact computation.
            log_pf_traj, log_pb_traj = self._resolve_logprobs(
                trajectories,
                recalculate_all_logprobs,
                log_pf_trajectories,
                log_pb_trajectories,
            )
            log_state_flows = self.calculate_log_state_flows(
                env, trajectories, log_pf_traj
            )
            # logF of the source state (row 0); -inf means sink — treat as 0
            logF_s0 = log_state_flows[0].masked_fill(log_state_flows[0].isinf(), 0.0)
            total_log_pf = log_pf_traj.sum(dim=0)
            total_log_pb = log_pb_traj.sum(dim=0)
            if log_rewards is None:
                # Guard behind debug: log_rewards is always set for terminating
                # trajectories; bare assert would cause a graph break in torch.compile.
                if self.debug:
                    assert trajectories.log_rewards is not None
                # cast: log_rewards is always set for terminating trajectories.
                log_rewards = cast(torch.Tensor, trajectories.log_rewards)
            if self.debug:
                assert log_rewards.shape == (trajectories.batch_size,)
            if math.isfinite(self.log_reward_clip_min):
                log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)
            tb_scores = logF_s0 + total_log_pf - total_log_pb - log_rewards
            return loss_reduce(self.loss_fn(tb_scores), reduction)

        # Get all scores and masks from the trajectories.
        scores, flattening_masks = self.get_scores(
            trajectories,
            log_rewards=log_rewards,
            recalculate_all_logprobs=recalculate_all_logprobs,
            env=env,
            log_pf_trajectories=log_pf_trajectories,
            log_pb_trajectories=log_pb_trajectories,
        )
        flattening_mask = torch.cat(flattening_masks)
        all_scores = torch.cat(scores, 0)

        if self.weighting == "DB":
            # Longer trajectories contribute more to the loss.
            # TODO: is this correct with `loss_reduce`?
            final_scores = self.loss_fn(scores[0][~flattening_masks[0]])
            return loss_reduce(final_scores, reduction)

        # TODO: we need to know what reductions are valid for each weighting method.
        weight_functions = {
            "equal_within": self.get_equal_within_contributions,
            "equal": self.get_equal_contributions,
            "TB": self.get_tb_contributions,
            "ModifiedDB": self.get_modified_db_contributions,
            "geometric_within": self.get_geometric_within_contributions,
            "geometric": self.get_geometric_contributions,
        }
        # Validate weighting unconditionally — this runs once per loss() call,
        # not per-element, so it doesn't affect torch.compile hot paths.
        contributions_fn = weight_functions.get(self.weighting)
        if contributions_fn is None:
            raise ValueError(f"Unknown weighting method {self.weighting}")
        contributions = contributions_fn(trajectories)

        flat_contributions = contributions[~flattening_mask]
        if self.debug:
            _assert_normalized(flat_contributions, f"{self.weighting!r} contributions")

        final_scores = flat_contributions * self.loss_fn(all_scores[~flattening_mask])

        # TODO: default was sum, should we allow mean?
        if reduction == "mean":
            if self.debug:
                warnings.warn(
                    "Mean reduction is not supported for SubTBGFlowNet with "
                    f"{self.weighting!r} weighting, using sum instead."
                )
            reduction = "sum"

        return loss_reduce(final_scores, reduction)
