from types import MethodType

import pytest
import torch

from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet


class _DummyTrajectories:
    """Minimal trajectories carrier for get_scores vectorization test."""

    def __init__(self, terminating_idx: torch.Tensor, max_length: int):
        self.terminating_idx = terminating_idx
        self.max_length = max_length
        self.n_trajectories = terminating_idx.shape[0]

    def __len__(self) -> int:
        return self.n_trajectories


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_subtb_get_scores_vectorized_matches_original(seed: int):
    torch.manual_seed(seed)
    max_len = 3
    n_traj = 4

    # Synthetic inputs for the get_scores pipeline.
    terminating_idx = torch.tensor([1, 2, 3, 2])
    log_pf_trajectories = torch.randn(max_len, n_traj)
    log_pb_trajectories = torch.randn(max_len, n_traj)
    log_state_flows = torch.randn(max_len, n_traj)
    sink_states_mask = torch.zeros(max_len, n_traj, dtype=torch.bool)
    is_terminal_mask = torch.zeros(max_len, n_traj, dtype=torch.bool)

    preds_list = [torch.randn(max_len + 1 - i, n_traj) for i in range(1, max_len + 1)]
    targets_list = [torch.randn(max_len + 1 - i, n_traj) for i in range(1, max_len + 1)]

    trajectories = _DummyTrajectories(
        terminating_idx=terminating_idx, max_length=max_len
    )
    env = object()  # Unused by the monkeypatched methods.

    # Build a SubTBGFlowNet instance without running its heavy __init__.
    model = SubTBGFlowNet.__new__(SubTBGFlowNet)
    torch.nn.Module.__init__(model)
    model.debug = False
    model.log_reward_clip_min = float("-inf")

    # Monkeypatch the dependencies used inside get_scores to deterministic tensors.
    model.get_pfs_and_pbs = MethodType(
        lambda self, traj, recalculate_all_logprobs=True: (
            log_pf_trajectories,
            log_pb_trajectories,
        ),
        model,
    )
    model.calculate_log_state_flows = MethodType(
        lambda self, _env, _traj, _log_pf: log_state_flows, model
    )
    model.calculate_masks = MethodType(
        lambda self, _log_state_flows, _traj: (sink_states_mask, is_terminal_mask),
        model,
    )
    model.calculate_preds = MethodType(
        lambda self, _log_pf_cum, _log_state_flows, i: preds_list[i - 1], model
    )
    model.calculate_targets = MethodType(
        lambda self, _traj, _preds, _log_pb_cum, _log_state_flows, _term_mask, _sink_mask, i: targets_list[
            i - 1
        ],
        model,
    )

    def original_get_scores(self, env, trajectories, recalculate_all_logprobs=True):
        log_pf_trajectories_, log_pb_trajectories_ = self.get_pfs_and_pbs(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        log_pf_trajectories_cum = self.cumulative_logprobs(
            trajectories, log_pf_trajectories_
        )
        log_pb_trajectories_cum = self.cumulative_logprobs(
            trajectories, log_pb_trajectories_
        )

        log_state_flows_ = self.calculate_log_state_flows(
            env, trajectories, log_pf_trajectories_
        )
        sink_states_mask_, is_terminal_mask_ = self.calculate_masks(
            log_state_flows_, trajectories
        )

        flattening_masks_orig = []
        scores_orig = []
        for i in range(1, 1 + trajectories.max_length):
            preds = self.calculate_preds(log_pf_trajectories_cum, log_state_flows_, i)
            targets = self.calculate_targets(
                trajectories,
                preds,
                log_pb_trajectories_cum,
                log_state_flows_,
                is_terminal_mask_,
                sink_states_mask_,
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
            if self.debug and torch.any(torch.isnan(flat_preds)):
                raise ValueError("NaN in preds")

            flat_targets = targets[~flattening_mask]
            if self.debug and torch.any(torch.isnan(flat_targets)):
                raise ValueError("NaN in targets")

            flattening_masks_orig.append(flattening_mask)
            scores_orig.append(preds - targets)

        return scores_orig, flattening_masks_orig

    orig_scores, orig_masks = original_get_scores(model, env, trajectories)
    vec_scores, vec_masks = model.get_scores(env, trajectories)  # type: ignore

    assert len(orig_scores) == len(vec_scores) == trajectories.max_length
    for orig, vec in zip(orig_scores, vec_scores):
        torch.testing.assert_close(vec, orig)
    for orig_m, vec_m in zip(orig_masks, vec_masks):
        assert torch.equal(vec_m, orig_m)
