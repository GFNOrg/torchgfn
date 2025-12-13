import pytest
import torch

from gfn.containers.trajectories import Trajectories
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet
from gfn.gym.hypergrid import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_subtb_get_scores_vectorized_matches_original(seed: int):
    torch.manual_seed(seed)
    n_traj = 4

    # Deterministic HyperGrid env and frozen estimators so real methods can run.
    env = HyperGrid(ndim=2, height=3, device="cpu", debug=False)
    preproc = KHotPreprocessor(height=env.height, ndim=env.ndim)

    # Tiny MLPs with random weights (frozen for determinism).
    module_pf = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions)
    module_pb = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions - 1)
    module_logF = MLP(input_dim=preproc.output_dim, output_dim=1)
    for mod in (module_pf, module_pb, module_logF):
        for p in mod.parameters():
            p.requires_grad_(False)

    pf = DiscretePolicyEstimator(
        module=module_pf,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    pb = DiscretePolicyEstimator(
        module=module_pb, n_actions=env.n_actions, preprocessor=preproc, is_backward=True
    )
    logF = ScalarEstimator(module=module_logF, preprocessor=preproc)

    # Initialize model via __init__ to set up real methods.
    model = SubTBGFlowNet(
        pf=pf, pb=pb, logF=logF, weighting="geometric_within", lamda=0.9
    )
    model.debug = False
    model.log_reward_clip_min = float("-inf")
    model.eval()
    pf.eval()
    pb.eval()
    logF.eval()

    # Sample a deterministic batch of trajectories with frozen estimators.
    sampler = Sampler(estimator=pf)
    trajectories: Trajectories = sampler.sample_trajectories(
        env,
        n=n_traj,
        epsilon=0.0,
        save_logprobs=True,
        save_estimator_outputs=False,
    )
    max_len = trajectories.max_length  # noqa: F841 used implicitly by shapes

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

    def normalize_scores_masks(
        scores, masks, trajectories: Trajectories
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert list outputs to padded tensors; pass tensors through unchanged."""
        if isinstance(scores, torch.Tensor):
            assert isinstance(masks, torch.Tensor)
            return scores, masks

        assert isinstance(scores, (list, tuple))
        assert isinstance(masks, (list, tuple))

        max_len = trajectories.max_length
        n_traj = (
            trajectories.n_trajectories
            if hasattr(trajectories, "n_trajectories")
            else len(trajectories)
        )
        device = trajectories.terminating_idx.device
        dtype = scores[0].dtype

        scores_padded = torch.zeros(
            (max_len, max_len, n_traj), dtype=dtype, device=device
        )
        masks_padded = torch.ones(
            (max_len, max_len, n_traj), dtype=torch.bool, device=device
        )

        for i, (s, m) in enumerate(zip(scores, masks), start=1):
            seq_len = s.shape[0]
            scores_padded[i - 1, :seq_len] = s
            masks_padded[i - 1, :seq_len] = m

        return scores_padded, masks_padded

    # Recompute logprobs to ensure PF/PB are evaluated for both paths.
    orig_scores_list, orig_masks_list = original_get_scores(
        model, env, trajectories, recalculate_all_logprobs=True
    )
    vec_scores, vec_masks = model.get_scores(
        env, trajectories, recalculate_all_logprobs=True
    )  # type: ignore

    vec_scores_t, vec_masks_t = normalize_scores_masks(
        vec_scores, vec_masks, trajectories
    )
    orig_scores_t, orig_masks_t = normalize_scores_masks(
        orig_scores_list, orig_masks_list, trajectories
    )

    valid_mask = ~orig_masks_t
    if not torch.allclose(
        vec_scores_t[valid_mask], orig_scores_t[valid_mask], equal_nan=True
    ):
        max_diff = (vec_scores_t[valid_mask] - orig_scores_t[valid_mask]).abs().max()
        raise AssertionError(
            f"Score mismatch on valid positions; max_abs_diff={max_diff.item()}"
        )

    torch.testing.assert_close(vec_masks_t, orig_masks_t, equal_nan=True)
