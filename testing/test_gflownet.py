import pytest
import torch

from gfn.containers import StatesContainer, Trajectories
from gfn.containers.base import Container
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import FMGFlowNet, TBGFlowNet
from gfn.gflownet.base import loss_reduce
from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet
from gfn.gym import Box, HyperGrid
from gfn.gym.helpers.box_utils import BoxPBEstimator, BoxPBMLP, BoxPFEstimator, BoxPFMLP
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates
from gfn.utils.handlers import (
    has_conditions_exception_handler,
    no_conditions_exception_handler,
)
from gfn.utils.modules import MLP


def test_trajectory_based_gflownet_generic():
    pf_module = BoxPFMLP(
        hidden_dim=32, n_hidden_layers=2, n_components=1, n_components_s0=1
    )
    pb_module = BoxPBMLP(
        hidden_dim=32, n_hidden_layers=2, n_components=1, trunk=pf_module.trunk
    )

    env = Box()

    pf_estimator = BoxPFEstimator(
        env=env, module=pf_module, n_components=1, n_components_s0=1
    )
    pb_estimator = BoxPBEstimator(env=env, module=pb_module, n_components=1)

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    mock_trajectories = Trajectories(env)

    result = gflownet.to_training_samples(mock_trajectories)

    # Assert that the result is of type `Trajectories`
    assert isinstance(
        result, Container
    ), f"Expected type Container, but got {type(result)}"
    assert isinstance(
        result, Trajectories
    ), f"Expected type Trajectories, but got {type(result)}"


def test_flow_matching_gflownet_generic():
    env = HyperGrid(ndim=2)
    preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)
    module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    estimator = DiscretePolicyEstimator(
        module, n_actions=env.n_actions, preprocessor=preprocessor
    )
    gflownet = FMGFlowNet(estimator)
    mock_trajectories = Trajectories(env)
    states_pairs = gflownet.to_training_samples(mock_trajectories)

    # Assert that the result is a StatesContainer[DiscreteStates]
    assert isinstance(states_pairs, StatesContainer)
    assert isinstance(states_pairs.intermediary_states, DiscreteStates)
    assert isinstance(states_pairs.terminating_states, DiscreteStates)


def test_pytorch_inheritance():
    pf_module = BoxPFMLP(
        hidden_dim=32, n_hidden_layers=2, n_components=1, n_components_s0=1
    )
    pb_module = BoxPBMLP(
        hidden_dim=32, n_hidden_layers=2, n_components=1, trunk=pf_module.trunk
    )

    env = Box()

    pf_estimator = BoxPFEstimator(
        env=env, module=pf_module, n_components=1, n_components_s0=1
    )
    pb_estimator = BoxPBEstimator(env=env, module=pb_module, n_components=1)

    tbgflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    assert hasattr(
        tbgflownet.parameters(), "__iter__"
    ), "Expected gflownet to have iterable parameters() method inherited from nn.Module"
    assert hasattr(
        tbgflownet.state_dict(), "__dict__"
    ), "Expected gflownet to have indexable state_dict() method inherited from nn.Module"

    estimator = DiscretePolicyEstimator(pf_module, n_actions=2)
    fmgflownet = FMGFlowNet(estimator)
    assert hasattr(
        fmgflownet.parameters(), "__iter__"
    ), "Expected gflownet to have iterable parameters() method inherited from nn.Module"
    assert hasattr(
        fmgflownet.state_dict(), "__dict__"
    ), "Expected gflownet to have indexable state_dict() method inherited from nn.Module"


@pytest.mark.parametrize("seed", [0, 12, 47, 67])
def test_flow_matching_vectorized_matches_original(seed):
    torch.manual_seed(seed)
    env = HyperGrid(ndim=2)
    preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)
    module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    estimator = DiscretePolicyEstimator(
        module, n_actions=env.n_actions, preprocessor=preprocessor
    )
    gflownet = FMGFlowNet(estimator)

    trajectories = gflownet.sample_trajectories(env, n=6)
    states_container = gflownet.to_training_samples(trajectories)
    states = states_container.intermediary_states
    conditions = states_container.intermediary_conditions

    if len(states) == 0:
        # If the sample produced only terminal states, resample with more trajectories.
        trajectories = gflownet.sample_trajectories(env, n=12)
        states_container = gflownet.to_training_samples(trajectories)
        states = states_container.intermediary_states
        conditions = states_container.intermediary_conditions

    assert len(states) > 0

    def flow_matching_loss_original(
        self, env, states, conditions, reduction: str = "mean"
    ):
        if len(states) == 0:
            return torch.tensor(0.0, device=states.device)
        assert len(states.batch_shape) == 1
        assert not torch.any(states.is_initial_state)
        incoming_log_flows = torch.full_like(
            states.backward_masks, -float("inf"), dtype=torch.get_default_dtype()
        )
        outgoing_log_flows = torch.full_like(
            states.forward_masks, -float("inf"), dtype=torch.get_default_dtype()
        )
        for action_idx in range(env.n_actions - 1):
            valid_backward_mask = states.backward_masks[:, action_idx]
            valid_forward_mask = states.forward_masks[:, action_idx]
            valid_backward_states = states[valid_backward_mask]
            valid_forward_states = states[valid_forward_mask]
            backward_actions = torch.full_like(
                valid_backward_states.backward_masks[:, 0], action_idx, dtype=torch.long
            ).unsqueeze(-1)
            backward_actions = env.actions_from_tensor(backward_actions)
            valid_backward_states_parents = env._backward_step(
                valid_backward_states, backward_actions
            )
            if conditions is not None:
                valid_backward_conditions = conditions[valid_backward_mask]
                valid_forward_conditions = conditions[valid_forward_mask]
                with has_conditions_exception_handler("logF", self.logF):
                    incoming_log_flows[valid_backward_mask, action_idx] = self.logF(
                        valid_backward_states_parents,
                        valid_backward_conditions,
                    )[:, action_idx]
                    outgoing_log_flows[valid_forward_mask, action_idx] = self.logF(
                        valid_forward_states,
                        valid_forward_conditions,
                    )[:, action_idx]
            else:
                with no_conditions_exception_handler("logF", self.logF):
                    incoming_log_flows[valid_backward_mask, action_idx] = self.logF(
                        valid_backward_states_parents,
                    )[:, action_idx]
                    outgoing_log_flows[valid_forward_mask, action_idx] = self.logF(
                        valid_forward_states,
                    )[:, action_idx]
        valid_forward_mask = states.forward_masks[:, -1]
        if conditions is not None:
            with has_conditions_exception_handler("logF", self.logF):
                outgoing_log_flows[valid_forward_mask, -1] = self.logF(
                    states[valid_forward_mask],
                    conditions[valid_forward_mask],
                )[:, -1]
        else:
            with no_conditions_exception_handler("logF", self.logF):
                outgoing_log_flows[valid_forward_mask, -1] = self.logF(
                    states[valid_forward_mask],
                )[:, -1]
        log_incoming_flows = torch.logsumexp(incoming_log_flows, dim=-1)
        log_outgoing_flows = torch.logsumexp(outgoing_log_flows, dim=-1)
        scores = (log_incoming_flows - log_outgoing_flows).pow(2)
        return loss_reduce(scores, reduction)

    loss_original = flow_matching_loss_original(
        gflownet, env, states, conditions, reduction="mean"
    )
    loss_vectorized = gflownet.flow_matching_loss(
        env, states, conditions, reduction="mean"
    )

    torch.testing.assert_close(loss_vectorized, loss_original)


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
