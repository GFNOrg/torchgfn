"""Extended tests for gfn.gflownet: losses, base, TB, DB, SubTB, FM, MLE."""

import warnings

import pytest
import torch

from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet.base import loss_reduce
from gfn.gflownet.losses import (
    HalfSquaredLoss,
    LinexLoss,
    ShiftedCoshLoss,
    SquaredLoss,
)
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env_and_estimators(ndim=2, height=4):
    env = HyperGrid(ndim=ndim, height=height, validate_modes=False)
    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)
    pf = DiscretePolicyEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions),
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    pb = DiscretePolicyEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions - 1),
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=True,
    )
    return env, pf, pb


def _sample_trajs(env, pf, n=8):
    torch.manual_seed(42)
    sampler = Sampler(estimator=pf)
    return sampler.sample_trajectories(env, n=n, save_logprobs=True)


# ---------------------------------------------------------------------------
# losses.py — RegressionLoss hierarchy
# ---------------------------------------------------------------------------


class TestSquaredLoss:
    def test_zero_residual(self):
        loss = SquaredLoss()
        assert loss(torch.tensor(0.0)).item() == 0.0

    def test_positive_negative(self):
        loss = SquaredLoss()
        r = torch.tensor([-2.0, 0.0, 3.0])
        expected = torch.tensor([4.0, 0.0, 9.0])
        assert torch.allclose(loss(r), expected)

    def test_repr_eq_hash(self):
        a, b = SquaredLoss(), SquaredLoss()
        assert repr(a) == "SquaredLoss()"
        assert a == b
        assert hash(a) == hash(b)


class TestHalfSquaredLoss:
    def test_basic(self):
        loss = HalfSquaredLoss()
        r = torch.tensor([2.0, -3.0])
        expected = torch.tensor([2.0, 4.5])
        assert torch.allclose(loss(r), expected)


class TestShiftedCoshLoss:
    def test_symmetry(self):
        loss = ShiftedCoshLoss()
        r = torch.tensor([1.5])
        assert torch.allclose(loss(r), loss(-r), atol=1e-6)

    def test_zero_at_zero(self):
        loss = ShiftedCoshLoss()
        assert loss(torch.tensor(0.0)).item() == pytest.approx(0.0, abs=1e-7)

    def test_clamps_extreme_values(self):
        """Values beyond +-80 should be clamped, not produce inf."""
        loss = ShiftedCoshLoss()
        result = loss(torch.tensor([100.0, -100.0]))
        assert torch.isfinite(result).all()


class TestLinexLoss:
    def test_alpha_1(self):
        loss = LinexLoss(alpha=1.0)
        r = torch.tensor([0.0])
        assert loss(r).item() == pytest.approx(0.0, abs=1e-7)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha must be nonzero"):
            LinexLoss(alpha=0.0)

    def test_repr_eq_hash(self):
        a = LinexLoss(alpha=0.5)
        b = LinexLoss(alpha=0.5)
        c = LinexLoss(alpha=1.0)
        assert repr(a) == "LinexLoss(alpha=0.5)"
        assert a == b
        assert a != c
        assert hash(a) == hash(b)
        assert hash(a) != hash(c)

    def test_non_negative(self):
        loss = LinexLoss(alpha=1.0)
        r = torch.linspace(-5, 5, 100)
        result = loss(r)
        assert (result >= -1e-6).all()  # non-negative (with float tolerance)


# ---------------------------------------------------------------------------
# base.py — loss_reduce
# ---------------------------------------------------------------------------


def test_loss_reduce_mean():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert loss_reduce(t, "mean").item() == pytest.approx(2.0)


def test_loss_reduce_sum():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert loss_reduce(t, "sum").item() == pytest.approx(6.0)


def test_loss_reduce_none():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(loss_reduce(t, "none"), t)


def test_loss_reduce_invalid_raises():
    with pytest.raises(ValueError, match="Invalid loss reduction"):
        loss_reduce(torch.tensor([1.0]), "bad")


# ---------------------------------------------------------------------------
# base.py — GFlowNet debug assertions
# ---------------------------------------------------------------------------


def test_gflownet_assert_finite_gradients_debug():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb, debug=True)
    trajs = _sample_trajs(env, pf)
    loss = gfn.loss(env, trajs)
    loss.backward()
    # Should not raise (gradients should be finite)
    gfn.assert_finite_gradients()
    gfn.assert_finite_parameters()


def test_gflownet_assert_finite_gradients_skipped_without_debug():
    _env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb, debug=False)
    # Should be a no-op, even with non-finite gradients
    gfn.assert_finite_gradients()
    gfn.assert_finite_parameters()


# ---------------------------------------------------------------------------
# base.py — PFBasedGFlowNet validation
# ---------------------------------------------------------------------------


def test_pf_based_gflownet_pb_none_without_constant_raises():
    _, pf, _ = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    with pytest.raises(ValueError, match="pb must be an Estimator"):
        TBGFlowNet(pf=pf, pb=None, constant_pb=False)


def test_pf_based_gflownet_pb_with_constant_warns():
    _, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TBGFlowNet(pf=pf, pb=pb, constant_pb=True)
        assert any("pb should be ignored" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# base.py — parameter accessors
# ---------------------------------------------------------------------------


def test_pf_pb_named_parameters():
    _, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb)
    params = gfn.pf_pb_named_parameters()
    assert len(params) > 0
    assert all("pf" in k or "pb" in k for k in params)


def test_logz_named_parameters():
    _, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb)
    params = gfn.logz_named_parameters()
    assert len(params) > 0
    assert all("logZ" in k for k in params)


def test_logz_parameters_list():
    _, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb)
    params = gfn.logz_parameters()
    assert len(params) > 0


# ---------------------------------------------------------------------------
# base.py — loss_from_trajectories
# ---------------------------------------------------------------------------


def test_loss_from_trajectories_delegates():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb)
    trajs = _sample_trajs(env, pf)
    loss_direct = gfn.loss(env, trajs)
    loss_helper = gfn.loss_from_trajectories(env, trajs)
    # Both should produce the same loss (same trajectories, same recalculate)
    assert torch.allclose(loss_direct, loss_helper, atol=1e-5)


# ---------------------------------------------------------------------------
# trajectory_balance.py — TBGFlowNet
# ---------------------------------------------------------------------------


def test_tb_loss_basic():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb)
    trajs = _sample_trajs(env, pf)
    loss = gfn.loss(env, trajs)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()


def test_tb_get_scores_with_custom_log_rewards():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb)
    trajs = _sample_trajs(env, pf)
    custom_rewards = torch.zeros(trajs.batch_size)
    scores = gfn.get_scores(trajs, log_rewards=custom_rewards)
    assert scores.shape == (trajs.batch_size,)


def test_tb_log_reward_clip_min():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb, log_reward_clip_min=-10.0)
    trajs = _sample_trajs(env, pf)
    loss = gfn.loss(env, trajs)
    assert torch.isfinite(loss)


def test_tb_with_constant_pb():
    env, pf, _ = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=None, constant_pb=True)
    trajs = _sample_trajs(env, pf)
    loss = gfn.loss(env, trajs)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# detailed_balance.py — DBGFlowNet
# ---------------------------------------------------------------------------


def test_db_loss_basic_discrete():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import DBGFlowNet

    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)
    logF = ScalarEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=1),
        preprocessor=preproc,
    )
    gfn = DBGFlowNet(pf=pf, pb=pb, logF=logF)
    trajs = _sample_trajs(env, pf)
    transitions = gfn.to_training_samples(trajs)
    loss = gfn.loss(env, transitions)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()


# ---------------------------------------------------------------------------
# sub_trajectory_balance.py — SubTBGFlowNet
# ---------------------------------------------------------------------------


def test_subtb_loss_basic():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import SubTBGFlowNet

    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)
    logF = ScalarEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=1),
        preprocessor=preproc,
    )
    gfn = SubTBGFlowNet(pf=pf, pb=pb, logF=logF, weighting="geometric_within")
    trajs = _sample_trajs(env, pf)
    loss = gfn.loss(env, trajs)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_subtb_weighting_functions():
    """Test that different weighting schemes produce finite loss."""
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import SubTBGFlowNet

    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)
    logF = ScalarEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=1),
        preprocessor=preproc,
    )
    trajs = _sample_trajs(env, pf)
    for weighting in (
        "DB",
        "ModifiedDB",
        "TB",
        "equal",
        "geometric_within",
    ):  # pyright: ignore[reportArgumentType]
        gfn = SubTBGFlowNet(pf=pf, pb=pb, logF=logF, weighting=weighting)
        loss = gfn.loss(env, trajs)
        assert torch.isfinite(loss), f"Non-finite loss for weighting={weighting}"


# ---------------------------------------------------------------------------
# flow_matching.py — FMGFlowNet
# ---------------------------------------------------------------------------


def test_fm_loss_basic():
    env, pf, _ = _make_env_and_estimators()
    from gfn.gflownet import FMGFlowNet

    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)
    logF = DiscretePolicyEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions),
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    gfn = FMGFlowNet(logF=logF)
    trajs = _sample_trajs(env, pf)
    states = gfn.to_training_samples(trajs)
    loss = gfn.loss(env, states)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()


# ---------------------------------------------------------------------------
# losses.py with GFlowNet — custom loss_fn integration
# ---------------------------------------------------------------------------


def test_tb_with_shifted_cosh_loss():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb, loss_fn=ShiftedCoshLoss())
    trajs = _sample_trajs(env, pf)
    loss = gfn.loss(env, trajs)
    assert torch.isfinite(loss)


def test_tb_with_linex_loss():
    env, pf, pb = _make_env_and_estimators()
    from gfn.gflownet import TBGFlowNet

    gfn = TBGFlowNet(pf=pf, pb=pb, loss_fn=LinexLoss(alpha=0.5))
    trajs = _sample_trajs(env, pf)
    loss = gfn.loss(env, trajs)
    assert torch.isfinite(loss)
