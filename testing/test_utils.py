"""Tests for gfn.utils: common, training, handlers, distributions."""

import logging
import random
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from gfn.utils.common import (
    Timer,
    default_fill_value_for_dtype,
    ensure_same_device,
    filter_kwargs_for_callable,
    get_available_cpus,
    is_int_dtype,
    set_seed,
    temporarily_set_seed,
)
from gfn.utils.distributions import IsotropicGaussian, UnsqueezedCategorical
from gfn.utils.handlers import (
    has_conditions_exception_handler,
    is_callable_exception_handler,
    no_conditions_exception_handler,
    warn_about_recalculating_logprobs,
)
from gfn.utils.training import grad_norm, lr_grad_ratio, param_norm

# ---------------------------------------------------------------------------
# common.py — is_int_dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.int32, True),
        (torch.int64, True),
        (torch.int16, True),
        (torch.uint8, True),
        (torch.float32, False),
        (torch.float64, False),
        (torch.bool, True),  # bool is treated as integer-like by torch
        (torch.complex64, False),
    ],
)
def test_is_int_dtype(dtype, expected):
    t = torch.zeros(1, dtype=dtype)
    assert is_int_dtype(t) == expected


# ---------------------------------------------------------------------------
# common.py — default_fill_value_for_dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.float32, -float("inf")),
        (torch.float64, -float("inf")),
        (torch.float16, -float("inf")),
        (torch.bfloat16, -float("inf")),
        (torch.complex64, -float("inf")),
        (torch.complex128, -float("inf")),
        (torch.bool, 0),
        (torch.int32, torch.iinfo(torch.int32).min),
        (torch.int64, torch.iinfo(torch.int64).min),
        (torch.uint8, torch.iinfo(torch.uint8).min),
    ],
)
def test_default_fill_value_for_dtype(dtype, expected):
    result = default_fill_value_for_dtype(dtype)
    assert result == expected


# ---------------------------------------------------------------------------
# common.py — ensure_same_device
# ---------------------------------------------------------------------------


def test_ensure_same_device_same_device_passes():
    d = torch.device("cpu")
    ensure_same_device(d, d)  # should not raise


def test_ensure_same_device_different_type_raises():
    with pytest.raises(ValueError, match="different types"):
        ensure_same_device(torch.device("cpu"), torch.device("meta"))


def test_ensure_same_device_mps_devices():
    """MPS devices should pass when both are mps (single-device)."""
    d1 = torch.device("mps")
    d2 = torch.device("mps")
    # Both are mps with no index — should pass via the early equality check.
    ensure_same_device(d1, d2)


# ---------------------------------------------------------------------------
# common.py — get_available_cpus
# ---------------------------------------------------------------------------


def test_get_available_cpus_returns_positive_int():
    result = get_available_cpus()
    assert isinstance(result, int)
    assert result >= 1


# ---------------------------------------------------------------------------
# common.py — filter_kwargs_for_callable
# ---------------------------------------------------------------------------


def test_filter_kwargs_for_callable_filters_correctly():
    def fn(a, b, c=3):
        pass

    result = filter_kwargs_for_callable(fn, {"a": 1, "b": 2, "d": 4, "e": 5})
    assert result == {"a": 1, "b": 2}


def test_filter_kwargs_for_callable_var_keyword_passthrough():
    def fn(a, **kwargs):
        pass

    kwargs = {"a": 1, "b": 2, "c": 3}
    result = filter_kwargs_for_callable(fn, kwargs)
    assert result == kwargs  # **kwargs means no filtering needed


# ---------------------------------------------------------------------------
# common.py — set_seed
# ---------------------------------------------------------------------------


def test_set_seed_non_distributed_deterministic():
    set_seed(42, deterministic_mode=False)
    a = torch.randn(5)
    set_seed(42, deterministic_mode=False)
    b = torch.randn(5)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# common.py — temporarily_set_seed
# ---------------------------------------------------------------------------


def test_temporarily_set_seed_restores_state():
    random.seed(0)
    before_python = random.getstate()

    with temporarily_set_seed(999):
        # Inside, the seed should be 999
        _ = random.random()

    after_python = random.getstate()
    # States should be restored
    assert before_python == after_python


# ---------------------------------------------------------------------------
# common.py — Timer
# ---------------------------------------------------------------------------


def test_timer_accumulates_elapsed_time():
    timing = {}
    with Timer(timing, "test_step"):
        # Do a small computation
        _ = sum(range(100))

    assert "test_step" in timing
    assert len(timing["test_step"]) == 1
    assert timing["test_step"][0] >= 0

    # Second invocation should append
    with Timer(timing, "test_step"):
        _ = sum(range(100))

    assert len(timing["test_step"]) == 2


def test_timer_disabled_returns_zero():
    timing = {}
    with Timer(timing, "disabled_step", enabled=False) as t:
        _ = sum(range(100))

    assert t.elapsed == 0.0
    assert "disabled_step" not in timing


# ---------------------------------------------------------------------------
# training.py — grad_norm, param_norm, lr_grad_ratio
# ---------------------------------------------------------------------------


def test_grad_norm_with_grads():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    loss = (p**2).sum()
    loss.backward()
    # grad = [6.0, 8.0], L2 norm = 10.0
    assert abs(grad_norm([p]) - 10.0) < 1e-5


def test_grad_norm_without_grads_returns_zero():
    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    # No backward() called, so no grad
    assert grad_norm([p]) == 0.0


def test_param_norm_basic():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    # L2 norm = 5.0
    assert abs(param_norm([p]) - 5.0) < 1e-5


def test_param_norm_empty_returns_zero():
    assert param_norm([]) == 0.0


def test_lr_grad_ratio_basic():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    loss = (p**2).sum()
    loss.backward()
    optimizer = torch.optim.SGD([p], lr=0.1)
    ratios = lr_grad_ratio(optimizer)
    # g_norm = 10.0, p_norm = 5.0, lr = 0.1 → ratio = 0.1 * 10 / 5 = 0.2
    assert len(ratios) == 1
    assert abs(ratios[0] - 0.2) < 1e-5


def test_lr_grad_ratio_zero_params():
    """When all params are zero and have no grad, ratio should be 0."""
    p = nn.Parameter(torch.tensor([0.0, 0.0]))
    optimizer = torch.optim.SGD([p], lr=0.1)
    ratios = lr_grad_ratio(optimizer)
    assert ratios == [0.0]


# ---------------------------------------------------------------------------
# training.py — states_actions_tns_to_traj
# ---------------------------------------------------------------------------


def test_states_actions_tns_to_traj_valid_input():
    from gfn.gym import HyperGrid
    from gfn.utils.training import states_actions_tns_to_traj

    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    # Build a simple trajectory: (0,0) → (1,0) → (1,1)
    states = torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.long)
    actions = torch.tensor([0, 1], dtype=torch.long)  # right, up
    traj = states_actions_tns_to_traj(states, actions, env)
    assert traj.batch_size == 1
    assert len(traj) == 1


def test_states_actions_tns_to_traj_shape_mismatch_raises():
    from gfn.gym import HyperGrid
    from gfn.utils.training import states_actions_tns_to_traj

    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    states = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
    actions = torch.tensor([0, 1, 2], dtype=torch.long)  # wrong length
    with pytest.raises(ValueError, match="same trajectory length"):
        states_actions_tns_to_traj(states, actions, env)


def test_states_actions_tns_to_traj_with_conditions():
    from gfn.gym import HyperGrid
    from gfn.utils.training import states_actions_tns_to_traj

    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    states = torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.long)
    actions = torch.tensor([0, 1], dtype=torch.long)
    conditions = torch.randn(3, 4)
    traj = states_actions_tns_to_traj(states, actions, env, conditions=conditions)
    assert traj.states.conditions is not None


# ---------------------------------------------------------------------------
# handlers.py
# ---------------------------------------------------------------------------


def test_has_conditions_exception_handler_passes_through():
    with has_conditions_exception_handler("test_fn", lambda: None):
        pass  # Should not raise


def test_has_conditions_exception_handler_logs_and_reraises(caplog):
    with pytest.raises(TypeError):
        with has_conditions_exception_handler("test_fn", "not_callable"):
            raise TypeError("test error")


def test_no_conditions_exception_handler_logs_and_reraises(caplog):
    with pytest.raises(TypeError):
        with no_conditions_exception_handler("test_fn", "not_callable"):
            raise TypeError("test error")


def test_is_callable_exception_handler_catches_all():
    with pytest.raises(RuntimeError):
        with is_callable_exception_handler("test_fn", "not_callable"):
            raise RuntimeError("generic error")


def test_warn_about_recalculating_logprobs_warns_when_has_logprobs(caplog):
    mock_container = MagicMock()
    mock_container.has_log_probs = True

    with caplog.at_level(logging.WARNING):
        warn_about_recalculating_logprobs(mock_container, recalculate_all_logprobs=True)

    assert "Recalculating logprobs" in caplog.text


def test_warn_about_recalculating_logprobs_silent_when_no_logprobs(caplog):
    mock_container = MagicMock()
    mock_container.has_log_probs = False

    with caplog.at_level(logging.WARNING):
        warn_about_recalculating_logprobs(mock_container, recalculate_all_logprobs=True)

    assert "Recalculating logprobs" not in caplog.text


# ---------------------------------------------------------------------------
# distributions.py — UnsqueezedCategorical
# ---------------------------------------------------------------------------


def test_unsqueezed_categorical_sample_shape():
    logits = torch.zeros(4, 5)  # batch_size=4, n_actions=5
    dist = UnsqueezedCategorical(logits=logits)
    sample = dist.sample()
    assert sample.shape == (4, 1)


def test_unsqueezed_categorical_log_prob():
    logits = torch.zeros(4, 5)  # uniform over 5 actions
    dist = UnsqueezedCategorical(logits=logits)
    sample = torch.tensor([[0], [1], [2], [3]])
    lp = dist.log_prob(sample)
    assert lp.shape == (4,)
    # Uniform: log(1/5) ≈ -1.6094
    assert torch.allclose(lp, torch.full((4,), -1.6094), atol=1e-3)


# ---------------------------------------------------------------------------
# distributions.py — IsotropicGaussian
# ---------------------------------------------------------------------------


def test_isotropic_gaussian_sample_shape():
    loc = torch.zeros(4, 3)
    scale = torch.ones(4, 1)
    dist = IsotropicGaussian(loc=loc, scale=scale)
    sample = dist.sample()
    assert sample.shape == (4, 3)


def test_isotropic_gaussian_log_prob_normal():
    loc = torch.zeros(4, 3)
    scale = torch.ones(4, 1)
    dist = IsotropicGaussian(loc=loc, scale=scale)
    actions = torch.zeros(4, 3)
    lp = dist.log_prob(actions)
    assert lp.shape == (4,)
    assert torch.isfinite(lp).all()


def test_isotropic_gaussian_log_prob_exit_action():
    """Exit actions (first dim is -inf) should get log_prob = 0."""
    loc = torch.zeros(4, 3)
    scale = torch.ones(4, 1)
    dist = IsotropicGaussian(loc=loc, scale=scale)
    actions = torch.zeros(4, 3)
    actions[0, 0] = float("-inf")  # exit sentinel
    lp = dist.log_prob(actions)
    assert lp[0].item() == 0.0
    assert torch.isfinite(lp[1:]).all()


def test_isotropic_gaussian_log_prob_zero_scale_no_nan():
    """B1: When scale < 1e-6 (deterministic), log_prob should be 0, not NaN."""
    loc = torch.zeros(4, 3)
    scale = torch.zeros(4, 1)  # zero scale
    dist = IsotropicGaussian(loc=loc, scale=scale)
    actions = torch.randn(4, 3)
    lp = dist.log_prob(actions)
    assert torch.isfinite(lp).all()
    assert (lp == 0.0).all()


def test_isotropic_gaussian_gradient_no_nan():
    """B1: Gradients must not contain NaN even when some scales are near-zero."""
    loc = torch.zeros(4, 3, requires_grad=True)
    scale = torch.tensor([[1.0], [0.0], [1e-8], [1.0]], requires_grad=True)
    dist = IsotropicGaussian(loc=loc, scale=scale, actions_detach=False)
    actions = dist.loc + 0.1  # close to loc
    lp = dist.log_prob(actions)
    loss = lp.sum()
    loss.backward()
    assert loc.grad is not None, "loc.grad is None"
    assert torch.isfinite(loc.grad).all(), f"loc.grad has NaN/inf: {loc.grad}"
    assert scale.grad is not None, "scale.grad is None"
    assert torch.isfinite(scale.grad).all(), f"scale.grad has NaN/inf: {scale.grad}"
