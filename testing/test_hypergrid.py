import math

import pytest
import torch

from gfn.gym.hypergrid import HyperGrid, get_reward_presets


def _make_env(reward_fn_str: str, kwargs: dict, **extra):
    return HyperGrid(
        ndim=extra.get("ndim", 2),
        height=extra.get("height", 16),
        reward_fn_str=reward_fn_str,
        reward_fn_kwargs=kwargs,
        store_all_states=extra.get("store_all_states", False),
        validate_modes=extra.get("validate_modes", False),
        mode_stats=extra.get("mode_stats", "none"),
        mode_stats_samples=extra.get("mode_stats_samples", 1000),
    )


# -------------------------
# Reward value tests
# -------------------------


def test_original_reward_values_small():
    R0, R1, R2 = 0.1, 0.5, 2.0
    env = _make_env("original", dict(R0=R0, R1=R1, R2=R2), ndim=2, height=16)

    def expected(x):
        ax = (x / (env.height - 1.0) - 0.5).abs()
        outer = (ax > 0.25).all(dim=-1)
        band = ((ax > 0.3) & (ax < 0.4)).all(dim=-1)
        return R0 + outer.float() * R1 + band.float() * R2

    xs = torch.tensor([[0, 0], [2, 2], [5, 5], [13, 13]], dtype=torch.long)
    r = env.reward(env.States(xs))
    exp = expected(xs.to(torch.get_default_dtype()))
    assert torch.allclose(r, exp, atol=1e-6)


def test_deceptive_reward_values_small():
    R0, R1, R2 = 1e-5, 0.1, 2.0
    env = _make_env("deceptive", dict(R0=R0, R1=R1, R2=R2), ndim=2, height=16)

    def expected(x):
        ax = (x / (env.height - 1.0) - 0.5).abs()
        term1 = R0 + R1
        cancel_outer = (ax > 0.1).all(dim=-1).float() * R1
        band = ((ax > 0.3) & (ax < 0.4)).all(dim=-1).float() * R2
        return term1 - cancel_outer + band

    xs = torch.tensor([[0, 0], [2, 2], [8, 8]], dtype=torch.long)
    r = env.reward(env.States(xs))
    exp = expected(xs.to(torch.get_default_dtype()))
    assert torch.allclose(r, exp, atol=1e-6)


def test_cosine_reward_values_small():
    R0, R1 = 0.1, 0.5
    env = _make_env("cosine", dict(R0=R0, R1=R1), ndim=2, height=16)
    xs = torch.tensor([[0, 0], [7, 7], [8, 8]], dtype=torch.long)
    ax = (xs.to(torch.get_default_dtype()) / (env.height - 1.0) - 0.5).abs()
    pdf = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * (5 * ax) ** 2)
    exp = R0 + (((torch.cos(50 * ax) + 1.0) * pdf).prod(dim=-1) * R1)
    r = env.reward(env.States(xs))
    assert torch.allclose(r, exp, atol=1e-6)


def test_sparse_reward_values_small():
    env = _make_env("sparse", {}, ndim=2, height=6)
    xs_match = torch.tensor([[1, env.height - 2]], dtype=torch.long)
    xs_nomatch = torch.tensor([[0, 0]], dtype=torch.long)
    r_match = env.reward(env.States(xs_match))[0].item()
    r_nomatch = env.reward(env.States(xs_nomatch))[0].item()
    assert r_match > 0.9
    assert r_nomatch < 1e-6


def test_bitwise_xor_reward_values_small():
    presets = get_reward_presets("bitwise_xor", 4, 16)["easy"]
    env = _make_env("bitwise_xor", presets, ndim=4, height=16)
    xs_zero = torch.zeros(1, 4, dtype=torch.long)
    r0 = env.reward(env.States(xs_zero))[0].item()
    expected = presets.get("R0", 0.0) + sum(
        presets["tier_weights"]
    )  # zero satisfies even parity
    assert abs(r0 - expected) < 1e-6


def test_multiplicative_coprime_reward_values_small():
    kwargs = dict(
        R0=0.0, tier_weights=[1.0], primes=[2, 3], exponent_caps=[1], active_dims=[0, 1]
    )
    env = _make_env("multiplicative_coprime", kwargs, ndim=2, height=16)
    # Note: coordinates are +1 shifted internally, so state [1,2] -> values [2,3].
    # 2=2^1 (ok, cap=1), 3=3^1 (ok, cap=1) -> passes tier 0.
    xs_ok = torch.tensor([[1, 2]], dtype=torch.long)
    # State [3,2] -> values [4,3]. 4=2^2 exceeds cap=1 -> fails tier 0.
    xs_bad = torch.tensor([[3, 2]], dtype=torch.long)
    r_ok = env.reward(env.States(xs_ok))[0].item()
    r_bad = env.reward(env.States(xs_bad))[0].item()
    assert abs(r_ok - 1.0) < 1e-6
    assert abs(r_bad - 0.0) < 1e-6


# -------------------------
# Mode counts and stats (small settings)
# -------------------------


@pytest.mark.parametrize(
    "reward_name,kwargs,height",
    [
        ("original", dict(R0=0.1, R1=0.5, R2=2.0), 16),
        ("cosine", dict(R0=0.1, R1=0.5), 32),
        ("sparse", {}, 16),
        ("deceptive", dict(R0=1e-5, R1=0.1, R2=2.0), 16),
        ("bitwise_xor", get_reward_presets("bitwise_xor", 3, 16)["easy"], 16),
        (
            "multiplicative_coprime",
            dict(
                R0=0.0,
                tier_weights=[1.0],
                primes=[2, 3],
                exponent_caps=[1],
                active_dims=[0, 1],
            ),
            16,
        ),
    ],
)
def test_mode_counts_small_exact(reward_name, kwargs, height):
    env = _make_env(
        reward_name,
        kwargs,
        ndim=3,
        height=height,
        store_all_states=True,
        validate_modes=True,
        mode_stats="exact",
    )
    assert env.n_mode_states is not None
    assert env.n_mode_states > 0
    assert env.n_modes == env.n_mode_states


# -------------------------
# Negative / invalid configurations should raise
# -------------------------


def test_original_validate_modes_raises_on_no_band():
    with pytest.raises(ValueError):
        _make_env(
            "original",
            dict(R0=0.1, R1=0.5, R2=2.0),
            ndim=2,
            height=4,
            validate_modes=True,
        )


def test_deceptive_validate_modes_raises_on_no_band():
    with pytest.raises(ValueError):
        _make_env(
            "deceptive",
            dict(R0=1e-5, R1=0.1, R2=2.0),
            ndim=2,
            height=4,
            validate_modes=True,
        )


def test_cosine_validate_modes_raises_on_tight_gamma_small_grid():
    with pytest.raises(ValueError):
        _make_env(
            "cosine",
            dict(R0=0.1, R1=0.5, mode_gamma=0.99),
            ndim=2,
            height=4,
            validate_modes=True,
        )


def test_sparse_validate_modes_raises_when_no_targets_in_grid():
    # height=1 -> targets contain -1 and 1, none inside the grid [0,0]
    with pytest.raises(ValueError):
        _make_env("sparse", {}, ndim=2, height=1, validate_modes=True)


def test_bitwise_xor_validate_modes_raises_on_inconsistent_parity():
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0],
        dims_constrained=[0],
        bits_per_tier=[(0, 0)],
        parity_checks=[
            {
                "A": torch.tensor([[0]], dtype=torch.long),
                "c": torch.tensor([1], dtype=torch.long),
            }
        ],
    )
    with pytest.raises(ValueError):
        _make_env("bitwise_xor", kwargs, ndim=2, height=2, validate_modes=True)


def test_multiplicative_validate_modes_raises_on_unrealizable_lcm():
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0],
        primes=[2],
        exponent_caps=[1],
        active_dims=[0, 1],
        target_lcms=[3],
    )
    with pytest.raises(ValueError):
        _make_env(
            "multiplicative_coprime", kwargs, ndim=2, height=8, validate_modes=True
        )


# -------------------------
# Smoke tests
# -------------------------


def test_step_and_backward_shapes():
    env = _make_env("original", dict(R0=0.1, R1=0.5, R2=2.0), ndim=3, height=10)
    states = env.make_random_states((5,))
    actions = torch.randint(0, env.n_actions - 1, (5, 1))
    next_states = env.step(states, env.Actions(actions))
    prev_states = env.backward_step(next_states, env.Actions(actions))
    assert next_states.tensor.shape == states.tensor.shape
    assert prev_states.tensor.shape == states.tensor.shape


def test_invalid_reward_name_raises():
    with pytest.raises(AssertionError):
        HyperGrid(ndim=2, height=4, reward_fn_str="does_not_exist")


# -------------------------
# Positive validation tests (non-negative)
# -------------------------


def test_validate_modes_succeeds_original():
    env = _make_env(
        "original", dict(R0=0.1, R1=0.5, R2=2.0), ndim=2, height=16, validate_modes=True
    )
    assert env.n_actions == env.ndim + 1


def test_validate_modes_succeeds_deceptive():
    env = _make_env(
        "deceptive",
        dict(R0=1e-5, R1=0.1, R2=2.0),
        ndim=2,
        height=16,
        validate_modes=True,
    )
    assert env.n_actions == env.ndim + 1


def test_validate_modes_succeeds_cosine():
    # Cosine's cos(50·ax) oscillates rapidly; height >= 32 needed for the
    # discrete grid to resolve peaks above the gamma=0.8 theoretical threshold.
    env = _make_env(
        "cosine",
        dict(R0=0.1, R1=0.5, mode_gamma=0.8),
        ndim=2,
        height=32,
        validate_modes=True,
    )
    assert env.n_actions == env.ndim + 1


def test_validate_modes_succeeds_sparse():
    env = _make_env("sparse", {}, ndim=2, height=6, validate_modes=True)
    assert env.n_actions == env.ndim + 1


def test_validate_modes_succeeds_bitwise_xor():
    presets = get_reward_presets("bitwise_xor", 4, 16)["easy"]
    env = _make_env("bitwise_xor", presets, ndim=4, height=16, validate_modes=True)
    assert env.n_actions == env.ndim + 1


def test_validate_modes_succeeds_multiplicative_coprime():
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        primes=[2, 3],
        exponent_caps=[2, 2],
        active_dims=[0, 1],
    )
    env = _make_env(
        "multiplicative_coprime", kwargs, ndim=2, height=16, validate_modes=True
    )
    assert env.n_actions == env.ndim + 1


# -------------------------
# Mode counts match enumeration across presets (easy/medium)
# -------------------------


@pytest.mark.parametrize(
    "reward_name,kwargs_fn,ndim,height",
    [
        ("original", lambda D, H: get_reward_presets("original", D, H)["easy"], 2, 16),
        ("original", lambda D, H: get_reward_presets("original", D, H)["medium"], 3, 16),
        ("cosine", lambda D, H: get_reward_presets("cosine", D, H)["easy"], 2, 32),
        ("cosine", lambda D, H: get_reward_presets("cosine", D, H)["medium"], 3, 32),
        ("sparse", lambda D, H: get_reward_presets("sparse", D, H)["easy"], 2, 6),
        ("deceptive", lambda D, H: get_reward_presets("deceptive", D, H)["easy"], 2, 16),
        (
            "deceptive",
            lambda D, H: get_reward_presets("deceptive", D, H)["medium"],
            3,
            16,
        ),
        (
            "bitwise_xor",
            lambda D, H: get_reward_presets("bitwise_xor", D, H)["easy"],
            3,
            16,
        ),
        (
            "bitwise_xor",
            lambda D, H: get_reward_presets("bitwise_xor", D, H)["medium"],
            3,
            16,
        ),
        (
            "multiplicative_coprime",
            lambda D, H: get_reward_presets("multiplicative_coprime", D, H)["easy"],
            3,
            16,
        ),
        (
            "multiplicative_coprime",
            lambda D, H: get_reward_presets("multiplicative_coprime", D, H)["medium"],
            3,
            16,
        ),
    ],
)
def test_mode_counts_match_enumeration(reward_name, kwargs_fn, ndim, height):
    kwargs = kwargs_fn(ndim, height)
    env = _make_env(
        reward_name,
        kwargs,
        ndim=ndim,
        height=height,
        store_all_states=True,
        validate_modes=True,
        mode_stats="exact",
    )
    all_states = env.all_states
    assert all_states is not None
    mask = env.mode_mask(all_states)
    expected = int(mask.sum().item())
    assert env.n_modes == expected
    assert expected >= 1
    # Verify modes_found returns the correct canonical indices.
    found = env.modes_found(all_states)
    assert len(found) == expected
