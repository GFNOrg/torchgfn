import math

import pytest
import torch

from gfn.gym.hypergrid import (
    BitwiseXORReward,
    ConditionalMultiScaleReward,
    CorruptedReward,
    HyperGrid,
    MultiplicativeCoprimeReward,
    _state_hash_uniform,
    get_reward_presets,
)


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


# =========================================================================
# UniformRandomReward and CorruptedReward tests
# =========================================================================


# -------------------------
# Hash utility tests
# -------------------------


def test_state_hash_determinism():
    xs = torch.randint(0, 16, (100, 3), dtype=torch.long)
    h1 = _state_hash_uniform(xs, seed=42)
    h2 = _state_hash_uniform(xs, seed=42)
    assert torch.equal(h1, h2), "Same seed must produce same hash"

    h3 = _state_hash_uniform(xs, seed=99)
    assert not torch.equal(h1, h3), "Different seeds must produce different hashes"

    # Values should be in [0, 1)
    assert (h1 >= 0).all() and (h1 < 1).all()


# -------------------------
# UniformRandom reward value tests
# -------------------------


def test_uniform_random_reward_values_small():
    kwargs = dict(R0=0.1, R_mode=2.0, mode_prob=0.5, seed=42)
    env = _make_env("uniform_random", kwargs, ndim=2, height=8)
    xs = torch.randint(0, 8, (50, 2), dtype=torch.long)
    r = env.reward(env.States(xs))
    # All rewards should be either R0 or R0+R_mode.
    for val in r.tolist():
        assert abs(val - 0.1) < 1e-6 or abs(val - 2.1) < 1e-6

    # Determinism: same states should give same rewards.
    r2 = env.reward(env.States(xs))
    assert torch.allclose(r, r2)


def test_uniform_random_raises_on_bad_mode_prob():
    with pytest.raises(AssertionError):
        _make_env("uniform_random", dict(mode_prob=0.0), ndim=2, height=8)
    with pytest.raises(AssertionError):
        _make_env("uniform_random", dict(mode_prob=1.0), ndim=2, height=8)


def test_validate_modes_succeeds_uniform_random():
    env = _make_env(
        "uniform_random",
        dict(R0=0.1, R_mode=2.0, mode_prob=0.01, seed=42),
        ndim=2,
        height=16,
        validate_modes=True,
    )
    assert env.n_actions == env.ndim + 1


# -------------------------
# CorruptedReward value tests
# -------------------------


def test_corrupted_reward_values_small():
    """Basic smoke test: corrupted reward produces finite positive values."""
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    kwargs = dict(
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=0.3,
        seed=137,
    )
    env = _make_env("corrupted", kwargs, ndim=3, height=16)
    xs = torch.randint(0, 16, (100, 3), dtype=torch.long)
    r = env.reward(env.States(xs))
    assert r.shape == (100,)
    assert (r >= 0).all()
    assert torch.isfinite(r).all()


def test_corrupted_rate_zero_matches_base():
    """With corruption_rate=0, corrupted reward should equal the base reward."""
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    kwargs = dict(
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=0.0,
        seed=137,
    )
    env_corrupted = _make_env("corrupted", kwargs, ndim=3, height=16)
    env_base = _make_env("bitwise_xor", base_kwargs, ndim=3, height=16)

    xs = torch.randint(0, 16, (200, 3), dtype=torch.long)
    r_corrupted = env_corrupted.reward(env_corrupted.States(xs))
    r_base = env_base.reward(env_base.States(xs))
    assert torch.allclose(r_corrupted, r_base, atol=1e-6)


def test_corrupted_different_seeds_differ():
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    kwargs1 = dict(
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=0.3,
        seed=137,
    )
    kwargs2 = dict(
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=0.3,
        seed=999,
    )
    env1 = _make_env("corrupted", kwargs1, ndim=3, height=16)
    env2 = _make_env("corrupted", kwargs2, ndim=3, height=16)

    xs = torch.randint(0, 16, (200, 3), dtype=torch.long)
    r1 = env1.reward(env1.States(xs))
    r2 = env2.reward(env2.States(xs))
    # With different seeds and corruption_rate > 0, rewards should differ
    # on at least some states.
    assert not torch.allclose(
        r1, r2
    ), "Different seeds should produce different corruption"


def test_validate_modes_succeeds_corrupted():
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    kwargs = dict(
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=0.3,
        seed=137,
    )
    env = _make_env("corrupted", kwargs, ndim=3, height=16, validate_modes=True)
    assert env.n_actions == env.ndim + 1


# -------------------------
# tier_indicators consistency tests
# -------------------------


@pytest.mark.parametrize(
    "reward_cls,reward_kwargs",
    [
        (
            BitwiseXORReward,
            get_reward_presets("bitwise_xor", 3, 16)["easy"],
        ),
        (
            MultiplicativeCoprimeReward,
            dict(
                R0=0.0,
                tier_weights=[1.0, 10.0],
                primes=[2, 3],
                exponent_caps=[2, 2],
                active_dims=[0, 1],
            ),
        ),
        (
            ConditionalMultiScaleReward,
            get_reward_presets("conditional_multiscale", 3, 16)["easy"],
        ),
    ],
)
def test_tier_indicators_match_reward(reward_cls, reward_kwargs):
    """Verify tier_indicators output is consistent with reward values."""
    height, ndim = 16, 3
    fn = reward_cls(height, ndim, **reward_kwargs)
    xs = torch.randint(0, height, (100, ndim), dtype=torch.long)

    indicators = fn.tier_indicators(xs)
    rewards = fn(xs)

    # A state passing all tiers cumulatively should get the full reward.
    all_pass = torch.ones(100, dtype=torch.bool)
    for ind in indicators:
        all_pass = all_pass & ind

    r0 = float(fn.R0)
    full_reward = r0 + sum(fn.tier_weights)
    for i in range(100):
        if all_pass[i]:
            assert abs(rewards[i].item() - full_reward) < 1e-5
        if not any(ind[i] for ind in indicators):
            assert abs(rewards[i].item() - r0) < 1e-5


# -------------------------
# Corruption mechanism unit tests
# -------------------------


def test_corrupted_demotion_rate():
    """On a small grid, verify demotion fraction is close to corruption_rate."""
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    fn_base = BitwiseXORReward(16, 3, **base_kwargs)

    # Enumerate all states.
    axes = [torch.arange(16, dtype=torch.long) for _ in range(3)]
    all_states = torch.cartesian_prod(*axes)
    base_indicators = fn_base.tier_indicators(all_states)

    corruption_rate = 0.3
    fn_corrupted = CorruptedReward(
        16,
        3,
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=corruption_rate,
        seed=137,
    )

    # For each tier, check that demotion fraction is close to corruption_rate.
    for t, base_ind in enumerate(base_indicators):
        n_pass = int(base_ind.sum().item())
        if n_pass == 0:
            continue
        h = _state_hash_uniform(all_states, fn_corrupted.seed + 2 * t)
        demoted = base_ind & (h < corruption_rate)
        actual_rate = float(demoted.sum().item()) / n_pass
        assert (
            abs(actual_rate - corruption_rate) < 0.1
        ), f"Tier {t}: expected demotion rate ~{corruption_rate}, got {actual_rate}"


def test_corrupted_promotion_rate():
    """Verify promotions roughly match demotions (mode count stable)."""
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    fn_base = BitwiseXORReward(16, 3, **base_kwargs)

    axes = [torch.arange(16, dtype=torch.long) for _ in range(3)]
    all_states = torch.cartesian_prod(*axes)
    base_indicators = fn_base.tier_indicators(all_states)

    corruption_rate = 0.3
    fn_corrupted = CorruptedReward(
        16,
        3,
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=corruption_rate,
        seed=137,
    )

    for t, base_ind in enumerate(base_indicators):
        n_pass = int(base_ind.sum().item())
        n_fail = int((~base_ind).sum().item())
        if n_pass == 0 or n_fail == 0:
            continue

        h_demote = _state_hash_uniform(all_states, fn_corrupted.seed + 2 * t)
        n_demoted = int((base_ind & (h_demote < corruption_rate)).sum().item())

        repl_rate = fn_corrupted._replacement_rates[t]
        h_promote = _state_hash_uniform(all_states, fn_corrupted.seed + 2 * t + 1)
        n_promoted = int(((~base_ind) & (h_promote < repl_rate)).sum().item())

        # Allow 50% tolerance for small grids.
        if n_demoted > 0:
            ratio = n_promoted / n_demoted
            assert (
                0.3 < ratio < 3.0
            ), f"Tier {t}: promoted/demoted ratio {ratio:.2f} out of range"


def test_corrupted_per_tier_independence():
    """Corruption at one tier shouldn't affect another tier's indicators."""
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    fn_corrupted = CorruptedReward(
        16,
        3,
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=0.5,
        seed=137,
    )

    xs = torch.randint(0, 16, (200, 3), dtype=torch.long)

    # Different tiers use different hash seeds, so corruption patterns
    # should be independent.
    h0 = _state_hash_uniform(xs, fn_corrupted.seed)
    h1 = _state_hash_uniform(xs, fn_corrupted.seed + 2)
    # The two hashes should not be identical.
    assert not torch.equal(h0, h1)


# -------------------------
# Integration tests
# -------------------------


def test_corrupted_mode_count_stable():
    """Mode count after corruption should be in a reasonable range."""
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    env_base = _make_env(
        "bitwise_xor",
        base_kwargs,
        ndim=3,
        height=16,
        store_all_states=True,
        validate_modes=True,
        mode_stats="exact",
    )
    n_base = env_base.n_modes
    assert n_base is not None and n_base > 0

    kwargs = dict(
        base_reward="bitwise_xor",
        base_kwargs=base_kwargs,
        corruption_rate=0.3,
        seed=137,
    )
    env_corrupted = _make_env(
        "corrupted",
        kwargs,
        ndim=3,
        height=16,
        store_all_states=True,
        validate_modes=True,
        mode_stats="exact",
    )
    n_corrupted = env_corrupted.n_modes
    assert n_corrupted is not None and n_corrupted > 0

    # Mode count should not be zero or wildly different.
    ratio = n_corrupted / n_base
    assert 0.1 < ratio < 10.0, f"Mode count ratio {ratio:.2f} is outside expected range"


@pytest.mark.parametrize(
    "base_str,base_kwargs_fn",
    [
        (
            "bitwise_xor",
            lambda: get_reward_presets("bitwise_xor", 3, 16)["easy"],
        ),
        (
            "multiplicative_coprime",
            lambda: dict(
                R0=0.0,
                tier_weights=[1.0, 10.0],
                primes=[2, 3],
                exponent_caps=[2, 2],
                active_dims=[0, 1],
            ),
        ),
        (
            "conditional_multiscale",
            lambda: get_reward_presets("conditional_multiscale", 3, 16)["easy"],
        ),
    ],
)
def test_corrupted_with_all_base_rewards(base_str, base_kwargs_fn):
    """CorruptedReward should work with all tiered base rewards."""
    base_kwargs = base_kwargs_fn()
    kwargs = dict(
        base_reward=base_str,
        base_kwargs=base_kwargs,
        corruption_rate=0.3,
        seed=137,
    )
    env = _make_env("corrupted", kwargs, ndim=3, height=16, validate_modes=True)
    xs = torch.randint(0, 16, (50, 3), dtype=torch.long)
    r = env.reward(env.States(xs))
    assert torch.isfinite(r).all()
    assert (r >= 0).all()


def test_corrupted_preset_environments_trainable():
    """Each corrupted preset should produce a valid, usable environment."""
    presets = get_reward_presets("corrupted", 3, 16)
    for name, kwargs in presets.items():
        env = _make_env(
            "corrupted",
            kwargs,
            ndim=3,
            height=16,
            store_all_states=True,
            validate_modes=True,
        )
        assert env.all_states is not None
        found = env.modes_found(env.all_states)
        assert len(found) > 0, f"Preset '{name}' has no modes"


# -------------------------
# Mode count tests for new rewards
# -------------------------


def test_mode_counts_uniform_random_exact():
    env = _make_env(
        "uniform_random",
        dict(R0=0.1, R_mode=2.0, mode_prob=0.1, seed=42),
        ndim=2,
        height=16,
        store_all_states=True,
        validate_modes=True,
        mode_stats="exact",
    )
    assert env.n_mode_states is not None
    assert env.n_mode_states > 0


def test_mode_counts_corrupted_exact():
    base_kwargs = get_reward_presets("bitwise_xor", 3, 16)["easy"]
    env = _make_env(
        "corrupted",
        dict(
            base_reward="bitwise_xor",
            base_kwargs=base_kwargs,
            corruption_rate=0.3,
            seed=137,
        ),
        ndim=3,
        height=16,
        store_all_states=True,
        validate_modes=True,
        mode_stats="exact",
    )
    assert env.n_mode_states is not None
    assert env.n_mode_states > 0


# -------------------------
# Preset smoke tests
# -------------------------


def test_get_reward_presets_uniform_random():
    presets = get_reward_presets("uniform_random", 3, 16)
    assert set(presets.keys()) == {"easy", "medium", "hard", "challenging", "impossible"}


def test_get_reward_presets_corrupted():
    presets = get_reward_presets("corrupted", 3, 16)
    assert set(presets.keys()) == {"easy", "medium", "hard", "challenging", "impossible"}
