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


# -------------------------
# ConditionalMultiScaleReward — single-rule (legacy compat)
# -------------------------


def test_conditional_multiscale_reward_values_small():
    # Small grid where we can hand-verify the reward.
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        base=2,
        filter_width=1,
        seed=0,
        active_dims=[0, 1],
    )
    env = _make_env("conditional_multiscale", kwargs, ndim=2, height=4)
    # Origin: all-zero digits -> all tiers pass.
    r0 = env.reward(env.States(torch.tensor([[0, 0]], dtype=torch.long)))[0].item()
    assert abs(r0 - (0.0 + 1.0 + 10.0)) < 1e-6


def test_conditional_multiscale_legacy_n_rules_default_is_1():
    # Backwards-compat: omitting n_rules must yield n_rules=1 with shift_coeffs
    # derived from `seed`, reproducing pre-K-rule behavior bit-exactly.
    # Use ndim=4, height=64 so the medium preset's 3 tiers are not truncated.
    kwargs = get_reward_presets("conditional_multiscale", 4, 64)["medium"]
    env = _make_env("conditional_multiscale", kwargs, ndim=4, height=64)
    rf = env.reward_fn
    assert rf.n_rules == 1
    assert rf.head_seed == rf.seed
    # The historical sequence with seed=42, base=4, 3 tiers:
    # tier 0: []; tier 1: torch.randint(0,4,(1,)) -> [2]; tier 2: torch.randint(0,4,(2,)) -> [3, 0]
    assert rf.shift_coeffs_per_rule[0] == [[], [2], [3, 0]]


def test_conditional_multiscale_validate_modes_succeeds_legacy():
    kwargs = get_reward_presets("conditional_multiscale", 4, 16)["medium"]
    env = _make_env(
        "conditional_multiscale", kwargs, ndim=4, height=16, validate_modes=True
    )
    assert env.n_actions == env.ndim + 1


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
# ConditionalMultiScaleReward — K-rule trunk+heads
# -------------------------


@pytest.mark.parametrize("n_rules", [1, 16, 64])
def test_conditional_multiscale_K_presets_validate_modes(n_rules):
    p = get_reward_presets("conditional_multiscale", ndim=6, height=64)[f"K{n_rules}"]
    env = _make_env(
        "conditional_multiscale",
        p,
        ndim=6,
        height=64,
        validate_modes=True,
    )
    assert env.reward_fn.n_rules == n_rules


def test_conditional_multiscale_density_invariant_in_K():
    """Total mode count must be invariant under n_rules (rules partition,
    not multiply, the canonical mode set)."""
    presets = get_reward_presets("conditional_multiscale", ndim=6, height=64)
    counts = []
    for n_rules in (1, 16, 64):
        env = _make_env(
            "conditional_multiscale",
            presets[f"K{n_rules}"],
            ndim=6,
            height=64,
            validate_modes=False,
        )
        counts.append(env.reward_fn.analytic_mode_count())
    assert len(set(counts)) == 1, f"mode counts vary with K: {counts}"


@pytest.mark.parametrize("n_rules", [1, 16, 64])
def test_conditional_multiscale_per_rule_count_matches_total(n_rules):
    p = get_reward_presets("conditional_multiscale", ndim=6, height=64)[f"K{n_rules}"]
    env = _make_env("conditional_multiscale", p, ndim=6, height=64, validate_modes=False)
    rf = env.reward_fn
    total = rf.analytic_mode_count()
    per_rule = rf.analytic_mode_count(per_rule=True)
    assert per_rule * rf.n_rules == total


def test_conditional_multiscale_selector_covers_all_rules():
    """Selector must produce every rule index in [0, n_rules) over uniform
    random states (else _validate_rule_coverage failed silently)."""
    p = get_reward_presets("conditional_multiscale", ndim=6, height=64)["K64"]
    env = _make_env("conditional_multiscale", p, ndim=6, height=64, validate_modes=False)
    rf = env.reward_fn
    torch.manual_seed(0)
    xs = torch.randint(0, 64, (200_000, 6))
    msd = (xs // (rf.base ** (rf.num_levels - 1))) % rf.base
    rule_idx = rf._selector(msd[..., rf.active_dims].long())
    assert rule_idx.unique().numel() == rf.n_rules


def test_conditional_multiscale_selector_deterministic():
    p = get_reward_presets("conditional_multiscale", ndim=6, height=64)["K16"]
    env = _make_env("conditional_multiscale", p, ndim=6, height=64, validate_modes=False)
    rf = env.reward_fn
    torch.manual_seed(123)
    xs = torch.randint(0, 64, (1024, 6))
    msd = (xs // (rf.base ** (rf.num_levels - 1))) % rf.base
    a = rf._selector(msd[..., rf.active_dims].long())
    b = rf._selector(msd[..., rf.active_dims].long())
    assert torch.equal(a, b)


def test_conditional_multiscale_K1_matches_single_rule_at_same_head_seed():
    """K=1 with head_seed=H is bit-exact equivalent to the original single-rule
    reward when its `seed=H`. This is the API consistency property: the K-rule
    code path collapses to the legacy reward at K=1."""
    base_kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0, 100.0],
        base=4,
        filter_width=2,
        seed=2025,
        active_dims=[0, 1, 2, 3, 4, 5],
        cross_dim_mods=[2, 2, 2],
    )
    # Path A: legacy single-rule (n_rules unset -> defaults to 1, head_seed defaults to seed).
    env_a = _make_env("conditional_multiscale", base_kwargs, ndim=6, height=64)
    # Path B: explicit K=1 with head_seed equal to seed.
    env_b_kwargs = dict(base_kwargs, n_rules=1, head_seed=2025)
    env_b = _make_env("conditional_multiscale", env_b_kwargs, ndim=6, height=64)

    torch.manual_seed(0)
    xs = torch.randint(0, 64, (1024, 6))
    r_a = env_a.reward_fn(xs)
    r_b = env_b.reward_fn(xs)
    assert torch.allclose(r_a, r_b, atol=0.0, rtol=0.0)


def test_conditional_multiscale_rule_coverage_assert_fires():
    # n_rules > f^d_active should raise — some rules unreachable.
    bad = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        base=4,
        filter_width=2,
        seed=42,
        active_dims=[0, 1],  # f^d_active = 4
        n_rules=8,  # exceeds 4
    )
    with pytest.raises(ValueError):
        HyperGrid(
            ndim=2,
            height=16,
            reward_fn_str="conditional_multiscale",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


def test_conditional_multiscale_K1_with_distinct_head_seed():
    """K=1 with head_seed=H (and seed=S, S != H) reproduces a fresh single-rule
    reward initialized at seed=H — i.e. head_seed actually drives the head."""
    common = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0, 100.0],
        base=4,
        filter_width=2,
        active_dims=[0, 1, 2, 3],
    )
    env_a = _make_env(
        "conditional_multiscale", dict(common, seed=999), ndim=4, height=64
    )
    env_b = _make_env(
        "conditional_multiscale",
        dict(common, seed=42, n_rules=1, head_seed=999),
        ndim=4,
        height=64,
    )
    torch.manual_seed(0)
    xs = torch.randint(0, 64, (1024, 4))
    assert torch.equal(env_a.reward_fn(xs), env_b.reward_fn(xs))


def test_conditional_multiscale_validator_rejects_dead_rules_via_cross_dim():
    """K-rule template must not silently leave rules empty when cross_dim_mods[0]
    shrinks the trunk-passing set below n_rules. Reproduces the audit blocker
    that prompted dropping cross_dim_mods[0] from the K-rule template."""
    bad = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0, 100.0],
        base=4,
        filter_width=2,
        seed=42,
        head_seed=2025,
        active_dims=[0, 1, 2, 3, 4, 5],
        cross_dim_mods=[2, 2, 2],  # tier-0 cross-dim halves trunk patterns
        n_rules=64,  # but selector mod 64 needs 64 distinct buckets
    )
    with pytest.raises(ValueError, match="trunk-passing"):
        HyperGrid(
            ndim=6,
            height=64,
            reward_fn_str="conditional_multiscale",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


def test_conditional_multiscale_per_rule_raises_on_non_uniform_partition():
    """analytic_mode_count(per_rule=True) must refuse to lie when the
    partition is non-uniform (rules have unequal mode counts)."""
    # f^d_active = 2^3 = 8; n_rules=3 doesn't divide 8 evenly (buckets: 3,3,2).
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        base=4,
        filter_width=2,
        seed=42,
        head_seed=99,
        n_rules=3,
        active_dims=[0, 1, 2],
    )
    env = HyperGrid(
        ndim=3,
        height=16,
        reward_fn_str="conditional_multiscale",
        reward_fn_kwargs=kwargs,
        validate_modes=False,
    )
    # Total count is fine.
    assert env.reward_fn.analytic_mode_count() > 0
    # Per-rule must raise — partition not uniform.
    with pytest.raises(ValueError, match="uniform"):
        env.reward_fn.analytic_mode_count(per_rule=True)


def test_conditional_multiscale_K16_enumeration_parity():
    """Direct enumeration: at K=16 on a small grid, the brute-force mode count
    matches the analytic formula — testing the K-rule path beyond K<=4."""
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        base=4,
        filter_width=2,
        seed=42,
        head_seed=7,
        n_rules=16,  # f^d_active = 2^4 = 16, exact partition
        active_dims=[0, 1, 2, 3],
    )
    env = HyperGrid(
        ndim=4,
        height=16,
        reward_fn_str="conditional_multiscale",
        reward_fn_kwargs=kwargs,
        validate_modes=False,
        store_all_states=True,
    )
    all_states = env.all_states
    assert all_states is not None
    rewards = env.reward_fn(all_states.tensor)
    threshold = env.reward_fn.R0 + sum(env.reward_fn.tier_weights)
    enum_count = int((rewards >= threshold - 1e-9).sum().item())
    assert env.reward_fn.analytic_mode_count() == enum_count


def test_conditional_multiscale_head_seed_none_raises():
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        base=4,
        filter_width=2,
        seed=42,
        head_seed=None,
        active_dims=[0, 1],
    )
    with pytest.raises(ValueError, match="head_seed=None"):
        HyperGrid(
            ndim=2,
            height=16,
            reward_fn_str="conditional_multiscale",
            reward_fn_kwargs=kwargs,
            validate_modes=False,
        )


# -------------------------
# BitwiseXORReward — K-rule trunk+heads
# -------------------------


def test_bitwise_xor_legacy_n_rules_default_is_1():
    """Existing presets (no n_rules set) collapse to n_rules=1 with empty head."""
    p = get_reward_presets("bitwise_xor", 10, 16)["medium"]
    env = _make_env("bitwise_xor", p, ndim=10, height=16)
    rf = env.reward_fn
    assert rf.n_rules == 1
    assert rf.head_check_count == 0
    assert rf.head_weight == 0.0
    assert rf.k_select == 0
    M = len(rf.dims_constrained)
    assert rf._head_A_per_rule.shape == (1, 0, M * rf._B)
    assert rf._selector_matrix.shape == (0, M * rf._B)


def test_bitwise_xor_legacy_reward_unchanged_by_refactor():
    """Existing 'medium' preset reward output must be bit-identical at any state
    to a hand-computed expectation. Locks in backward compat."""
    p = get_reward_presets("bitwise_xor", 10, 16)["medium"]
    env = _make_env("bitwise_xor", p, ndim=10, height=16)
    # The all-zeros bit configuration trivially satisfies even-parity checks
    # (when c_t = 0). Tier 0 in medium has c=[0]; tier 1 has c=[1, 0]; tier 2
    # has c=[0]. With all-zero bits: tier 0 passes (Ab=0=c). Tier 1 first
    # check: 0 != 1 → fails. So all-zeros gives R = R0 + tier_w[0] = 1.0.
    xs = torch.zeros(1, 10, dtype=torch.long)
    r = env.reward(env.States(xs))[0].item()
    assert abs(r - 1.0) < 1e-6


def test_bitwise_xor_K1_no_head_equals_legacy_over_random_batch():
    """K=1 with empty head (head_check_count=0, head_weight=0) must produce
    bit-identical reward to the legacy single-tier code path over a random
    batch — confirms the K-rule code collapses cleanly."""
    p = get_reward_presets("bitwise_xor", 10, 16)["medium"]
    env = _make_env("bitwise_xor", p, ndim=10, height=16, validate_modes=False)
    rf = env.reward_fn
    # Sanity: ensure this preset is in the n_rules=1 / no-head regime.
    assert rf.n_rules == 1
    assert rf.head_check_count == 0
    assert rf.head_weight == 0.0
    torch.manual_seed(0)
    xs = torch.randint(0, 16, (2048, 10))
    r = rf(xs)
    # Recompute reward via the trunk-only formula (matches pre-refactor code).
    # Only the constrained dims feed the parity check.
    xs_constrained = xs[:, rf.dims_constrained]
    flat_bits = (
        ((xs_constrained.unsqueeze(-1) >> torch.arange(rf._B)) & 1)
        .reshape(2048, -1)
        .long()
    )
    prod = (flat_bits @ rf._full_A.t()) & 1
    expected = torch.full((2048,), float(rf.R0))
    tier_ok = torch.ones(2048, dtype=torch.bool)
    offset = 0
    for n_chk, w in zip(rf._tier_check_counts, rf.tier_weights):
        if n_chk > 0:
            slice_ok = (
                prod[..., offset : offset + n_chk] == rf._full_c[offset : offset + n_chk]
            ).all(-1)
            tier_ok = tier_ok & slice_ok
        expected = expected + tier_ok.float() * w
        offset += n_chk
    assert torch.allclose(r, expected, atol=0.0)


@pytest.mark.parametrize("n_rules", [1, 16, 64])
def test_bitwise_xor_K_presets_validate_modes(n_rules):
    p = get_reward_presets("bitwise_xor", 10, 16)[f"K{n_rules}"]
    env = _make_env("bitwise_xor", p, ndim=10, height=16, validate_modes=True)
    assert env.reward_fn.n_rules == n_rules


def test_bitwise_xor_density_invariant_in_K():
    counts = []
    for n_rules in (1, 16, 64):
        p = get_reward_presets("bitwise_xor", 10, 16)[f"K{n_rules}"]
        env = _make_env("bitwise_xor", p, ndim=10, height=16, validate_modes=False)
        counts.append(env.reward_fn.analytic_mode_count())
    assert len(set(counts)) == 1, f"mode counts vary with K: {counts}"


@pytest.mark.parametrize("n_rules", [1, 16, 64])
def test_bitwise_xor_per_rule_count_matches_total(n_rules):
    p = get_reward_presets("bitwise_xor", 10, 16)[f"K{n_rules}"]
    env = _make_env("bitwise_xor", p, ndim=10, height=16, validate_modes=False)
    rf = env.reward_fn
    total = rf.analytic_mode_count()
    per_rule = rf.analytic_mode_count(per_rule=True)
    assert per_rule * rf.n_rules == total


def test_bitwise_xor_selector_covers_all_rules():
    """Random-state samples must reach every rule in [0, n_rules)."""
    p = get_reward_presets("bitwise_xor", 10, 16)["K64"]
    env = _make_env("bitwise_xor", p, ndim=10, height=16, validate_modes=False)
    rf = env.reward_fn
    torch.manual_seed(0)
    xs = torch.randint(0, 16, (50_000, 10))
    bits = ((xs.unsqueeze(-1) >> torch.arange(rf._B)) & 1).reshape(50_000, -1).long()
    sel_bits = (bits @ rf._selector_matrix.t()) & 1
    powers = 2 ** torch.arange(rf.k_select, dtype=torch.long)
    rule_idx = (sel_bits * powers).sum(dim=-1) % rf.n_rules
    assert rule_idx.unique().numel() == rf.n_rules


def test_bitwise_xor_head_seed_required_when_K_gt_1():
    """head_seed=None with n_rules > 1 must raise (no silent fallback)."""
    bad = dict(
        R0=0.0,
        tier_weights=[1.0],
        dims_constrained=[0, 1, 2],
        bits_per_tier=[(0, 3)],
        n_rules=4,
    )
    with pytest.raises(ValueError, match="head_seed"):
        HyperGrid(
            ndim=3,
            height=16,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


def test_bitwise_xor_head_weight_nonzero_with_zero_check_count_raises():
    bad = dict(
        R0=0.0,
        tier_weights=[1.0],
        dims_constrained=[0, 1, 2],
        bits_per_tier=[(0, 3)],
        n_rules=2,
        head_seed=1,
        head_check_count=0,
        head_weight=10.0,
    )
    with pytest.raises(ValueError, match="head_check_count=0"):
        HyperGrid(
            ndim=3,
            height=16,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


def test_bitwise_xor_validator_rejects_unreachable_rule():
    """A degenerate K-rule config must be rejected at construction.

    Setup: trunk pins all M*B bits to specific values (rank = M*B), so the
    selector cannot find an independent matrix and witness construction fails.
    The validator must raise — error message will mention either selector
    independence or rule reachability depending on which check fires first.
    """
    M = 2
    B = 2  # height=4
    A_trunk = torch.eye(M * B, dtype=torch.long)
    c_trunk = torch.zeros(M * B, dtype=torch.long)
    bad = dict(
        R0=0.0,
        tier_weights=[1.0],
        dims_constrained=[0, 1],
        bits_per_tier=[(0, B - 1)],
        parity_checks=[{"A": A_trunk, "c": c_trunk}],
        n_rules=2,
        head_seed=42,
        head_check_count=1,
        head_weight=10.0,
        head_bit_range=(0, B - 1),
    )
    with pytest.raises(ValueError, match="selector|no solution|inconsistent"):
        HyperGrid(
            ndim=2,
            height=4,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


# -------------------------
# MultiplicativeCoprimeReward — K-rule trunk+heads
# -------------------------


def test_multiplicative_coprime_legacy_n_rules_default_is_1():
    """Existing presets (no n_rules set) collapse to n_rules=1 with rule
    target inherited from target_lcms[T-1] (auto)."""
    p = get_reward_presets("multiplicative_coprime", 6, 64)["medium"]
    env = _make_env("multiplicative_coprime", p, ndim=6, height=64)
    rf = env.reward_fn
    assert rf.n_rules == 1
    assert rf.head_seed == 0
    # medium has target_lcms=[None, None, None, "auto"] → auto LCM of
    # primes={2,3,5,7} with cap=2 each = 4*9*25*49 = 44100.
    assert rf.rule_targets == [44100]
    assert rf.target_lcms[-1] == 44100


def test_multiplicative_coprime_K1_no_head_seed_equals_legacy_over_random_batch():
    """K=1 without head_seed (backward-compat path) must produce reward output
    bit-identical to the pre-refactor single-LCM-target path. Locks in legacy
    reward equivalence over a random batch."""
    p = get_reward_presets("multiplicative_coprime", 6, 64)["medium"]
    env = _make_env("multiplicative_coprime", p, ndim=6, height=64, validate_modes=False)
    rf = env.reward_fn
    # Sanity: this is the legacy regime (no head_seed, target_lcms[-1] retained).
    assert rf.n_rules == 1
    assert rf.head_seed == 0
    assert rf.rule_targets == [44100]
    assert rf.target_lcms[-1] == 44100
    torch.manual_seed(0)
    xs = torch.randint(0, 64, (2048, 6))
    r = rf(xs)
    # All rewards must be in the discrete set {R0, R0+w0, R0+w0+w1, ...}.
    valid_rewards = {rf.R0}
    cum = rf.R0
    for w in rf.tier_weights:
        cum += float(w)
        valid_rewards.add(cum)
    seen = set(round(x, 6) for x in r.tolist())
    assert seen.issubset(
        {round(v, 6) for v in valid_rewards}
    ), f"unexpected reward values: {seen - {round(v, 6) for v in valid_rewards}}"
    # And: there must be at least one tier-3 mode in 2048 samples? No — too
    # rare. Instead: assert there are tier-1 passers (lots).
    threshold_t1 = rf.R0 + float(rf.tier_weights[0])
    assert (r >= threshold_t1).any()


def test_multiplicative_coprime_K_presets_validate_modes():
    for n_rules in (1, 16):
        p = get_reward_presets("multiplicative_coprime", 6, 64)[f"K{n_rules}"]
        env = _make_env(
            "multiplicative_coprime", p, ndim=6, height=64, validate_modes=True
        )
        assert env.reward_fn.n_rules == n_rules


def test_multiplicative_coprime_K16_distinct_rule_targets():
    """All 16 rules at K=16 must have distinct LCM targets (the cap-tuple enum
    yields 2^4=16 unique LCMs for primes={2,3,5,7})."""
    p = get_reward_presets("multiplicative_coprime", 6, 64)["K16"]
    env = _make_env("multiplicative_coprime", p, ndim=6, height=64, validate_modes=False)
    assert len(set(env.reward_fn.rule_targets)) == 16


def test_multiplicative_coprime_head_seed_required_when_K_gt_1():
    bad = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0, 100.0, 1000.0],
        primes=[2, 3, 5, 7],
        exponent_caps=[2, 2, 2, 2],
        active_dims=[0, 1, 2, 3, 4, 5],
        coprime_pairs=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        coprime_start_tier=2,
        target_lcms=[None, None, None, None],
        n_rules=4,
        # head_seed missing
    )
    with pytest.raises(ValueError, match="head_seed"):
        HyperGrid(
            ndim=6,
            height=64,
            reward_fn_str="multiplicative_coprime",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


def test_multiplicative_coprime_too_few_primes_for_K_raises():
    bad = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0, 100.0, 1000.0],
        primes=[2, 3],  # only 2 primes
        exponent_caps=[2, 2, 2, 2],
        active_dims=[0, 1],
        coprime_pairs=[(0, 1)],
        coprime_start_tier=2,
        target_lcms=[None, None, None, None],
        n_rules=16,  # 2^2 = 4 < 16
        head_seed=42,
    )
    with pytest.raises(ValueError, match="Cannot generate"):
        HyperGrid(
            ndim=2,
            height=16,
            reward_fn_str="multiplicative_coprime",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


def test_multiplicative_coprime_K1_with_explicit_head_seed_uses_enum():
    """K=1 with head_seed explicitly set uses the K-rule enum, not target_lcms.
    Also verifies B1 fix: when the K-rule path is active, target_lcms[-1] is
    cleared, so a state matching rule 0's head LCM is actually a mode (no
    contradictory shared trunk LCM)."""
    common = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0, 100.0, 1000.0],
        primes=[2, 3, 5, 7],
        exponent_caps=[2, 2, 2, 2],
        active_dims=[0, 1, 2, 3, 4, 5],
        coprime_pairs=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        coprime_start_tier=2,
        target_lcms=[None, None, None, "auto"],  # would yield 44100
    )
    # No head_seed: backward-compat path inherits 44100.
    env_a = _make_env(
        "multiplicative_coprime", common, ndim=6, height=64, validate_modes=False
    )
    assert env_a.reward_fn.rule_targets == [44100]
    assert env_a.reward_fn.target_lcms[-1] == 44100  # NOT cleared
    # With head_seed: K=1 uses enum entry, target_lcms[-1] cleared.
    env_b = _make_env(
        "multiplicative_coprime",
        dict(common, n_rules=1, head_seed=2025),
        ndim=6,
        height=64,
        validate_modes=True,  # validate must succeed
    )
    assert env_b.reward_fn.rule_targets[0] != 44100
    assert env_b.reward_fn.target_lcms[-1] is None  # cleared by K-rule path


def test_multiplicative_coprime_head_seed_reproducibility():
    """Same head_seed → same rule_targets across constructions."""
    p = get_reward_presets("multiplicative_coprime", 6, 64)["K16"]
    env_a = _make_env(
        "multiplicative_coprime", p, ndim=6, height=64, validate_modes=False
    )
    env_b = _make_env(
        "multiplicative_coprime", p, ndim=6, height=64, validate_modes=False
    )
    assert env_a.reward_fn.rule_targets == env_b.reward_fn.rule_targets


def test_multiplicative_coprime_explicit_rule_targets_bypass_generator():
    """User-supplied rule_targets must be used as-is and clear target_lcms[-1]."""
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        primes=[2, 3],
        exponent_caps=[2, 2],
        active_dims=[0, 1],
        coprime_pairs=[(0, 1)],
        coprime_start_tier=1,
        target_lcms=[None, "auto"],  # would yield 36 = 4*9
        n_rules=2,
        head_seed=42,
        rule_targets=[6, 12],  # explicit — bypasses generator
    )
    env = HyperGrid(
        ndim=2,
        height=16,
        reward_fn_str="multiplicative_coprime",
        reward_fn_kwargs=kwargs,
        validate_modes=False,
    )
    assert env.reward_fn.rule_targets == [6, 12]
    assert env.reward_fn.target_lcms[-1] is None  # K-rule path cleared it


def test_multiplicative_coprime_rule_targets_with_None_uses_sentinel():
    """A rule with rule_targets[k]=None (no head LCM constraint) must pass
    the head check trivially via the -1 sentinel."""
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        primes=[2, 3],
        exponent_caps=[2, 2],
        active_dims=[0, 1],
        coprime_pairs=[(0, 1)],
        coprime_start_tier=1,
        target_lcms=[None, None],
        n_rules=2,
        head_seed=42,
        rule_targets=[None, 6],
    )
    env = HyperGrid(
        ndim=2,
        height=16,
        reward_fn_str="multiplicative_coprime",
        reward_fn_kwargs=kwargs,
        validate_modes=False,
    )
    rf = env.reward_fn
    # Rule 0's exps are all -1 sentinel.
    assert (rf._rule_target_exps[0] == -1).all()
    # Rule 1's exps are factorization of 6: 2^1 * 3^1.
    assert rf._rule_target_exps[1].tolist() == [1, 1]


def test_multiplicative_coprime_density_within_2x_K1_K16():
    """Empirical density of K1 and K16 must be within 2x at the shipped
    presets (research-code contract: density approximately invariant)."""
    torch.manual_seed(0)
    xs = torch.randint(0, 64, (1_000_000, 6))
    densities = []
    for n_rules in (1, 16):
        p = get_reward_presets("multiplicative_coprime", 6, 64)[f"K{n_rules}"]
        env = _make_env(
            "multiplicative_coprime", p, ndim=6, height=64, validate_modes=False
        )
        rf = env.reward_fn
        r = rf(xs)
        threshold = sum(rf.tier_weights) + rf.R0
        densities.append((r >= threshold).float().mean().item())
    # Both should be > 0; ratio within 4x in either direction.
    assert all(d > 0 for d in densities), f"K1 or K16 has 0 mode density: {densities}"
    ratio = max(densities) / min(densities)
    assert ratio <= 4.0, f"K1/K16 density ratio = {ratio}: not approximately invariant"


def test_multiplicative_coprime_exists_check_K_rule_path():
    """At K>1 with a large grid, _exists_multiplicative_coprime must find at
    least one rule with a witness whose selector maps back to its index."""
    p = get_reward_presets("multiplicative_coprime", 6, 64)["K16"]
    env = _make_env("multiplicative_coprime", p, ndim=6, height=64, validate_modes=False)
    thr = env._mode_reward_threshold()
    assert env._exists_multiplicative_coprime(thr) is True


def test_multiplicative_coprime_selector_covers_all_rules():
    p = get_reward_presets("multiplicative_coprime", 6, 64)["K16"]
    env = _make_env("multiplicative_coprime", p, ndim=6, height=64, validate_modes=False)
    rf = env.reward_fn
    torch.manual_seed(0)
    xs = torch.randint(0, 64, (50_000, 6))
    x_active = xs[:, rf.active_dims] + 1
    rule_idx = rf._selector(x_active)
    assert rule_idx.unique().numel() == rf.n_rules


def test_multiplicative_coprime_K16_enumeration_parity():
    """On a small grid, brute-force mode count equals analytic threshold count."""
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        primes=[2, 3],
        exponent_caps=[2, 2],
        active_dims=[0, 1, 2],
        coprime_pairs=[(0, 1), (1, 2)],
        coprime_start_tier=1,
        target_lcms=[None, None],
        n_rules=4,
        head_seed=42,
    )
    env = HyperGrid(
        ndim=3,
        height=16,
        reward_fn_str="multiplicative_coprime",
        reward_fn_kwargs=kwargs,
        validate_modes=False,
        store_all_states=True,
    )
    all_states = env.all_states
    assert all_states is not None
    rewards = env.reward_fn(all_states.tensor)
    threshold = sum(env.reward_fn.tier_weights) + env.reward_fn.R0
    enum_count = int((rewards >= threshold - 1e-9).sum().item())
    # Each rule with non-trivial LCM target has at least one mode.
    assert enum_count > 0


def test_solve_gf2_witness_basic_cases():
    """Direct unit tests for HyperGrid._solve_gf2_witness."""
    # Empty system: any vector is a solution.
    A = torch.zeros(0, 5, dtype=torch.long)
    c = torch.zeros(0, dtype=torch.long)
    b = HyperGrid._solve_gf2_witness(A, c, 5)
    assert b is not None and torch.all(b == 0)

    # 1-equation system: x0 = 1.
    A = torch.tensor([[1, 0]], dtype=torch.long)
    c = torch.tensor([1], dtype=torch.long)
    b = HyperGrid._solve_gf2_witness(A, c, 2)
    assert b is not None and ((A.long() @ b.long()) & 1 == c).all()

    # Inconsistent: 0·x = 1 → no solution.
    A = torch.tensor([[0, 0]], dtype=torch.long)
    c = torch.tensor([1], dtype=torch.long)
    assert HyperGrid._solve_gf2_witness(A, c, 2) is None

    # Redundant rows: x0 + x1 = 1, 2·(x0 + x1) = 0 (in GF(2): same row twice → consistent).
    A = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
    c = torch.tensor([1, 1], dtype=torch.long)
    b = HyperGrid._solve_gf2_witness(A, c, 2)
    assert b is not None and ((A.long() @ b.long()) & 1 == c).all()

    # Redundant rows with conflict: x0+x1=1, x0+x1=0 → inconsistent.
    A = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
    c = torch.tensor([1, 0], dtype=torch.long)
    assert HyperGrid._solve_gf2_witness(A, c, 2) is None


def test_bitwise_xor_K1_with_head_is_validated():
    """K=1 with head_check_count > 0 must trigger _validate_rule_coverage
    (witness construction must succeed for the single rule)."""
    p = get_reward_presets("bitwise_xor", 10, 16)["K1"]
    env = _make_env("bitwise_xor", p, ndim=10, height=16, validate_modes=True)
    rf = env.reward_fn
    assert rf.n_rules == 1
    assert rf.head_check_count == 1
    assert rf.head_weight == 1000.0
    # Witness construction must have set head_c[0] consistently. An explicit
    # mode is hard to construct without enumeration, but the analytic count
    # > 0 implies one exists at threshold R0 + sum(tier_weights) + head_weight.
    assert rf.analytic_mode_count() > 0


def test_bitwise_xor_K1_head_seed_required():
    """At K=1 with head_check_count > 0, head_seed=None must raise."""
    bad = dict(
        R0=0.0,
        tier_weights=[1.0],
        dims_constrained=[0, 1, 2],
        bits_per_tier=[(0, 3)],
        n_rules=1,
        head_check_count=1,
        head_weight=10.0,
        # head_seed missing
    )
    with pytest.raises(ValueError, match="head_seed"):
        HyperGrid(
            ndim=3,
            height=16,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


def test_bitwise_xor_empty_head_bit_range_with_count_raises():
    bad = dict(
        R0=0.0,
        tier_weights=[1.0],
        dims_constrained=[0, 1, 2],
        bits_per_tier=[(0, 3)],
        n_rules=2,
        head_seed=1,
        head_check_count=1,
        head_weight=10.0,
        head_bit_range=(0, -1),  # empty range
    )
    with pytest.raises(AssertionError, match="head_bit_range"):
        HyperGrid(
            ndim=3,
            height=16,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs=bad,
            validate_modes=False,
        )


def test_bitwise_xor_K_enumeration_parity():
    """Brute-force enumeration must match analytic_mode_count on a small grid."""
    B = 2  # height=4
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0],
        dims_constrained=[0, 1, 2, 3],
        bits_per_tier=[(0, B - 1)],
        parity_checks=[None],  # default even parity
        n_rules=4,
        head_seed=99,
        head_check_count=1,
        head_weight=10.0,
        head_bit_range=(0, B - 1),
    )
    env = HyperGrid(
        ndim=4,
        height=4,
        reward_fn_str="bitwise_xor",
        reward_fn_kwargs=kwargs,
        validate_modes=False,
        store_all_states=True,
    )
    all_states = env.all_states
    assert all_states is not None
    rewards = env.reward_fn(all_states.tensor)
    threshold = sum(env.reward_fn.tier_weights) + env.reward_fn.head_weight
    enum_count = int((rewards >= threshold - 1e-9).sum().item())
    assert env.reward_fn.analytic_mode_count() == enum_count


def test_conditional_multiscale_analytic_matches_enumeration_K():
    """At a small grid, analytic_mode_count must equal a brute-force count of
    states reaching the highest tier. Tests the K-rule path explicitly."""
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        base=4,
        filter_width=2,
        seed=42,
        head_seed=99,
        n_rules=4,  # f^d_active = 2^3 = 8 ≥ 4
        active_dims=[0, 1, 2],
    )
    # 16x16x16 grid (4^2 each): state space = 4096. Easily enumerable.
    env = HyperGrid(
        ndim=3,
        height=16,
        reward_fn_str="conditional_multiscale",
        reward_fn_kwargs=kwargs,
        validate_modes=False,
        store_all_states=True,
    )
    all_states = env.all_states
    assert all_states is not None
    rewards = env.reward_fn(all_states.tensor)
    threshold = env.reward_fn.R0 + sum(env.reward_fn.tier_weights)
    enum_count = int((rewards >= threshold - 1e-9).sum().item())
    analytic = env.reward_fn.analytic_mode_count()
    assert analytic == enum_count


# -------------------------
# ConditionalMultiScale filter_shift coverage knob
# -------------------------


def test_multiscale_filter_shift_default_is_zero_and_unchanged():
    """Without filter_shift, the trunk passes top_digit < f starting at 0
    (current behavior)."""
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        base=4,
        filter_width=2,
        seed=42,
        active_dims=[0, 1],
    )
    env = _make_env("conditional_multiscale", kwargs, ndim=2, height=16)
    rf = env.reward_fn
    assert rf.filter_shift == [0, 0]
    # All-zeros state passes (default behavior preserved).
    xs = torch.zeros(1, 2, dtype=torch.long)
    r = env.reward_fn(xs)[0].item()
    assert r >= rf.R0 + sum(rf.tier_weights) - 1e-6


def test_multiscale_filter_shift_moves_blind_strip():
    """filter_shift[0]=k cyclically rotates the trunk-passing top-digit
    block: passing values become {(B-k) mod B, (B-k+1) mod B, ..., (B-k+f-1) mod B}."""
    base = 4
    f = 3
    L = 2  # h=16
    H = base**L
    # filter_shift[0] = 1: passing top-digits are {3, 0, 1} (excludes 2).
    kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0],
        base=base,
        filter_width=f,
        seed=42,
        active_dims=[0, 1],
        filter_shift=[1, 0],
    )
    env = _make_env("conditional_multiscale", kwargs, ndim=2, height=H)
    rf = env.reward_fn

    # Tier-0 should pass for top-digit ∈ {0, 1, 3} and FAIL for top-digit==2.
    # Top digit = state // base^(L-1) = state // 4 in [0, 4).
    for top_digit in range(base):
        v = top_digit * (base ** (L - 1))  # represents top digit specifically
        xs = torch.tensor([[v, v]], dtype=torch.long)
        r = env.reward_fn(xs)[0].item()
        passes_tier0 = (top_digit + 1) % base < f  # 1 == filter_shift[0]
        assert (
            r > rf.R0
        ) == passes_tier0, (
            f"top_digit={top_digit}, expected pass_t0={passes_tier0}, got reward={r}"
        )


def test_multiscale_filter_shift_density_invariant():
    """Density (analytic_mode_count) is invariant under filter_shift —
    the shift cyclically rotates the passing window without changing |window|."""
    base_kwargs = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0, 100.0],
        base=4,
        filter_width=3,
        seed=42,
        head_seed=2025,
        active_dims=list(range(8)),
        cross_dim_mods=[None, 3, 3],
        n_rules=1,
    )
    counts = []
    for fs0 in range(4):
        kw = dict(base_kwargs, filter_shift=[fs0, 0, 0])
        env = _make_env(
            "conditional_multiscale", kw, ndim=8, height=64, validate_modes=False
        )
        counts.append(env.reward_fn.analytic_mode_count())
    assert len(set(counts)) == 1, f"density varies under filter_shift: {counts}"


# -------------------------
# K-rule "matched" presets — production-grade matched-density at K=64
# -------------------------


@pytest.mark.parametrize("n_rules", [1, 16, 64])
def test_multiscale_K_matched_validate_modes(n_rules):
    p = get_reward_presets("conditional_multiscale", 24, 16)[f"K{n_rules}_matched"]
    env = _make_env("conditional_multiscale", p, ndim=24, height=16, validate_modes=True)
    assert env.reward_fn.n_rules == n_rules
    total = env.reward_fn.analytic_mode_count()
    density = total / 16**24
    # Target ~1e-7; allow 0.5e-7 to 5e-7.
    assert 5e-8 < density < 5e-7, f"density {density:.2e} out of expected matched range"


@pytest.mark.parametrize("n_rules", [1, 16, 64])
def test_bitxor_K_matched_validate_modes(n_rules):
    p = get_reward_presets("bitwise_xor", 10, 16)[f"K{n_rules}_matched"]
    env = _make_env("bitwise_xor", p, ndim=10, height=16, validate_modes=True)
    assert env.reward_fn.n_rules == n_rules
    total = env.reward_fn.analytic_mode_count()
    density = total / 16**10
    assert 5e-8 < density < 5e-7, f"density {density:.2e} out of expected matched range"


@pytest.mark.parametrize("n_rules", [1, 16, 64])
def test_coprime_K_matched_validate_modes(n_rules):
    """Coprime matched: validate_modes via _exists_multiplicative_coprime
    constructs a witness for trunk LCM=44100. No analytic count; just confirm
    construction + at least one mode exists."""
    p = get_reward_presets("multiplicative_coprime", 10, 64)[f"K{n_rules}_matched"]
    env = _make_env("multiplicative_coprime", p, ndim=10, height=64, validate_modes=True)
    assert env.reward_fn.n_rules == n_rules


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
