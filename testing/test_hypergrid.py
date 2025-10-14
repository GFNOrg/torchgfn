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


def _adjust_template_minkowski_kwargs_for_grid(
    kwargs: dict, ndim: int, height: int
) -> dict:
    """Return a copy of TemplateMinkowski preset kwargs adjusted to the grid.

    Purpose
    -------
    TemplateMinkowski presets specify L1 radius bands per tier, e.g.,
    ``r_bands=[(r_0^min, r_0^max), (r_1^min, r_1^max), ...]``.
    On a finite grid with dimension ``D`` and height ``H``, the maximum
    reachable L1 sum over a subset of dimensions of size ``D`` is
    $r_{max,reachable} = (H-1) cdot D$.
    This helper ensures each configured band is reachable on the given grid
    by clamping band endpoints to ``cap_sum = (H-1) * D`` and discarding bands
    whose lower endpoint exceeds ``cap_sum``. If no band remains, it inserts a
    single band at ``(cap_sum, cap_sum)``.

    Additionally, to keep the preset consistent, it aligns the lengths of
    ``tier_weights`` and (if present) ``sum_mods`` with the resulting number of
    bands by truncating their lists if necessary. This guarantees the invariant
    $|\text{r_bands}| = |\text{tier_weights}| = |\text{sum_mods}|$
    (when ``sum_mods`` is provided), as expected by the reward implementation.

    Parameters
    ----------
    kwargs: dict
        Original preset dictionary (may contain ``r_bands``, ``tier_weights``,
        and ``sum_mods``).
    ndim: int
        Number of grid dimensions ``D``.
    height: int
        Grid height ``H`` (each coordinate in ``{0, …, H-1}``).

    Returns
    -------
    dict
        A safe-to-use copy of ``kwargs`` where all radius bands are reachable
        and list-valued fields are length-aligned.
    """
    # Ensure r_bands are reachable on this (D,H) by clamping to cap_sum and
    # keeping only bands that are ≤ cap_sum; if none remain, set a single band at cap_sum.
    cap_sum = (height - 1) * ndim
    new_kwargs = dict(kwargs)
    r_bands = list(new_kwargs.get("r_bands", []))
    adjusted = [(min(a, cap_sum), min(b, cap_sum)) for (a, b) in r_bands if a <= cap_sum]
    if not adjusted:
        adjusted = [(cap_sum, cap_sum)]
    new_kwargs["r_bands"] = adjusted
    # Align tier_weights length with adjusted bands if needed by truncating
    tw = list(new_kwargs.get("tier_weights", []))
    if tw and len(tw) != len(adjusted):
        k = min(len(tw), len(adjusted))
        new_kwargs["tier_weights"] = tw[:k]
        # Also align sum_mods if present
        sm = list(new_kwargs.get("sum_mods", []))
        if sm and len(sm) != k:
            new_kwargs["sum_mods"] = sm[:k]
    # Sum mods remain as-is; will be checked against adjusted bands during validation
    return new_kwargs


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
    xs_ok = torch.tensor([[2, 3]], dtype=torch.long)
    xs_bad = torch.tensor([[4, 3]], dtype=torch.long)  # 4 exceeds cap for prime 2
    r_ok = env.reward(env.States(xs_ok))[0].item()
    r_bad = env.reward(env.States(xs_bad))[0].item()
    assert abs(r_ok - 1.0) < 1e-6
    assert abs(r_bad - 0.0) < 1e-6


def test_template_minkowski_reward_values_small():
    kwargs = dict(R0=0.0, tier_weights=[1.0], r_bands=[(2, 2)], sum_mods=[None])
    env = _make_env("template_minkowski", kwargs, ndim=3, height=5)
    xs_ok = torch.tensor([[2, 0, 0]], dtype=torch.long)
    xs_bad = torch.tensor([[1, 0, 0]], dtype=torch.long)
    r_ok = env.reward(env.States(xs_ok))[0].item()
    r_bad = env.reward(env.States(xs_bad))[0].item()
    assert abs(r_ok - 1.0) < 1e-6
    assert abs(r_bad - 0.0) < 1e-6


# -------------------------
# Mode counts and stats (small settings)
# -------------------------


@pytest.mark.parametrize(
    "reward_name,kwargs",
    [
        ("original", dict(R0=0.1, R1=0.5, R2=2.0)),
        ("cosine", dict(R0=0.1, R1=0.5)),
        ("sparse", {}),
        ("deceptive", dict(R0=1e-5, R1=0.1, R2=2.0)),
        ("bitwise_xor", get_reward_presets("bitwise_xor", 3, 16)["easy"]),
        (
            "multiplicative_coprime",
            dict(
                R0=0.0,
                tier_weights=[1.0],
                primes=[2, 3],
                exponent_caps=[1],
                active_dims=[0, 1],
            ),
        ),
        (
            "template_minkowski",
            dict(R0=0.0, tier_weights=[1.0], r_bands=[(2, 2)], sum_mods=[None]),
        ),
    ],
)
def test_mode_counts_small_exact(reward_name, kwargs):
    env = _make_env(
        reward_name,
        kwargs,
        ndim=3,
        height=16,
        store_all_states=True,
        validate_modes=True,
        mode_stats="exact",
    )
    assert env.n_mode_states is not None
    assert env.n_modes >= 1
    assert env.n_mode_states > 0
    assert isinstance(env.n_modes, int)


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


def test_template_minkowski_validate_modes_raises_on_unreachable_band():
    # Max reachable sum is (H-1)*D = 4*2 = 8 < 100
    kwargs = dict(R0=0.0, tier_weights=[1.0], r_bands=[(100, 100)], sum_mods=[None])
    with pytest.raises(ValueError):
        _make_env("template_minkowski", kwargs, ndim=2, height=5, validate_modes=True)


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
    env = _make_env(
        "cosine",
        dict(R0=0.1, R1=0.5, mode_gamma=0.8),
        ndim=2,
        height=16,
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


def test_validate_modes_succeeds_template_minkowski():
    kwargs = dict(R0=0.0, tier_weights=[1.0], r_bands=[(2, 2)], sum_mods=[None])
    env = _make_env("template_minkowski", kwargs, ndim=3, height=5, validate_modes=True)
    assert env.n_actions == env.ndim + 1


# -------------------------
# Mode counts match enumeration across presets (easy/medium)
# -------------------------


@pytest.mark.parametrize(
    "reward_name,kwargs_fn,ndim,height",
    [
        ("original", lambda D, H: get_reward_presets("original", D, H)["easy"], 2, 16),
        ("original", lambda D, H: get_reward_presets("original", D, H)["medium"], 3, 16),
        ("cosine", lambda D, H: get_reward_presets("cosine", D, H)["easy"], 2, 16),
        ("cosine", lambda D, H: get_reward_presets("cosine", D, H)["medium"], 3, 16),
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
        (
            "template_minkowski",
            lambda D, H: _adjust_template_minkowski_kwargs_for_grid(
                get_reward_presets("template_minkowski", D, H)["easy"], D, H
            ),
            3,
            32,
        ),
        (
            "template_minkowski",
            lambda D, H: _adjust_template_minkowski_kwargs_for_grid(
                get_reward_presets("template_minkowski", D, H)["medium"], D, H
            ),
            3,
            64,
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
    ids = env.mode_ids(all_states)
    ids = ids[mask]
    ids = ids[ids >= 0]
    expected = int(torch.unique(ids).numel())
    assert env.n_modes == expected
    assert expected >= 1
