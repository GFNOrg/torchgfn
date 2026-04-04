"""Tests for dependency-free logic in hard-to-test modules.

Covers: hypergrid reward classes, mode existence checks, GF(2) solver,
distributed report functions, and utils/graphs edge index functions.
"""

import logging

import pytest
import torch

from gfn.gym import HyperGrid

# ===========================================================================
# Hypergrid reward classes — direct __call__ on tensors
# ===========================================================================


class TestOriginalReward:
    def test_base_reward(self):
        """States far from any ring should get only R0."""
        env = HyperGrid(ndim=2, height=16, validate_modes=False)
        # Center state: (7,7) => ax = |7/15 - 0.5| ≈ 0.033 => no rings
        center = torch.tensor([[7, 7]], dtype=torch.long)
        r = env.reward_fn(center)
        assert r.item() == pytest.approx(0.1, abs=0.01)  # just R0

    def test_mode_state_gets_high_reward(self):
        """Known mode at (2,2) for height=16 should get R0+R1+R2."""
        env = HyperGrid(ndim=2, height=16, validate_modes=False)
        mode = torch.tensor([[2, 2]], dtype=torch.long)
        r = env.reward_fn(mode)
        assert r.item() == pytest.approx(0.1 + 0.5 + 2.0, abs=0.01)

    def test_batch(self):
        env = HyperGrid(ndim=2, height=16, validate_modes=False)
        states = torch.randint(0, 16, (32, 2))
        r = env.reward_fn(states)
        assert r.shape == (32,)
        assert (r >= 0.1).all()  # at least R0


class TestCosineReward:
    def test_batch_output(self):
        env = HyperGrid(ndim=2, height=16, reward_fn_str="cosine", validate_modes=False)
        states = torch.randint(0, 16, (32, 2))
        r = env.reward_fn(states)
        assert r.shape == (32,)
        assert (r > 0).all()

    def test_center_gets_peak(self):
        """Center of the grid should be near a reward peak."""
        env = HyperGrid(ndim=2, height=16, reward_fn_str="cosine", validate_modes=False)
        center = torch.tensor([[8, 8]], dtype=torch.long)  # ax ≈ 0.03
        r_center = env.reward_fn(center)
        # Compare with far corner
        corner = torch.tensor([[0, 0]], dtype=torch.long)  # ax = 0.5
        r_corner = env.reward_fn(corner)
        assert r_center.item() > r_corner.item()


class TestSparseReward:
    def test_target_gets_high_reward(self):
        """Exact target states should get reward ≈ 1."""
        env = HyperGrid(ndim=2, height=8, reward_fn_str="sparse", validate_modes=False)
        # The zero vector is one of the target states (all 1s case).
        # Actually let's use a permutation of [1, height-2] = [1, 6]
        target = torch.tensor([[1, 6]], dtype=torch.long)
        r = env.reward_fn(target)
        assert r.item() > 0.5  # should match a target

    def test_non_target_gets_eps(self):
        """Non-target states should get only eps."""
        env = HyperGrid(ndim=2, height=8, reward_fn_str="sparse", validate_modes=False)
        non_target = torch.tensor([[3, 3]], dtype=torch.long)
        r = env.reward_fn(non_target)
        assert r.item() < 0.01


class TestDeceptiveReward:
    def test_center_boosted(self):
        """Center region should get R0+R1 (outer ring not canceled)."""
        env = HyperGrid(
            ndim=2, height=16, reward_fn_str="deceptive", validate_modes=False
        )
        center = torch.tensor([[7, 7]], dtype=torch.long)
        r_center = env.reward_fn(center)
        # Center: ax ≈ 0.03, cancel_outer=0 (ax < 0.1), ring_band=0
        # Corner: ax = 0.5, cancel_outer = R1, ring_band = 0
        corner = torch.tensor([[0, 0]], dtype=torch.long)
        r_corner = env.reward_fn(corner)
        # Center should have higher reward (keeps R1, corner loses it)
        assert r_center.item() > r_corner.item()

    def test_batch(self):
        env = HyperGrid(
            ndim=2, height=16, reward_fn_str="deceptive", validate_modes=False
        )
        states = torch.randint(0, 16, (32, 2))
        r = env.reward_fn(states)
        assert r.shape == (32,)
        assert (r > 0).all()


class TestBitwiseXORReward:
    def test_zero_state_satisfies_even_parity(self):
        """All-zero state has even parity at every bit-plane => max reward."""
        env = HyperGrid(
            ndim=2, height=64, reward_fn_str="bitwise_xor", validate_modes=False
        )
        zero = torch.tensor([[0, 0]], dtype=torch.long)
        r = env.reward_fn(zero)
        # R0=0.1 (default from HyperGrid) + sum(tier_weights) = 0.1 + 1+10+100
        # Actually R0 comes from reward_fn_kwargs; BitwiseXOR uses R0 from kwargs
        # Default: R0=0.1, tier_weights=[1,10,100] => total ~111.1
        assert r.item() > 100.0  # satisfies all tiers

    def test_odd_parity_fails(self):
        """A state with odd bit parity at bit 0 should fail tier 0."""
        env = HyperGrid(
            ndim=2, height=64, reward_fn_str="bitwise_xor", validate_modes=False
        )
        # State (1, 0): bit 0 of dim 0 is 1, bit 0 of dim 1 is 0 => odd parity
        odd = torch.tensor([[1, 0]], dtype=torch.long)
        r = env.reward_fn(odd)
        # Should only get R0 (failed first tier, cumulative so no tiers)
        assert r.item() < 1.0  # only base reward

    def test_batch(self):
        env = HyperGrid(
            ndim=2, height=64, reward_fn_str="bitwise_xor", validate_modes=False
        )
        states = torch.randint(0, 64, (32, 2))
        r = env.reward_fn(states)
        assert r.shape == (32,)


class TestMultiplicativeCoprimeReward:
    def test_all_ones_satisfies_all_tiers(self):
        """All-ones state: each dim=1, which has zero exponents for all primes."""
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="multiplicative_coprime",
            validate_modes=False,
        )
        ones = torch.tensor([[1, 1]], dtype=torch.long)
        r = env.reward_fn(ones)
        # Should get full reward since 1 trivially satisfies coprime constraints
        assert r.item() > 0

    def test_zero_state_gets_base(self):
        """State with 0 should fail prime-support (can't factorize 0)."""
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="multiplicative_coprime",
            validate_modes=False,
        )
        zero = torch.tensor([[0, 0]], dtype=torch.long)
        r = env.reward_fn(zero)
        # 0 is invalid for prime-support, so only base R0
        assert r.item() < 1.0

    def test_batch(self):
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="multiplicative_coprime",
            validate_modes=False,
        )
        states = torch.randint(0, 64, (32, 2))
        r = env.reward_fn(states)
        assert r.shape == (32,)


class TestConditionalMultiScaleReward:
    def test_zero_state(self):
        """All-zero state: all digits are zero, should satisfy base constraints."""
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="conditional_multiscale",
            validate_modes=False,
        )
        zero = torch.tensor([[0, 0]], dtype=torch.long)
        r = env.reward_fn(zero)
        assert r.item() > 0

    def test_batch(self):
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="conditional_multiscale",
            validate_modes=False,
        )
        states = torch.randint(0, 64, (32, 2))
        r = env.reward_fn(states)
        assert r.shape == (32,)

    def test_mode_threshold(self):
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="conditional_multiscale",
            validate_modes=False,
        )
        t = env._mode_reward_threshold()
        assert t > 0

    def test_analytic_mode_count(self):
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="conditional_multiscale",
            validate_modes=False,
        )
        count = env.reward_fn.analytic_mode_count()
        assert count > 0

    def test_analytic_log_partition(self):
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="conditional_multiscale",
            validate_modes=False,
        )
        lp = env.reward_fn.analytic_log_partition()
        assert lp > 0


# ===========================================================================
# GF(2) Gaussian elimination solver
# ===========================================================================


class TestSolveGF2:
    def test_trivial_system(self):
        """1x=1 in GF(2) has solution x=1."""
        A = torch.tensor([[1]])
        c = torch.tensor([1])
        assert HyperGrid._solve_gf2_has_solution(A, c) is True

    def test_inconsistent_system(self):
        """0x=1 in GF(2) has no solution."""
        A = torch.tensor([[0]])
        c = torch.tensor([1])
        assert HyperGrid._solve_gf2_has_solution(A, c) is False

    def test_2x2_consistent(self):
        """x1 XOR x2 = 0; x1 = 1 => x1=1, x2=1."""
        A = torch.tensor([[1, 1], [1, 0]])
        c = torch.tensor([0, 1])
        assert HyperGrid._solve_gf2_has_solution(A, c) is True

    def test_2x2_inconsistent(self):
        """x1=0 AND x1=1 => contradiction."""
        A = torch.tensor([[1, 0], [1, 0]])
        c = torch.tensor([0, 1])
        assert HyperGrid._solve_gf2_has_solution(A, c) is False

    def test_empty_matrix(self):
        """Empty system is always consistent."""
        A = torch.empty(0, 3)
        c = torch.empty(0)
        assert HyperGrid._solve_gf2_has_solution(A, c) is True

    def test_underdetermined_system(self):
        """1 equation, 3 variables: always solvable (many solutions)."""
        A = torch.tensor([[1, 1, 0]])
        c = torch.tensor([1])
        assert HyperGrid._solve_gf2_has_solution(A, c) is True


# ===========================================================================
# Mode existence checks (via HyperGrid init with validate_modes=True)
# ===========================================================================


class TestModeExistenceChecks:
    def test_original_modes_exist(self):
        env = HyperGrid(ndim=2, height=16, reward_fn_str="original", validate_modes=True)
        assert env is not None

    def test_deceptive_modes_exist(self):
        env = HyperGrid(
            ndim=2, height=16, reward_fn_str="deceptive", validate_modes=True
        )
        assert env is not None

    def test_sparse_modes_exist(self):
        env = HyperGrid(ndim=2, height=8, reward_fn_str="sparse", validate_modes=True)
        assert env is not None

    def test_bitwise_xor_modes_exist(self):
        env = HyperGrid(
            ndim=2, height=64, reward_fn_str="bitwise_xor", validate_modes=True
        )
        assert env is not None

    def test_multiplicative_coprime_modes_exist(self):
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="multiplicative_coprime",
            validate_modes=True,
        )
        assert env is not None

    def test_no_modes_raises(self):
        """Tiny grid should fail validation."""
        with pytest.raises(ValueError, match="No states"):
            HyperGrid(ndim=2, height=2, reward_fn_str="original", validate_modes=True)

    def test_fallback_random_check(self):
        """Exercise the random fallback path on a custom reward."""
        # conditional_multiscale uses a different code path
        env = HyperGrid(
            ndim=2,
            height=64,
            reward_fn_str="conditional_multiscale",
            validate_modes=True,
        )
        assert env is not None


# ===========================================================================
# Distributed report functions (pure list/dict math, no MPI)
# ===========================================================================


class TestDistributedReports:
    def test_report_load_imbalance(self, caplog):
        # Import the function but it normally needs mpi4py at module level.
        # We import only the functions, which are pure python.
        # Skip if mpi4py not available
        try:
            from gfn.utils.distributed import report_load_imbalance
        except ImportError:
            pytest.skip("mpi4py not installed")

        timing = [
            {"step_a": [1.0, 2.0, 3.0], "step_b": [0.5, 0.5, 0.5]},
            {"step_a": [1.5, 2.5, 3.5], "step_b": [0.4, 0.6, 0.5]},
        ]
        with caplog.at_level(logging.INFO):
            report_load_imbalance(timing, world_size=2)
        assert "step_a" in caplog.text
        assert "step_b" in caplog.text

    def test_report_time_info(self, caplog):
        try:
            from gfn.utils.distributed import report_time_info
        except ImportError:
            pytest.skip("mpi4py not installed")

        timing = [
            {"train": [0.1, 0.2, 0.3]},
            {"train": [0.15, 0.25, 0.35]},
        ]
        with caplog.at_level(logging.INFO):
            report_time_info(timing, world_size=2)
        assert "train" in caplog.text


# ===========================================================================
# utils/graphs.py — edge index functions (pure torch)
# ===========================================================================


class TestGraphEdgeIndices:
    """Test get_edge_indices and from_edge_indices (torch-only, no geometric)."""

    def test_get_edge_indices_undirected(self):
        try:
            from gfn.utils.graphs import get_edge_indices
        except ImportError:
            pytest.skip("torch_geometric not installed")

        ei0, ei1 = get_edge_indices(
            n_nodes=3, is_directed=False, device=torch.device("cpu")
        )
        # 3 nodes, undirected: 3 choose 2 = 3 edges
        assert ei0.shape[0] == 3
        assert ei1.shape[0] == 3

    def test_get_edge_indices_directed(self):
        try:
            from gfn.utils.graphs import get_edge_indices
        except ImportError:
            pytest.skip("torch_geometric not installed")

        ei0, _ei1 = get_edge_indices(
            n_nodes=3, is_directed=True, device=torch.device("cpu")
        )
        # 3 nodes, directed: 3*2 = 6 edges (no self-loops)
        assert ei0.shape[0] == 6

    def test_from_edge_indices_roundtrip_undirected(self):
        try:
            from gfn.utils.graphs import from_edge_indices, get_edge_indices
        except ImportError:
            pytest.skip("torch_geometric not installed")

        n = 4
        ei0, ei1 = get_edge_indices(
            n_nodes=n, is_directed=False, device=torch.device("cpu")
        )
        idx = from_edge_indices(ei0, ei1, n_nodes=n, is_directed=False)
        assert idx.shape == (ei0.shape[0],)
        assert idx.unique().shape[0] == idx.shape[0]

    def test_from_edge_indices_roundtrip_directed(self):
        try:
            from gfn.utils.graphs import from_edge_indices, get_edge_indices
        except ImportError:
            pytest.skip("torch_geometric not installed")

        n = 4
        ei0, ei1 = get_edge_indices(
            n_nodes=n, is_directed=True, device=torch.device("cpu")
        )
        idx = from_edge_indices(ei0, ei1, n_nodes=n, is_directed=True)
        assert idx.shape == (ei0.shape[0],)
        assert idx.unique().shape[0] == idx.shape[0]

    def test_single_node(self):
        try:
            from gfn.utils.graphs import get_edge_indices
        except ImportError:
            pytest.skip("torch_geometric not installed")

        ei0, _ = get_edge_indices(
            n_nodes=1, is_directed=False, device=torch.device("cpu")
        )
        assert ei0.shape[0] == 0

    def test_two_nodes_undirected(self):
        try:
            from gfn.utils.graphs import get_edge_indices
        except ImportError:
            pytest.skip("torch_geometric not installed")

        ei0, _ = get_edge_indices(
            n_nodes=2, is_directed=False, device=torch.device("cpu")
        )
        assert ei0.shape[0] == 1


# ===========================================================================
# Hypergrid enumerate + partition for small grids
# ===========================================================================


class TestHyperGridEnumeration:
    def test_enumerate_all_states_small(self):
        env = HyperGrid(ndim=2, height=3, store_all_states=True, validate_modes=False)
        assert env._all_states_tensor is not None
        assert env._all_states_tensor.shape == (9, 2)  # 3^2 = 9

    def test_log_partition_consistency(self):
        """log_partition from enumeration should match manual sum."""
        env = HyperGrid(ndim=2, height=3, store_all_states=True, validate_modes=False)
        # Manually compute: sum of all rewards, then log
        all_r = env.reward_fn(env._all_states_tensor)
        manual_lp = all_r.sum().log().item()
        assert env._log_partition == pytest.approx(manual_lp, abs=1e-5)
