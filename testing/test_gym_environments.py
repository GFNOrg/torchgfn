"""Extended tests for gfn.gym environments: hypergrid, line, box, bitSequence."""

import numpy as np
import pytest
import torch

from gfn.gym import HyperGrid
from gfn.gym.line import Line

# ===========================================================================
# HyperGrid — biggest coverage impact (57% → 75%+, 307 missed stmts)
# ===========================================================================


class TestHyperGridInit:
    def test_default_init(self):
        env = HyperGrid(ndim=2, height=8)
        assert env.ndim == 2
        assert env.height == 8
        assert env.n_actions == 3  # ndim + 1

    def test_small_height_warns(self):
        """Height <= 4 should log a warning but not crash."""
        # Just verify it doesn't crash
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        assert env.height == 4

    def test_store_all_states(self):
        env = HyperGrid(ndim=2, height=4, store_all_states=True, validate_modes=False)
        assert env._all_states_tensor is not None
        assert env._log_partition is not None

    def test_calculate_partition(self):
        env = HyperGrid(ndim=2, height=4, calculate_partition=True, validate_modes=False)
        assert env._log_partition is not None

    # ------------------------------------------------------------------
    # multiprocessing.Pool with the ``spawn`` start method
    # ------------------------------------------------------------------
    #
    # ``HyperGrid._generate_combinations_in_batches`` previously used
    # ``fork`` so it could send a bound method to the worker pool, which
    # was incompatible with MPI/CUDA contexts.  We switched to ``spawn``
    # plus a module-level ``_hypergrid_worker``.  These tests pin the new
    # contract: enumerating all states must succeed under spawn, the worker
    # must remain a picklable module-level callable, and the Pool size must
    # be capped (so a 64-core node hosting many co-located ranks doesn't
    # explode into ``ranks * num_cores`` simultaneous worker processes).

    def test_enumerate_via_spawned_pool(self):
        """End-to-end: store_all_states triggers the Pool path under spawn."""
        env = HyperGrid(ndim=3, height=6, store_all_states=True, validate_modes=False)
        assert env._all_states_tensor is not None
        # 6**3 = 216 unique states.
        assert env._all_states_tensor.shape == (216, 3)
        # Every coordinate value 0..5 should be present in each dimension.
        for d in range(3):
            assert set(env._all_states_tensor[:, d].tolist()) == set(range(6))

    def test_hypergrid_worker_is_picklable_module_level_function(self):
        """The worker must be picklable so spawn can ship it to children."""
        import pickle

        from gfn.gym.hypergrid import _hypergrid_worker

        # Module-level functions pickle by qualified name.
        round_tripped = pickle.loads(pickle.dumps(_hypergrid_worker))
        assert round_tripped is _hypergrid_worker

        # And the worker actually does the right thing on a small task.
        task = ([0, 1, 2], 2, 0, 4)
        result = _hypergrid_worker(task)
        assert isinstance(result, list)
        assert result == [(0, 0), (0, 1), (0, 2), (1, 0)]
        # Result must round-trip too (pool sends it back over a pipe).
        assert pickle.loads(pickle.dumps(result)) == result

    def test_pool_worker_count_is_capped(self):
        """Sanity-check the cap so a future refactor can't accidentally
        regress to ``Pool()`` with no ``processes=`` argument."""
        from gfn.gym.hypergrid import _MAX_POOL_WORKERS

        assert isinstance(_MAX_POOL_WORKERS, int)
        assert 1 <= _MAX_POOL_WORKERS <= 32, (
            f"Cap {_MAX_POOL_WORKERS} should be small to prevent fork/spawn "
            f"storms when many MPI ranks are co-located on one node."
        )

    def test_start_method_is_spawn(self):
        """``set_start_method('spawn')`` must take effect for safety inside
        MPI/CUDA contexts.  Skipped if another framework already pinned the
        start method to something else before pytest started."""
        import multiprocessing

        method = multiprocessing.get_start_method(allow_none=True)
        # If unset, importing hypergrid should set it to 'spawn'.
        # If already set (e.g. by another import), accept whatever's there
        # but flag if it's still 'fork' on POSIX.
        # This next line only exists to induce the side effect.
        import gfn.gym.hypergrid  # noqa: F401  # pyright: ignore[reportUnusedImport]

        method = multiprocessing.get_start_method()
        if method == "fork":
            pytest.skip(
                "start method was pinned to 'fork' by another framework "
                "before pytest started; the production code does call "
                "set_start_method('spawn') but it was a no-op here."
            )
        assert method == "spawn", f"expected spawn, got {method}"


class TestHyperGridRewardFunctions:
    """Test all reward function variants produce valid rewards."""

    @pytest.mark.parametrize(
        "reward_fn_str,height",
        [
            ("original", 8),
            ("cosine", 8),
            ("sparse", 8),
            ("deceptive", 8),
            ("bitwise_xor", 8),
            ("multiplicative_coprime", 8),
            ("conditional_multiscale", 64),  # needs height=base^levels, base=4, 3 tiers
        ],
    )
    def test_reward_fn_produces_valid_output(self, reward_fn_str, height):
        env = HyperGrid(
            ndim=2, height=height, reward_fn_str=reward_fn_str, validate_modes=False
        )
        states = env.make_random_states((16,))
        rewards = env.reward(states)
        assert rewards.shape == (16,)
        assert torch.isfinite(rewards).all()
        assert (rewards > 0).all()  # Rewards must be positive

    def test_invalid_reward_fn_raises(self):
        with pytest.raises(AssertionError, match="Invalid reward function"):
            HyperGrid(ndim=2, height=8, reward_fn_str="nonexistent")


class TestHyperGridStepAndBackward:
    def test_step_increments_dimension(self):
        env = HyperGrid(ndim=2, height=8, validate_modes=False)
        s0_batch = env.States(env.s0.unsqueeze(0).expand(4, -1).clone())
        # Action 0: increment dim 0
        actions = env.Actions(torch.zeros(4, 1, dtype=torch.long))
        next_states = env.step(s0_batch, actions)
        assert next_states.tensor[0, 0].item() == 1
        assert next_states.tensor[0, 1].item() == 0

    def test_backward_step_decrements_dimension(self):
        env = HyperGrid(ndim=2, height=8, validate_modes=False)
        state_tensor = torch.tensor([[3, 2]], dtype=torch.long)
        states = env.States(state_tensor)
        actions = env.Actions(torch.tensor([[0]], dtype=torch.long))
        prev_states = env.backward_step(states, actions)
        assert prev_states.tensor[0, 0].item() == 2
        assert prev_states.tensor[0, 1].item() == 2

    def test_make_random_states(self):
        env = HyperGrid(ndim=2, height=8, validate_modes=False)
        states = env.make_random_states((32,))
        assert states.tensor.shape == (32, 2)
        assert (states.tensor >= 0).all()
        assert (states.tensor < 8).all()


class TestHyperGridModes:
    def test_mode_mask(self):
        env = HyperGrid(ndim=2, height=8, validate_modes=False)
        states = env.make_random_states((100,))
        mask = env.mode_mask(states)
        assert mask.shape == (100,)
        assert mask.dtype == torch.bool

    def test_modes_found(self):
        env = HyperGrid(ndim=2, height=8, validate_modes=False)
        states = env.make_random_states((1000,))
        found = env.modes_found(states)
        assert isinstance(found, set)

    def test_n_mode_states_with_store_all(self):
        env = HyperGrid(ndim=2, height=8, store_all_states=True, validate_modes=False)
        n = env.n_mode_states
        assert n is not None
        assert n >= 0

    def test_n_mode_states_exact(self):
        env = HyperGrid(
            ndim=2,
            height=8,
            store_all_states=True,
            mode_stats="exact",
            validate_modes=False,
        )
        n = env.n_mode_states
        assert isinstance(n, int)
        assert n >= 0

    def test_n_mode_states_approx(self):
        env = HyperGrid(
            ndim=2,
            height=8,
            mode_stats="approx",
            mode_stats_samples=1000,
            validate_modes=False,
        )
        n = env.n_mode_states
        assert isinstance(n, float)
        assert n >= 0

    def test_n_modes_alias(self):
        env = HyperGrid(ndim=2, height=8, store_all_states=True, validate_modes=False)
        assert env.n_modes == env.n_mode_states

    def test_validate_modes_raises_on_bad_config(self):
        """Very small grid with validate_modes=True should raise."""
        with pytest.raises(ValueError):
            HyperGrid(ndim=2, height=2, validate_modes=True)


class TestHyperGridModeThresholds:
    """Test _mode_reward_threshold for various reward functions."""

    def test_original_threshold(self):
        env = HyperGrid(ndim=2, height=8, reward_fn_str="original", validate_modes=False)
        t = env._mode_reward_threshold()
        assert t == 0.1 + 0.5 + 2.0  # R0 + R1 + R2

    def test_deceptive_threshold(self):
        env = HyperGrid(
            ndim=2, height=8, reward_fn_str="deceptive", validate_modes=False
        )
        t = env._mode_reward_threshold()
        assert t == 0.1 + 2.0  # R0 + R2

    def test_cosine_threshold(self):
        env = HyperGrid(ndim=2, height=8, reward_fn_str="cosine", validate_modes=False)
        t = env._mode_reward_threshold()
        assert t > 0

    def test_sparse_threshold(self):
        env = HyperGrid(ndim=2, height=8, reward_fn_str="sparse", validate_modes=False)
        t = env._mode_reward_threshold()
        assert t == 0.5


class TestHyperGridGetStatesIndices:
    def test_indices_unique(self):
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        states = env.make_random_states((100,))
        indices = env.get_states_indices(states)
        assert indices.shape == (100,)
        assert indices.dtype == torch.long

    def test_indices_from_tensor(self):
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        tensor = torch.tensor([[0, 0], [1, 0], [0, 1], [3, 3]], dtype=torch.long)
        indices = env.get_states_indices(tensor)
        assert indices.shape == (4,)
        # s0 should have index 0
        assert indices[0].item() == 0

    # -----------------------------------------------------------------
    # Bigint fallback path: ndim * ceil(log2(height)) > 62
    # -----------------------------------------------------------------
    #
    # When the canonical index space exceeds int64, the historical
    # ``self.height ** torch.arange(...)`` path silently overflows and
    # different states map to the same int64 value.  ``get_states_indices``
    # detects this regime and falls back to a numpy object array of
    # arbitrary-precision Python ints.  These tests pin down both paths.

    def test_safe_path_returns_int64_tensor(self):
        """Small grids: int64 tensor with the historical contract."""
        env = HyperGrid(ndim=2, height=8, validate_modes=False)
        states = env.make_random_states((50,))
        indices = env.get_states_indices(states)
        assert isinstance(indices, torch.Tensor)
        assert indices.dtype == torch.long
        assert indices.shape == (50,)

    def test_safe_path_boundary_height128_ndim9(self):
        """Boundary case: height=128, ndim=9 → max index = 128**9 - 1 = 2**63 - 1,
        which exactly fits in int64.  The exact overflow check must keep this
        on the fast int64 tensor path (128**9 == 2**63, which is NOT > 2**63)."""
        env = HyperGrid(ndim=9, height=128, validate_modes=False)
        states = torch.zeros((4, 9), dtype=torch.long)
        indices = env.get_states_indices(states)
        assert isinstance(
            indices, torch.Tensor
        ), "height=128, ndim=9 should stay on the int64 path (max index = 2**63-1)"
        assert indices.dtype == torch.long

        # The all-max state should map to exactly INT64_MAX = 2**63 - 1.
        max_state = torch.full((1, 9), 127, dtype=torch.long)
        max_idx = int(env.get_states_indices(max_state).item())
        assert max_idx == 2**63 - 1

    def test_safe_path_canonical_encoding_matches_python(self):
        """Safe path encodes states as base-``height`` numbers."""
        ndim, height = 3, 8
        env = HyperGrid(ndim=ndim, height=height, validate_modes=False)
        states = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [7, 7, 7]],
            dtype=torch.long,
        )
        indices = env.get_states_indices(states).tolist()
        expected = [
            0,  # 0*64 + 0*8 + 0
            1 * height**2,  # 64
            1 * height,  # 8
            1,  # 1
            7 * height**2 + 7 * height + 7,  # 511 = n_states - 1
        ]
        assert indices == expected

    def test_bigint_path_triggered_for_large_grids(self):
        """ndim=10, height=128 needs 70 bits → numpy object array."""
        env = HyperGrid(
            ndim=10,
            height=128,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs={"R0": 0.125},
            validate_modes=False,
        )
        states = torch.zeros((4, 10), dtype=torch.long)
        indices = env.get_states_indices(states)
        assert isinstance(indices, np.ndarray)
        assert indices.dtype == object
        assert indices.shape == (4,)
        # All-zero states all encode to Python int 0.
        assert all(int(v) == 0 for v in indices)

    def test_bigint_path_no_collisions_on_states_that_overflow_int64(self):
        """The exact pairs that DO collide under the buggy int64 path
        (states differing only in the high bits of dim 0) MUST produce
        distinct indices in the bigint path."""
        env = HyperGrid(
            ndim=10,
            height=128,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs={"R0": 0.125},
            validate_modes=False,
        )
        # 128**9 == 2**63 overflows int64.  Under the OLD int64 path, only
        # the LSB of s[0] survives, so s[0] in {0, 2, 4, 6, ...} all collide
        # to the same canonical "0".  The bigint path must produce 4 distinct
        # indices for these 4 distinct states.
        states = torch.zeros((4, 10), dtype=torch.long)
        states[:, 0] = torch.tensor([0, 2, 4, 6])
        indices = env.get_states_indices(states)
        unique = {int(v) for v in indices}
        assert len(unique) == 4

    def test_bigint_path_matches_python_reference(self):
        """Cross-check the vectorized bigint path against a naive per-row
        Python-int reference encoder over a randomized batch."""
        ndim, height = 10, 128
        env = HyperGrid(
            ndim=ndim,
            height=height,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs={"R0": 0.125},
            validate_modes=False,
        )
        torch.manual_seed(0)
        states = torch.randint(0, height, (256, ndim), dtype=torch.long)
        indices = env.get_states_indices(states)

        # Reference: O(batch * ndim) per-row Python loop using arbitrary-
        # precision ints.  This is the unvectorized form of the same algorithm
        # the bigint fallback runs internally.
        flat = states.cpu().numpy()
        expected = []
        for row in flat:
            k = 0
            for v in row:
                k = k * height + int(v)
            expected.append(k)
        assert [int(v) for v in indices] == expected

    def test_int64_path_is_demonstrably_wrong_in_overflow_regime(self):
        """Sanity check: simulate what the OLD int64-only encoder would have
        produced and confirm it really does collapse distinct states.  This
        documents *why* the bigint fallback is needed and locks in the
        regression test."""
        ndim, height = 10, 128
        states = torch.zeros((4, ndim), dtype=torch.long)
        states[:, 0] = torch.tensor([0, 2, 4, 6])

        # Recreate the buggy old path verbatim.
        canonical_base = height ** torch.arange(ndim - 1, -1, -1)
        buggy_indices = (canonical_base * states).sum(-1).long().tolist()

        # In the int64 regime, these 4 distinct states collapse to ≤2 unique
        # values (parity of s[0]).  This guarantees the bug is real and that
        # the new path is fixing it, not papering over an irrelevance.
        assert len(set(buggy_indices)) < 4

        # Meanwhile the bigint path keeps them distinct.
        env = HyperGrid(
            ndim=ndim,
            height=height,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs={"R0": 0.125},
            validate_modes=False,
        )
        good_indices = env.get_states_indices(states)
        assert len({int(v) for v in good_indices}) == 4

    def test_bigint_path_preserves_batch_shape(self):
        """Multi-dimensional batch shapes round-trip through the bigint path."""
        env = HyperGrid(
            ndim=10,
            height=128,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs={"R0": 0.125},
            validate_modes=False,
        )
        states = torch.randint(0, 128, (3, 5, 10), dtype=torch.long)
        indices = env.get_states_indices(states)
        assert isinstance(indices, np.ndarray)
        assert indices.shape == (3, 5)

    def test_modes_found_works_in_bigint_regime(self):
        """End-to-end: ``modes_found`` returns hashable Python ints in the
        bigint regime so the downstream mode tracker stays correct."""
        env = HyperGrid(
            ndim=10,
            height=128,
            reward_fn_str="bitwise_xor",
            reward_fn_kwargs={"R0": 0.125},
            validate_modes=False,
        )
        torch.manual_seed(0)
        # Sample enough states that some are likely to be modes; we don't
        # need any specific count, just that the call succeeds and returns
        # a set of ints.
        states_tensor = torch.randint(0, 128, (1024, 10), dtype=torch.long)
        states = env.States(states_tensor)
        modes = env.modes_found(states)
        assert isinstance(modes, set)
        for m in modes:
            assert isinstance(m, int)


class TestHyperGridValidation:
    """Test the validate() method on HyperGrid."""

    def test_validate_runs(self):
        from gfn.estimators import DiscretePolicyEstimator
        from gfn.gflownet import TBGFlowNet
        from gfn.preprocessors import KHotPreprocessor
        from gfn.utils.modules import MLP

        torch.manual_seed(42)
        env = HyperGrid(ndim=2, height=8, store_all_states=True)
        preproc = KHotPreprocessor(env.height, env.ndim)
        assert isinstance(preproc.output_dim, int)
        pf = DiscretePolicyEstimator(
            module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions),
            n_actions=env.n_actions,
            preprocessor=preproc,
            is_backward=False,
        )
        gfn = TBGFlowNet(pf=pf, pb=None, constant_pb=True)
        metrics, _visited = env.validate(gfn, n_validation_samples=100)
        assert isinstance(metrics, dict)
        # Should have at least l1_dist
        assert "l1_dist" in metrics


# ===========================================================================
# Line environment
# ===========================================================================


class TestLine:
    def test_init(self):
        env = Line(mus=[0.0, 1.0], sigmas=[0.1, 0.1], init_value=0.5)
        assert env.n_steps_per_trajectory == 5
        assert len(env.mixture) == 2

    def test_step_forward(self):
        env = Line(mus=[0.0], sigmas=[0.1], init_value=0.0)
        s0 = env.States(env.s0.unsqueeze(0).clone())
        action = env.Actions(torch.tensor([[0.1]]))
        next_state = env.step(s0, action)
        assert next_state.tensor[0, 0].item() == pytest.approx(0.1, abs=1e-5)
        assert next_state.tensor[0, 1].item() == 1.0  # step counter

    def test_backward_step(self):
        env = Line(mus=[0.0], sigmas=[0.1], init_value=0.0)
        state = env.States(torch.tensor([[0.5, 3.0]]))
        action = env.Actions(torch.tensor([[0.2]]))
        prev = env.backward_step(state, action)
        assert prev.tensor[0, 0].item() == pytest.approx(0.3, abs=1e-5)
        assert prev.tensor[0, 1].item() == 2.0

    def test_is_action_valid_forward(self):
        env = Line(mus=[0.0], sigmas=[0.1], init_value=0.0)
        state = env.States(torch.tensor([[0.2, 2.0]]))
        action = env.Actions(torch.tensor([[0.1]]))
        assert env.is_action_valid(state, action, backward=False) is True

    def test_is_action_valid_backward_from_s0(self):
        """Backward from initial state should be invalid."""
        env = Line(mus=[0.0], sigmas=[0.1], init_value=0.0)
        s0 = env.States(env.s0.unsqueeze(0).clone())
        action = env.Actions(torch.tensor([[0.1]]))
        assert env.is_action_valid(s0, action, backward=True) is False

    def test_log_reward_mixture(self):
        env = Line(mus=[0.0, 1.0], sigmas=[0.2, 0.2], init_value=0.5)
        states = env.States(torch.tensor([[0.0, 5.0], [1.0, 5.0]]))
        lr = env.log_reward(states)
        assert lr.shape == (2,)
        assert torch.isfinite(lr).all()

    def test_log_partition(self):
        env = Line(mus=[0.0, 1.0], sigmas=[0.1, 0.1], init_value=0.5)
        lp = env.log_partition()
        # log(2 modes) = log(2) ≈ 0.693
        assert lp.item() == pytest.approx(0.693, abs=0.01)


# ===========================================================================
# Box environments
# ===========================================================================


class TestBoxPolar:
    def test_init(self):
        from gfn.gym.box import BoxPolar

        env = BoxPolar(delta=0.1)
        assert env.delta == 0.1
        assert env.state_shape == (2,)

    def test_step_and_backward(self):
        from gfn.gym.box import BoxPolar

        env = BoxPolar(delta=0.1)
        s0 = env.States(env.s0.unsqueeze(0).clone())
        # Take a step with action [0.05, 0.05]
        action = env.Actions(torch.tensor([[0.05, 0.05]]))
        next_state = env.step(s0, action)
        assert torch.allclose(next_state.tensor, s0.tensor + action.tensor, atol=1e-6)

    def test_reward(self):
        from gfn.gym.box import BoxPolar

        env = BoxPolar(delta=0.1, R0=0.1, R1=0.5, R2=2.0)
        states = env.States(torch.tensor([[0.5, 0.5], [0.1, 0.1]]))
        rewards = env.reward(states)
        assert rewards.shape == (2,)
        assert (rewards > 0).all()


class TestBoxCartesian:
    def test_init(self):
        from gfn.gym import Box

        env = Box(delta=0.1)
        assert env.delta == 0.1

    def test_is_action_valid_forward(self):
        from gfn.gym import Box

        env = Box(delta=0.1)
        s0 = env.States(env.s0.unsqueeze(0).clone())
        # Valid forward action from s0
        action = env.Actions(torch.tensor([[0.05, 0.05]]))
        result = env.is_action_valid(s0, action, backward=False)
        assert isinstance(result, bool)


# ===========================================================================
# BitSequence environment
# ===========================================================================


class TestBitSequence:
    def test_init(self):
        from gfn.gym.bitSequence import BitSequence

        env = BitSequence(word_size=4, seq_size=120)
        assert env.word_size == 4
        assert env.words_per_seq == 30  # 120 / 4

    def test_step_appends_word(self):
        from gfn.gym.bitSequence import BitSequence

        env = BitSequence(word_size=4, seq_size=120)
        s0 = env.States(env.s0.unsqueeze(0).clone())
        # Action 0 appends word "0" as the first word
        action = env.Actions(torch.tensor([[0]]))
        next_state = env.step(s0, action)
        assert next_state.tensor.shape == s0.tensor.shape

    def test_reward_positive(self):
        from gfn.gym.bitSequence import BitSequence

        env = BitSequence(word_size=4, seq_size=120)
        # Create complete sequences (no -1 values)
        state_tensor = torch.zeros(4, env.words_per_seq, dtype=torch.long)
        states = env.States(state_tensor)
        rewards = env.reward(states)
        assert rewards.shape == (4,)
        assert (rewards > 0).all()

    def test_integers_to_binary_roundtrip(self):
        from gfn.gym.bitSequence import BitSequence

        env = BitSequence(word_size=4, seq_size=120)
        ints = torch.tensor([0, 1, 7, 15])
        binary = env.integers_to_binary(ints, k=env.word_size)
        back = env.binary_to_integers(binary, k=env.word_size)
        assert torch.equal(ints, back)

    def test_hamming_distance(self):
        from gfn.gym.bitSequence import BitSequence

        env = BitSequence(word_size=4, seq_size=120)
        a = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        b = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        dist = env.hamming_distance(a, b)
        assert dist.item() == 2

    def test_reset(self):
        from gfn.gym.bitSequence import BitSequence

        env = BitSequence(word_size=4, seq_size=120)
        states = env.reset(batch_shape=(16,))
        assert states.tensor.shape[0] == 16
        # Reset should return s0-like states
        assert (states.tensor == -1).all()

    def test_states_to_str(self):
        from gfn.gym.bitSequence import BitSequence

        env = BitSequence(word_size=4, seq_size=120)
        state_tensor = torch.zeros(2, env.words_per_seq, dtype=torch.long)
        states = env.States(state_tensor)
        result = states.to_str()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_create_test_set(self):
        from gfn.gym.bitSequence import BitSequence

        env = BitSequence(word_size=4, seq_size=120)
        test_set = env.create_test_set(k=env.word_size)
        assert test_set is not None
        assert len(test_set) > 0


# ===========================================================================
# BitSequenceNonAutoregressive
# ===========================================================================


class TestBitSequenceNonAutoregressive:
    def test_init(self):
        from gfn.gym.bitSequenceNonAutoregressive import NonAutoregressiveBitSequence

        env = NonAutoregressiveBitSequence(word_size=4, seq_size=120)
        assert env.word_size == 4

    def test_step(self):
        from gfn.gym.bitSequenceNonAutoregressive import NonAutoregressiveBitSequence

        env = NonAutoregressiveBitSequence(word_size=4, seq_size=120)
        s0 = env.States(env.s0.unsqueeze(0).clone())
        # Action: set position 0 to value 0 (action index encodes position+value)
        action = env.Actions(torch.tensor([[0]]))
        next_state = env.step(s0, action)
        assert next_state.tensor.shape == s0.tensor.shape

    def test_reward(self):
        from gfn.gym.bitSequenceNonAutoregressive import NonAutoregressiveBitSequence

        env = NonAutoregressiveBitSequence(word_size=4, seq_size=120)
        states = env.make_random_states(batch_shape=(8,))
        rewards = env.reward(states)
        assert rewards.shape == (8,)
        assert (rewards > 0).all()


# ===========================================================================
# HyperGrid — additional coverage for properties and validate
# ===========================================================================


class TestHyperGridProperties:
    def test_n_states(self):
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        assert env.n_states == 16  # 4^2

    def test_n_terminating_states(self):
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        assert env.n_terminating_states == 16

    def test_all_states_with_store(self):
        env = HyperGrid(ndim=2, height=4, store_all_states=True, validate_modes=False)
        assert env.all_states is not None
        assert len(env.all_states) == 16

    def test_true_dist(self):
        env = HyperGrid(ndim=2, height=4, store_all_states=True, validate_modes=False)
        dist = env.true_dist()  # method, not property
        assert dist is not None
        assert dist.shape == (16,)
        assert torch.allclose(dist.sum(), torch.tensor(1.0), atol=1e-5)

    def test_log_partition(self):
        env = HyperGrid(ndim=2, height=8, calculate_partition=True, validate_modes=False)
        lp = env.log_partition()
        assert lp is not None
        lp_tensor = torch.tensor(lp) if isinstance(lp, (float, int)) else lp
        assert torch.isfinite(lp_tensor)

    def test_get_terminating_state_dist(self):
        env = HyperGrid(ndim=2, height=4, store_all_states=True, validate_modes=False)
        states = env.make_random_states((100,))
        dist = env.get_terminating_state_dist(states)
        assert dist.shape == (16,)

    def test_log_reward(self):
        env = HyperGrid(ndim=2, height=8, validate_modes=False)
        states = env.make_random_states((16,))
        lr = env.log_reward(states)
        assert lr.shape == (16,)
        assert torch.isfinite(lr).all()


class TestHyperGridModeExistence:
    """Test mode existence checks for various reward functions."""

    @pytest.mark.parametrize(
        "reward_fn_str",
        ["original", "sparse", "deceptive"],
    )
    def test_modes_exist(self, reward_fn_str):
        env = HyperGrid(
            ndim=2, height=8, reward_fn_str=reward_fn_str, validate_modes=True
        )
        # If we get here without error, modes exist
        assert env is not None

    def test_bitwise_xor_modes_exist(self):
        env = HyperGrid(
            ndim=2, height=8, reward_fn_str="bitwise_xor", validate_modes=True
        )
        assert env is not None

    def test_multiplicative_coprime_modes_exist(self):
        env = HyperGrid(
            ndim=2,
            height=8,
            reward_fn_str="multiplicative_coprime",
            validate_modes=True,
        )
        assert env is not None


class TestHyperGridForwardBackwardMasks:
    def test_forward_masks_at_height_limit(self):
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        # State at (3, 3) — at height limit in both dims
        state_tensor = torch.tensor([[3, 3]], dtype=torch.long)
        states = env.States(state_tensor)
        fwd_masks = states.forward_masks
        # Non-exit actions for dim0 and dim1 should be False (at limit)
        assert fwd_masks[0, 0].item() is False  # dim 0 blocked
        assert fwd_masks[0, 1].item() is False  # dim 1 blocked
        assert fwd_masks[0, 2].item() is True  # exit always allowed

    def test_backward_masks(self):
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        state_tensor = torch.tensor([[0, 3]], dtype=torch.long)
        states = env.States(state_tensor)
        bwd_masks = states.backward_masks
        assert bwd_masks[0, 0].item() is False  # dim 0 at 0, can't go back
        assert bwd_masks[0, 1].item() is True  # dim 1 at 3, can go back


# ===========================================================================
# DiscreteEBM
# ===========================================================================


class TestDiscreteEBM:
    def test_init(self):
        from gfn.gym.discrete_ebm import DiscreteEBM, IsingModel

        ising = IsingModel(J=torch.ones(4, 4))
        env = DiscreteEBM(ndim=4, energy=ising)
        assert env.ndim == 4

    def test_init_default_energy(self):
        from gfn.gym.discrete_ebm import DiscreteEBM

        env = DiscreteEBM(ndim=4)
        assert env.ndim == 4

    def test_step(self):
        from gfn.gym.discrete_ebm import DiscreteEBM

        env = DiscreteEBM(ndim=4)
        s0 = env.States(env.s0.unsqueeze(0).clone())
        # Action 0: set position 0 to 0
        action = env.Actions(torch.tensor([[0]]))
        next_state = env.step(s0, action)
        assert next_state.tensor.shape == s0.tensor.shape

    def test_reward_positive(self):
        from gfn.gym.discrete_ebm import DiscreteEBM

        env = DiscreteEBM(ndim=4)
        # Complete states (0s and 1s, no -1s)
        state_tensor = torch.ones(4, 4, dtype=torch.long)
        states = env.States(state_tensor)
        rewards = env.reward(states)
        assert rewards.shape == (4,)
        assert (rewards > 0).all()

    def test_all_states(self):
        from gfn.gym.discrete_ebm import DiscreteEBM

        env = DiscreteEBM(ndim=3)
        assert env.n_states == 3**3  # 27 states for ndim=3 (values in {-1,0,1})
        assert env.all_states is not None
