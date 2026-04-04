"""Tests for the ChipDesign environment."""

import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from gfn.gym.chip_design import ChipDesign, ChipDesignStates, CostStats
from gfn.gym.helpers.chip_design import SAMPLE_INIT_PLACEMENT, SAMPLE_NETLIST_FILE
from gfn.gym.helpers.chip_design import utils as placement_util
from gfn.gym.helpers.chip_design.plc_client import _find_singularity, _resolve_plc_binary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(vals):
    """Long tensor with trailing dim-1 for actions."""
    return torch.tensor(vals, dtype=torch.long).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Module-scoped fixture -- ONE env for the entire module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    try:
        e = ChipDesign(device="cpu", cd_finetune=False, reward_norm=None)
    except FileNotFoundError:
        pytest.skip("plc_wrapper_main not available (Linux x86-64 only)")
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Function-scoped fixture: env + cached placement & reward (ONE log_reward call)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def env_with_reward():
    """Creates env, places macros [0, 3], computes log_reward ONCE."""
    try:
        e = ChipDesign(device="cpu", cd_finetune=False, reward_norm=None)
    except FileNotFoundError:
        pytest.skip("plc_wrapper_main not available (Linux x86-64 only)")

    states = e.reset(batch_shape=1)
    # Place macro 0 at grid cell 0
    actions = e.actions_from_tensor(_fmt([0]))
    states = e.step(states, actions)
    # Place macro 1 at grid cell 3
    actions = e.actions_from_tensor(_fmt([3]))
    states = e.step(states, actions)
    # Exit
    actions = e.actions_from_tensor(_fmt([e.n_actions - 1]))
    final = e.step(states, actions)

    lr = e.log_reward(final)
    yield e, final, lr
    e.close()


# ===================================================================
# A. ChipDesignStates (9 tests)
# ===================================================================


class TestChipDesignStates:

    def test_states_init_all_unplaced(self, env):
        """Reset -> current_node_idx == 0 for all."""
        states = env.reset(batch_shape=2)
        assert torch.all(states.current_node_idx == 0)

    def test_states_init_partially_placed(self, env):
        """After one step, constructing states from tensor infers node_idx == 1."""
        states = env.reset(batch_shape=1)
        actions = env.actions_from_tensor(_fmt([0]))
        states = env.step(states, actions)
        # Construct fresh from the raw tensor (no explicit node_idx)
        new = env.States(tensor=states.tensor.clone())
        assert new.current_node_idx.item() == 1

    def test_states_init_all_placed(self, env):
        """All macros placed -> node_idx == n_macros."""
        states = env.reset(batch_shape=1)
        for i in range(env.n_macros):
            actions = env.actions_from_tensor(_fmt([i]))
            states = env.step(states, actions)
        new = env.States(tensor=states.tensor.clone())
        assert new.current_node_idx.item() == env.n_macros

    def test_states_init_explicit_node_idx(self, env):
        """Explicit current_node_idx is stored as-is."""
        tensor = env.s0.unsqueeze(0).clone()
        idx = torch.tensor([42])
        s = env.States(tensor=tensor, current_node_idx=idx)
        assert s.current_node_idx.item() == 42

    def test_states_clone(self, env):
        """Clone then mutate -- original unchanged."""
        states = env.reset(batch_shape=1)
        actions = env.actions_from_tensor(_fmt([0]))
        states = env.step(states, actions)
        original_tensor = states.tensor.clone()
        original_idx = states.current_node_idx.clone()

        cloned = states.clone()
        cloned.tensor[0, 0] = 999
        cloned.current_node_idx[0] = 999

        assert torch.equal(states.tensor, original_tensor)
        assert torch.equal(states.current_node_idx, original_idx)

    def test_states_getitem(self, env):
        """Batch of 4, slice [1:3]."""
        states = env.reset(batch_shape=4)
        subset = states[1:3]
        assert subset.tensor.shape[0] == 2
        assert subset.current_node_idx.shape[0] == 2
        assert torch.equal(subset.tensor, states.tensor[1:3])

    def test_states_setitem(self, env):
        """Set index 0 to a different state."""
        states = env.reset(batch_shape=2)
        actions = env.actions_from_tensor(_fmt([1, 2]))
        stepped = env.step(states, actions)
        # Overwrite index 0 with index 1
        stepped[0] = stepped[1]
        assert torch.equal(stepped.tensor[0], stepped.tensor[1])
        assert stepped.current_node_idx[0] == stepped.current_node_idx[1]

    def test_states_extend(self, env):
        """Extend two batches."""
        a = env.reset(batch_shape=2)
        b = env.reset(batch_shape=3)
        a.extend(b)
        assert a.tensor.shape[0] == 5
        assert a.current_node_idx.shape[0] == 5

    def test_states_stack(self, env):
        """Stack two states -> leading dim."""
        a = env.reset(batch_shape=2)
        b = env.reset(batch_shape=2)
        stacked = env.States.stack([a, b])
        assert stacked.tensor.shape == (2, 2, env.n_macros)
        assert stacked.current_node_idx.shape == (2, 2)


# ===================================================================
# B. Environment Core (8 tests)
# ===================================================================


class TestEnvironmentCore:

    def test_reset_initial(self, env):
        states = env.reset(batch_shape=1)
        assert torch.all(states.tensor == -1)
        assert torch.all(states.is_initial_state)

    def test_reset_sink(self, env):
        states = env.reset(batch_shape=1, sink=True)
        assert torch.all(states.is_sink_state)

    def test_step_places_macro(self, env):
        """Action 2 on first macro -> tensor[0]==2, node_idx==1."""
        states = env.reset(batch_shape=1)
        actions = env.actions_from_tensor(_fmt([2]))
        new = env.step(states, actions)
        assert new.tensor[0, 0].item() == 2
        assert new.current_node_idx.item() == 1

    def test_step_exit_produces_sink(self, env):
        """Place all macros then exit -> sink."""
        states = env.reset(batch_shape=1)
        for i in range(env.n_macros):
            actions = env.actions_from_tensor(_fmt([i]))
            states = env._step(states, actions)
        actions = env.actions_from_tensor(_fmt([env.n_actions - 1]))
        final = env._step(states, actions)
        assert torch.all(final.is_sink_state)

    def test_step_empty_batch(self, env):
        """batch_shape=0 should not crash."""
        states = env.reset(batch_shape=0)
        actions = env.actions_from_tensor(torch.zeros(0, 1, dtype=torch.long))
        new = env.step(states, actions)
        assert new.tensor.shape[0] == 0

    def test_backward_step(self, env):
        """Place one macro, step back, tensor goes back to -1."""
        states = env.reset(batch_shape=1)
        actions = env.actions_from_tensor(_fmt([1]))
        stepped = env.step(states, actions)
        assert stepped.tensor[0, 0].item() == 1

        back_actions = env.actions_from_tensor(_fmt([1]))
        back = env.backward_step(stepped, back_actions)
        assert back.tensor[0, 0].item() == -1
        assert back.current_node_idx.item() == 0

    def test_forward_backward_roundtrip(self, env):
        """Step forward then backward with same action -> back to start."""
        states = env.reset(batch_shape=1)
        original = states.tensor.clone()
        actions = env.actions_from_tensor(_fmt([2]))
        fwd = env.step(states, actions)
        back_actions = env.actions_from_tensor(_fmt([2]))
        back = env.backward_step(fwd, back_actions)
        assert torch.equal(back.tensor, original)

    def test_full_trajectory_roundtrip(self, env):
        """Place all, undo all, back to s0."""
        states = env.reset(batch_shape=1)
        original = states.tensor.clone()
        placements = list(range(env.n_macros))
        # Forward
        for a in placements:
            actions = env.actions_from_tensor(_fmt([a]))
            states = env.step(states, actions)
        # Backward
        for a in reversed(placements):
            actions = env.actions_from_tensor(_fmt([a]))
            states = env.backward_step(states, actions)
        assert torch.equal(states.tensor, original)
        assert states.current_node_idx.item() == 0


# ===================================================================
# C. Masks (4 tests)
# ===================================================================


class TestMasks:

    def test_masks_initial(self, env):
        """Initial: fwd allows grid cells (not exit), bwd all false."""
        states = env.reset(batch_shape=1)
        fwd = states.forward_masks[0]
        bwd = states.backward_masks[0]
        # Exit (last action) should be False
        assert not fwd[-1].item()
        # At least some grid cells allowed
        assert fwd[:-1].any()
        # Backward: all false (no macro placed)
        assert not bwd.any()

    def test_masks_all_placed(self, env):
        """All placed: fwd only exit, bwd has last placed cell."""
        states = env.reset(batch_shape=1)
        for i in range(env.n_macros):
            actions = env.actions_from_tensor(_fmt([i]))
            states = env._step(states, actions)
        fwd = states.forward_masks[0]
        bwd = states.backward_masks[0]
        # Only exit allowed
        assert fwd[-1].item()
        assert not fwd[:-1].any()
        # Backward: last placed cell should be True
        last_cell = env.n_macros - 1  # grid cell of last placement
        assert bwd[last_cell].item()

    def test_masks_sink(self, env):
        """Sink state: all masks false."""
        states = env.reset(batch_shape=1, sink=True)
        assert not states.forward_masks.any()
        assert not states.backward_masks.any()

    def test_masks_mid_trajectory(self, env):
        """After 1 placement: fwd allows grid cells, bwd has placed cell."""
        states = env.reset(batch_shape=1)
        actions = env.actions_from_tensor(_fmt([2]))
        states = env._step(states, actions)
        fwd = states.forward_masks[0]
        bwd = states.backward_masks[0]
        # Not exit yet (still macros to place if n_macros > 1, or exit if n_macros == 1)
        if env.n_macros > 1:
            assert fwd[:-1].any()
        # Backward: cell 2 should be True
        assert bwd[2].item()


# ===================================================================
# D. Sentinel Handling (3 tests)
# ===================================================================


class TestSentinelHandling:

    def test_apply_state_unplaced(self, env):
        """All -1 -> no macros placed in plc."""
        state = torch.full((env.n_macros,), -1, dtype=torch.long)
        env._apply_state_to_plc(state)
        for idx in env._hard_macro_indices:
            assert not env.plc.is_node_placed(idx)

    def test_apply_state_sf(self, env):
        """All -2 (sf sentinel) -> no macros placed (loc >= 0 guard)."""
        state = torch.full((env.n_macros,), -2, dtype=torch.long)
        env._apply_state_to_plc(state)
        for idx in env._hard_macro_indices:
            assert not env.plc.is_node_placed(idx)

    def test_apply_state_partial(self, env):
        """[2, -1] -> only first macro placed at cell 2."""
        state = torch.tensor([2] + [-1] * (env.n_macros - 1), dtype=torch.long)
        env._apply_state_to_plc(state)
        assert env.plc.is_node_placed(env._hard_macro_indices[0])
        for idx in env._hard_macro_indices[1:]:
            assert not env.plc.is_node_placed(idx)


# ===================================================================
# E. Reward & Normalization (7 tests)
# ===================================================================


class TestRewardNormalization:

    def test_log_reward_finite(self, env_with_reward):
        """Cached reward is finite and non-NaN."""
        _, _, lr = env_with_reward
        assert torch.isfinite(lr).all()
        assert not torch.isnan(lr).any()

    def test_log_reward_identical(self, env_with_reward):
        """Batch of 2 identical placements -> equal rewards."""
        e, final, lr_single = env_with_reward
        # Build batch of 2 identical final states
        t = final.tensor.expand(2, -1).clone()
        batch_states = e.States(tensor=t)
        lr = e.log_reward(batch_states)
        assert torch.allclose(lr[0], lr[1])

    def test_normalize_cost_none(self, env):
        """reward_norm=None -> raw cost returned."""
        assert env._normalize_cost(0.8) == 0.8

    def test_normalize_cost_zscore(self, env):
        """Z-score normalization formula."""
        env._cost_mean = 0.5
        env._cost_std = 0.2
        old_norm = env.reward_norm
        env.reward_norm = "zscore"
        try:
            result = env._normalize_cost(0.7)
            expected = (0.7 - 0.5) / 0.2
            assert abs(result - expected) < 1e-9
        finally:
            env.reward_norm = old_norm
            env._cost_mean = 0.0
            env._cost_std = 1.0

    def test_normalize_cost_minmax(self, env):
        """Min-max normalization formula."""
        env._cost_min = 0.1
        env._cost_max = 0.5
        old_norm = env.reward_norm
        env.reward_norm = "minmax"
        try:
            result = env._normalize_cost(0.3)
            expected = (0.3 - 0.1) / 0.5
            assert abs(result - expected) < 1e-9
        finally:
            env.reward_norm = old_norm
            env._cost_min = 0.0
            env._cost_max = 1.0

    def test_reward_temper(self, env_with_reward):
        """temper=2.0 -> log_reward is 2x the raw."""
        e, final, lr_raw = env_with_reward
        old_temper = e.reward_temper
        e.reward_temper = 2.0
        try:
            lr_tempered = e.log_reward(final)
            # raw uses temper=1.0, tempered uses 2.0
            assert torch.allclose(lr_tempered, lr_raw * 2.0, atol=1e-5)
        finally:
            e.reward_temper = old_temper

    def test_cost_stats_passed_to_env(self):
        """CostStats passed to env -> attrs match."""
        stats = CostStats(mean=1.0, std=2.0, min=0.5, range=3.0)
        try:
            e = ChipDesign(
                device="cpu",
                cd_finetune=False,
                reward_norm="zscore",
                cost_stats=stats,
            )
        except FileNotFoundError:
            pytest.skip("plc_wrapper_main not available")
        try:
            assert e._cost_mean == 1.0
            assert e._cost_std == 2.0
            assert e._cost_min == 0.5
            assert e._cost_max == 3.0
        finally:
            e.close()


# ===================================================================
# F. PLC Resolution (4 tests, pure Python, monkeypatch)
# ===================================================================


class TestPLCResolution:

    def test_resolve_plc_binary_env_var(self, tmp_path, monkeypatch):
        """PLC_WRAPPER_MAIN env var pointing to valid file -> returned."""
        binary = tmp_path / "plc_wrapper_main"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setenv("PLC_WRAPPER_MAIN", str(binary))
        assert _resolve_plc_binary() == str(binary)

    def test_resolve_plc_binary_not_found(self, monkeypatch):
        """No binary anywhere -> FileNotFoundError."""
        monkeypatch.delenv("PLC_WRAPPER_MAIN", raising=False)
        monkeypatch.setattr(
            "gfn.gym.helpers.chip_design.plc_client._PKG_DIR", "/nonexistent"
        )
        monkeypatch.setattr(
            "gfn.gym.helpers.chip_design.plc_client._find_singularity",
            lambda: None,
        )
        with pytest.raises(FileNotFoundError):
            _resolve_plc_binary()

    def test_find_singularity_none(self, monkeypatch):
        """shutil.which returns None for both -> None."""
        monkeypatch.setattr(
            "gfn.gym.helpers.chip_design.plc_client.shutil.which",
            lambda name: None,
        )
        assert _find_singularity() is None

    def test_find_singularity_apptainer(self, monkeypatch):
        """shutil.which returns path only for apptainer."""

        def fake_which(name):
            if name == "apptainer":
                return "/usr/bin/apptainer"
            return None

        monkeypatch.setattr(
            "gfn.gym.helpers.chip_design.plc_client.shutil.which",
            fake_which,
        )
        assert _find_singularity() == "/usr/bin/apptainer"


# ===================================================================
# G. File Parsing (8 tests, pure Python)
# ===================================================================


class TestFileParsing:

    def test_extract_attribute_block_name(self):
        """Parse Block name from SAMPLE_INIT_PLACEMENT."""
        result = placement_util.extract_attribute_from_comments(
            "Block", [SAMPLE_INIT_PLACEMENT]
        )
        assert result == "sample_clustered"

    def test_extract_attribute_missing(self):
        """Missing attribute returns None."""
        result = placement_util.extract_attribute_from_comments(
            "NonExistent", [SAMPLE_INIT_PLACEMENT]
        )
        assert result is None

    def test_extract_sizes(self):
        """Parse canvas/grid sizes from SAMPLE_INIT_PLACEMENT."""
        w, h, cols, rows = placement_util.extract_sizes_from_comments(
            [SAMPLE_INIT_PLACEMENT]
        )
        assert w == 500.0
        assert h == 500.0
        assert cols == 2
        assert rows == 2

    def test_get_blockages_none(self):
        """Sample file has no blockage lines -> None."""
        result = placement_util.get_blockages_from_comments([SAMPLE_INIT_PLACEMENT])
        assert result is None

    def test_get_blockages_parses(self, tmp_path):
        """Temp file with blockage line -> parsed correctly."""
        f = tmp_path / "test.plc"
        f.write_text(
            "# Blockage : 10.0 20.0 30.0 40.0 0.5\n"
            "# Blockage : 50.0 60.0 70.0 80.0 0.9\n"
            "data line\n"
        )
        result = placement_util.get_blockages_from_comments([str(f)])
        assert result is not None
        assert len(result) == 2
        assert result[0] == [10.0, 20.0, 30.0, 40.0, 0.5]
        assert result[1] == [50.0, 60.0, 70.0, 80.0, 0.9]

    def test_cost_info_not_done(self):
        """done=False -> cost is 0."""
        plc = MagicMock()
        cost, info = placement_util.cost_info_function(plc, done=False)
        assert cost == 0.0

    def test_cost_info_done(self):
        """done=True -> weighted sum of costs."""
        plc = MagicMock()
        plc.get_cost.return_value = 0.3
        plc.get_congestion_cost.return_value = 0.4
        plc.get_density_cost.return_value = 0.5
        cost, info = placement_util.cost_info_function(
            plc,
            done=True,
            wirelength_weight=1.0,
            density_weight=1.0,
            congestion_weight=0.5,
        )
        expected = 1.0 * 0.3 + 0.5 * 0.4 + 1.0 * 0.5
        assert abs(cost - expected) < 1e-9
        assert info["wirelength"] == 0.3
        assert info["congestion"] == 0.4
        assert info["density"] == 0.5

    def test_cost_info_zero_weights(self):
        """All weights 0 -> cost 0, methods not called."""
        plc = MagicMock()
        cost, info = placement_util.cost_info_function(
            plc,
            done=True,
            wirelength_weight=0.0,
            density_weight=0.0,
            congestion_weight=0.0,
        )
        assert cost == 0.0
        plc.get_cost.assert_not_called()
        plc.get_congestion_cost.assert_not_called()
        plc.get_density_cost.assert_not_called()


# ===================================================================
# H. Integration (5 tests, require binary)
# ===================================================================


class TestIntegration:

    def test_placement_cost_lifecycle(self, env):
        """PlacementCost: query grid size, close."""
        cols, rows = env.plc.get_grid_num_columns_rows()
        assert cols == 2
        assert rows == 2

    def test_create_placement_cost_sample(self, env):
        """Canvas size matches sample netlist."""
        w, h = env.plc.get_canvas_width_height()
        assert w == 500.0
        assert h == 500.0

    def test_node_ordering_descending_size(self, env):
        """Hard macros come first in ordering."""
        ordered = placement_util.get_ordered_node_indices(
            mode="descending_size_macro_first", plc=env.plc
        )
        hard = [m for m in ordered if not env.plc.is_node_soft_macro(m)]
        soft = [m for m in ordered if env.plc.is_node_soft_macro(m)]
        # Hard macros should precede soft macros
        if hard and soft:
            assert max(ordered.index(h) for h in hard) < min(
                ordered.index(s) for s in soft
            )

    def test_nodes_of_types_macro(self, env):
        """nodes_of_types with MACRO returns MACRO type nodes."""
        macro_indices = list(placement_util.nodes_of_types(env.plc, ["MACRO"]))
        assert len(macro_indices) >= env.n_macros
        for idx in macro_indices:
            assert env.plc.get_node_type(idx) == "MACRO"

    def test_get_restore_orientations_roundtrip(self, env):
        """Save orientations, change, restore, verify."""
        # Place macros so orientations can be queried
        states = env.reset(batch_shape=1)
        for i in range(env.n_macros):
            actions = env.actions_from_tensor(_fmt([i]))
            states = env.step(states, actions)
        env._apply_state_to_plc(states.tensor[0])

        original = placement_util.get_macro_orientations(env.plc)
        if not original:
            pytest.skip("No macros with orientations")

        # Change one orientation
        first_idx = next(iter(original))
        env.plc.update_macro_orientation(first_idx, "S")

        # Restore
        placement_util.restore_macro_orientations(env.plc, original)

        restored = placement_util.get_macro_orientations(env.plc)
        assert restored == original
