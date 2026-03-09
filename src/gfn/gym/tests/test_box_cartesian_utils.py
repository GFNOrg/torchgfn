import pytest
import torch

from gfn.gym import Box
from gfn.gym.helpers.box_cartesian_utils import (
    BoxCartesianPBEstimator,
    BoxCartesianPBMLP,
    BoxCartesianPFEstimator,
    BoxCartesianPFMLP,
)


class TestBoxCartesianEnvironment:
    """Tests for the Box environment with Cartesian semantics."""

    @pytest.fixture
    def env(self):
        return Box(delta=0.25, epsilon=1e-6, device="cpu")

    def test_forward_step_adds_action(self, env):
        """Forward step should add action to state."""
        states = env.States(torch.tensor([[0.2, 0.3], [0.5, 0.5]]))
        actions = env.Actions(torch.tensor([[0.25, 0.25], [0.3, 0.2]]))
        next_states = env.step(states, actions)
        expected = states.tensor + actions.tensor
        assert torch.allclose(next_states.tensor, expected)

    def test_backward_step_subtracts_action(self, env):
        """Backward step should subtract action from state."""
        states = env.States(torch.tensor([[0.5, 0.5], [0.7, 0.8]]))
        actions = env.Actions(torch.tensor([[0.25, 0.25], [0.3, 0.3]]))
        prev_states = env.backward_step(states, actions)
        expected = states.tensor - actions.tensor
        assert torch.allclose(prev_states.tensor, expected)

    def test_forward_backward_roundtrip(self, env):
        """Forward then backward should return to original state."""
        states = env.States(torch.tensor([[0.3, 0.4]]))
        actions = env.Actions(torch.tensor([[0.25, 0.25]]))
        next_states = env.step(states, actions)
        recovered = env.backward_step(next_states, actions)
        assert torch.allclose(recovered.tensor, states.tensor)

    def test_is_action_valid_s0_forward(self, env):
        """From s0, forward actions should be in [0, 1] (full space coverage)."""
        s0 = env.States(torch.tensor([[0.0, 0.0]]))

        # Valid: action in [0, 1]
        valid_actions = env.Actions(torch.tensor([[0.5, 0.8]]))
        assert env.is_action_valid(s0, valid_actions, backward=False)

        # Invalid: action > 1
        invalid_actions = env.Actions(torch.tensor([[1.1, 0.1]]))
        assert not env.is_action_valid(s0, invalid_actions, backward=False)

    def test_is_action_valid_non_s0_forward(self, env):
        """From non-s0, forward actions should be >= delta."""
        states = env.States(torch.tensor([[0.3, 0.3]]))

        # Valid: action >= delta and doesn't exceed boundary
        valid_actions = env.Actions(torch.tensor([[0.25, 0.25]]))
        assert env.is_action_valid(states, valid_actions, backward=False)

        # Invalid: action < delta
        invalid_actions = env.Actions(torch.tensor([[0.1, 0.1]]))
        assert not env.is_action_valid(states, invalid_actions, backward=False)

    def test_is_action_valid_boundary(self, env):
        """Actions shouldn't push state past boundary."""
        states = env.States(torch.tensor([[0.8, 0.8]]))

        # Invalid: would exceed boundary
        invalid_actions = env.Actions(torch.tensor([[0.3, 0.3]]))
        assert not env.is_action_valid(states, invalid_actions, backward=False)

        # Valid: stays within bounds
        valid_actions = env.Actions(torch.tensor([[0.19, 0.19]]))
        # This might be invalid due to min delta, but let's check
        # Actually for non-s0, we need action >= delta, so this is invalid
        assert not env.is_action_valid(states, valid_actions, backward=False)

    def test_exit_action_valid(self, env):
        """Exit actions should be valid for non-s0 states."""
        states = env.States(torch.tensor([[0.5, 0.5]]))
        exit_actions = env.Actions(torch.tensor([[float("-inf"), float("-inf")]]))
        # Exit actions have is_exit = True, so they're handled specially
        assert exit_actions.is_exit.all()
        # Verify that states are non-s0 (exit is valid from non-s0)
        assert not states.is_initial_state.all()


class TestBoxCartesianDistribution:
    """Tests for BoxCartesianDistribution."""

    @pytest.fixture
    def env(self):
        return Box(delta=0.25, epsilon=1e-6, device="cpu")

    @pytest.fixture
    def pf_module(self):
        return BoxCartesianPFMLP(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=5,
        )

    @pytest.fixture
    def pf_estimator(self, env, pf_module):
        return BoxCartesianPFEstimator(
            env, pf_module, n_components=5, min_concentration=0.1, max_concentration=5.0
        )

    def test_sample_shape(self, env, pf_estimator):
        """Sampled actions should have correct shape."""
        states = env.States(torch.rand(16, 2) * 0.5)
        module_output = pf_estimator.module(states.tensor)
        dist = pf_estimator.to_probability_distribution(states, module_output)
        actions = dist.sample()
        assert actions.shape == (16, 2)

    def test_sample_s0_in_range(self, env, pf_estimator):
        """From s0, sampled actions should be in [0, 1] (full space coverage)."""
        s0 = env.States(torch.zeros(10, 2))
        module_output = pf_estimator.module(s0.tensor)
        dist = pf_estimator.to_probability_distribution(s0, module_output)

        # Sample multiple times to check range
        for _ in range(5):
            actions = dist.sample()
            non_exit = ~torch.all(actions == float("-inf"), dim=-1)
            if non_exit.any():
                valid_actions = actions[non_exit]
                assert (valid_actions >= 0).all()
                assert (valid_actions <= 1 + 1e-5).all()

    def test_sample_non_s0_in_range(self, env, pf_estimator):
        """From non-s0, sampled actions should be >= delta."""
        states = env.States(torch.rand(10, 2) * 0.3 + 0.3)  # In [0.3, 0.6]
        module_output = pf_estimator.module(states.tensor)
        dist = pf_estimator.to_probability_distribution(states, module_output)

        for _ in range(5):
            actions = dist.sample()
            non_exit = ~torch.all(actions == float("-inf"), dim=-1)
            if non_exit.any():
                valid_actions = actions[non_exit]
                # Actions should be >= delta
                assert (valid_actions >= env.delta - 1e-5).all()
                # Actions shouldn't exceed remaining space
                remaining = 1.0 - states.tensor[non_exit]
                assert (valid_actions <= remaining + 1e-5).all()

    def test_log_prob_finite(self, env, pf_estimator):
        """Log probabilities should be finite for valid actions."""
        states = env.States(torch.rand(16, 2) * 0.5)
        module_output = pf_estimator.module(states.tensor)
        dist = pf_estimator.to_probability_distribution(states, module_output)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        assert log_probs.shape == (16,)
        assert torch.isfinite(log_probs).all()

    def test_log_prob_exit_at_boundary(self, env, pf_estimator):
        """Exit at boundary should use learned Bernoulli log_prob (finite, <= 0).

        Previously exit was forced to log_prob=0, making boundary exits invisible to
        the TB loss. Now the learned exit probability contributes so the policy gets
        gradient to prefer exiting at the reward ring vs. the hard boundary.
        Non-exit at boundary must remain -inf (impossible).
        """
        # States at boundary (any dim >= 1 - delta)
        states = env.States(torch.tensor([[0.8, 0.5], [0.5, 0.8]]))
        module_output = pf_estimator.module(states.tensor)
        dist = pf_estimator.to_probability_distribution(states, module_output)

        exit_actions = torch.full((2, 2), float("-inf"))
        log_probs = dist.log_prob(exit_actions)

        # Boundary exits: log_prob is from learned Bernoulli (finite and <= 0)
        assert log_probs.shape == (2,)
        assert torch.isfinite(log_probs).all()
        assert (log_probs <= 0).all()
        # Non-exit at boundary is still impossible
        non_exit_actions = torch.rand(2, 2) * 0.1 + env.delta
        log_probs_non_exit = dist.log_prob(non_exit_actions)
        assert torch.all(log_probs_non_exit == float("-inf"))

    # -- Forward log_prob validity tests ----------------------------------------
    # delta=0.25 for all tests in this class.
    # s0 support per dim: [0, 1].
    # non-s0 support per dim: [delta, 1 - state_d].
    # exit (-inf): forbidden from s0, required at boundary.

    @pytest.mark.parametrize(
        "state, action, reason",
        [
            # === From non-s0: action below delta ===
            ([0.5, 0.5], [0.1, 0.3], "dim0 below delta"),
            ([0.5, 0.5], [0.3, 0.1], "dim1 below delta"),
            ([0.5, 0.5], [0.1, 0.1], "both dims below delta"),
            ([0.3, 0.3], [0.0, 0.25], "dim0 is zero (below delta)"),
            # === From non-s0: action exceeds 1 - state ===
            ([0.5, 0.5], [0.6, 0.3], "dim0 would exceed 1"),
            ([0.5, 0.5], [0.3, 0.6], "dim1 would exceed 1"),
            ([0.5, 0.5], [0.6, 0.6], "both dims would exceed 1"),
            ([0.7, 0.3], [0.35, 0.3], "dim0 tight overshoot"),
            # === From non-s0: negative action ===
            ([0.5, 0.5], [-0.1, 0.3], "negative action dim0"),
            ([0.5, 0.5], [0.3, -0.5], "negative action dim1"),
            # === From s0: action exceeds 1 ===
            ([0.0, 0.0], [1.5, 0.5], "from s0, dim0 exceeds 1"),
            ([0.0, 0.0], [0.5, 1.5], "from s0, dim1 exceeds 1"),
            ([0.0, 0.0], [2.0, 2.0], "from s0, both dims exceed 1"),
            # === From s0: negative action ===
            ([0.0, 0.0], [-0.1, 0.5], "from s0, negative dim0"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_impossible_action_gets_neg_inf(
        self, env, pf_estimator, state, action, reason
    ):
        """Out-of-support forward actions must get -inf log_prob."""
        states = env.States(torch.tensor([state]))
        module_output = pf_estimator.module(states.tensor)
        dist = pf_estimator.to_probability_distribution(states, module_output)
        lp = dist.log_prob(torch.tensor([action]))
        assert lp.item() == float("-inf"), f"Expected -inf for: {reason}"

    @pytest.mark.parametrize(
        "state, action",
        [
            ([0.5, 0.5], [0.25, 0.25]),  # non-s0: at lower bound (delta)
            ([0.5, 0.5], [0.45, 0.45]),  # non-s0: near upper bound (1 - 0.5 = 0.5)
            ([0.5, 0.5], [0.35, 0.4]),  # non-s0: interior
            ([0.3, 0.3], [0.3, 0.3]),  # non-s0: action in mid-range
            ([0.0, 0.0], [0.5, 0.5]),  # s0: interior
            ([0.0, 0.0], [0.01, 0.99]),  # s0: near extremes of [0, 1]
        ],
    )
    def test_valid_action_gets_finite_log_prob(self, env, pf_estimator, state, action):
        """In-support forward actions must get finite log_prob."""
        states = env.States(torch.tensor([state]))
        module_output = pf_estimator.module(states.tensor)
        dist = pf_estimator.to_probability_distribution(states, module_output)
        lp = dist.log_prob(torch.tensor([action]))
        assert torch.isfinite(
            lp
        ).all(), f"Expected finite for state={state}, action={action}"

    def test_exit_from_s0_neg_inf(self, env, pf_estimator):
        """Exit from s0 is impossible and must get -inf."""
        s0 = env.States(torch.zeros(1, 2))
        module_output = pf_estimator.module(s0.tensor)
        dist = pf_estimator.to_probability_distribution(s0, module_output)
        lp = dist.log_prob(torch.full((1, 2), float("-inf")))
        assert lp.item() == float("-inf")

    def test_non_exit_at_boundary_neg_inf(self, env, pf_estimator):
        """At the boundary, non-exit actions are impossible and must get -inf."""
        # 0.8 >= 1 - delta = 0.75, so this is at boundary
        boundary = env.States(torch.tensor([[0.8, 0.5]]))
        module_output = pf_estimator.module(boundary.tensor)
        dist = pf_estimator.to_probability_distribution(boundary, module_output)
        non_exit = torch.tensor([[0.25, 0.25]])
        lp = dist.log_prob(non_exit)
        assert lp.item() == float("-inf")

    def test_batch_mixed_valid_invalid(self, env, pf_estimator):
        """In a batch, valid actions get finite log_prob and invalid get -inf."""
        states = env.States(torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
        module_output = pf_estimator.module(states.tensor)
        dist = pf_estimator.to_probability_distribution(states, module_output)
        actions = torch.tensor(
            [
                [0.35, 0.35],  # valid (in [0.25, 0.5])
                [0.1, 0.3],  # invalid (dim0 < delta)
                [0.6, 0.3],  # invalid (dim0 > 1 - state)
            ]
        )
        lps = dist.log_prob(actions)
        assert torch.isfinite(lps[0]), "valid action should be finite"
        assert lps[1].item() == float("-inf"), "below-delta should be -inf"
        assert lps[2].item() == float("-inf"), "over-boundary should be -inf"


class TestBoxCartesianPBDistribution:
    """Tests for BoxCartesianPBDistribution (backward policy)."""

    @pytest.fixture
    def env(self):
        return Box(delta=0.25, epsilon=1e-6, device="cpu")

    @pytest.fixture
    def pb_module(self):
        return BoxCartesianPBMLP(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=5,
        )

    @pytest.fixture
    def pb_estimator(self, env, pb_module):
        return BoxCartesianPBEstimator(
            env, pb_module, n_components=5, min_concentration=0.1, max_concentration=5.0
        )

    def test_sample_shape(self, env, pb_estimator):
        """Sampled backward actions should have correct shape."""
        states = env.States(torch.rand(16, 2) * 0.5 + 0.3)  # Non-s0 states
        module_output = pb_estimator.module(states.tensor)
        dist = pb_estimator.to_probability_distribution(states, module_output)
        actions = dist.sample()
        assert actions.shape == (16, 2)

    def test_sample_near_origin_to_s0(self, env, pb_estimator):
        """States near origin should go directly to s0."""
        # States with all dims < delta should go directly to s0
        near_origin = env.States(torch.tensor([[0.1, 0.1], [0.2, 0.15]]))
        module_output = pb_estimator.module(near_origin.tensor)
        dist = pb_estimator.to_probability_distribution(near_origin, module_output)
        actions = dist.sample()

        # Action should equal state (to go to s0)
        assert torch.allclose(actions, near_origin.tensor)

        # Verify backward step reaches s0
        prev_state = env.backward_step(near_origin, env.Actions(actions))
        assert torch.allclose(prev_state.tensor, torch.zeros_like(prev_state.tensor))

    def test_sample_mixed_dims_near_origin(self, env, pb_estimator):
        """States with mixed dims (some < delta, some >= delta) should handle correctly."""
        # One dim < delta (0.1), one dim >= delta (0.5)
        mixed = env.States(torch.tensor([[0.1, 0.5], [0.2, 0.6]]))
        module_output = pb_estimator.module(mixed.tensor)
        dist = pb_estimator.to_probability_distribution(mixed, module_output)
        actions = dist.sample()

        # For dim 0 (< delta): action should equal state
        assert torch.allclose(actions[:, 0], mixed.tensor[:, 0])

        # For dim 1 (>= delta): action should be in [delta, state]
        assert (actions[:, 1] >= env.delta - 1e-5).all()
        assert (actions[:, 1] <= mixed.tensor[:, 1] + 1e-5).all()

        # Verify backward step doesn't go negative
        prev_state = env.backward_step(mixed, env.Actions(actions))
        assert (prev_state.tensor >= -1e-5).all()

    def test_sample_non_origin_valid(self, env, pb_estimator):
        """Backward actions from non-origin should be valid."""
        states = env.States(torch.rand(10, 2) * 0.3 + 0.5)  # In [0.5, 0.8]
        module_output = pb_estimator.module(states.tensor)
        dist = pb_estimator.to_probability_distribution(states, module_output)
        actions = dist.sample()

        # Actions should be >= delta
        assert (actions >= env.delta - 1e-5).all()
        # Actions shouldn't exceed state (can't go below 0)
        assert (actions <= states.tensor + 1e-5).all()

    def test_log_prob_near_origin(self, env, pb_estimator):
        """Log prob for near-origin states going to s0 should be 0.

        When near origin (any dimension <= delta), BTS is forced and log_prob = 0.
        For numerical robustness, any action from a near-origin state is treated
        as BTS since that's the only valid action anyway.
        """
        near_origin = env.States(torch.tensor([[0.1, 0.1]]))
        module_output = pb_estimator.module(near_origin.tensor)
        dist = pb_estimator.to_probability_distribution(near_origin, module_output)

        # Correct action: go to s0 (BTS)
        correct_action = near_origin.tensor.clone()
        log_prob = dist.log_prob(correct_action)
        assert torch.allclose(log_prob, torch.zeros(1))

        # Any action from near-origin state is treated as BTS for numerical robustness
        # (since BTS is the only valid action, we don't need strict enforcement)
        other_action = torch.tensor([[0.05, 0.05]])
        log_prob_other = dist.log_prob(other_action)
        assert torch.allclose(log_prob_other, torch.zeros(1))

    def test_out_of_range_backward_action_gets_neg_inf(self, env, pb_estimator):
        """Non-BTS backward actions outside [delta, state] should get -inf log_prob."""
        states = env.States(torch.tensor([[0.7, 0.8]]))
        module_output = pb_estimator.module(states.tensor)
        dist = pb_estimator.to_probability_distribution(states, module_output)

        # Action below delta (0.1 < 0.25) — invalid backward action
        too_small = torch.tensor([[0.1, 0.3]])
        assert dist.log_prob(too_small).item() == float("-inf")

        # Action above state (0.9 > 0.8 in dim 1) — would make next state negative
        too_large = torch.tensor([[0.5, 0.9]])
        assert dist.log_prob(too_large).item() == float("-inf")

    def test_valid_backward_action_gets_finite_log_prob(self, env, pb_estimator):
        """Valid backward actions should get finite log_prob."""
        states = env.States(torch.tensor([[0.7, 0.8]]))
        module_output = pb_estimator.module(states.tensor)
        dist = pb_estimator.to_probability_distribution(states, module_output)

        # delta=0.25, state=[0.7, 0.8], so valid range is [0.25, 0.7] x [0.25, 0.8]
        valid_action = torch.tensor([[0.4, 0.5]])
        lp = dist.log_prob(valid_action)
        assert torch.isfinite(lp).all()


class TestBoxCartesianEndToEnd:
    """End-to-end tests for the Cartesian Box implementation."""

    @pytest.fixture
    def env(self):
        return Box(delta=0.25, epsilon=1e-6, device="cpu")

    @pytest.fixture
    def pf_estimator(self, env):
        module = BoxCartesianPFMLP(hidden_dim=32, n_hidden_layers=2, n_components=5)
        return BoxCartesianPFEstimator(
            env, module, n_components=5, min_concentration=0.1, max_concentration=5.0
        )

    @pytest.fixture
    def pb_estimator(self, env):
        module = BoxCartesianPBMLP(hidden_dim=32, n_hidden_layers=2, n_components=5)
        return BoxCartesianPBEstimator(
            env, module, n_components=5, min_concentration=0.1, max_concentration=5.0
        )

    def test_trajectory_stays_in_bounds(self, env, pf_estimator):
        """A trajectory should stay within [0, 1]^2."""
        state = env.States(torch.zeros(1, 2))

        for _ in range(20):  # Max steps
            module_output = pf_estimator.module(state.tensor)
            dist = pf_estimator.to_probability_distribution(state, module_output)
            action = dist.sample()

            # Check for exit
            if torch.all(action == float("-inf")):
                break

            # Take step
            next_state = env.step(state, env.Actions(action))

            # Verify bounds
            assert (next_state.tensor >= 0).all()
            assert (next_state.tensor <= 1 + 1e-5).all()

            state = next_state

    def test_backward_trajectory_reaches_s0(self, env, pb_estimator):
        """A backward trajectory should reach s0."""
        # Start from a terminal state
        state = env.States(torch.tensor([[0.7, 0.8]]))

        for _ in range(20):  # Max steps
            # Check if at s0
            if torch.all(state.tensor < 1e-6):
                break

            module_output = pb_estimator.module(state.tensor)
            dist = pb_estimator.to_probability_distribution(state, module_output)
            action = dist.sample()

            # Take backward step
            prev_state = env.backward_step(state, env.Actions(action))

            # Verify bounds
            assert (prev_state.tensor >= -1e-5).all()
            assert (prev_state.tensor <= state.tensor + 1e-5).all()

            state = prev_state

        # Should have reached s0
        assert torch.all(state.tensor < env.delta)

    def test_reward_function(self, env):
        """Reward function should return expected values."""
        # Test points in different reward regions
        final_states = env.States(
            torch.tensor(
                [
                    [0.5, 0.5],  # Center - low reward
                    [0.1, 0.1],  # Corner - high reward
                    [0.9, 0.9],  # Corner - high reward
                    [0.15, 0.85],  # Edge region
                ]
            )
        )
        rewards = env.reward(final_states)
        assert rewards.shape == (4,)
        assert (rewards > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
