from typing import cast

import pytest
import torch

from gfn.estimators import DiscretePolicyEstimator, LogitBasedEstimator
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP


class SimpleDiscreteStates(DiscreteStates):
    """Simple discrete states for testing purposes."""

    n_actions = 5

    def __init__(self, tensor, forward_masks=None, backward_masks=None):
        super().__init__(tensor, forward_masks, backward_masks)

    @property
    def state_shape(self):
        return (3,)

    @property
    def device(self):
        return self.tensor.device

    @property
    def s0(self):
        return torch.zeros(3, device=self.device)

    @property
    def sf(self):
        return torch.ones(3, device=self.device) * 2


def reference_probability_computation(
    logits: torch.Tensor,
    masks: torch.Tensor,
    sf_bias: float = 0.0,
    temperature: float = 1.0,
    epsilon: float = 0.0,
    is_backward: bool = False,
) -> torch.Tensor:
    """Reference implementation for probability computation as provided in the user query."""
    assert masks.any(dim=-1).all(), "No possible actions"

    # Clone and mask logits
    masked_logits = logits.clone()
    masked_logits[~masks] = -float("inf")

    # Apply sf_bias to the last action (exit action)
    if sf_bias != 0.0:
        masked_logits[:, -1] -= sf_bias

    # Apply temperature
    if temperature != 1.0:
        masked_logits /= temperature

    # Compute probabilities
    probs = torch.softmax(masked_logits, dim=-1)

    # Apply epsilon-greedy exploration
    if epsilon != 0.0:
        uniform_dist_probs = masks / masks.sum(dim=-1, keepdim=True)
        probs = (1 - epsilon) * probs + epsilon * uniform_dist_probs

    return probs


def test_prepare_logits_basic():
    """Test _prepare_logits with basic parameters."""
    # Create test data
    batch_size, n_actions = 4, 5
    logits = torch.randn(batch_size, n_actions)
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)

    # Mask some actions
    masks[0, 1] = False  # Mask action 1 for first state
    masks[1, [2, 3]] = False  # Mask actions 2,3 for second state

    # Test basic functionality
    prepared = LogitBasedEstimator._prepare_logits(
        logits=logits, masks=masks, sf_index=None, sf_bias=0.0, temperature=1.0
    )

    # Check that masked actions have -inf logits
    assert torch.isinf(prepared[0, 1]) and prepared[0, 1] < 0
    assert torch.isinf(prepared[1, 2]) and prepared[1, 2] < 0
    assert torch.isinf(prepared[1, 3]) and prepared[1, 3] < 0

    # Check that unmasked actions remain finite
    assert torch.isfinite(prepared[0, 0])
    assert torch.isfinite(prepared[1, 0])
    assert torch.isfinite(prepared[1, 1])


def test_prepare_logits_with_sf_bias():
    """Test _prepare_logits with sf_bias."""
    batch_size, n_actions = 3, 4
    logits = torch.randn(batch_size, n_actions)
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)
    sf_bias = 2.0
    sf_index = n_actions - 1  # Last action is exit action

    original_exit_logits = logits[:, sf_index].clone()

    prepared = LogitBasedEstimator._prepare_logits(
        logits=logits, masks=masks, sf_index=sf_index, sf_bias=sf_bias, temperature=1.0
    )

    # Check that sf_bias was subtracted from exit action
    expected_exit_logits = original_exit_logits - sf_bias
    assert torch.allclose(prepared[:, sf_index], expected_exit_logits)


def test_prepare_logits_with_temperature():
    """Test _prepare_logits with temperature scaling."""
    batch_size, n_actions = 3, 4
    logits = torch.randn(batch_size, n_actions)
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)
    temperature = 2.0

    prepared = LogitBasedEstimator._prepare_logits(
        logits=logits, masks=masks, sf_index=None, sf_bias=0.0, temperature=temperature
    )

    # Check that temperature scaling was applied
    expected = logits / temperature
    expected[~masks] = -float("inf")
    assert torch.allclose(prepared, expected)


def test_prepare_logits_all_masked_row_with_sf_index():
    """Test _prepare_logits behavior when entire row is masked but sf_index is provided."""
    batch_size, n_actions = 3, 4
    logits = torch.randn(batch_size, n_actions)
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)

    # Mask all actions for the second state
    masks[1, :] = False
    sf_index = n_actions - 1

    prepared = LogitBasedEstimator._prepare_logits(
        logits=logits, masks=masks, sf_index=sf_index, sf_bias=0.0, temperature=1.0
    )

    # Check that the sf_index was set to 0.0 for the all-masked row
    assert prepared[1, sf_index] == 0.0
    # All other actions in that row should be -inf
    for i in range(n_actions - 1):
        assert torch.isinf(prepared[1, i]) and prepared[1, i] < 0


def test_prepare_logits_all_masked_row_without_sf_index():
    """Test _prepare_logits behavior when entire row is masked and no sf_index."""
    batch_size, n_actions = 3, 4
    logits = torch.randn(batch_size, n_actions)
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)

    # Mask all actions for the second state
    masks[1, :] = False

    prepared = LogitBasedEstimator._prepare_logits(
        logits=logits, masks=masks, sf_index=None, sf_bias=0.0, temperature=1.0
    )

    # Check that the first column was set to 0.0 for the all-masked row
    assert prepared[1, 0] == 0.0
    # All other actions in that row should be -inf
    for i in range(1, n_actions):
        assert torch.isinf(prepared[1, i]) and prepared[1, i] < 0


def test_compute_logits_for_distribution_no_epsilon():
    """Test _compute_logits_for_distribution without epsilon exploration."""
    batch_size, n_actions = 4, 5
    logits = torch.randn(batch_size, n_actions)
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)

    # Mask some actions
    masks[0, 1] = False
    masks[1, [2, 3]] = False

    result = LogitBasedEstimator._compute_logits_for_distribution(
        logits=logits,
        masks=masks,
        sf_index=n_actions - 1,
        sf_bias=0.0,
        temperature=1.0,
        epsilon=0.0,
    )

    # Result should be log-softmax of prepared logits
    prepared = LogitBasedEstimator._prepare_logits(
        logits=logits, masks=masks, sf_index=n_actions - 1, sf_bias=0.0, temperature=1.0
    )
    expected = torch.log_softmax(prepared, dim=-1)

    assert torch.allclose(result, expected)


def test_compute_logits_for_distribution_with_epsilon():
    """Test _compute_logits_for_distribution with epsilon exploration."""
    batch_size, n_actions = 4, 5
    logits = torch.randn(batch_size, n_actions)
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)
    epsilon = 0.1

    # Mask some actions
    masks[0, 1] = False
    masks[1, [2, 3]] = False

    result = LogitBasedEstimator._compute_logits_for_distribution(
        logits=logits,
        masks=masks,
        sf_index=n_actions - 1,
        sf_bias=0.0,
        temperature=1.0,
        epsilon=epsilon,
    )

    # Convert to probabilities and check they sum to 1
    probs = torch.exp(result)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))

    # Check that masked actions have zero probability
    assert torch.allclose(probs[0, 1], torch.zeros(1))
    assert torch.allclose(probs[1, 2], torch.zeros(1))
    assert torch.allclose(probs[1, 3], torch.zeros(1))


@pytest.mark.parametrize("sf_bias", [0.0, 1.0, 2.5])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.3])
def test_against_reference_implementation(sf_bias, temperature, epsilon):
    """Test LogitBasedEstimator methods against reference implementation."""
    torch.manual_seed(42)  # For reproducible results

    batch_size, n_actions = 6, 8
    logits = torch.randn(batch_size, n_actions)

    # Create diverse mask patterns
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)
    masks[0, [1, 3]] = False  # Partially masked
    masks[1, :3] = False  # First few actions masked
    masks[2, -2:] = False  # Last few actions masked
    masks[3, [0, 2, 4, 6]] = False  # Alternating pattern
    # Keep masks[4] and masks[5] fully unmasked

    # Ensure all rows have at least one valid action
    assert masks.any(dim=-1).all()

    # Get probabilities from LogitBasedEstimator
    logits_for_dist = LogitBasedEstimator._compute_logits_for_distribution(
        logits=logits,
        masks=masks,
        sf_index=n_actions - 1,
        sf_bias=sf_bias,
        temperature=temperature,
        epsilon=epsilon,
    )
    estimator_probs = torch.exp(logits_for_dist)

    # Get probabilities from reference implementation
    reference_probs = reference_probability_computation(
        logits=logits,
        masks=masks,
        sf_bias=sf_bias,
        temperature=temperature,
        epsilon=epsilon,
    )

    # Compare probabilities (allowing for small numerical differences)
    assert torch.allclose(
        estimator_probs, reference_probs, atol=1e-6, rtol=1e-5
    ), f"Probabilities don't match for sf_bias={sf_bias}, temp={temperature}, eps={epsilon}"

    # Additional checks
    # 1. Probabilities should sum to 1
    assert torch.allclose(estimator_probs.sum(dim=-1), torch.ones(batch_size))
    assert torch.allclose(reference_probs.sum(dim=-1), torch.ones(batch_size))

    # 2. Masked actions should have zero probability
    assert torch.allclose(
        estimator_probs[~masks], torch.zeros_like(estimator_probs[~masks])
    )
    assert torch.allclose(
        reference_probs[~masks], torch.zeros_like(reference_probs[~masks])
    )

    # 3. All probabilities should be non-negative
    assert (estimator_probs >= 0).all()
    assert (reference_probs >= 0).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_numerical_stability_different_dtypes(dtype):
    """Test numerical stability with different dtypes."""
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bfloat16 requires CUDA")

    torch.manual_seed(42)
    batch_size, n_actions = 4, 6

    # Create logits with extreme values to test numerical stability
    logits = torch.tensor(
        [
            [100.0, -100.0, 50.0, -50.0, 0.0, 25.0],
            [-100.0, 100.0, -50.0, 50.0, 0.0, -25.0],
            [1e-10, 1e10, -1e10, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )

    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)
    masks[0, 1] = False  # Mask the -100.0 logit
    masks[1, 0] = False  # Mask the -100.0 logit
    masks[2, 2] = False  # Mask the -1e10 logit

    # Test that computation doesn't produce NaN or Inf in final probabilities
    result = LogitBasedEstimator._compute_logits_for_distribution(
        logits=logits,
        masks=masks,
        sf_index=n_actions - 1,
        sf_bias=1.0,
        temperature=0.5,
        epsilon=0.1,
    )

    probs = torch.exp(result)

    # Check no NaN values
    assert not torch.isnan(probs).any(), f"NaN values found with dtype {dtype}"

    # Check probabilities sum to 1
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(
        prob_sums, torch.ones_like(prob_sums), atol=1e-5
    ), f"Probabilities don't sum to 1 with dtype {dtype}"

    # Check masked actions have zero probability
    assert torch.allclose(probs[~masks], torch.zeros_like(probs[~masks]), atol=1e-6)


def test_discrete_policy_estimator_integration():
    """Test integration with DiscretePolicyEstimator."""
    # Create a simple environment and estimator
    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    preprocessor = KHotPreprocessor(env.height, env.ndim)
    module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)

    estimator = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
        is_backward=False,
    )

    # Create test states
    batch_size = 8
    states_tensor = torch.randint(0, env.height, (batch_size, env.ndim))
    forward_masks = torch.ones(batch_size, env.n_actions, dtype=torch.bool)
    backward_masks = torch.ones(batch_size, env.n_actions - 1, dtype=torch.bool)

    # Create states using environment's States class
    states = env.States(states_tensor, forward_masks, backward_masks)
    env.update_masks(states)

    # Test different parameter combinations
    test_params = [
        {"sf_bias": 0.0, "temperature": 1.0, "epsilon": 0.0},
        {"sf_bias": 1.5, "temperature": 0.8, "epsilon": 0.0},
        {"sf_bias": 0.0, "temperature": 2.0, "epsilon": 0.1},
        {"sf_bias": 2.0, "temperature": 0.5, "epsilon": 0.2},
    ]

    for params in test_params:
        # Get module output first
        module_output = estimator(states)
        # Get distribution from estimator
        dist = estimator.to_probability_distribution(states, module_output, **params)

        # Sample from distribution to ensure it works
        actions = dist.sample()
        assert actions.shape == (batch_size, 1)

        # Check that sampled actions are within valid range
        assert (actions >= 0).all()
        assert (actions < env.n_actions).all()

        # Check that probabilities sum to 1
        probs = cast(torch.Tensor, dist.probs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))

        # Check that masked actions have zero probability
        assert torch.allclose(
            probs[~states.forward_masks], torch.zeros_like(probs[~states.forward_masks])
        )


def test_edge_case_single_valid_action():
    """Test behavior when only one action is valid per state."""
    batch_size, n_actions = 4, 5
    logits = torch.randn(batch_size, n_actions)

    # Create masks where only one action is valid per state
    masks = torch.zeros(batch_size, n_actions, dtype=torch.bool)
    masks[0, 0] = True  # Only action 0 valid
    masks[1, 2] = True  # Only action 2 valid
    masks[2, 4] = True  # Only action 4 valid
    masks[3, 1] = True  # Only action 1 valid

    result = LogitBasedEstimator._compute_logits_for_distribution(
        logits=logits,
        masks=masks,
        sf_index=n_actions - 1,
        sf_bias=0.0,
        temperature=1.0,
        epsilon=0.2,  # Test with epsilon > 0
    )

    probs = torch.exp(result)

    # Check that only valid actions have non-zero probability
    assert probs[0, 0] > 0 and probs[0, [1, 2, 3, 4]].sum() == 0
    assert probs[1, 2] > 0 and probs[1, [0, 1, 3, 4]].sum() == 0
    assert probs[2, 4] > 0 and probs[2, [0, 1, 2, 3]].sum() == 0
    assert probs[3, 1] > 0 and probs[3, [0, 2, 3, 4]].sum() == 0

    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))


def test_uniform_log_probs_method():
    """Test the _uniform_log_probs static method."""
    batch_size, n_actions = 3, 4

    # Test case 1: All actions valid
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)
    log_uniform = LogitBasedEstimator._uniform_log_probs(masks)

    expected_log_prob = -torch.log(torch.tensor(n_actions, dtype=torch.float))
    assert torch.allclose(log_uniform, expected_log_prob.expand_as(log_uniform))

    # Test case 2: Some actions masked
    masks[0, [1, 3]] = False  # 2 valid actions
    masks[1, [0, 1, 2]] = False  # 1 valid action
    # masks[2] remains all valid (4 actions)

    log_uniform = LogitBasedEstimator._uniform_log_probs(masks)

    # Check first row (2 valid actions)
    expected_0 = -torch.log(torch.tensor(2.0))
    assert torch.allclose(log_uniform[0, [0, 2]], expected_0.expand(2))
    assert (
        torch.isinf(log_uniform[0, [1, 3]]).all() and (log_uniform[0, [1, 3]] < 0).all()
    )

    # Check second row (1 valid action)
    expected_1 = -torch.log(torch.tensor(1.0))  # Should be 0
    assert torch.allclose(log_uniform[1, 3], expected_1)
    assert (
        torch.isinf(log_uniform[1, [0, 1, 2]]).all()
        and (log_uniform[1, [0, 1, 2]] < 0).all()
    )

    # Check third row (4 valid actions)
    expected_2 = -torch.log(torch.tensor(4.0))
    assert torch.allclose(log_uniform[2], expected_2.expand(4))


def test_mix_with_uniform_in_log_space():
    """Test the _mix_with_uniform_in_log_space static method."""
    batch_size, n_actions = 3, 4
    set_seed(123)

    # Create log-softmax values
    logits = torch.randn(batch_size, n_actions)
    masks = torch.ones(batch_size, n_actions, dtype=torch.bool)
    masks[0, 1] = False  # Mask one action

    # Manual masking.
    masked_logits_manual = logits.clone()
    masked_logits_manual[~masks] = -float("inf")
    lsm = torch.log_softmax(masked_logits_manual, dim=-1)

    # Automatic masking.
    lsm_automatic = torch.log_softmax(
        LogitBasedEstimator._prepare_logits(
            logits=logits, masks=masks, sf_index=None, sf_bias=0.0, temperature=1.0
        ),
        dim=-1,
    )

    # Check that the two methods produce the same result.
    assert torch.allclose(lsm, lsm_automatic)

    epsilon = 0.2

    mixed = LogitBasedEstimator._mix_with_uniform_in_log_space(lsm, masks, epsilon)
    mixed_probs = torch.exp(mixed)
    # Check probabilities sum to 1
    assert torch.allclose(mixed_probs.sum(dim=-1), torch.ones(batch_size))

    # Check masked actions have zero probability
    assert torch.allclose(mixed_probs[~masks], torch.zeros_like(mixed_probs[~masks]))

    # Test epsilon = 0 case (should return original)
    mixed_zero_eps = LogitBasedEstimator._mix_with_uniform_in_log_space(lsm, masks, 0.0)
    assert torch.allclose(mixed_zero_eps, lsm)


if __name__ == "__main__":
    test_mix_with_uniform_in_log_space()
