import pytest
import torch


# Fixture that runs automatically for all tests.
@pytest.fixture(scope="function", autouse=True)
def cpu_only():
    torch.set_default_device("cpu")

# Fixture to reset torch deterministic algorithms state after each test.
@pytest.fixture(autouse=True)
def reset_torch_determinism():
    """Ensure deterministic algorithms don't leak between tests"""
    original_state = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(original_state)
