import pytest
import torch


# Fixture that runs automatically for all tests.
@pytest.fixture(scope="function", autouse=True)
def cpu_only():
    torch.set_default_device("cpu")
