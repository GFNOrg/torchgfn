import random

import numpy as np
import torch


def set_seed(seed: int, performance_mode: bool = False) -> None:
    """Used to control randomness."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # These are only set when we care about reproducibility over performance.
    if not performance_mode:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_same_device(device1: torch.device, device2: torch.device) -> None:
    """Ensure that two tensors are on the same device."""
    if device1 == device2:
        return

    if device1.type != device2.type:
        raise ValueError(f"The devices have different types: {device1}, {device2}")

    index1, index2 = device1.index, device2.index

    # Same type and same index.
    if index1 == index2:
        return

    # Both have not-None index but they are different.
    if index1 is not None and index2 is not None:
        raise ValueError(f"Device index mismatch: {device1}, {device2}")

    # If one device index is None and the other is not,
    # the None index defaults to torch.cuda.current_device().
    # Check that the not-None index matches the current device index.
    current_device = torch.cuda.current_device()
    for idx in (index1, index2):
        if idx is not None and idx != current_device:
            raise ValueError(f"Device index mismatch: {device1}, {device2}")
