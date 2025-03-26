import random

import numpy as np
import torch


def set_seed(seed: int, performance_mode: bool = False) -> None:
    """Used to control randomness."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    # Devices are different due to the different indices.
    if device1.type == device2.type:
        index1, index2 = device1.index, device2.index

        # Case 1: They have different indices, which is problematic.
        if index1 is not None and index2 is not None:
            raise ValueError(f"The devices have different indices: {device1}, {device2}")
        # Case 2: At least one of them has None index, which is fine for now.
        else:
            return  # FIXME: This could be problematic if we use multiple GPUs.

    raise ValueError(f"The devices are different: {device1}, {device2}")
