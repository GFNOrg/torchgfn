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

    if device1.type != device2.type:
        raise ValueError(f"The devices have different types: {device1}, {device2}")

    # Same type but different indices or one is None.
    index1, index2 = device1.index, device2.index
    if index1 == index2:
        return

    current_device = torch.cuda.current_device()
    # Any non-None index must match the current device.
    for idx in (index1, index2):
        if idx is not None and idx != current_device:
            raise ValueError(f"Device index mismatch: {device1}, {device2}")
