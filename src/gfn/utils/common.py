import random

import numpy as np
import torch


def set_seed(seed: int, performance_mode: bool = False) -> None:
    """Used to control randomness."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # These are only set when we care about reproducibility over performance.
    if not performance_mode:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
