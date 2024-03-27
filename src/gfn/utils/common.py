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


def has_log_probs(obj):
    """Returns True if the submitted object has the log_probs attribute populated."""
    if not hasattr(obj, "log_probs"):
        return False

    return obj.log_probs is not None and obj.log_probs.nelement() > 0
