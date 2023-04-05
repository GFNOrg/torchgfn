from typing import cast

import torch
from torchtyping import TensorType

# Typing.
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]


def correct_cast(
    forward_masks: ForwardMasksTensor | None,
    backward_masks: BackwardMasksTensor | None,
) -> tuple[ForwardMasksTensor, BackwardMasksTensor]:
    """
    Casts the given masks to the correct type, if they are not None.
    This function is to help with type checking only.
    """
    forward_masks = cast(ForwardMasksTensor, forward_masks)
    backward_masks = cast(BackwardMasksTensor, backward_masks)
    return forward_masks, backward_masks