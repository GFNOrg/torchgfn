import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import torch

from gfn.containers import Container
from gfn.states import States

if TYPE_CHECKING:
    from gfn.estimators import Estimator  # type: ignore


def check_cond_forward(
    module: "Estimator",
    module_name: str,
    states: States,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Call estimator forward with or without conditioning using error handlers.

    Uses the same exception handling policy as the legacy utility to keep
    behavior consistent across adapters and probability code paths.
    """
    if condition is not None:
        with has_conditioning_exception_handler(module_name, module):
            return module(states, condition)
    else:
        with no_conditioning_exception_handler(module_name, module):
            return module(states)


@contextmanager
def has_conditioning_exception_handler(
    target_name: str,
    target: Any,
):
    """A context manager that handles exceptions when conditioning is passed.

    Args:
        target_name: The name of the target.
        target: The target object.
    """
    try:
        yield
    except TypeError as e:
        print(f"conditioning was passed but {target_name} is {type(target)}")
        print(f"error: {str(e)}")
        raise


@contextmanager
def no_conditioning_exception_handler(
    target_name: str,
    target: Any,
):
    """A context manager that handles exceptions when no conditioning is passed.

    Args:
        target_name: The name of the target.
        target: The target object.
    """
    try:
        yield
    except TypeError as e:
        print(f"conditioning was not passed but {target_name} is {type(target)}")
        print(f"error: {str(e)}")
        raise


@contextmanager
def is_callable_exception_handler(
    target_name: str,
    target: Any,
):
    """A context manager that handles exceptions when a target is not callable.

    Args:
        target_name: The name of the target.
        target: The target object.
    """
    try:
        yield
    except:  # noqa
        print(
            f"conditioning was passed but {target_name} is not callable: {type(target)}"
        )
        raise


def warn_about_recalculating_logprobs(
    obj: Container,
    recalculate_all_logprobs: bool,
):
    """Warns the user if logprobs are being recalculated.

    Args:
        obj: The container to check for logprobs.
        recalculate_all_logprobs: Whether to recalculate all logprobs.
    """
    if recalculate_all_logprobs and obj.has_log_probs:
        warnings.warn(
            "Recalculating logprobs for a container that already has them. "
            "This might be intended, if the log_probs were calculated off-policy. "
            "However, this is inefficient when training on-policy. In this case, "
            "you should instead call loss() or loss_from_trajectories() with "
            "recalculate_all_logprobs=False."
        )
