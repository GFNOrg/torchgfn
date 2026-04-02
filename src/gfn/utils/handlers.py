import logging
from contextlib import contextmanager
from typing import Any

from gfn.containers import Container

logger = logging.getLogger(__name__)


@contextmanager
def has_conditions_exception_handler(
    target_name: str,
    target: Any,
):
    """Catches TypeError when calling a target with conditions and logs context."""
    try:
        yield
    except TypeError as e:
        logger.error(
            f"Failed calling {target_name} ({type(target).__name__}) with conditions: {e}"
        )
        raise


@contextmanager
def no_conditions_exception_handler(
    target_name: str,
    target: Any,
):
    """Catches TypeError when calling a target without conditions and logs context."""
    try:
        yield
    except TypeError as e:
        logger.error(
            f"Failed calling {target_name} ({type(target).__name__}) "
            f"without conditions: {e}"
        )
        raise


@contextmanager
def is_callable_exception_handler(
    target_name: str,
    target: Any,
):
    """Catches exceptions when calling a target that may not be callable."""
    try:
        yield
    except Exception as e:
        logger.error(f"Failed calling {target_name} ({type(target).__name__}): {e}")
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
        logger.warning(
            "Recalculating logprobs for a container that already has them. "
            "This might be intended, if the log_probs were calculated off-policy. "
            "However, this is inefficient when training on-policy. In this case, "
            "you should instead call loss() or loss_from_trajectories() with "
            "recalculate_all_logprobs=False."
        )
