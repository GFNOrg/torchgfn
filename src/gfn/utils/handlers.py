import warnings
from contextlib import contextmanager
from typing import Any

from gfn.containers import Container


@contextmanager
def has_conditioning_exception_handler(
    target_name: str,
    target: Any,
):
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
    if recalculate_all_logprobs and obj.has_log_probs:
        warnings.warn(
            "Recalculating logprobs for a container that already has them. "
            "This might be intended, if the log_probs were calculated off-policy."
            "However, this is inefficient when training on-policy. In this case, "
            "you should instead call loss() or loss_from_trajectories() with "
            "recalculate_all_logprobs=False "
        )
