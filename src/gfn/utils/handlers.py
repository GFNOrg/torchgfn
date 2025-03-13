import warnings
from contextlib import contextmanager
from typing import Any

from gfn.containers import Container, has_log_probs


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
    if recalculate_all_logprobs and has_log_probs(obj):
        warnings.warn(
            "Recalculating logprobs for a container that already has them. "
            "This is inefficient when training on-policy.You should instead "
            "call loss() or loss_from_trajectories() with "
            "recalculate_all_logprobs=False "
        )
