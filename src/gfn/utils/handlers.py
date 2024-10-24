from contextlib import contextmanager
from typing import Any


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
