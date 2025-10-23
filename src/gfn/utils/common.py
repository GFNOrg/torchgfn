import inspect
import os
import random
import threading
import time
import warnings
from typing import Any, Callable, Tuple

import numpy as np
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def is_int_dtype(tensor: torch.Tensor) -> bool:
    """Check if a tensor is an integer dtype."""
    return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)


def default_fill_value_for_dtype(dtype: torch.dtype) -> int | float:
    """Return default fill value for dtype.

    - Float and complex dtypes → ``-inf``
    - Integer dtypes → ``torch.iinfo(dtype).min``
    - Bool dtype → ``0``
    """

    # Floats
    if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return -float("inf")

    # Complex
    if dtype in (torch.complex64, torch.complex128):
        return -float("inf")

    # Bool
    if dtype == torch.bool:
        return 0

    # Integers (incl. uint8)
    try:
        return torch.iinfo(dtype).min
    except (TypeError, ValueError):
        # Fallback for any unusual/unsupported dtype
        return -float("inf")


def get_available_cpus() -> int:
    """Return the number of *usable* CPUs for the current process.

    The naive ``os.cpu_count()`` often reports the host's total logical cores, which
    can be misleading inside containers, job schedulers, or when CPU affinity is
    restricted.  This helper tries to detect the real quota:

    1. On Linux and recent *BSD it queries ``os.sched_getaffinity`` which already
       respects cgroups and task-set masks.
    2. If that is not available it looks at common thread-limiting environment
       variables (``OMP_NUM_THREADS``/``MKL_NUM_THREADS``/``NUMBA_NUM_THREADS``).
    3. Finally it falls back to ``os.cpu_count()`` and ensures the return value is
       at least ``1``.
    """

    # 1. Affinity mask – most accurate in containers / cgroups.
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            # Can happen in some restricted environments; fall back.
            pass

    # 2. Honour explicit user limits via env-vars.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        val = os.environ.get(var)
        if val and val.isdigit():
            n = int(val)
            if n > 0:
                return n

    # 3. Last resort.
    return os.cpu_count() or 1


def filter_kwargs_for_callable(
    callable_obj: Any, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Filter a kwargs dict to only the parameters accepted by callable_obj."""
    sig = inspect.signature(callable_obj)

    # If the callable accepts **kwargs, no filtering is necessary.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    accepted_names = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    # Remove common non-forwarded parameters if present in kwargs.
    accepted_names -= {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted_names}


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int, performance_mode: bool = False) -> None:
    """Used to control randomness for both single and distributed training.

    Args:
        seed: The seed to use for all random number generators
        performance_mode: If True, disables deterministic behavior for better performance.
            In multi-GPU settings, this only affects cuDNN. In multi-CPU settings,
            this allows parallel processing in NumPy.
    """
    if dist.is_initialized():
        # Get process-specific seed for distributed training
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        process_seed = seed + rank

        # Set all seeds with the process-specific seed
        torch.manual_seed(process_seed)
        random.seed(process_seed)
        np.random.seed(process_seed)

        # Handle GPU-specific seeding
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(process_seed)

        # Handle MPS (Apple Silicon) specific seeding
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(process_seed)

        # Set environment variable for distributed data loader
        os.environ["PYTHONHASHSEED"] = str(process_seed)

        # Thread-level metadata for reproducibility debugging.
        threading.current_thread()._seed = process_seed

        backend = dist.get_backend()

        # Get number of CPUs available to this process
        num_cpus = get_available_cpus()

        # Set device-specific environment variables
        if torch.cuda.is_available():
            # For GPU training, we can use multiple threads for CPU operations
            if performance_mode:
                os.environ["OMP_NUM_THREADS"] = str(num_cpus)
                os.environ["MKL_NUM_THREADS"] = str(num_cpus)
            else:
                # For reproducibility in GPU training, we still want deterministic
                # CPU operations
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
        else:
            # For CPU-only training, we need to be more careful with threading
            if performance_mode:

                # Allow parallel processing but with controlled number of threads
                # Different backends might handle threading differently
                if backend in ["mpi", "ccl"]:
                    # MPI and CCL backends often handle their own thread management
                    # Use a conservative thread count
                    num_threads = max(1, min(num_cpus, 4))
                else:
                    # For other backends, divide threads among processes
                    num_threads = max(1, num_cpus // world_size)

                os.environ["OMP_NUM_THREADS"] = str(num_threads)
                os.environ["MKL_NUM_THREADS"] = str(num_threads)
            else:
                # For perfect reproducibility in CPU training, disable parallel processing
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"

    else:
        # Non-distributed training - use the global seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Handle GPU-specific seeding for non-distributed case
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Handle MPS (Apple Silicon) specific seeding for non-distributed case
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        # Set environment variable for deterministic behavior
        os.environ["PYTHONHASHSEED"] = str(seed)

        # Thread-level metadata for reproducibility debugging.
        threading.current_thread()._seed = seed

    # These are only set when we care about reproducibility over performance
    if not performance_mode:
        # GPU-specific settings
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Enforce deterministic algorithms in PyTorch. This will raise an
        # error at runtime if a used operation does not have a deterministic
        # implementation.
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Older PyTorch (<1.8) fallback: do nothing.
            warnings.warn(
                "PyTorch is older than 1.8, deterministic algorithms are not supported.",
                UserWarning,
            )

        # CPU-specific settings for non-distributed case
        if not dist.is_initialized():
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"


def ensure_same_device(device1: torch.device, device2: torch.device) -> None:
    """Ensure that two tensors are on the same device.

    Args:
        device1: The first device.
        device2: The second device.

    Raises:
        ValueError: If the devices are not the same.
    """
    try:
        assert device1 == device2
        return
    except AssertionError:
        pass

    if device1.type != device2.type:
        raise ValueError(f"The devices have different types: {device1}, {device2}")

    # Device indices must differ since the device types are the same.
    index1, index2 = device1.index, device2.index

    # Both have not-None index but they are different.
    if index1 is not None and index2 is not None:
        raise ValueError(f"Device index mismatch: {device1}, {device2}")

    # If one device index is None and the other is not,
    # the None index defaults to torch.cuda.current_device().
    # Check that the not-None index matches the current device index.
    current_device = torch.cuda.current_device()
    for idx in (index1, index2):
        if idx is not None and idx != current_device:
            raise ValueError(f"Device index mismatch: {device1}, {device2}")


# -----------------------------------------------------------------------------
# DataLoader helpers
# -----------------------------------------------------------------------------


def make_dataloader_seed_fns(
    base_seed: int,
) -> Tuple[Callable[[int], None], torch.Generator]:
    """Return `(worker_init_fn, generator)` for DataLoader reproducibility.

    Example
    -------
    >>> w_init, g = make_dataloader_seed_fns(process_seed)
    >>> DataLoader(dataset,
    ...            num_workers=4,
    ...            worker_init_fn=w_init,
    ...            generator=g)

    Every worker receives its own deterministic seed ``base_seed + worker_id``.
    The returned ``torch.Generator`` is seeded with ``base_seed`` so that
    shuffling the order of the dataset is deterministic across runs.
    """

    def _worker_init_fn(worker_id: int) -> None:  # pragma: no cover
        # Each worker gets a distinct seed in the same pattern used for ranks.
        set_seed(base_seed + worker_id, performance_mode=False)

    gen = torch.Generator()
    gen.manual_seed(base_seed)

    return _worker_init_fn, gen


class Timer:
    r"""
    Helper class for timing code execution blocks and accumulating elapsed time in a dictionary.

    This class is designed to be used as a context manager to measure the execution time of code blocks.
    Upon entering the context, it records the start time, and upon exiting, it adds the elapsed time to a
    specified key in a provided timing dictionary. This is useful for profiling and tracking the time spent
    in different parts of a program, such as during training loops or data processing steps.

        timing_dict (dict): A dictionary where timing results will be accumulated.
        key (str): The key in the timing_dict under which to accumulate elapsed time.

    Example:
        for name in ["step1", "step2"]:
            timing[name] = 0

        with Timer(timing, "step1"):
            # Code block to time
            do_something()

        print(f"Elapsed time for step1: {timing['step1']} seconds")
    """

    def __init__(self, timing_dict, key, enabled=True):
        self.timing_dict = timing_dict
        self.key = key
        self.enabled = enabled
        self.elapsed = None

    def __enter__(self):
        if self.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.elapsed = time.perf_counter() - self.start
            if self.key not in self.timing_dict:
                self.timing_dict[self.key] = []
            self.timing_dict[self.key].append(self.elapsed)
        else:
            self.elapsed = 0.0
