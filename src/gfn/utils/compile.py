from __future__ import annotations

import logging
from typing import Iterable

import torch

logger = logging.getLogger(__name__)


def try_compile_gflownet(
    gfn,
    *,
    mode: str = "default",
    components: Iterable[str] = ("pf", "pb", "logZ", "logF"),
) -> None:
    """Best-effort compilation of estimator modules attached to a GFlowNet.
    Args:
        gfn: The GFlowNet instance to compile.
        mode: Compilation mode forwarded to ``torch.compile``.
        components: Attribute names to attempt compilation on (e.g., ``pf``).
    Returns:
        Mapping from component name to compilation success status.
    """
    results: dict[str, bool] = {}

    if not hasattr(torch, "compile"):
        results = {name: False for name in components}

    else:
        for name in components:

            # If the estimator does not exist, we cannot compile it.
            if not hasattr(gfn, name):
                results[name] = False
                continue

            estimator = getattr(gfn, name)
            module = getattr(estimator, "module", None)

            # If the estimator does not have a module, we cannot compile it.
            if module is None:
                results[name] = False
                continue

        try:
            # Attempt to compile the module.
            assert isinstance(estimator.module, torch.nn.Module)
            estimator.module = torch.compile(module, mode=mode)
            results[name] = True
        except Exception:
            results[name] = False

    # Print the results.
    formatted = ", ".join(
        f"{name}:{'✓' if success else 'x'}" for name, success in results.items()
    )
    logger.info(f"[compile] {formatted}")
