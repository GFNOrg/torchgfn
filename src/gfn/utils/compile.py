from __future__ import annotations

from typing import Iterable

import torch


def try_compile_gflownet(
    gfn,
    *,
    mode: str = "default",
    components: Iterable[str] = ("pf", "pb", "logZ", "logF"),
) -> dict[str, bool]:
    """Best-effort compilation of estimator modules attached to a GFlowNet.
    Args:
        gfn: The GFlowNet instance to compile.
        mode: Compilation mode forwarded to ``torch.compile``.
        components: Attribute names to attempt compilation on (e.g., ``pf``).
    Returns:
        Mapping from component name to compilation success status.
    """

    if not hasattr(torch, "compile"):
        return {name: False for name in components}

    results: dict[str, bool] = {}
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

    return results
