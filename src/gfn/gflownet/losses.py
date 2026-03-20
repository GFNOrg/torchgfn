"""Pluggable regression losses for GFlowNet balance conditions.

All GFlowNet training objectives (TB, DB, SubTB, FM, RTB, etc.) minimize a
balance condition residual.  The standard approach squares this residual,
which corresponds to minimizing the reverse KL divergence between the
learned and target distributions.

This module provides alternative loss functions that correspond to different
divergence measures, following Hu et al. "Beyond Squared Error: Exploring
Loss Design for Enhanced Training of Generative Flow Networks"
(ICLR 2025, arXiv:2410.02596).

Each loss ``g(t)`` is applied elementwise to the residual ``t`` and satisfies:
  - ``g(0) = 0`` (zero loss at balance)
  - ``g(t) >= 0`` for all ``t`` (non-negative)
  - ``g'(0) = 0`` (stationary point at balance)

Hu et al. Theorem 4.1 shows that each regression loss ``g`` induces an
f-divergence between the learned flow and the target, where the f-divergence
generator is ``f(u) = u * integral_1^u [g'(log s) / s^2] ds``.

**Zero-forcing** losses (like squared error) penalize the learner for
placing mass where the target has none — they tend to *undercover* modes.
**Zero-avoiding** losses penalize the learner for missing mass where the
target has some — they tend to *overcover* and explore more modes.

Usage::

    from gfn.gflownet import TBGFlowNet
    from gfn.gflownet.losses import ShiftedCoshLoss

    gfn = TBGFlowNet(pf=pf, pb=pb, loss_fn=ShiftedCoshLoss())
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class RegressionLoss(ABC):
    """Abstract base for regression losses on GFlowNet balance residuals.

    Subclasses implement ``__call__`` mapping a residual tensor to a
    non-negative loss tensor of the same shape.
    """

    @abstractmethod
    def __call__(self, residuals: torch.Tensor) -> torch.Tensor:
        """Apply the loss elementwise.

        Args:
            residuals: Balance condition residuals (any shape).

        Returns:
            Non-negative tensor of the same shape.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


class SquaredLoss(RegressionLoss):
    r"""Standard squared loss: :math:`g(t) = t^2`.

    Corresponds to the **reverse KL** divergence (Malkin et al. 2022).
    This is **zero-forcing** (mode-seeking): it penalizes the learner for
    placing probability mass where the target has none, but does not penalize
    missing modes.  This can lead to mode collapse in multi-modal targets.

    This is the default loss for TB, DB, SubTB, LPV, and FM classes,
    reproducing the standard behavior from the literature.
    """

    def __call__(self, residuals: torch.Tensor) -> torch.Tensor:
        return residuals.pow(2)


class HalfSquaredLoss(RegressionLoss):
    r"""Half squared loss: :math:`g(t) = \tfrac{1}{2} t^2`.

    The :math:`\tfrac{1}{2}` factor ensures the gradient equals the residual
    itself: :math:`g'(t) = t` rather than :math:`2t`.  This is the standard
    least-squares convention (minimizing :math:`\tfrac{1}{2}\|r\|^2` so the
    normal equations have no factor of 2), and matches the RTB formulation
    in Venkatraman et al. (2024).

    This is the default loss for :class:`RelativeTrajectoryBalanceGFlowNet`
    and :class:`RelativeLogPartitionVarianceGFlowNet`.
    """

    def __call__(self, residuals: torch.Tensor) -> torch.Tensor:
        return 0.5 * residuals.pow(2)


class ShiftedCoshLoss(RegressionLoss):
    r"""Shifted hyperbolic cosine: :math:`g(t) = e^t + e^{-t} - 2 = 2(\cosh(t) - 1)`.

    This is the **only** loss in the family that is simultaneously
    **zero-forcing** (penalizes spurious mass) and **zero-avoiding**
    (penalizes missing modes).  It is symmetric: ``g(t) = g(-t)``.

    Near ``t = 0`` it behaves like ``t^2`` (same curvature as squared loss),
    but for large ``|t|`` it grows exponentially, providing stronger gradients
    for poorly-matched trajectories.

    Hu et al. (ICLR 2025) found this loss generally outperforms squared error
    on convergence speed and mode coverage across HyperGrid, bit-sequence,
    and sEH molecule benchmarks.

    References:
        Hu et al. "Beyond Squared Error: Exploring Loss Design for Enhanced
        Training of Generative Flow Networks" (ICLR 2025, arXiv:2410.02596).
    """

    def __call__(self, residuals: torch.Tensor) -> torch.Tensor:
        # Clamp to prevent overflow in exp() for extreme residuals.
        # exp(80) ≈ 5.5e34 which is within float32 range (~3.4e38)
        # exp(89) would overflow to inf.
        r = residuals.clamp(-80.0, 80.0)
        return torch.exp(r) + torch.exp(-r) - 2.0


class LinexLoss(RegressionLoss):
    r"""Linear-exponential (Linex) loss: :math:`g(t) = \frac{1}{\alpha^2}(e^{\alpha t} - \alpha t - 1)`.

    The ``alpha`` parameter controls the asymmetry:

    - ``alpha = 1``: corresponds to the **forward KL** divergence.
      **Zero-avoiding** (mass-covering / exploration-favoring): penalizes
      the learner for missing mass where the target has support, encouraging
      broader mode coverage at the cost of some spurious mass.

    - ``alpha = 0.5``: corresponds to the **alpha-divergence** with
      ``alpha = 0.5``. **Balanced**: neither purely zero-forcing nor
      zero-avoiding.

    - ``alpha < 0``: becomes zero-forcing (mode-seeking), similar to but
      distinct from squared loss.

    The :math:`1/\alpha^2` normalization ensures ``g''(0) = 1`` for all
    ``alpha``, matching the curvature of squared loss near zero.

    References:
        Hu et al. "Beyond Squared Error: Exploring Loss Design for Enhanced
        Training of Generative Flow Networks" (ICLR 2025, arXiv:2410.02596).

        The Linex loss originates from Bayesian decision theory:
        Varian (1975), Zellner (1986).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha == 0.0:
            raise ValueError("alpha must be nonzero (alpha=0 degenerates to 0.5*t^2)")
        self.alpha = alpha

    def __call__(self, residuals: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        # Clamp the exponent to prevent overflow.
        # exp(80) ≈ 5.5e34 which is within float32 range (~3.4e38)
        # exp(89) would overflow to inf.
        exp_term = torch.exp((a * residuals).clamp(-80.0, 80.0))
        return (1.0 / (a * a)) * (exp_term - a * residuals - 1.0)

    def __repr__(self) -> str:
        return f"LinexLoss(alpha={self.alpha})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.alpha == other.alpha  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self), self.alpha))
