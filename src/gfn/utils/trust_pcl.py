r"""Trust-PCL ↔ RTB parameter conversion utilities.

Deleu et al. (2025, arXiv:2509.01632) proved that Relative Trajectory
Balance (RTB) is mathematically equivalent to Trust-PCL, an off-policy
reinforcement learning method with KL regularization toward a reference
policy (Nachum et al., NeurIPS 2017).

**The core identity** (Proposition 3.1):

.. math::

    \mathcal{L}_{\text{Trust-PCL}}(\phi, \psi)
        = \alpha^2 \,\mathcal{L}_{\text{RTB}}(\phi, \psi)

where :math:`\alpha = 1/\beta` is the Trust-PCL temperature.

**What this means:**  Training a GFlowNet with RTB is *exactly* the same
optimization problem as training a policy with Trust-PCL.  The same
parameters are updated, the same gradients flow, and the same fixed point
is reached.  Only the loss scale differs by the constant :math:`\alpha^2`.

**Parameter correspondence:**

+-----------------------+------------------------------------+-------------------------------------+
| Concept               | RTB (GFlowNet)                     | Trust-PCL (RL)                      |
+=======================+====================================+=====================================+
| Temperature           | :math:`\beta`                      | :math:`\alpha = 1/\beta`            |
+-----------------------+------------------------------------+-------------------------------------+
| Learned scalar        | :math:`\log Z_\psi`               | :math:`V^{\text{soft}}_\psi(s_0)    |
|                       |                                    | = \alpha \cdot \log Z_\psi`          |
+-----------------------+------------------------------------+-------------------------------------+
| Trainable model       | Posterior :math:`p_\phi`           | Policy :math:`\pi_\phi`             |
+-----------------------+------------------------------------+-------------------------------------+
| Fixed reference       | Prior :math:`p_\theta`             | Reference :math:`\pi_{\text{ref}}`  |
+-----------------------+------------------------------------+-------------------------------------+
| Target distribution   | :math:`p(x) \propto p_\theta(x)    | :math:`\pi^*(a|s) \propto           |
|                       | \cdot r(x)^\beta`                  | \pi_{\text{ref}}(a|s)               |
|                       |                                    | \exp(Q^{\text{soft}}/\alpha)`        |
+-----------------------+------------------------------------+-------------------------------------+

**Derivation sketch:**

The RTB balance condition for a trajectory :math:`\tau` is:

.. math::

    \log Z_\psi + \log p_\phi(\tau) = \beta \log r(x_T) + \log p_\theta(\tau)

Multiplying both sides by :math:`\alpha = 1/\beta`:

.. math::

    \alpha \log Z_\psi + \alpha \log p_\phi(\tau)
        = \log r(x_T) + \alpha \log p_\theta(\tau)

Rearranging with :math:`V^{\text{soft}}_\psi(s_0) = \alpha \log Z_\psi`:

.. math::

    -V^{\text{soft}}_\psi(s_0)
    + \sum_t r_t
    + \alpha \sum_t \log \frac{\pi_{\text{ref}}(a_t|s_t)}{\pi_\phi(a_t|s_t)}
    = 0

This is exactly the Trust-PCL consistency condition (Nachum et al. 2017,
Equation 3).  The KL regularization term
:math:`\alpha \sum_t \log(\pi_{\text{ref}} / \pi_\phi)` emerges naturally
from the ratio of prior to posterior trajectory log-probabilities in the
original RTB equation — no separate KL penalty is added; it is an intrinsic
consequence of the balance condition.

References:
    Deleu et al. "Relative Trajectory Balance is equivalent to Trust-PCL"
    (2025, arXiv:2509.01632).

    Nachum et al. "Trust-PCL: An Off-Policy Trust Region Method for
    Continuous Control" (NeurIPS 2017, arXiv:1707.01891).

    Venkatraman et al. "Amortizing intractable inference in diffusion
    models for vision, language, and control" (NeurIPS 2024,
    arXiv:2405.20971).
"""

from __future__ import annotations

import torch


def rtb_to_trust_pcl_params(
    logZ: torch.Tensor | float,
    beta: torch.Tensor | float,
) -> dict[str, torch.Tensor | float]:
    r"""Convert RTB parameters to Trust-PCL parameters.

    Args:
        logZ: RTB log-partition function :math:`\log Z_\psi`.
        beta: RTB reward scaling :math:`\beta`.

    Returns:
        Dictionary with keys ``"alpha"`` and ``"v_soft_s0"``:

        - ``alpha = 1 / beta`` — Trust-PCL temperature
        - ``v_soft_s0 = alpha * logZ`` — soft value function at :math:`s_0`

    Example::

        >>> rtb_to_trust_pcl_params(logZ=2.0, beta=0.5)
        {'alpha': 2.0, 'v_soft_s0': 4.0}
    """
    alpha = 1.0 / beta
    v_soft_s0 = alpha * logZ
    return {"alpha": alpha, "v_soft_s0": v_soft_s0}


def trust_pcl_to_rtb_params(
    alpha: torch.Tensor | float,
    v_soft_s0: torch.Tensor | float,
) -> dict[str, torch.Tensor | float]:
    r"""Convert Trust-PCL parameters to RTB parameters.

    Args:
        alpha: Trust-PCL temperature :math:`\alpha`.
        v_soft_s0: Soft value function at :math:`s_0`,
            i.e. :math:`V^{\text{soft}}_\psi(s_0)`.

    Returns:
        Dictionary with keys ``"beta"`` and ``"logZ"``:

        - ``beta = 1 / alpha`` — RTB reward scaling
        - ``logZ = v_soft_s0 / alpha`` — RTB log-partition function

    Example::

        >>> trust_pcl_to_rtb_params(alpha=2.0, v_soft_s0=4.0)
        {'beta': 0.5, 'logZ': 2.0}
    """
    beta = 1.0 / alpha
    logZ = v_soft_s0 * beta  # v / alpha = v * beta
    return {"beta": beta, "logZ": logZ}
