"""Shared constants for the gfn package."""

# Relative tolerance for detecting terminal time in diffusion processes.
#
# Applied as: eps = dt * DIFFUSION_TERMINAL_TIME_EPS, where dt is the
# discretization step size. Used for:
#   - Initial state detection: t < eps
#   - Terminal state detection: t >= 1.0 - eps
#   - Exit action trigger:     t + dt >= 1.0 - eps  (next step reaches terminal)
#
# This constant must be consistent across estimators, environments, and loss
# functions. Changing it here updates all three automatically.
DIFFUSION_TERMINAL_TIME_EPS: float = 1e-2


# Numerical tolerances used by quick mode-existence checks.
#
# - EPS_REWARD_CMP: tolerance for comparing scalar rewards to thresholds. It
#   guards against small floating-point rounding errors when checking
#   inequalities like r >= thr.
# - EPS_INDEX_CMP: tolerance for floating-point-to-index boundary calculations,
#   used when turning fractional bands into integer indices.
EPS_REWARD_CMP = 1e-12
EPS_INDEX_CMP = 1e-9
