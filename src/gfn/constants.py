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
