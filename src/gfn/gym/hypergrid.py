"""Adapted from https://github.com/Tikquuss/GflowNets_Tutorial"""

import itertools
import logging
import multiprocessing
import platform
import warnings
from abc import ABC, abstractmethod
from decimal import Decimal
from functools import reduce
from math import gcd, log, pi, sqrt
from time import time
from typing import List, Literal, Tuple

import torch

from gfn.actions import Actions
from gfn.constants import EPS_INDEX_CMP, EPS_REWARD_CMP
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates

logger = logging.getLogger(__name__)

# Only set the multiprocessing start method when not already configured
# (e.g. by Jupyter or another framework). force=True can crash notebook kernels.
try:
    if platform.system() == "Windows":
        multiprocessing.set_start_method("spawn")
    else:
        multiprocessing.set_start_method("fork")
except RuntimeError:
    pass  # Already set — don't override.


def lcm(a, b):
    """Returns the lowest common multiple between a and b."""
    return a * b // gcd(a, b)


def lcm_multiple(numbers):
    """Find the lowest common multiple across a list of numbers"""
    return reduce(lcm, numbers)


def smallest_multiplier_to_integers(float_vector, precision=3):
    """Used to calculate a scale factor to avoid imprecise floating point arithmetic."""
    denominators = []

    for num in float_vector:
        dec = Decimal(str(num))  # Convert to Decimal for precise arithmetic.
        fraction = dec.as_integer_ratio()
        denominators.append(fraction[1])

    smallest_multiplier = lcm_multiple(denominators)

    return smallest_multiplier


class HyperGrid(DiscreteEnv):
    """HyperGrid environment from the GFlowNets paper.

    The states are represented as 1-d tensors of length `ndim` with values in
    `{0, 1, ..., height - 1}`.

    Attributes:
        ndim: The dimension of the grid.
        height: The height of the grid.
        reward_fn: The reward function.
        calculate_partition: Whether to calculate the log partition function.
        store_all_states: Whether to store all states.
        validate_modes: Whether to check that at least one state reaches the
            mode threshold at init; raises if not.
        mode_stats: One of {"none", "approx", "exact"}. If not "none",
            computes (exact or approximate) `n_modes` and `n_mode_states`.
            "exact" requires `store_all_states=True` and enumerates all states.
        mode_stats_samples: Number of random samples when `mode_stats="approx"`.
    """

    def __init__(
        self,
        ndim: int = 2,
        # Smallest height that satisfies `validate_modes=True` for the original reward.
        height: int = 8,
        reward_fn_str: str = "original",
        reward_fn_kwargs: dict | None = None,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        calculate_partition: bool = False,
        store_all_states: bool = False,
        debug: bool = False,
        validate_modes: bool = True,
        mode_stats: Literal["none", "approx", "exact"] = "none",
        mode_stats_samples: int = 20000,
    ):
        """Initializes the HyperGrid environment.

        Args:
            ndim: The dimension of the grid.
            height: The height of the grid. The default value is the smallest height
                that satisfies `validate_modes=True` for the original reward.
            reward_fn_str: The reward function string to use.
            reward_fn_kwargs: The keyword arguments for the reward function.
            device: The device to use.
            calculate_partition: Whether to calculate the log partition function.
            store_all_states: Whether to store all states. If True, the true distribution
                can be accessed via the `true_dist` property.
            debug: If True, emit States with debug guards (not compile-friendly).
            validate_modes: If True, verifies at initialization that at least one
                state achieves the reward-defined mode threshold; raises
                `ValueError` when no such state is found.
            mode_stats: Level of mode statistics to compute: "none" (disabled),
                "approx" (uniform sampling), or "exact" (full enumeration).
                "exact" requires `store_all_states=True` and may be expensive for
                large `height` or `ndim`.
            mode_stats_samples: Number of random samples used when
                `mode_stats="approx"`.
        """
        if height <= 4:
            logger.warning("+ Warning: height <= 4 can lead to unsolvable environments.")

        reward_functions = {
            "original": OriginalReward,
            "cosine": CosineReward,
            "sparse": SparseReward,
            "deceptive": DeceptiveReward,
            # New compositional environments (see classes below)
            "bitwise_xor": BitwiseXORReward,
            "multiplicative_coprime": MultiplicativeCoprimeReward,
            "conditional_multiscale": ConditionalMultiScaleReward,
        }

        self.ndim = ndim
        self.height = height

        # Default reward function kwargs.
        if reward_fn_kwargs is None:
            reward_fn_kwargs = {"R0": 0.1, "R1": 0.5, "R2": 2.0}
        self.reward_fn_kwargs = reward_fn_kwargs
        assert (
            reward_fn_str in reward_functions
        ), f"Invalid reward function string: {reward_fn_str} not in {reward_functions.keys()}"
        self.reward_fn = reward_functions[reward_fn_str](
            height, ndim, **reward_fn_kwargs
        )

        self._all_states_tensor = None  # Populated optionally in init.
        self._log_partition = None  # Populated optionally in init.
        self._true_dist = None  # Populated at first request.

        # If we store the all states, the partition function is calculated automatically.
        self.calculate_partition = calculate_partition or store_all_states
        self.store_all_states = store_all_states

        # Pre-computes these values when printing.
        if self.store_all_states or self.calculate_partition:
            self._enumerate_all_states_tensor()

        if self.store_all_states:
            assert self._all_states_tensor is not None
            logger.info(f"+ Environment has {len(self._all_states_tensor)} states")
        if self.calculate_partition:
            assert self._log_partition is not None
            logger.info(f"+ Environment log partition is {self._log_partition}")

        if isinstance(device, str):
            device = torch.device(device)

        s0 = torch.zeros(ndim, dtype=torch.long, device=device)
        sf = torch.full((ndim,), fill_value=-1, device=device)
        n_actions = ndim + 1

        state_shape = (self.ndim,)

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            state_shape=state_shape,
            sf=sf,
            debug=debug,
        )
        self.States: type[DiscreteStates] = self.States  # for type checking

        # Optionally validate that modes exist under the configured reward.
        if validate_modes:
            ok, msg = self._modes_exist_quick_check_info()
            if not ok:
                raise ValueError(msg)

        # Optional mode statistics (expensive when exact)
        self._n_mode_states_exact: int | None = None
        self._n_mode_states_estimate: float | None = None
        self._mode_stats_kind: str = "none"

        if mode_stats != "none":
            try:
                if mode_stats == "exact":
                    if not self.store_all_states:
                        raise ValueError(
                            "Exact mode_stats requires store_all_states=True to enumerate states."
                        )
                    if self._all_states_tensor is None:
                        raise ValueError(
                            "Failed to access all_states for exact mode_stats."
                        )
                    # Compute on CPU to avoid device-mismatch in reward fns.
                    cpu_tensor = self._all_states_tensor.cpu()
                    rewards = self.reward_fn(cpu_tensor)
                    threshold = self._mode_reward_threshold()
                    mask = rewards >= threshold
                    self._n_mode_states_exact = int(mask.sum().item())
                    self._mode_stats_kind = "exact"
                else:
                    # Approximate via uniform sampling.
                    with torch.no_grad():
                        B = int(max(1, mode_stats_samples))
                        xs = self.make_random_states((B,)).tensor
                        mask = self.mode_mask(self.States(xs))
                        frac = float(mask.float().mean().item())
                        self._n_mode_states_estimate = frac * float(self.n_states)
                        self._mode_stats_kind = "approx"
            except Exception:
                warnings.warn("+ Warning: Failed to compute mode_stats (skipping).")

    def make_states_class(self) -> type[DiscreteStates]:
        """Returns the DiscreteStates class for the HyperGrid environment."""
        env = self

        class HyperGridStates(DiscreteStates):
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states
            n_actions = env.n_actions

            def _compute_forward_masks(self) -> torch.Tensor:
                """Computes forward masks for HyperGrid states.

                Not allowed to take any action beyond the environment height,
                but allow early termination.
                """
                # Create mask: True where action would go beyond height
                at_height_limit = self.tensor == env.height - 1
                # Forward masks: all True except where at height limit
                forward_masks = torch.ones(
                    (*self.batch_shape, self.n_actions),
                    dtype=torch.bool,
                    device=self.device,
                )
                # Set non-exit actions to False where at height limit
                # Exit action (last action) remains True
                exit_mask = torch.zeros(
                    self.batch_shape + (1,), device=self.device, dtype=torch.bool
                )
                full_mask = torch.cat([at_height_limit, exit_mask], dim=-1)
                forward_masks[full_mask] = False
                return forward_masks

            def _compute_backward_masks(self) -> torch.Tensor:
                """Computes backward masks for HyperGrid states."""
                return self.tensor != 0

        return HyperGridStates

    def make_random_states(
        self,
        batch_shape: Tuple[int, ...],
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> DiscreteStates:
        """Creates a batch of random states.

        Args:
            batch_shape: The shape of the batch.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: The device to use.
            debug: If True, emit States with debug guards (not compile-friendly).

        Returns:
            A `DiscreteStates` object with random states.
        """
        device = self.device if device is None else device
        tensor = torch.randint(
            0, self.height, batch_shape + self.s0.shape, device=device
        )
        return self.States(tensor, conditions=conditions, debug=debug)

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Performs a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The next states.
        """
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")
        if self.debug:
            assert new_states_tensor.shape == states.tensor.shape
        return self.States(new_states_tensor)

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Performs a backward step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The previous states.
        """
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, -1, reduce="add")
        if self.debug:
            assert new_states_tensor.shape == states.tensor.shape
        return self.States(new_states_tensor)

    def reward(self, states: DiscreteStates) -> torch.Tensor:
        r"""Computes the reward for a batch of final states.

        In the normal setting, the reward is:
        `R(s) = R_0 + 0.5 \prod_{d=1}^D \mathbf{1} \left( \left\lvert \frac{s^d}{H-1}
          - 0.5 \right\rvert \in (0.25, 0.5] \right)
          + 2 \prod_{d=1}^D \mathbf{1} \left( \left\lvert \frac{s^d}{H-1} - 0.5 \right\rvert \in (0.3, 0.4) \right)`

        Args:
            final_states: The final states.

        Returns:
            The reward of the final states.
        """
        reward = self.reward_fn(states.tensor)
        if self.debug:
            assert (
                reward.shape == states.batch_shape
            ), f"reward.shape is {reward.shape} and states.batch_shape is {states.batch_shape}"
        return reward

    # -------------------------
    # Mode utilities
    # -------------------------
    def _mode_reward_threshold(self) -> float:
        """Returns the reward threshold used to define a mode.

        By default, a state is considered in a mode if its reward is at least
        the schema-defined threshold derived from the configured reward.
        """
        # We branch on the concrete reward to derive a principled threshold.

        # Original reward: ring band adds R2 on top of base R0 and outer ring R1.
        if isinstance(self.reward_fn, OriginalReward):
            for key in ("R0", "R1", "R2"):
                if key not in self.reward_fn_kwargs:
                    raise ValueError(
                        f"Missing '{key}' in reward_fn_kwargs for Original reward; "
                        "please provide R0, R1, and R2."
                    )
            r0 = float(self.reward_fn_kwargs["R0"])  # type: ignore[index]
            r1 = float(self.reward_fn_kwargs["R1"])  # type: ignore[index]
            r2 = float(self.reward_fn_kwargs["R2"])  # type: ignore[index]

            # Modes are the thin ring where both outer ring and band conditions hold.
            return r0 + r1 + r2

        # Deceptive reward: ring band adds R2 while the outer region cancels R1.
        if isinstance(self.reward_fn, DeceptiveReward):
            for key in ("R0", "R2"):
                if key not in self.reward_fn_kwargs:
                    raise ValueError(
                        f"Missing '{key}' in reward_fn_kwargs for Deceptive reward; "
                        "please provide R0 and R2."
                    )
            r0 = float(self.reward_fn_kwargs["R0"])  # type: ignore[index]
            r2 = float(self.reward_fn_kwargs["R2"])  # type: ignore[index]
            # Modes are the band where R1 is cancelled and R2 dominates.
            return r0 + r2

        # Cosine reward: peak at center with oscillatory local maxima along each axis.
        # Treat "modes" as states whose per-dimension factor is near its theoretical
        # maximum f_max = 2 / sqrt(2*pi). We allow a tunable closeness factor `mode_gamma`
        # (default 0.8). The product structure implies a threshold of (gamma*f_max)^ndim.
        # On coarse grids the discrete samples may not reach this threshold, which
        # correctly causes validate_modes to fail — the grid is too coarse for modes.
        if isinstance(self.reward_fn, CosineReward):
            r0 = float(self.reward_fn_kwargs.get("R0", 0.1))
            r1 = float(self.reward_fn_kwargs.get("R1", 0.5))
            gamma = float(self.reward_fn_kwargs.get("mode_gamma", 0.8))
            per_dim_peak = 2.0 / sqrt(2 * pi)  # ~0.79788456
            return r0 + (gamma * per_dim_peak) ** self.ndim * r1

        # Sparse reward: modes are exactly the target points (reward ~ 1+eps vs eps).
        # Any value strictly above 0.5 cleanly separates modes from non-modes.
        if isinstance(self.reward_fn, SparseReward):
            return 0.5

        # ConditionalMultiScale uses an adaptive tier based on dimensionality:
        # at low d, deeper tiers are needed for modes to be sparse.
        if isinstance(self.reward_fn, ConditionalMultiScaleReward):
            return self.reward_fn.mode_threshold()

        # For other compositional rewards with tiered structure, mark mode as
        # achieving the highest tier.
        if isinstance(
            self.reward_fn,
            (BitwiseXORReward, MultiplicativeCoprimeReward),
        ):
            r0 = float(getattr(self.reward_fn, "R0", 0.0))
            tw = getattr(self.reward_fn, "tier_weights", [])
            if not isinstance(tw, (list, tuple)) or len(tw) == 0:
                raise ValueError(
                    "Tiered reward missing `tier_weights`; cannot derive mode threshold."
                )
            return r0 + float(sum(tw))

        # Other reward schemas are not supported for mode counting via threshold.
        raise NotImplementedError(
            "Mode threshold is only defined for known reward schemas."
        )

    def mode_mask(self, states: DiscreteStates) -> torch.Tensor:
        """Boolean mask indicating which states are in a mode.

        A state is flagged as mode if its reward is greater-or-equal to
        the threshold based on `reward_fn_kwargs` (R0+R1+R2 by default).
        """
        rewards = self.reward(states)
        threshold = self._mode_reward_threshold()
        return rewards >= threshold

    def modes_found(self, states: DiscreteStates) -> set[int]:
        """Returns the set of canonical state indices for mode states in the batch.

        Each mode state is identified by its unique canonical index (from
        ``get_states_indices``), not by a quadrant-based grouping. This allows
        correct mode-state tracking for all reward functions.
        """
        mask = self.mode_mask(states)
        if not mask.any():
            return set()
        indices = self.get_states_indices(states)
        return set(indices[mask].tolist())

    @property
    def n_mode_states(self) -> int | float | None:
        """Number of states inside a mode (exact, approx, or None).

        - If mode_stats="exact", returns an exact integer count.
        - If mode_stats="approx", returns a floating-point estimate.
        - If store_all_states is True (but mode_stats was "none"), computes on
          demand from all_states.
        - Otherwise, returns None.
        """
        if self._mode_stats_kind == "exact" and self._n_mode_states_exact is not None:
            return int(self._n_mode_states_exact)
        if (
            self._mode_stats_kind == "approx"
            and self._n_mode_states_estimate is not None
        ):
            return float(self._n_mode_states_estimate)
        # On-demand computation when all states are available.
        if self.store_all_states and self.all_states is not None:
            mask = self.mode_mask(self.all_states)
            return int(mask.sum().item())
        return None

    @property
    def n_modes(self) -> int | float | None:
        """Returns the total number of mode states for this environment.

        Equivalent to ``n_mode_states``. Each individual grid cell whose reward
        meets the mode threshold counts as one mode.
        """
        return self.n_mode_states

    def get_states_indices(self, states: DiscreteStates | torch.Tensor) -> torch.Tensor:
        """Get the indices of the states in the canonical ordering.

        Args:
            states: The states to get the indices of.

        Returns:
            The indices of the states in the canonical ordering.
        """
        if isinstance(states, DiscreteStates):
            states_raw = states.tensor
        else:
            states_raw = states

        canonical_base = self.height ** torch.arange(
            self.ndim - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        if isinstance(states, DiscreteStates):
            assert (
                indices.shape == states.batch_shape
            ), f"indices.shape is {indices.shape} and states.batch_shape is {states.batch_shape}"
        else:
            n_dims = len(indices.shape)
            assert (
                indices.shape[:n_dims] == states.shape[:n_dims]
            ), f"indices.shape is {indices.shape} and states.shape is {states.shape}"
        return indices

    def get_terminating_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """Get the indices of the terminating states in the canonical ordering.

        Args:
            states: The states to get the indices of.

        Returns:
            The indices of the terminating states in the canonical ordering.
        """
        return self.get_states_indices(states)

    @property
    def n_states(self) -> int:
        """Returns the number of states in the environment."""
        return self.height**self.ndim

    @property
    def n_terminating_states(self) -> int:
        """Returns the number of terminating states in the environment."""
        return self.n_states

    # -------------------------
    # Mode existence validation
    # -------------------------
    def _modes_exist_quick_check(self) -> bool:
        """Lightweight check that a mode-level state exists.

        In simple terms, this answers: "Is there at least one state whose reward
        reaches the mode threshold?" without enumerating all states. It proceeds
        in three stages:
        1) If the grid is small (or pre-enumerated), it computes rewards exactly
           and checks against the threshold.
        2) Otherwise, it dispatches to reward-specific constructive tests that
           are sufficient to guarantee at least one state reaches the threshold.
        3) As a last resort, it samples a small batch of random states.
        """
        thr = self._mode_reward_threshold()

        # If the grid is small enough, prefer an exact check to avoid fragile heuristics.
        # Also prefer exact when all states are already stored.
        # All work is done on CPU to avoid device-mismatch issues (e.g. MPS).
        try:
            if self.store_all_states and self.all_states is not None:
                cpu_states = self.all_states.tensor.cpu()
                rewards = self.reward_fn(cpu_states)
                return bool((rewards >= thr - EPS_REWARD_CMP).any().item())
            if self.n_states <= 200_000:
                axes = [
                    torch.arange(self.height, dtype=torch.long) for _ in range(self.ndim)
                ]
                grid = torch.cartesian_prod(*axes)
                rewards = self.reward_fn(grid)
                return bool((rewards >= thr - EPS_REWARD_CMP).any().item())
        except Exception:
            # Fall back to heuristic paths below
            pass
        if isinstance(self.reward_fn, (OriginalReward, DeceptiveReward)):
            return self._exists_original_or_deceptive(thr)
        if isinstance(self.reward_fn, CosineReward):
            return self._exists_cosine(thr)
        if isinstance(self.reward_fn, SparseReward):
            return self._exists_sparse(thr)
        if isinstance(self.reward_fn, BitwiseXORReward):
            return self._exists_bitwise_xor(thr)
        if isinstance(self.reward_fn, MultiplicativeCoprimeReward):
            return self._exists_multiplicative_coprime(thr)
        return self._exists_fallback_random(thr)

    def _modes_exist_quick_check_info(self) -> tuple[bool, str]:
        """Same as _modes_exist_quick_check but returns (ok, message)."""
        try:
            ok = self._modes_exist_quick_check()
            if ok:
                return True, "OK"
        except (ValueError, ArithmeticError):
            # Genuine validation failures — fall through to the message below.
            pass
        except Exception as e:
            # Unexpected errors (device mismatch, OOM, etc.) — re-raise so
            # they are not silently converted into a misleading "no modes" message.
            raise RuntimeError(
                f"validate_modes failed with an unexpected error: {e}"
            ) from e

        return (
            False,
            "No states satisfy the mode threshold for the current reward and parameters.",
        )

    def _exists_original_or_deceptive(self, thr: float) -> bool:
        """Constructive check for ``OriginalReward`` and ``DeceptiveReward``.

        Intuition:
        - These rewards form rings/bands around the center when each coordinate
          is normalized to [0,1]. The mode lies on a thin band at specific
          normalized distances from the center.
        - We translate those fractional band boundaries into integer indices via
          small inside/outside nudges (using ``EPS_INDEX_CMP``) and test one
          candidate index from any non-empty feasible interval.
        - If the reward at that candidate exceeds the threshold (with
          ``EPS_REWARD_CMP`` tolerance), we return True.
        """
        Hm1 = self.height - 1
        if Hm1 <= 0:
            return False

        # The band condition |x/(H-1) - 0.5| in (0.3, 0.4) maps to two
        # symmetric raw-coordinate intervals in normalized [0,1]:
        #   lower band: [0.1, 0.2]   (center-left)
        #   upper band: [0.8, 0.9]   (center-right)
        # Convert each to integer index ranges. The +1 after int() is a ceiling
        # to find the first index strictly inside the open boundary.
        lower_band_lo = int((0.1 + EPS_INDEX_CMP) * Hm1) + 1
        lower_band_hi = int((0.2 - EPS_INDEX_CMP) * Hm1)
        upper_band_lo = int((0.8 + EPS_INDEX_CMP) * Hm1) + 1
        upper_band_hi = int((0.9 - EPS_INDEX_CMP) * Hm1)

        # Pick the first feasible index from either band.
        candidate_idx = None
        for lo, hi in [(lower_band_lo, lower_band_hi), (upper_band_lo, upper_band_hi)]:
            if lo <= hi:
                candidate_idx = lo
                break
        if candidate_idx is None:
            return False

        x = torch.full((self.ndim,), candidate_idx, dtype=torch.long)
        r = float(self.reward_fn(x.unsqueeze(0))[0])
        return r >= thr - EPS_REWARD_CMP

    def _exists_cosine(self, thr: float) -> bool:
        """Analytic upper-bound check for ``CosineReward``.

        Idea:
        - The per-dimension factor is ``(cos(50·ax) + 1) · N(0,1)(5·ax)`` with
          ax in [0,0.5]. We estimate its maximum over the discrete grid by
          evaluating all candidate ax and taking the maximum value ``m``.
        - The full reward upper bound is ``R0 + m^D * R1``. If this is at least
          the mode target and the given threshold, a mode-level state must exist.
        - We also compute a theoretical per-dimension peak (at ax≈0) to form a
          slightly conservative target scaled by ``mode_gamma``.
        """
        R0 = float(self.reward_fn.kwargs.get("R0", 0.1))
        R1 = float(self.reward_fn.kwargs.get("R1", 0.5))
        gamma = float(self.reward_fn.kwargs.get("mode_gamma", 0.8))
        Hm1 = max(1, self.height - 1)

        # Evaluate the per-dimension factor at every discrete grid index.
        idx = torch.arange(0, self.height, dtype=torch.get_default_dtype())
        ax = (idx / Hm1 - 0.5).abs()
        pdf = (1.0 / sqrt(2 * pi)) * torch.exp(-0.5 * (5 * ax) ** 2)
        per_dim_values = (torch.cos(50 * ax) + 1.0) * pdf
        max_per_dim_factor = float(per_dim_values.max())

        # The theoretical continuous peak is 2/sqrt(2*pi) (~0.798).
        # The gamma-scaled target requires the grid to resolve oscillations
        # well enough that max_per_dim_factor^D exceeds (gamma * peak)^D.
        theoretical_peak = 2.0 / sqrt(2 * pi)
        target = R0 + (gamma * theoretical_peak) ** self.ndim * R1
        rmax = R0 + (max_per_dim_factor**self.ndim) * R1

        # Must exceed both the gamma-scaled target (grid resolves peaks)
        # and the caller's threshold (mode definition is satisfiable).
        return rmax >= target - EPS_REWARD_CMP and rmax >= thr - EPS_REWARD_CMP

    def _exists_sparse(self, thr: float) -> bool:
        """Constructive check for ``SparseReward``.

        This reward assigns positive mass only to a finite set of target
        configurations. When ``H>=2`` and ``D>=1``, a known target is the zero
        vector except for certain coordinates fixed at 1 or ``H-2``. We probe a
        canonical target and confirm the threshold is not above its reward.
        """
        probe = torch.zeros(self.ndim, dtype=torch.long)
        r = float(self.reward_fn(probe.unsqueeze(0))[0])
        # SparseReward gives ~1+eps to exact targets and ~eps to non-targets.
        # The zero vector may or may not be a target. We check: either it
        # already exceeds the threshold, or the maximum possible reward
        # (r + 1.0, the target spike) would exceed it.
        return (self.height >= 2 and self.ndim >= 1) and (r >= thr or r + 1.0 >= thr)

    def _exists_bitwise_xor(self, thr: float) -> bool:
        """Feasibility and constructive check for ``BitwiseXORReward``.

        Steps:
        - For each tier, verify the GF(2) parity system has at least one
          solution using Gaussian elimination modulo 2. If any tier is
          infeasible, no mode exists.
        - The all-zero configuration satisfies even-parity constraints, so if
          tiers are feasible we evaluate that point against the threshold with
          tolerance.
        """
        if self.reward_fn.parity_checks is not None:
            for t in range(len(self.reward_fn.tier_weights)):
                cfg = self.reward_fn.parity_checks[t]
                if cfg is None:
                    continue
                A = cfg.get("A", None)
                c = cfg.get("c", None)
                if A is None or c is None:
                    continue
                if not self._solve_gf2_has_solution(A, c):
                    return False

        x = torch.zeros(self.ndim, dtype=torch.long)
        r = float(self.reward_fn(x.unsqueeze(0))[0])
        return r >= thr - EPS_REWARD_CMP

    def _exists_multiplicative_coprime(self, thr: float) -> bool:
        """Number-theoretic constructive check for ``MultiplicativeCoprimeReward``.

        Constructs a candidate state by factoring the target LCM (if any) over
        the allowed primes, assigning each prime power to a separate active
        dimension, and verifying coprimality and grid-bound constraints.
        """
        primes: list[int] = [int(p) for p in self.reward_fn.primes]
        caps: list[int] = [int(c) for c in self.reward_fn.exponent_caps]
        active = list(self.reward_fn.active_dims)
        coprime_pairs = self.reward_fn.coprime_pairs or []
        max_exponent = int(caps[-1])
        target_lcms = self.reward_fn.target_lcms
        target = None if target_lcms is None else target_lcms[-1]

        # Start with all-ones (valid for prime-support: 1 has zero exponents).
        x = torch.ones(self.ndim, dtype=torch.long)

        if target is not None:
            target = int(target)

            # Factor target LCM over the allowed primes.
            required_prime_powers: list[tuple[int, int]] = []
            unfactored_remainder = target
            for p in primes:
                exp = 0
                while unfactored_remainder % p == 0:
                    unfactored_remainder //= p
                    exp += 1
                if exp > 0:
                    if exp > max_exponent or (p**exp) > (self.height - 1):
                        return False
                    required_prime_powers.append((p, exp))

            # All prime factors must come from the allowed set.
            if unfactored_remainder != 1:
                return False
            # Need at most one dimension per distinct prime power.
            if len(required_prime_powers) > len(active):
                return False

            # Assign each prime power to a separate active dimension.
            for (p, exp), dim in zip(required_prime_powers, active):
                x[dim] = p**exp

            # Verify that dimension pairs designated as coprime have gcd == 1.
            for i, j in coprime_pairs:
                if torch.gcd(x[active[i]], x[active[j]]).item() != 1:
                    return False

        if int(x.max()) >= self.height:
            return False
        r = float(self.reward_fn(x.unsqueeze(0))[0])
        return r >= thr - EPS_REWARD_CMP

    def _exists_fallback_random(self, thr: float) -> bool:
        """Random sampling fallback.

        Draw a modest batch of random states on CPU and accept if any exceed the
        threshold with a small tolerance. This is a last resort to avoid
        expensive enumeration for large grids.
        """
        with torch.no_grad():
            device = torch.device("cpu")
            B = min(2048, max(128, 8 * self.ndim))
            xs = torch.randint(0, self.height, (B, self.ndim), device=device)
            rr = self.reward_fn(xs)
            return bool((rr >= thr - EPS_REWARD_CMP).any().item())

    @staticmethod
    def _solve_gf2_has_solution(A: torch.Tensor, c: torch.Tensor) -> bool:
        """Return True if A x = c over GF(2) has at least one solution.

        Performs Gaussian elimination modulo 2 (XOR arithmetic) without
        constructing a specific solution. A row that reduces to all-zero
        coefficients with a non-zero RHS (``0 = 1``) indicates inconsistency.
        """
        if A.numel() == 0:
            return True

        # Reduce to GF(2): keep only the least-significant bit.
        A = A.clone().detach().to(dtype=torch.uint8, device=torch.device("cpu")) & 1
        c = c.clone().detach().to(dtype=torch.uint8, device=torch.device("cpu")) & 1

        n_equations, n_variables = A.shape
        pivot_row = 0
        for col in range(n_variables):
            # Find a row at or below pivot_row with a 1 in this column.
            pivot_candidate = None
            for row_idx in range(pivot_row, n_equations):
                if A[row_idx, col]:
                    pivot_candidate = row_idx
                    break
            if pivot_candidate is None:
                continue  # No pivot in this column; move to the next.

            # Swap the pivot candidate into the pivot position.
            if pivot_candidate != pivot_row:
                A[[pivot_row, pivot_candidate]] = A[[pivot_candidate, pivot_row]]
                c[[pivot_row, pivot_candidate]] = c[[pivot_candidate, pivot_row]]

            # Eliminate all rows below the pivot via XOR (addition in GF(2)).
            for row_idx in range(pivot_row + 1, n_equations):
                if A[row_idx, col]:
                    A[row_idx, :] ^= A[pivot_row, :]
                    c[row_idx] ^= c[pivot_row]

            pivot_row += 1
            if pivot_row == n_equations:
                break

        # A row with all-zero coefficients but c=1 means 0=1 (inconsistent).
        for row_idx in range(n_equations):
            if not A[row_idx, :].any() and c[row_idx]:
                return False
        return True

    def _enumerate_all_states_tensor(self, batch_size: int = 20_000):
        """Enumerate all grid states, optionally storing them and computing log Z.

        Iterates over the full Cartesian product ``{0, ..., H-1}^D`` in batches
        (via multiprocessing) to avoid materializing all ``H^D`` states at once.

        Args:
            batch_size: Number of states per batch.
        """
        need_to_enumerate = (
            self.store_all_states and self._all_states_tensor is None
        ) or (self.calculate_partition and self._log_partition is None)

        if need_to_enumerate:
            start_time = time()
            all_states_tensor = []
            reward_sum = 0.0
            n_expected = self.height**self.ndim
            n_found = 0

            # max_val = height - 1 because coordinates are 0-indexed.
            for batch_tuples in self._generate_combinations_in_batches(
                self.ndim, self.height - 1, batch_size
            ):
                batch_tensor = torch.LongTensor(list(batch_tuples))
                n_found += batch_tensor.shape[0]
                if self.store_all_states:
                    all_states_tensor.append(batch_tensor)
                if self.calculate_partition:
                    reward_sum += self.reward_fn(batch_tensor).sum().item()

            assert (
                n_expected == n_found
            ), f"Enumeration visited {n_found} states, expected {n_expected}"

            end_time = time()
            logger.info(
                "Enumerated all states in {} minutes".format(
                    (end_time - start_time) / 60.0,
                )
            )

            if self.store_all_states:
                self._all_states_tensor = torch.cat(all_states_tensor, dim=0)

            if self.calculate_partition:
                self._log_partition = log(reward_sum)

    def true_dist(self, condition=None) -> torch.Tensor | None:  # condition is ignored
        """Returns the pmf over all states in the hypergrid."""
        if self._true_dist is None:
            assert (
                self.all_states is not None
            ), "true_dist is not available without all_states"
            all_rewards = self.reward(self.all_states)
            self._true_dist = all_rewards / all_rewards.sum()

        return self._true_dist

    def all_indices(self) -> List[Tuple[int, ...]]:
        """Generate all possible indices for the grid.

        Returns:
            A list of all possible indices for the grid.
        """

        def _all_indices(dim: int, height: int) -> List[Tuple[int, ...]]:
            if dim == 1:
                return [(i,) for i in range(height)]
            return [
                (i, *j) for i in range(height) for j in _all_indices(dim - 1, height)
            ]

        return _all_indices(self.ndim, self.height)

    def log_partition(self, condition=None) -> float | None:  # condition is ignored
        """Returns the log partition of the reward function."""
        return self._log_partition

    @property
    def all_states(self) -> DiscreteStates | None:
        """Returns a tensor of all hypergrid states as a `DiscreteStates` instance."""
        if not self.store_all_states:
            return None

        if self._all_states_tensor is None:
            self._enumerate_all_states_tensor()

        assert self._all_states_tensor is not None
        self._all_states_tensor = self._all_states_tensor.to(self.device)
        assert torch.all(
            self.get_states_indices(self._all_states_tensor)
            == torch.arange(self.n_states, device=self.device)
        )
        return self.States(self._all_states_tensor)

    @property
    def terminating_states(self) -> DiscreteStates | None:
        """Returns all terminating states of the environment."""
        return self.all_states

    def _worker(self, task: tuple) -> itertools.islice:
        """Return a slice of the Cartesian product for one batch.

        Args:
            task: (values, ndim, start_idx, end_idx) where values is the list
                  of coordinate values, ndim is the number of dimensions, and
                  [start_idx, end_idx) is the range within the full product.
        """
        values, ndim, start_idx, end_idx = task
        return itertools.islice(
            itertools.product(values, repeat=ndim), start_idx, end_idx
        )

    def _generate_combinations_in_batches(
        self, ndim: int, max_val: int, batch_size: int
    ):
        """Yield batches of the Cartesian product {0, ..., max_val}^ndim.

        Uses multiprocessing to avoid materializing the full product
        (size ``(max_val+1)^ndim``) in memory.

        Args:
            ndim: Number of dimensions (tuple length).
            max_val: Maximum coordinate value (inclusive).
            batch_size: Number of tuples per batch.

        Yields:
            An iterator of tuples for each batch.
        """
        values = list(range(max_val + 1))
        total_combinations = (max_val + 1) ** ndim
        tasks = [
            (values, ndim, i, min(i + batch_size, total_combinations))
            for i in range(0, total_combinations, batch_size)
        ]

        with multiprocessing.Pool() as pool:
            for result in pool.imap(self._worker, tasks):
                yield result


####################
# Reward functions #
####################


class GridReward(ABC):
    """Base class for reward functions that can be pickled."""

    def __init__(self, height: int, ndim: int, **kwargs):
        self.height = height
        self.ndim = ndim
        self.kwargs = kwargs
        self._EPS = 1e-12

    @abstractmethod
    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OriginalReward(GridReward):
    """The reward function from the original GFlowNet paper (Bengio et al., 2021;
    https://arxiv.org/abs/2106.04399)."""

    # Original reward: center ring adds R2 on top of base R0 and outer ring R1.
    # Modes are the thin ring where both outer ring and band conditions hold.
    # Modes will have a reward greater than R2+R1+R0.
    # Impl: https://github.com/bengioy/gfn/blob/master/gfn/gym/hypergrid.py#L28
    # For example, using the default kwargs, R0=0.1, R1=0.5, R2=2.0:
    # A 2D grid with height 16. ax = |i/(15) − 0.5|. The band 0.3 < ax < 0.4 holds only
    # at indices i ∈ {2, 13}. The outer ring (0.25 < ax ≤ 0.5) holds at indices
    # i ∈ {0,1,2,3,12,13,14,15}. Legend: X=band (R2), O=outer-ring (R1), .=base (R0).
    # Modes are at the 2×2 Cartesian product of band indices: (2,2), (2,13), (13,2), (13,13).
    # y=15: O O O O . . . . . . . . O O O O
    # y=14: O O O O . . . . . . . . O O O O
    # y=13: O O X O . . . . . . . . O X O O
    # y=12: O O O O . . . . . . . . O O O O
    # y=11: . . . . . . . . . . . . . . . .
    # y=10: . . . . . . . . . . . . . . . .
    # y=09: . . . . . . . . . . . . . . . .
    # y=08: . . . . . . . . . . . . . . . .
    # y=07: . . . . . . . . . . . . . . . .
    # y=06: . . . . . . . . . . . . . . . .
    # y=05: . . . . . . . . . . . . . . . .
    # y=04: . . . . . . . . . . . . . . . .
    # y=03: O O O O . . . . . . . . O O O O
    # y=02: O O X O . . . . . . . . O X O O
    # y=01: O O O O . . . . . . . . O O O O
    # y=00: O O O O . . . . . . . . O O O O
    #       0 1 2 3 4 5 6 7 8 9 101112131415 (x-axis)
    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R0 = self.kwargs.get("R0", 0.1)
        R1 = self.kwargs.get("R1", 0.5)
        R2 = self.kwargs.get("R2", 2.0)

        ax = abs(states_tensor / (self.height - 1) - 0.5)
        return (
            R0
            + (0.25 + self._EPS < ax).prod(-1) * R1
            + ((0.3 + self._EPS < ax) * (ax < 0.4 - self._EPS)).prod(-1) * R2
        )


class CosineReward(GridReward):
    """Cosine reward function."""

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R0 = self.kwargs.get("R0", 0.1)
        R1 = self.kwargs.get("R1", 0.5)

        ax = abs(states_tensor / (self.height - 1) - 0.5)
        pdf_input = ax * 5
        pdf = 1.0 / (2 * torch.pi) ** 0.5 * torch.exp(-(pdf_input**2) / 2)
        reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1
        return reward


class SparseReward(GridReward):
    """Sparse reward function from the GAFN paper (Pan et al., 2022;
    https://arxiv.org/abs/2210.03308)."""

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        targets = []
        for number_of_1s in range(ndim):
            targets.extend(
                itertools.permutations(
                    [1] * number_of_1s + [self.height - 2] * (self.ndim - number_of_1s)
                )
            )
        self.targets = torch.tensor(list(set(targets)), dtype=torch.long)

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        self.targets = self.targets.to(states_tensor.device)
        reward = (
            (states_tensor.unsqueeze(1) == self.targets.unsqueeze(0)).prod(-1).sum(-1)
        ) + self._EPS  # Avoid log(0)
        return reward


class DeceptiveReward(GridReward):
    """Deceptive reward function from Adaptive Teachers (Kim et al., 2025).

    Note that the reward definition in the paper (eq. (9)) is incorrect, and we follow
    the official implementation (https://github.com/alstn12088/adaptive-teacher/blob/8cfcb2298fce3f46eb36ead03791eeee75b7d066/grid/env.py#L27)
    while modifying it to use EPS = 1e-12 to handle inequalities with floating points.
    """

    # Deceptive reward: R0 + (1 - 1[outer-ring])·R1 + 1[band]·R2.
    # Outer-ring cancels R1, center keeps R1, band adds R2. Compared to Original,
    # corners are de-emphasized (no +R1), while the center square is emphasized (+R1).
    # Legend: X=band (R2), +=center (+R1), .=base (R0).
    # For example, with height 16: band indices {2,13}; center indices {6,7,8,9}.
    # Modes remain at (2,2), (2,13), (13,2), (13,13).
    # y=15: . . . . . . + + + + . . . . . .
    # y=14: . . . . . . + + + + . . . . . .
    # y=13: . . X . . . + + + + . . . X . .
    # y=12: . . . . . . + + + + . . . . . .
    # y=11: . . . . . . + + + + . . . . . .
    # y=10: . . . . . . + + + + . . . . . .
    # y=09: + + + + + + + + + + + + + + + +
    # y=08: + + + + + + + + + + + + + + + +
    # y=07: + + + + + + + + + + + + + + + +
    # y=06: + + + + + + + + + + + + + + + +
    # y=05: . . . . . . + + + + . . . . . .
    # y=04: . . . . . . + + + + . . . . . .
    # y=03: . . . . . . + + + + . . . . . .
    # y=02: . . X . . . + + + + . . . X . .
    # y=01: . . . . . . + + + + . . . . . .
    # y=00: . . . . . . + + + + . . . . . .
    #       0 1 2 3 4 5 6 7 8 9 101112131415 (x-axis)

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R0 = self.kwargs.get("R0", 1e-5)
        R1 = self.kwargs.get("R1", 0.1)
        R2 = self.kwargs.get("R2", 2.0)

        ax = abs(states_tensor / (self.height - 1) - 0.5)
        term1 = R0 + R1
        cancel_outer = (0.1 + self._EPS < ax).prod(-1) * R1
        ring_band = ((0.3 + self._EPS < ax) * (ax < 0.4 - self._EPS)).prod(-1) * R2
        return term1 - cancel_outer + ring_band


#########################
# Conditional HyperGrid #
#########################


class ConditionalHyperGrid(HyperGrid):
    """HyperGrid environment with condition-aware rewards.

    Let condition 'c' be a real value in [0, 1]. It defines the reward as a linear
    interpolation between the uniform reward and the original reward. Special cases are:
    - c = 0: Uniform reward (all terminal states get reward=R0+R1+R2)
    - c = 1: Original HyperGrid reward (original multi-modal reward landscape)
    """

    is_conditional: bool = True
    condition_dim: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_reward_fn = self.reward_fn  # Rename, just to avoid confusion
        self._max_reward: float = (
            self.reward_fn_kwargs.get("R0", 0.1)
            + self.reward_fn_kwargs.get("R1", 0.5)
            + self.reward_fn_kwargs.get("R2", 2.0)
        )
        self._log_partition_cache: dict[torch.Tensor, float] = {}
        self._true_dist_cache: dict[torch.Tensor, torch.Tensor] = {}

    def sample_conditions(self, batch_shape: int | tuple[int, ...]) -> torch.Tensor:
        """Sample conditions for the environment."""

        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        return torch.rand(batch_shape + (self.condition_dim,), device=self.device)

    def reward(self, states: DiscreteStates) -> torch.Tensor:
        """Compute rewards for the conditional environment.

        A condition is continuous from 0 to 1:
        - 0: Fully uniform reward (all states get R0+R1+R2)
        - 1: Fully original HyperGrid reward
        - In between: Linear interpolation between uniform and original

        Args:
            states: The states to compute rewards for.
                states.tensor.shape should be (*batch_shape, *state_shape)

        Returns:
            A tensor of shape (*batch_shape,) containing the rewards.
        """
        # Get original rewards
        original_rewards = self._original_reward_fn(states.tensor)
        # shape: (*batch_shape,)

        assert states.conditions is not None
        # Remove feature dimension
        cond = states.conditions.squeeze(-1)  # shape: (*batch_shape,)

        # For uniform, all states get the max reward (R0+R1+R2)
        uniform_rewards = torch.full_like(original_rewards, self._max_reward)

        # Linear interpolation between uniform and original based on conditions
        rewards = (1 - cond) * uniform_rewards + cond * original_rewards
        return rewards

    def log_partition(self, condition: torch.Tensor) -> float:
        """Compute the log partition for the given condition.

        Args:
            condition: The condition to compute the log partition for.
                condition.shape should be (1,)

        Returns:
            The log partition function, as a float.
        """
        if condition not in self._log_partition_cache:
            assert self.all_states is not None
            # Attach conditions to states for reward computation
            states_with_cond = self.all_states.clone()
            states_with_cond.conditions = condition.repeat(self.n_states, 1)
            all_rewards = self.reward(states_with_cond)
            self._log_partition_cache[condition] = all_rewards.sum().log().item()
        return self._log_partition_cache[condition]

    def true_dist(self, condition: torch.Tensor) -> torch.Tensor:
        """Compute the true distribution for the given condition.

        Args:
            condition: The condition to compute the true distribution for.
            condition.shape should be (1,)

        Returns:
            The true distribution for the given condition as a 1-dimensional tensor.
        """
        if condition not in self._true_dist_cache:
            assert self.all_states is not None
            # Attach conditions to states for reward computation
            states_with_cond = self.all_states.clone()
            states_with_cond.conditions = condition.repeat(self.n_states, 1)
            all_rewards = self.reward(states_with_cond)
            self._true_dist_cache[condition] = all_rewards / all_rewards.sum()
        return self._true_dist_cache[condition]


####################################
# New compositional reward classes #
####################################


class BitwiseXORReward(GridReward):
    """Tiered, compositional reward based on bitwise XOR/parity constraints.

    This class implements the "Bitwise/XOR fractal" environment family: where tiers
    progressively constrain bit-planes across a subset of dimensions via linear parity
    checks over GF(2). It supports easy sharding by high-bit prefixes, and difficulty
    control by adjusting which bit-planes and how many dimensions are constrained per tier.

    GF(2) is the finite field with two elements {0, 1}, where addition and
    multiplication are performed modulo 2. In this context, vector addition is
    equivalent to bitwise XOR, and matrix-vector products (A @ b) are evaluated
    entrywise modulo 2.

    Reward form:
        R(s) = R0 + Σ_t tier_weights[t] · 1[ state satisfies all constraints up to tier t ]

    Key kwargs (with reasonable defaults):
        - R0: float, base reward (default 0.0)
        - tier_weights: list[float], strictly increasing weights for each tier
        - dims_constrained: Optional[list[int]] subset of dims to constrain
          (default: all dims)
        - bits_per_tier: list[tuple[int,int]]; for each tier t, inclusive bit range
          (low_bit, high_bit). Example: [(0,5), (0,7), (0,9)].
        - parity_checks: Optional[list[dict]]; per tier, optional parity system:
            Each entry may contain:
              { "A": IntTensor[num_checks, m], "c": IntTensor[num_checks] }
            where m = len(dims_constrained). Constraints apply identically to every
            bit-plane specified for that tier: A @ b(mod2) == c, where b are the
            bit values across constrained dimensions at the tested bit-plane.
            If omitted for a tier, a single even-parity check across all constrained
            dims is used by default: sum(b) mod 2 == 0.

    Difficulty presets align with step ranges by controlling the highest bit used
    and the number of constrained dimensions. Typical distance from origin for
    valid modes scales roughly like (constrained_dims · 2^{highest_bit}).

    Comparison with other compositional rewards:
        - MultiplicativeCoprimeReward: number-theoretic (prime factorization);
          same constraint type per tier (tighter exponent caps), no cross-scale
          dependency.
        - ConditionalMultiScaleReward: base-B digit decomposition with conditional
          constraints across scales; each tier's rule is a function of prior tiers,
          introducing qualitatively different structure at each level.
        - This class: GF(2) linear algebra on bit-planes; same parity check type
          per tier, but applied to progressively wider bit windows. Constraints at
          each bit-plane are independent of other planes.
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.0))
        tier_weights = kwargs.get("tier_weights", [1.0, 10.0, 100.0])
        assert isinstance(tier_weights, (list, tuple)) and len(tier_weights) > 0
        self.tier_weights: list[float] = [float(w) for w in tier_weights]

        dims_constrained = kwargs.get("dims_constrained", None)
        if dims_constrained is None:
            dims_constrained = list(range(ndim))
        assert len(dims_constrained) > 0
        self.dims_constrained: list[int] = list(map(int, dims_constrained))

        bits_per_tier = kwargs.get("bits_per_tier", None)
        if bits_per_tier is None:
            # Default: widen the bit window gradually
            bits_per_tier = [(0, 5), (0, 7), (0, 9)]
        assert len(bits_per_tier) == len(self.tier_weights)
        self.bits_per_tier: list[tuple[int, int]] = [
            (int(lo), int(hi)) for (lo, hi) in bits_per_tier
        ]

        self.parity_checks = kwargs.get("parity_checks", None)
        if self.parity_checks is not None:
            assert len(self.parity_checks) == len(self.tier_weights)

    def _even_parity_mask(self, bits: torch.Tensor) -> torch.Tensor:
        """bits: (..., m) int/bool -> returns (...,) bool for even parity."""
        if bits.dtype != torch.long:
            bits = bits.long()
        return (bits.sum(dim=-1) & 1) == 0

    def _apply_parity_checks(
        self, bits_plane: torch.Tensor, tier_idx: int
    ) -> torch.Tensor:
        """Apply GF(2) linear parity checks at a single bit-plane.

        bits_plane: (..., m) with m=len(dims_constrained), integer in {0,1}.
        Returns mask (...,) bool.
        """
        if self.parity_checks is None or self.parity_checks[tier_idx] is None:
            return self._even_parity_mask(bits_plane)

        cfg = self.parity_checks[tier_idx]
        A: torch.Tensor | None = cfg.get("A")
        c: torch.Tensor | None = cfg.get("c")
        if A is None or c is None:
            return self._even_parity_mask(bits_plane)

        # Ensure device/dtype
        A = A.to(bits_plane.device).long()
        c = c.to(bits_plane.device).long()
        # Compute (A @ bits) mod 2 for each batch element
        # reshape bits to (..., m, 1) for bmm if needed, but here we can do matmul
        # by flattening the batch.
        flat = bits_plane.reshape(-1, bits_plane.shape[-1]).long()
        prod = (flat @ A.t()) & 1  # shape (B, num_checks)
        target = c.unsqueeze(0).expand_as(prod)
        ok = (prod == target).all(dim=-1)
        return ok.reshape(bits_plane.shape[:-1])

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R = torch.zeros(
            states_tensor.shape[:-1],
            device=states_tensor.device,
            dtype=torch.get_default_dtype(),
        )
        if self.R0 != 0.0:
            R += self.R0

        # Select constrained dims
        x = states_tensor.index_select(
            dim=-1,
            index=torch.tensor(self.dims_constrained, device=states_tensor.device),
        )

        # Cumulative tiers: a state gets tier-t reward only if it satisfies
        # all constraints from tiers 0..t.
        valid_up_to_t = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
        for t, w in enumerate(self.tier_weights):
            lo_bit, hi_bit = self.bits_per_tier[t]
            # All bit-planes in this tier's window must pass parity checks.
            all_planes_ok = torch.ones_like(valid_up_to_t)
            for b in range(lo_bit, hi_bit + 1):
                # Extract the b-th bit of each coordinate: 0 or 1.
                bits = ((x >> b) & 1).long()
                plane_ok = self._apply_parity_checks(bits, t)
                all_planes_ok = all_planes_ok & plane_ok
            valid_up_to_t = valid_up_to_t & all_planes_ok
            R = R + (valid_up_to_t.to(R.dtype) * float(w))

        return R


class MultiplicativeCoprimeReward(GridReward):
    """Tiered reward based on prime-support and coprimality/lcm composition.

    Each tier enforces that per-dimension values use only a small shared prime set
    with bounded exponents, plus optional cross-dimension constraints (pairwise
    coprime pairs and/or target lcm). Higher tiers tighten exponent caps or add
    additional global targets. This encourages information sharing to learn the
    latent prime/exponent structure.

    Reward form:
        R(s) = R0 + Σ_t tier_weights[t] · 1[ constraints_0..t all satisfied ]

    Key kwargs:
        - R0: float, base reward (default 0.0)
        - tier_weights: list[float]
        - primes: list[int], e.g., [2,3,5,7,11]
        - exponent_caps: list[int], same length as tier_weights. Cap for every prime
          at tier t (uniform cap across primes for simplicity).
        - active_dims: Optional[list[int]]; constraints only apply to these dims
          (default: all dims). Other dims are ignored in constraints.
        - coprime_pairs: Optional[list[tuple[int,int]]]; indices relative to active_dims.
        - target_lcms: Optional[list[int | None]]; per-tier target lcm across active dims.

    Notes:
    - Values 0 are treated as invalid for prime-support constraints (cannot factorize);
      value 1 is valid with all-zero exponents.
    - Implementation removes primes up to the current tier cap and checks residue == 1.
      Exponent counts are accumulated to evaluate LCM targets.

    Comparison with other compositional rewards:
        - BitwiseXORReward: GF(2) parity checks on bit-planes; same constraint
          type per tier (wider bit window), no cross-scale dependency.
        - ConditionalMultiScaleReward: base-B digit decomposition with conditional
          constraints across scales; each tier's rule depends on prior tiers.
        - This class: prime factorization with bounded exponents and optional
          coprimality/LCM targets. Same constraint type per tier (tighter caps),
          but cross-dimension coupling via coprime pairs and LCM targets.
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.0))
        tier_weights = kwargs.get("tier_weights", [1.0, 10.0, 100.0])
        assert isinstance(tier_weights, (list, tuple)) and len(tier_weights) > 0
        self.tier_weights: list[float] = [float(w) for w in tier_weights]

        primes = kwargs.get("primes", [2, 3, 5])
        assert isinstance(primes, (list, tuple)) and len(primes) > 0
        self.primes: list[int] = [int(p) for p in primes]

        exponent_caps = kwargs.get("exponent_caps", [2] * len(self.tier_weights))
        assert len(exponent_caps) == len(self.tier_weights)
        self.exponent_caps: list[int] = [int(c) for c in exponent_caps]

        active_dims = kwargs.get("active_dims", None)
        if active_dims is None:
            active_dims = list(range(ndim))
        self.active_dims: list[int] = list(map(int, active_dims))

        self.coprime_pairs = kwargs.get("coprime_pairs", None)
        self.target_lcms = kwargs.get("target_lcms", [None] * len(self.tier_weights))
        assert isinstance(self.target_lcms, (list, tuple)) and len(
            self.target_lcms
        ) == len(self.tier_weights)

    def _factor_exponents_up_to_cap(self, v: torch.Tensor, cap: int):
        """Trial-divide each element by allowed primes, returning residue and exponents.

        Args:
            v: (...,) LongTensor of non-negative values to factorize.
            cap: Maximum number of times each prime may divide a value.

        Returns:
            residue: (...,) values after stripping allowed primes (1 if fully factored).
            exps: [num_primes, ...] exponent counts per prime (leading axis is primes).
        """
        residue = v.clone()
        exps = []
        for p_int in [int(p) for p in self.primes]:
            count = torch.zeros_like(residue)
            for _ in range(cap):
                divisible = (residue % p_int) == 0
                if not torch.any(divisible):
                    break
                residue = torch.where(divisible, residue // p_int, residue)
                count = count + divisible.long()
            exps.append(count)
        # Shape: [num_primes, ...] — prime axis is leading.
        exps = torch.stack(exps, dim=0)
        return residue, exps

    def _pairwise_coprime_ok(self, v: torch.Tensor) -> torch.Tensor:
        """Check that configured dimension pairs share no common allowed prime.

        Args:
            v: (..., num_active_dims) coordinate values.

        Returns:
            (...,) bool mask, True where all coprime pair constraints hold.
        """
        if not self.coprime_pairs:
            return torch.ones(v.shape[:-1], dtype=torch.bool, device=v.device)
        ok = torch.ones(v.shape[:-1], dtype=torch.bool, device=v.device)
        for p_int in [int(p) for p in self.primes]:
            divisible = (v % p_int) == 0  # (..., num_active_dims)
            for i, j in self.coprime_pairs:
                num_active = divisible.shape[-1]
                # Skip pairs that reference dims beyond active_dims (guard).
                if i >= num_active or j >= num_active:
                    continue
                both_divisible = divisible[..., i] & divisible[..., j]
                ok = ok & (~both_divisible)
        return ok

    def _lcm_ok(self, exps: torch.Tensor, target_lcm: int) -> torch.Tensor:
        """Check whether max exponents across dims match target LCM's factorization.

        Args:
            exps: [num_primes, ..., num_active_dims] exponent counts.
            target_lcm: The target LCM value to match.

        Returns:
            (...,) bool mask, True where the LCM of active-dim values equals target.
        """
        # Factor target_lcm over allowed primes; reject if leftover > 1.
        remaining = int(target_lcm)
        target_exp_counts: list[int] = []
        for p_int in [int(p) for p in self.primes]:
            exp_count = 0
            while remaining % p_int == 0:
                remaining //= p_int
                exp_count += 1
            target_exp_counts.append(exp_count)
        if remaining != 1:
            # Target contains primes outside the allowed set — impossible.
            batch_shape = exps.shape[1:-1]
            return torch.zeros(batch_shape, dtype=torch.bool, device=exps.device)

        target_vec = torch.tensor(
            target_exp_counts, dtype=torch.long, device=exps.device
        )
        max_exp = exps.max(dim=-1).values  # [num_primes, ...]

        # Unsqueeze target to broadcast against batch dimensions of max_exp.
        while target_vec.dim() < max_exp.dim():
            target_vec = target_vec.unsqueeze(-1)
        return (max_exp == target_vec).all(dim=0)

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R = torch.zeros(
            states_tensor.shape[:-1],
            device=states_tensor.device,
            dtype=torch.get_default_dtype(),
        )
        if self.R0 != 0.0:
            R += self.R0

        x = states_tensor.index_select(
            dim=-1, index=torch.tensor(self.active_dims, device=states_tensor.device)
        )
        # Zero values cannot be factored over primes — exclude them upfront.
        base_valid = (x != 0).all(dim=-1)

        valid_up_to_t = base_valid
        for t, w in enumerate(self.tier_weights):
            cap = self.exponent_caps[t]
            # Flatten all active-dim values for batch factorization, then reshape.
            residue, exps = self._factor_exponents_up_to_cap(x.reshape(-1), cap)
            residue = residue.reshape(x.shape)
            exps = exps.reshape((len(self.primes),) + x.shape)

            # residue==1 means the value is fully explained by allowed primes.
            # x==1 trivially satisfies (zero exponents for all primes).
            support_ok = ((residue == 1) | (x == 1)).all(dim=-1)

            # Coprime check is tier-independent but placed here for uniformity.
            pairs_ok = self._pairwise_coprime_ok(x)
            lcm_ok = torch.ones_like(pairs_ok)
            if self.target_lcms[t] is not None:
                lcm_ok = self._lcm_ok(exps, int(self.target_lcms[t]))

            tier_ok = support_ok & pairs_ok & lcm_ok
            valid_up_to_t = valid_up_to_t & tier_ok
            R = R + (valid_up_to_t.to(R.dtype) * float(w))

        return R


class ConditionalMultiScaleReward(GridReward):
    """Tiered reward via conditional digit constraints across spatial scales.

    Each coordinate is decomposed in base B into L = log_B(H) digits. Tier t
    constrains digit t-1 via a shifted filter that depends on all finer-scale
    digits, creating a hierarchy where learning lower-scale structure is
    prerequisite for predicting higher-scale constraints.

    Per-dimension constraint at tier t:
        (d_{t-1}(i) + sigma_t(i)) mod B < f_t

    where sigma_t(i) = sum_{k=0}^{t-2} a_{t,k} * d_k(i)  mod B  is a linear
    function of finer-scale digits with seed-derived coefficients a_{t,k}.

    Optional cross-dimensional constraint at tier t:
        sum_i d_{t-1}(i) ≡ 0  (mod m_t)

    Reward form (cumulative — tier t requires all tiers 1..t):
        R(s) = R0 + sum_t tier_weights[t] * 1[s satisfies tiers 1..t]

    Mode count (exact closed form):
        Without cross-dim:  modes_T = (prod_{t=1}^T f_t)^d * B^{(L-T)*d}
        With cross-dim:     modes_T = (prod_t f_t)^d * B^{(L-T)*d} / prod_t m_t

    Partition function (analytic, no enumeration):
        Z = R0 * H^d + sum_t w_t * modes_t

    Key kwargs:
        - R0: float, base reward (default 0.0).
        - tier_weights: list[float], reward weight per tier.
        - base: int, digit base B (default 4). H must be a power of B.
        - filter_width: int, number of passing digit values per tier (default B//2).
          Constant across tiers to avoid mode collapse at deep tiers.
        - seed: int, PRNG seed for generating shift coefficients (default 42).
        - cross_dim_mods: Optional[list[int|None]], per-tier modular cross-dim
          constraint. m_t must divide filter_width for exact mode counts.
          Default: no cross-dim constraints.
        - active_dims: Optional[list[int]], subset of dims to constrain
          (default: all dims).

    Comparison with other compositional rewards:
        - BitwiseXORReward: GF(2) parity checks on bit-planes; each tier widens
          the bit window but uses the same rule type. No cross-scale dependency.
        - MultiplicativeCoprimeReward: prime factorization with tightening exponent
          caps. Same constraint type at every tier.
        - This class: each tier introduces a qualitatively different constraint
          whose form depends on what was learned at prior tiers (conditional
          structure across scales).
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.0))

        tier_weights = kwargs.get("tier_weights", [1.0, 10.0, 100.0])
        assert isinstance(tier_weights, (list, tuple)) and len(tier_weights) > 0
        self.tier_weights: list[float] = [float(w) for w in tier_weights]

        self.base: int = int(kwargs.get("base", 4))
        assert self.base >= 2, f"base must be >= 2, got {self.base}"

        # Compute number of digit levels
        self.num_levels: int = 0
        h = height
        while h > 1:
            assert (
                h % self.base == 0
            ), f"height={height} must be a power of base={self.base}"
            h //= self.base
            self.num_levels += 1
        assert self.num_levels > 0

        assert len(self.tier_weights) <= self.num_levels, (
            f"Too many tiers ({len(self.tier_weights)}) for "
            f"base={self.base}, height={height} ({self.num_levels} digit levels)"
        )

        self.filter_width: int = int(kwargs.get("filter_width", self.base // 2))
        assert (
            1 <= self.filter_width <= self.base
        ), f"filter_width={self.filter_width} must be in [1, {self.base}]"

        self.seed: int = int(kwargs.get("seed", 42))

        # Generate shift coefficients from seed
        # a_{t,k} for t=1..T, k=0..t-2; each in {0, ..., B-1}
        rng = torch.Generator().manual_seed(self.seed)
        self.shift_coeffs: list[list[int]] = []
        for t in range(len(self.tier_weights)):
            if t == 0:
                self.shift_coeffs.append([])  # tier 1 has no prior digits
            else:
                coeffs = torch.randint(0, self.base, (t,), generator=rng).tolist()
                self.shift_coeffs.append(coeffs)

        # Cross-dimensional modular constraints (optional)
        cross_dim_mods = kwargs.get("cross_dim_mods", None)
        if cross_dim_mods is None:
            cross_dim_mods = [None] * len(self.tier_weights)
        assert len(cross_dim_mods) == len(self.tier_weights)
        self.cross_dim_mods: list[int | None] = []
        for m in cross_dim_mods:
            if m is not None:
                m = int(m)
                assert m >= 2
                assert self.filter_width % m == 0, (
                    f"filter_width={self.filter_width} must be divisible by "
                    f"cross_dim_mod={m} for exact mode counts"
                )
            self.cross_dim_mods.append(m)

        active_dims = kwargs.get("active_dims", None)
        if active_dims is None:
            active_dims = list(range(ndim))
        self.active_dims: list[int] = list(map(int, active_dims))

    def _extract_digits(self, x: torch.Tensor, num_levels: int) -> list[torch.Tensor]:
        """Extract base-B digits from x.

        Args:
            x: (..., m) integer tensor with coordinate values.
            num_levels: how many digit levels to extract.

        Returns:
            List of num_levels tensors, each (..., m), with digit values in [0, B).
            digits[k] is the k-th digit (scale k), i.e., floor(x / B^k) mod B.
        """
        digits = []
        remaining = x.clone()
        for _ in range(num_levels):
            digits.append(remaining % self.base)
            remaining = remaining // self.base
        return digits

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R = torch.zeros(
            states_tensor.shape[:-1],
            device=states_tensor.device,
            dtype=torch.get_default_dtype(),
        )
        if self.R0 != 0.0:
            R += self.R0

        # Select active dims
        x = states_tensor.index_select(
            dim=-1,
            index=torch.tensor(self.active_dims, device=states_tensor.device),
        ).long()

        # Extract all needed digits upfront
        num_tiers = len(self.tier_weights)
        digits = self._extract_digits(x, num_tiers)

        valid_up_to_t = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
        for t, w in enumerate(self.tier_weights):
            # Compute shift: sigma_t(i) = sum_{k<t} a_{t,k} * d_k(i)  mod B
            if t == 0:
                shift = torch.zeros_like(x)
            else:
                shift = torch.zeros_like(x)
                for k, a_tk in enumerate(self.shift_coeffs[t]):
                    if a_tk != 0:
                        shift = shift + int(a_tk) * digits[k]
                shift = shift % self.base

            # Filter: shifted digit must be in [0, filter_width).
            shifted = (digits[t] + shift) % self.base
            per_dim_ok = (shifted < self.filter_width).all(dim=-1)

            # Optional cross-dim modular constraint on the digit sum.
            cross_ok = torch.ones_like(per_dim_ok)
            cross_mod = self.cross_dim_mods[t]
            if cross_mod is not None:
                digit_sum = digits[t].sum(dim=-1)
                cross_ok = (digit_sum % int(cross_mod)) == 0

            tier_ok = per_dim_ok & cross_ok
            valid_up_to_t = valid_up_to_t & tier_ok
            R = R + (valid_up_to_t.to(R.dtype) * float(w))

        return R

    def mode_tier(self, target_sparsity: float = 0.10) -> int:
        """Return the lowest tier whose mode coverage is below target_sparsity.

        Coverage at tier t = (f/B)^(t*d) (before cross-dim constraints).
        This adapts the mode definition to dimensionality: at low d, deeper
        tiers are needed for modes to be sparse; at high d, even tier 1 is
        already a needle in a haystack.

        Args:
            target_sparsity: Fraction of total state space below which modes
                are considered "interesting" (default 0.10 = 10%).

        Returns:
            1-indexed tier number. Clamped to [1, num_tiers].
        """
        d = len(self.active_dims)
        ratio = self.filter_width / self.base  # f/B
        for t in range(1, len(self.tier_weights) + 1):
            if ratio ** (t * d) < target_sparsity:
                return t
        return len(self.tier_weights)

    def mode_threshold(self, target_sparsity: float = 0.10) -> float:
        """Return the reward threshold for mode counting at the adaptive tier.

        States with reward >= this value are counted as modes. The tier is
        chosen via ``mode_tier(target_sparsity)`` so that mode coverage adapts
        to dimensionality.

        Args:
            target_sparsity: Passed to ``mode_tier()``.

        Returns:
            Reward threshold (R0 + sum of weights up to the mode tier).
        """
        t = self.mode_tier(target_sparsity)
        return self.R0 + sum(self.tier_weights[:t])

    def analytic_mode_count(self, tier: int | None = None) -> int:
        """Compute exact mode count for a given tier (1-indexed) or all tiers.

        Args:
            tier: 1-indexed tier number. If None, returns count for the
                  highest tier (most constrained).

        Returns:
            Number of states satisfying all constraints up to the given tier.
        """
        if tier is None:
            tier = len(self.tier_weights)
        assert 1 <= tier <= len(self.tier_weights)

        d = len(self.active_dims)
        T = tier
        L = self.num_levels
        B = self.base
        f = self.filter_width

        # Per-dim: (f)^T valid constrained digit combos × B^(L-T) free digits
        choices_per_coord = (f**T) * (B ** (L - T))
        modes = choices_per_coord**d

        # Cross-dim modular constraints divide out uniformly.
        for t in range(T):
            cross_mod = self.cross_dim_mods[t]
            if cross_mod is not None:
                modes //= cross_mod

        return modes

    def analytic_log_partition(self) -> float:
        """Compute log(Z) analytically.

        Z = R0 * H^d + sum_t w_t * modes_t
        """
        d = len(self.active_dims)
        Z = self.R0 * (self.height**d)
        for t in range(len(self.tier_weights)):
            Z += self.tier_weights[t] * self.analytic_mode_count(tier=t + 1)
        return log(Z) if Z > 0 else float("-inf")


# -------------------------
# Difficulty preset factories
# -------------------------


def _first_k_dims(k: int, ndim: int) -> list[int]:
    """Return indices [0, 1, ..., min(k, ndim)-1] for the first k dimensions."""
    k = min(max(1, k), ndim)
    return list(range(k))


def get_bitwise_xor_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for BitwiseXORReward.

    The presets target approximate L1 distance bands by selecting the highest
    constrained bit and number of constrained dimensions. Typical distance scales
    like m · 2^b, where m is the number of constrained dims and b the highest bit.

    Bands (steps from s0):
      - easy:        ~50-100
      - medium:      ~250-500
      - hard:        ~1k-2.5k
      - challenging: ~2.5k-5k
      - impossible:  5k+

    Notes
    - You may tweak m (dims) and bit windows to fine-tune distances for your D,H.
    - Tier weights are geometric to encourage reaching higher tiers.
    - Parity checks default to even parity across constrained dims per bit-plane.
    """

    presets = {
        "easy": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0],
            dims_constrained=_first_k_dims(3, ndim),
            bits_per_tier=[(0, 4), (0, 5), (0, 5)],
        ),
        "medium": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            dims_constrained=_first_k_dims(4, ndim),
            bits_per_tier=[(0, 6), (0, 7), (0, 7), (0, 7)],
        ),
        "hard": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            dims_constrained=_first_k_dims(8, ndim),
            bits_per_tier=[(0, 8), (0, 8), (0, 8), (0, 8)],
        ),
        "challenging": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            dims_constrained=_first_k_dims(6, ndim),
            bits_per_tier=[(0, 9), (0, 9), (0, 9), (0, 9)],
        ),
        "impossible": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0, 10000.0],
            dims_constrained=_first_k_dims(12, ndim),
            bits_per_tier=[(0, 9), (0, 10), (0, 10), (0, 10), (0, 10)],
        ),
    }
    return presets


def get_multiplicative_coprime_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for MultiplicativeCoprimeReward.

    Bands (steps from s0):
      - easy:        ~50-100 (small primes, small exponents, few active dims)
      - medium:      ~250-500 (adds one prime, caps=2, more dims, light coupling)
      - hard:        ~1k-2.5k (primes up to 11, caps=3, more dims, LCM target)
      - challenging: ~2.5k-5k (primes up to 13, caps=3-4, 10-12 dims, tighter)
      - impossible:  5k+ (primes up to 29, caps=4, 12-16 dims, multiple targets)

    Notes
    - Distances are approximate; increase primes and exponent caps to push further.
    - `active_dims` indexes are relative to state dims; we pick first k for simplicity.
    - `coprime_pairs` are pairs within `active_dims` index space.
    - Tier weights are geometric.
    """

    def chain_pairs(k: int) -> list[tuple[int, int]]:
        """Return adjacent-index coprime pairs: (0,1), (1,2), ..., (k-2, k-1)."""
        return [(i, i + 1) for i in range(max(0, k - 1))]

    presets = {
        "easy": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0],
            primes=[2, 3, 5],
            exponent_caps=[2, 2, 2],
            active_dims=_first_k_dims(3, ndim),
            coprime_pairs=chain_pairs(3),
            target_lcms=[None, None, None],
        ),
        "medium": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7],
            exponent_caps=[2, 2, 2, 2],
            active_dims=_first_k_dims(5, ndim),
            coprime_pairs=chain_pairs(5),
            target_lcms=[None, None, None, None],
        ),
        "hard": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7, 11],
            exponent_caps=[3, 3, 3, 3],
            active_dims=_first_k_dims(8, ndim),
            coprime_pairs=chain_pairs(8),
            target_lcms=[
                None,
                None,
                2**3 * 3**2 * 5 * 7 * 11,
                2**3 * 3**2 * 5 * 7 * 11,
            ],  # = 9240
        ),
        "challenging": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7, 11, 13],
            exponent_caps=[3, 3, 4, 4],
            active_dims=_first_k_dims(10, ndim),
            coprime_pairs=chain_pairs(10),
            target_lcms=[None, None, None, 2**3 * 3**2 * 5**2 * 13],  # = 5850
        ),
        "impossible": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0, 10000.0],
            primes=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            exponent_caps=[4, 4, 4, 4, 4],
            active_dims=_first_k_dims(12, ndim),
            coprime_pairs=chain_pairs(12),
            target_lcms=[None, None, None, None, 2**4 * 3**3 * 5**2 * 7 * 11],
        ),
    }
    return presets


def get_conditional_multiscale_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for ConditionalMultiScaleReward.

    All presets use base=4 (requiring H to be a power of 4). The number of
    available digit levels is L = log_4(H). Difficulty is controlled by:
      - Number of tiers (more tiers = deeper compositional hierarchy)
      - Number of active dims (more = exponentially sparser modes)
      - Cross-dim modular constraints (further sparsification)

    Mode counts are computed via:
        modes_T = (f^T * 4^{L-T})^d / prod_t m_t

    with f = filter_width = 2 (i.e. B//2), so each tier halves modes per coord.

    Presets (assuming H=256, i.e. L=4 digit levels):
      - easy:        2 tiers, 3 active dims -> ~2M modes at tier 2
      - medium:      3 tiers, 4 active dims -> ~1M modes at tier 3
      - hard:        3 tiers, 6 active dims, cross-dim -> ~260K modes at tier 3
      - challenging: 4 tiers, 8 active dims, cross-dim -> ~65K modes at tier 4
      - impossible:  4 tiers, 12 active dims, cross-dim -> ~4K modes at tier 4
    """

    presets = {
        "easy": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0],
            base=4,
            filter_width=2,
            seed=42,
            active_dims=_first_k_dims(3, ndim),
        ),
        "medium": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0],
            base=4,
            filter_width=2,
            seed=42,
            active_dims=_first_k_dims(4, ndim),
        ),
        "hard": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0],
            base=4,
            filter_width=2,
            seed=42,
            active_dims=_first_k_dims(6, ndim),
            cross_dim_mods=[None, 2, 2],
        ),
        "challenging": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            base=4,
            filter_width=2,
            seed=42,
            active_dims=_first_k_dims(8, ndim),
            cross_dim_mods=[None, None, 2, 2],
        ),
        "impossible": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            base=4,
            filter_width=2,
            seed=42,
            active_dims=_first_k_dims(12, ndim),
            cross_dim_mods=[None, 2, 2, 2],
        ),
    }
    return presets


def get_original_presets(ndim: int, height: int) -> dict:
    """Return five presets for OriginalReward.

    These presets primarily control the relative importance of the outer ring (R1)
    and thin band (R2). Exploration difficulty (distance from s0) is more a function
    of (D, H) than of these weights; tune D and H externally to match your distance
    bands.
    """
    presets = {
        "easy": dict(R0=0.1, R1=0.3, R2=1.0),
        "medium": dict(R0=0.1, R1=0.5, R2=2.0),
        "hard": dict(R0=0.05, R1=0.6, R2=3.0),
        "challenging": dict(R0=0.01, R1=0.6, R2=4.0),
        "impossible": dict(R0=0.0, R1=0.7, R2=5.0),
    }
    return presets


def get_cosine_presets(ndim: int, height: int) -> dict:
    """Return five presets for CosineReward.

    R1 scales the oscillatory product, and `mode_gamma` (used only for mode
    detection thresholding) tightens what is considered a "mode-like" maximum.
    """
    presets = {
        "easy": dict(R0=0.1, R1=0.3, mode_gamma=0.7),
        "medium": dict(R0=0.1, R1=0.5, mode_gamma=0.8),
        "hard": dict(R0=0.05, R1=0.6, mode_gamma=0.85),
        "challenging": dict(R0=0.01, R1=0.7, mode_gamma=0.9),
        "impossible": dict(R0=0.0, R1=0.8, mode_gamma=0.92),
    }
    return presets


def get_sparse_presets(ndim: int, height: int) -> dict:
    """Return five presets for SparseReward.

    SparseReward has built-in targets; it ignores most kwargs. Presets are provided
    for API symmetry and future extensibility.
    """
    empty: dict = {}
    presets = {
        "easy": empty,
        "medium": empty,
        "hard": empty,
        "challenging": empty,
        "impossible": empty,
    }
    return presets


def get_deceptive_presets(ndim: int, height: int) -> dict:
    """Return five presets for DeceptiveReward.

    Increase R2 to accentuate the thin band, and set a small but non-zero R0.
    R1 controls the center emphasis vs. the cancelled outer region.
    """
    presets = {
        "easy": dict(R0=1e-5, R1=0.05, R2=1.0),
        "medium": dict(R0=1e-5, R1=0.1, R2=2.0),
        "hard": dict(R0=1e-5, R1=0.15, R2=3.0),
        "challenging": dict(R0=1e-5, R1=0.2, R2=4.0),
        "impossible": dict(R0=1e-5, R1=0.25, R2=5.0),
    }
    return presets


def get_reward_presets(reward_fn_str: str, ndim: int, height: int) -> dict:
    """Return presets for a given reward name: 'bitwise_xor', 'multiplicative_coprime', 'conditional_multiscale'.

    Usage
    ----
    presets = get_reward_presets("bitwise_xor", D, H)
    kwargs = presets["hard"]
    env = HyperGrid(ndim=D, height=H, reward_fn_str="bitwise_xor", reward_fn_kwargs=kwargs)
    """
    if reward_fn_str == "bitwise_xor":
        return get_bitwise_xor_presets(ndim, height)
    if reward_fn_str == "multiplicative_coprime":
        return get_multiplicative_coprime_presets(ndim, height)
    if reward_fn_str == "conditional_multiscale":
        return get_conditional_multiscale_presets(ndim, height)
    if reward_fn_str == "original":
        return get_original_presets(ndim, height)
    if reward_fn_str == "cosine":
        return get_cosine_presets(ndim, height)
    if reward_fn_str == "sparse":
        return get_sparse_presets(ndim, height)
    if reward_fn_str == "deceptive":
        return get_deceptive_presets(ndim, height)
    raise ValueError(f"Unknown reward_fn_str for presets: {reward_fn_str}")
