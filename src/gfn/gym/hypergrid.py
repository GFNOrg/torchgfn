"""Adapted from https://github.com/Tikquuss/GflowNets_Tutorial"""

import hashlib
import itertools
import logging
import math
import multiprocessing
import os
import warnings
from abc import ABC, abstractmethod
from decimal import Decimal
from functools import reduce
from math import gcd, log, pi, sqrt
from time import time
from typing import List, Literal, Tuple, Union

import numpy as np
import torch

from gfn.actions import Actions
from gfn.constants import EPS_INDEX_CMP, EPS_REWARD_CMP
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates

logger = logging.getLogger(__name__)

# Use ``spawn`` everywhere.  Historically this module set the start method to
# ``fork`` on POSIX so that ``HyperGrid._worker`` (a bound method) could be
# sent to a multiprocessing.Pool without pickling — fork inherits the parent
# via memory copy, sidestepping the fact that ``HyperGridStates`` is a local
# class and therefore unpicklable.
#
# Fork has three serious problems in modern usage:
#   1. **MPI:** OpenMPI and Intel MPI both forbid ``fork()`` after MPI_Init.
#      Fork inside an MPI rank duplicates libfabric/UCX provider state and
#      causes silent corruption, hangs, or crashes.
#   2. **CUDA:** CUDA contexts cannot be forked.  Any fork inside a process
#      that has touched CUDA poisons the child.
#   3. **Threads:** POSIX only allows async-signal-safe operations between
#      ``fork()`` and ``exec()`` in a multi-threaded program.  Fork while
#      another thread holds an allocator lock can deadlock the child.
#
# We now require all multiprocessing entry points in this module to be
# picklable (``_hypergrid_worker`` is a module-level function, not a bound
# method) and we use spawn unconditionally.
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass  # Already set — don't override (e.g. Jupyter, pytest-xdist).

# Cap on the multiprocessing.Pool size used by ``_generate_combinations_in_batches``.
# Without a cap, ``Pool()`` defaults to ``os.cpu_count()`` workers — on a 64-core
# node hosting many co-located MPI ranks, that becomes ``ranks * 64`` worker
# processes spawned simultaneously, which can wedge the kernel scheduler.
_MAX_POOL_WORKERS = 8


def _hypergrid_worker(task):
    """Module-level worker for ``HyperGrid._generate_combinations_in_batches``.

    Returns the requested slice of the Cartesian product as a concrete
    ``list``.  Lives at module level (rather than as a bound method) so it
    can be pickled to a spawned ``multiprocessing.Pool`` worker — bound
    methods of ``HyperGrid`` are not picklable because the env's States
    subclass is created locally inside ``make_states_class``.

    Args:
        task: ``(values, ndim, start_idx, end_idx)`` where ``values`` is the
            list of coordinate values, ``ndim`` is the number of dimensions,
            and ``[start_idx, end_idx)`` is the index range within the full
            Cartesian product.

    Returns:
        A list of length ``end_idx - start_idx`` containing tuples of length
        ``ndim``.  Returning a concrete list (rather than an
        ``itertools.islice``) keeps the result picklable across workers and
        future-proofs against the Python 3.14 removal of itertools pickle
        support.
    """
    values, ndim, start_idx, end_idx = task
    return list(
        itertools.islice(itertools.product(values, repeat=ndim), start_idx, end_idx)
    )


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


def _state_hash_uniform(states_tensor: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic hash mapping each grid state to a float in [0, 1).

    Uses a polynomial rolling hash over coordinate values computed in int64
    arithmetic. Suitable for pseudo-random but reproducible per-state decisions
    (e.g., mode assignment, corruption masks).

    Args:
        states_tensor: (..., ndim) integer tensor of coordinates.
        seed: Integer seed for determinism.

    Returns:
        Tensor of shape states_tensor.shape[:-1] with values in [0.0, 1.0).
    """
    PRIME_A = 6364136223846793005  # Knuth LCG multiplier
    PRIME_B = 1442695040888963407  # Knuth LCG increment
    LARGE_PRIME = 2147483647  # 2^31 - 1 (Mersenne prime)
    h = torch.full(
        states_tensor.shape[:-1],
        seed,
        dtype=torch.int64,
        device=states_tensor.device,
    )
    for d in range(states_tensor.shape[-1]):
        h = h * PRIME_A + states_tensor[..., d].long() * PRIME_B
    return (h.abs() % LARGE_PRIME).float() / LARGE_PRIME


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
            # Compositional environments (see classes below)
            "bitwise_xor": BitwiseXORReward,
            "multiplicative_coprime": MultiplicativeCoprimeReward,
            "conditional_multiscale": ConditionalMultiScaleReward,
            # Random / corrupted environments
            "uniform_random": UniformRandomReward,
            "corrupted": CorruptedReward,
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
        # achieving the highest tier (plus head bonus for K-rule bitxor).
        if isinstance(self.reward_fn, BitwiseXORReward):
            r0 = float(self.reward_fn.R0)
            tw = self.reward_fn.tier_weights
            if len(tw) == 0:
                raise ValueError(
                    "BitwiseXORReward missing `tier_weights`; cannot derive mode threshold."
                )
            return r0 + float(sum(tw)) + float(self.reward_fn.head_weight)
        if isinstance(self.reward_fn, MultiplicativeCoprimeReward):
            r0 = float(self.reward_fn.R0)
            tw = self.reward_fn.tier_weights
            if len(tw) == 0:
                raise ValueError(
                    "MultiplicativeCoprimeReward missing `tier_weights`; cannot derive mode threshold."
                )
            return r0 + float(sum(tw))

        # UniformRandomReward: mode threshold is R0 + R_mode.
        if isinstance(self.reward_fn, UniformRandomReward):
            return self.reward_fn.R0 + self.reward_fn.R_mode

        # CorruptedReward: delegate to base reward's threshold.
        if isinstance(self.reward_fn, CorruptedReward):
            return self.reward_fn.mode_threshold()

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
        # ``get_states_indices`` returns either a torch.Tensor (small grids
        # whose indices fit in int64, i.e. height**ndim <= 2**63) or a numpy
        # object array of Python ints (larger grids where height**ndim > 2**63).
        # Handle both so the mode tracker stays correct in either regime.
        if isinstance(indices, np.ndarray):
            return set(indices[mask.cpu().numpy()].tolist())
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

    def get_states_indices(
        self, states: Union[DiscreteStates, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Get the canonical ordering indices for a batch of states.

        Returns one canonical index per state computed from the base-``height``
        encoding ``sum(s[j] * height^(ndim-1-j))``.  The maximum index is
        ``height^ndim - 1``.

        - **Safe regime** (``height ** ndim <= 2 ** 63``): the index fits
          in signed int64 and we return a ``torch.Tensor`` of shape
          ``batch_shape`` with dtype ``torch.int64`` (the historical behaviour).
        - **Overflow regime** (``height ** ndim > 2 ** 63``): the index
          would overflow int64 and silently wrap, producing collisions between
          distinct states (a real bug we hit at e.g. ndim=10, height=128 where
          ``128**10 == 2**70``).  In this regime we fall back to per-row Python
          ``int`` arithmetic and return a ``numpy.ndarray`` of dtype ``object``
          containing arbitrary-precision Python ints.  Each element is a
          unique, hashable canonical index.

        The two return types support the same downstream usages we care about
        (``set(...tolist())`` for mode tracking, boolean masking with
        ``[mask]`` after converting the mask to numpy if needed).  Code paths
        that need an ``int64`` tensor for tensor indexing (e.g.
        ``EnumPreprocessor``) implicitly require the safe regime — they'll see
        the numpy fallback and fail loudly, which is the correct behavior
        because such grids are too large to enumerate anyway.

        Args:
            states: The states to get the indices of.

        Returns:
            Indices in canonical ordering. ``torch.Tensor[int64]`` of shape
            ``batch_shape`` in the safe regime; ``np.ndarray[object]`` of shape
            ``batch_shape`` containing Python ints in the overflow regime.
        """
        if isinstance(states, DiscreteStates):
            states_raw = states.tensor
        else:
            states_raw = states

        # Exact overflow guard using Python's arbitrary-precision integers.
        # The int64 path is safe iff height**ndim <= 2**63, which guarantees
        # that both the per-column products (height^k * s_k) and the running
        # sum stay within signed int64.  This scalar check is negligible in
        # cost regardless of batch size.
        if self.height**self.ndim > 2**63:
            indices_obj = self._get_states_indices_bigint(states_raw)
            expected_shape = (
                tuple(states.batch_shape)
                if isinstance(states, DiscreteStates)
                else tuple(states_raw.shape[:-1])
            )
            assert (
                indices_obj.shape == expected_shape
            ), f"indices.shape is {indices_obj.shape} but expected {expected_shape}"
            return indices_obj

        canonical_base = self.height ** torch.arange(
            self.ndim - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        if isinstance(states, DiscreteStates):
            assert (
                indices.shape == states.batch_shape
            ), f"indices.shape is {indices.shape} and states.batch_shape is {states.batch_shape}"
        else:
            assert (
                indices.shape == states.shape[:-1]
            ), f"indices.shape is {indices.shape} but expected {states.shape[:-1]}"
        return indices

    def _get_states_indices_bigint(self, states_raw: torch.Tensor) -> np.ndarray:
        """Compute canonical indices using arbitrary-precision Python ints.

        Used by :meth:`get_states_indices` when ``height ** ndim > 2 ** 63``
        and the int64 path would overflow.

        Vectorized over the (potentially large) batch dimension via numpy
        object-dtype broadcasting: the inner Python loop iterates only over
        the small feature dimension ``ndim``, and each ``k * h + col``
        operation dispatches a single C-level loop over all rows that calls
        Python ``int.__mul__`` / ``int.__add__`` per element.  This is a few
        times faster than a nested Python loop while still preserving
        arbitrary-precision correctness.

        Returns a numpy ``object`` array of shape ``states_raw.shape[:-1]``
        containing one Python ``int`` per state.
        """
        batch_shape = tuple(states_raw.shape[:-1])
        # Convert the whole (n_rows, ndim) block to object dtype once so each
        # column slice we read below is already a Python-int array.
        flat = states_raw.reshape(-1, self.ndim).cpu().numpy().astype(object)
        h = int(self.height)

        # k starts as a Python-int 0 for every row (np.zeros with object
        # dtype fills with int(0)).
        k = np.zeros(flat.shape[0], dtype=object)
        for j in range(self.ndim):
            k = k * h + flat[:, j]
        return k.reshape(batch_shape)

    def get_terminating_states_indices(
        self, states: DiscreteStates
    ) -> Union[torch.Tensor, np.ndarray]:
        """Get the indices of the terminating states in the canonical ordering.

        See :meth:`get_states_indices` for the return-type contract: a
        ``torch.Tensor[int64]`` for grids small enough to fit in 62 bits, or a
        ``numpy.ndarray[object]`` of Python ints for larger grids that would
        otherwise overflow.

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
        if isinstance(self.reward_fn, ConditionalMultiScaleReward):
            return self._exists_conditional_multiscale(thr)
        if isinstance(self.reward_fn, (UniformRandomReward, CorruptedReward)):
            return self._exists_random_or_corrupted(thr)
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
        """Deterministic feasibility check for ``BitwiseXORReward``.

        Builds the combined GF(2) system [trunk; selector→0; head_0]·b =
        [c_trunk; 0; c_head_0] for rule 0 and verifies consistency. This works
        uniformly for n_rules=1 (k_select=0, selector empty) and for K-rule
        (per-rule coverage was already verified at __init__ by
        _validate_rule_coverage; this re-checks rule 0 as a defense-in-depth).

        Feasibility of this combined GF(2) system is necessary and sufficient
        for a mode to exist; no random sampling is required. Presets use
        power-of-two heights so every feasible bit-assignment is a valid
        state (raw coord < height).
        """
        rf = self.reward_fn
        sel_c = torch.zeros(rf.k_select, dtype=torch.long)
        head_A_0 = rf._head_A_per_rule[0]
        head_c_0 = rf._head_c_per_rule[0]
        full_A = torch.cat([rf._full_A, rf._selector_matrix, head_A_0], dim=0)
        full_c = torch.cat([rf._full_c, sel_c, head_c_0], dim=0)
        if full_A.numel() == 0:
            return True
        return self._solve_gf2_has_solution(full_A, full_c)

    def _exists_multiplicative_coprime(self, thr: float) -> bool:
        """Number-theoretic constructive check for ``MultiplicativeCoprimeReward``.

        For each rule, factors the rule's target LCM over allowed primes,
        tries permutations of prime-to-active-dim assignments, and checks
        coprime + grid-bound + selector-match. Returns True iff at least one
        rule has a witness state whose selector maps back to that rule's
        index AND whose reward reaches the mode threshold.

        The reward shifts raw coords by +1 internally (raw 0 → internal 1),
        so witness states are constructed in raw space as ``p**exp - 1`` per
        active dim, with coprime pair checks evaluated on the post-shift
        internal values.

        At n_rules=1 the selector is trivially 0 and only rule 0 is tried,
        recovering the legacy behavior.
        """
        rf = self.reward_fn
        primes: list[int] = [int(p) for p in rf.primes]
        caps: list[int] = [int(c) for c in rf.exponent_caps]
        active = list(rf.active_dims)
        coprime_pairs = rf.coprime_pairs or []
        max_exponent = int(caps[-1])
        rule_targets = rf.rule_targets

        # No per-rule head targets → mode is determined by trunk's tier-T
        # target_lcms[-1] (or trivially the all-zeros state if both are None).
        # In this regime we ignore the selector match (head trivially passes
        # for every rule), so any state achieving the trunk LCM is a witness.
        if not rule_targets or all(t is None for t in rule_targets):
            trunk_target = rf.target_lcms[-1] if rf.target_lcms else None
            if trunk_target is None:
                x = torch.zeros(self.ndim, dtype=torch.long)
                r = float(rf(x.unsqueeze(0))[0])
                return r >= thr - EPS_REWARD_CMP
            # Factor trunk_target and search permutations like the per-rule
            # branch, but with no selector constraint.
            required: list[tuple[int, int]] = []
            remaining = int(trunk_target)
            for p in primes:
                exp = 0
                while remaining % p == 0:
                    remaining //= p
                    exp += 1
                if exp > 0:
                    if exp > max_exponent or (p**exp) > self.height:
                        return False
                    required.append((p, exp))
            if remaining != 1 or len(required) > len(active):
                return False
            for perm in itertools.permutations(active, len(required)):
                x = torch.zeros(self.ndim, dtype=torch.long)
                for (p, e), dim in zip(required, perm):
                    x[dim] = p**e - 1
                if int(x.max()) >= self.height:
                    continue
                coprime_ok = True
                for i, j in coprime_pairs:
                    if i >= len(active) or j >= len(active):
                        continue
                    vi = int(x[active[i]]) + 1
                    vj = int(x[active[j]]) + 1
                    if math.gcd(vi, vj) != 1:
                        coprime_ok = False
                        break
                if not coprime_ok:
                    continue
                r = float(rf(x.unsqueeze(0))[0])
                if r >= thr - EPS_REWARD_CMP:
                    return True
            return False

        for k, target in enumerate(rule_targets):
            if target is None:
                # Rule with no LCM constraint passes head trivially. Try
                # all-zeros and check selector.
                x = torch.zeros(self.ndim, dtype=torch.long)
                if int(rf._selector((x[active] + 1).unsqueeze(0))[0]) == k:
                    r = float(rf(x.unsqueeze(0))[0])
                    if r >= thr - EPS_REWARD_CMP:
                        return True
                continue

            # Factor target LCM over allowed primes.
            required: list[tuple[int, int]] = []
            remaining = int(target)
            for p in primes:
                exp = 0
                while remaining % p == 0:
                    remaining //= p
                    exp += 1
                if exp > 0:
                    if exp > max_exponent or (p**exp) > self.height:
                        break
                    required.append((p, exp))
            else:
                if remaining != 1 or len(required) > len(active):
                    continue
                # Try permutations of prime → dim assignment until one matches
                # the selector and passes coprime/grid constraints.
                for perm in itertools.permutations(active, len(required)):
                    x = torch.zeros(self.ndim, dtype=torch.long)
                    for (p, e), dim in zip(required, perm):
                        x[dim] = p**e - 1
                    if int(x.max()) >= self.height:
                        continue
                    # Coprime check on post-shift values.
                    coprime_ok = True
                    for i, j in coprime_pairs:
                        if i >= len(active) or j >= len(active):
                            continue
                        vi = int(x[active[i]]) + 1
                        vj = int(x[active[j]]) + 1
                        if math.gcd(vi, vj) != 1:
                            coprime_ok = False
                            break
                    if not coprime_ok:
                        continue
                    # Selector must map back to this rule.
                    if int(rf._selector((x[active] + 1).unsqueeze(0))[0]) != k:
                        continue
                    r = float(rf(x.unsqueeze(0))[0])
                    if r >= thr - EPS_REWARD_CMP:
                        return True
        return False

    def _exists_conditional_multiscale(self, thr: float) -> bool:
        """Constructive existence check for ConditionalMultiScaleReward.

        With filter_shift=[0,...,0] (default) the all-zeros state is always a
        mode: every per-tier filter passes 0 since (0 + 0) mod B = 0 < f. With
        non-zero filter_shift, we try a few "all-same-v" candidate states
        chosen so the MSD passes tier 0; one of them typically passes all
        deeper tiers when the per-rule shift_coeffs map zero lower digits to
        zero, leaving the constant filter_shift[t] as the only contribution.
        """
        rf = self.reward_fn
        with torch.no_grad():
            # Try all-same v over digit values that pass tier 0's filter.
            # MSD passes iff (msd + filter_shift[0]) mod B < f.
            base = rf.base
            f = rf.filter_width
            fs0 = rf.filter_shift[0] if hasattr(rf, "filter_shift") else 0
            msb_factor = base ** (rf.num_levels - 1)
            for msd in range(base):
                if (msd + fs0) % base >= f:
                    continue
                v = msd * msb_factor
                xs = torch.full(
                    (1, self.ndim), v, dtype=torch.long, device=torch.device("cpu")
                )
                r = rf(xs)[0].item()
                if r >= thr - EPS_REWARD_CMP:
                    return True
        return self._exists_fallback_random(thr)

    def _exists_random_or_corrupted(self, thr: float) -> bool:
        """Check for UniformRandomReward or CorruptedReward.

        For UniformRandomReward the probe budget is sized so that
        P(miss all modes | at least one mode exists) < 1e-9, using
        n = ceil(log(1e-9) / log(1 - mode_prob)).  For CorruptedReward a
        fixed budget of 10 000 is used (mode density is approximately
        preserved by the promotion/demotion calibration).

        A seeded generator derived from the reward seed and grid shape makes
        the result reproducible across calls with the same configuration.
        """
        # Cap at total states to avoid over-probing small grids.
        total_states = int(min(float(self.height) ** self.ndim, 2**53))

        # Derive a deterministic seed from reward config and grid shape.
        reward_seed = getattr(self.reward_fn, "seed", 42)
        gen_seed = (reward_seed * 1_000_003 + self.height * 31 + self.ndim) & (2**63 - 1)

        if isinstance(self.reward_fn, UniformRandomReward):
            effective_p = max(1e-15, self.reward_fn.mode_prob)
            # n so that (1 - p)^n < 1e-9  <=>  n > log(1e-9) / log(1-p)
            n_needed = math.ceil(math.log(1e-9) / math.log1p(-effective_p))
            n_probes = int(min(total_states, max(2048, n_needed)))
        elif isinstance(self.reward_fn, CorruptedReward):
            # Mode density is roughly preserved by demotion/promotion calibration.
            n_probes = int(min(total_states, max(10_000, 8 * self.ndim)))
        else:
            n_probes = int(min(total_states, 2048))

        gen = torch.Generator().manual_seed(gen_seed)
        with torch.no_grad():
            xs = torch.randint(
                0,
                self.height,
                (n_probes, self.ndim),
                generator=gen,
                device=torch.device("cpu"),
            )
            rr = self.reward_fn(xs)
            return bool((rr >= thr - EPS_REWARD_CMP).any().item())

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
    def _solve_gf2_witness(
        A: torch.Tensor, c: torch.Tensor, n_vars: int
    ) -> torch.Tensor | None:
        """Return a witness solution to A·b = c over GF(2), or None if none exists.

        b has length n_vars. A is reduced via Gaussian elimination; free
        variables are set to 0.
        """
        if A.numel() == 0:
            return torch.zeros(n_vars, dtype=torch.long)
        A = A.clone().detach().to(dtype=torch.uint8, device=torch.device("cpu")) & 1
        c = c.clone().detach().to(dtype=torch.uint8, device=torch.device("cpu")) & 1

        n_eq, n_vars_check = A.shape
        assert n_vars == n_vars_check
        pivot_cols: list[int] = []
        pivot_row = 0
        for col in range(n_vars):
            cand = None
            for r in range(pivot_row, n_eq):
                if A[r, col]:
                    cand = r
                    break
            if cand is None:
                continue
            if cand != pivot_row:
                A[[pivot_row, cand]] = A[[cand, pivot_row]]
                c[[pivot_row, cand]] = c[[cand, pivot_row]]
            for r in range(n_eq):
                if r != pivot_row and A[r, col]:
                    A[r] ^= A[pivot_row]
                    c[r] ^= c[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row == n_eq:
                break

        # Inconsistent if any zero row has nonzero RHS.
        for r in range(pivot_row, n_eq):
            if c[r]:
                return None

        b = torch.zeros(n_vars, dtype=torch.long)
        for r, col in enumerate(pivot_cols):
            b[col] = int(c[r])
        return b

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

    def _generate_combinations_in_batches(
        self, ndim: int, max_val: int, batch_size: int
    ):
        """Yield batches of the Cartesian product {0, ..., max_val}^ndim.

        Uses multiprocessing to avoid materializing the full product
        (size ``(max_val+1)^ndim``) in memory.

        Workers are created via the spawn start method and execute the
        module-level :func:`_hypergrid_worker` function so the call is safe
        inside MPI ranks and CUDA contexts (see the start-method comment near
        the top of this file).  Pool size is capped at ``MAX_POOL_WORKERS``
        because larger pools just multiply per-rank fork/spawn overhead
        without shrinking the per-task work — and a 64-core node hosting many
        co-located MPI ranks can otherwise blow up to thousands of worker
        processes simultaneously.

        Args:
            ndim: Number of dimensions (tuple length).
            max_val: Maximum coordinate value (inclusive).
            batch_size: Number of tuples per batch.

        Yields:
            A list of tuples for each batch.
        """
        values = list(range(max_val + 1))
        total_combinations = (max_val + 1) ** ndim
        tasks = [
            (values, ndim, i, min(i + batch_size, total_combinations))
            for i in range(0, total_combinations, batch_size)
        ]

        n_workers = min(_MAX_POOL_WORKERS, max(1, (os.cpu_count() or 1)))
        with multiprocessing.Pool(processes=n_workers) as pool:
            for result in pool.imap(_hypergrid_worker, tasks):
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

    Curriculum motivation — rule reuse:
        This reward tests whether a GFlowNet can learn a global algebraic rule
        (GF(2) parity) and reuse it across tiers of increasing strictness. Unlike
        the other compositional rewards, modes are NOT spatially concentrated near
        the origin — they are distributed non-locally across the grid according to
        algebraic structure. This is intentional: it probes the model's ability to
        learn abstract, non-spatial compositionality.

        The curriculum operates through constraint accumulation: tier 0 applies few
        parity checks (many modes, easy to discover), tier 1 adds more checks
        (fewer modes, same rule type), etc. A model that learns the parity
        computation at tier 0 can reuse that same computation to satisfy tier 1+
        constraints, providing a form of compositional transfer for long-horizon
        credit assignment.

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

    K-rule structure (n_rules >= 1):
        - Trunk: the per-tier parity stack above, shared across all rules.
        - Selector: a fixed GF(2) matrix S of shape (k_select, M*B) projects
          bits to a rule index r = pack(S·b mod 2) ∈ [0, n_rules). Here
          k_select = ceil(log2(n_rules)); for n_rules=1, k_select=0 and r is
          always 0.
        - Head: per-rule parity matrix H_r of shape (head_check_count, M),
          applied at every bit-plane in head_bit_range. Mode iff trunk passes
          AND H_r·b == c_r at the head's bit-planes.

        Reward:
            R = R0 + Σ_t w_t·1[trunk_0..t pass]
                  + head_weight · 1[trunk all pass ∧ head_{σ(b)} pass]

        At n_rules=1 with head_check_count=0 and head_weight=0 (defaults), the
        head is empty and the K-rule code path collapses to the legacy reward
        bit-exactly.

        Total mode count is invariant in n_rules when it's a power of 2 — the
        selector adds k_select bits that partition the trunk-passing space,
        and per-rule head adds head_check_count·n_head_bits bits per coset.

    Comparison with other compositional rewards:
        - MultiplicativeCoprimeReward: number-theoretic (prime factorization);
          knowledge composition — learning prime structure enables coprimality
          and LCM constraints at higher tiers.
        - ConditionalMultiScaleReward: base-B digit decomposition with conditional
          constraints across scales; conditional hierarchy — coarse-scale structure
          predicts fine-scale constraints.
        - This class: GF(2) linear algebra on bit-planes; rule reuse — the same
          parity check type is applied with increasing strictness per tier.
          Modes are non-local (algebraic, not spatial).
    """

    _RULE_SEED_STRIDE: int = 1_000_003

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

        # Available bits for this height
        B = max(1, int(math.ceil(math.log2(max(height, 2)))))

        bits_per_tier = kwargs.get("bits_per_tier", None)
        if bits_per_tier is None:
            # Default: widen the bit window gradually
            bits_per_tier = [(0, 5), (0, 7), (0, 9)]
        assert len(bits_per_tier) == len(self.tier_weights)
        # Cap hi_bit to B - 1 to avoid vacuous checks
        self.bits_per_tier: list[tuple[int, int]] = [
            (int(lo), min(int(hi), B - 1)) for (lo, hi) in bits_per_tier
        ]

        self.parity_checks = kwargs.get("parity_checks", None)
        if self.parity_checks is not None:
            assert len(self.parity_checks) == len(self.tier_weights)

        # --- Pre-compute tensors for compile-friendly __call__ ---
        M = len(self.dims_constrained)
        self._dim_idx = torch.tensor(self.dims_constrained, dtype=torch.long)
        self._bit_positions = torch.arange(B, dtype=torch.long)

        # Build the full combined A matrix and c vector.
        # For each tier t and each bit b in its range, the tier's parity
        # check matrix A_t (n_checks_t, M) is placed into the full matrix
        # at columns corresponding to bit b of each constrained dim.
        # Dim-major bit ordering: index = d * B + b.
        full_A_rows: list[torch.Tensor] = []
        full_c_parts: list[torch.Tensor] = []
        tier_check_counts: list[int] = []

        for t in range(len(self.tier_weights)):
            lo_bit, hi_bit = self.bits_per_tier[t]
            n_bits_tier = hi_bit - lo_bit + 1
            if n_bits_tier <= 0:
                tier_check_counts.append(0)
                continue

            # Get per-tier A_t and c_t
            if self.parity_checks is not None and self.parity_checks[t] is not None:
                cfg = self.parity_checks[t]
                A_t = cfg.get("A", None)
                c_t = cfg.get("c", None)
                if A_t is None or c_t is None:
                    # Fall back to even parity
                    A_t = torch.ones(1, M, dtype=torch.long)
                    c_t = torch.zeros(1, dtype=torch.long)
                else:
                    A_t = A_t.long()
                    c_t = c_t.long()
            else:
                # Default even parity: sum(b) mod 2 == 0
                A_t = torch.ones(1, M, dtype=torch.long)
                c_t = torch.zeros(1, dtype=torch.long)

            n_checks = A_t.shape[0]

            # For each bit in this tier's range, place A_t into full
            # matrix at the appropriate columns.
            for b in range(lo_bit, hi_bit + 1):
                # Row in full matrix for this (tier, bit): A_t placed
                # at columns d * B + b for d in 0..M-1
                row_block = torch.zeros(n_checks, M * B, dtype=torch.long)
                for d in range(M):
                    row_block[:, d * B + b] = A_t[:, d]
                full_A_rows.append(row_block)
                full_c_parts.append(c_t)

            tier_check_counts.append(n_checks * n_bits_tier)

        if full_A_rows:
            self._full_A = torch.cat(full_A_rows, dim=0)
            self._full_c = torch.cat(full_c_parts, dim=0)
        else:
            self._full_A = torch.zeros(0, M * B, dtype=torch.long)
            self._full_c = torch.zeros(0, dtype=torch.long)

        self._tier_check_counts = tier_check_counts
        self._tier_weights_t = torch.tensor(
            self.tier_weights, dtype=torch.get_default_dtype()
        )
        self._B = B

        # --- Degeneracy detection ---
        if M < 2:
            logger.warning(
                "BitwiseXORReward: only %d constrained dim(s). GF(2) parity "
                "checks require M >= 2 for meaningful tier structure.",
                M,
            )

        # Check if all bit ranges collapse to the same range.
        unique_ranges = set(self.bits_per_tier)
        if len(unique_ranges) == 1 and len(self.tier_weights) > 1:
            lo, hi = next(iter(unique_ranges))
            if hi - lo < 1:
                logger.warning(
                    "BitwiseXORReward: all tiers use the same 1-bit range "
                    "[%d, %d]. Tier structure may be degenerate for height=%d.",
                    lo,
                    hi,
                    height,
                )

        # Check if cumulative checks saturate (>= M), killing all modes.
        cumulative = 0
        for t, n_chk in enumerate(tier_check_counts):
            cumulative += n_chk
            if cumulative >= M * (
                self.bits_per_tier[t][1] - self.bits_per_tier[t][0] + 1
            ):
                logger.warning(
                    "BitwiseXORReward: cumulative checks (%d) at tier %d "
                    "may saturate M=%d constrained dims, leaving very few "
                    "or no valid modes.",
                    cumulative,
                    t,
                    M,
                )
                break

        # --- K-rule selector + per-rule heads ---
        self.n_rules: int = int(kwargs.get("n_rules", 1))
        assert self.n_rules >= 1, f"n_rules must be >= 1, got {self.n_rules}"
        self.head_check_count: int = int(kwargs.get("head_check_count", 0))
        head_seed = kwargs.get("head_seed", None)
        # head_seed is required whenever K-rule structure is non-trivial:
        # either n_rules > 1 (selector exists) or head_check_count > 0 (head matrix).
        if head_seed is None and (self.n_rules > 1 or self.head_check_count > 0):
            raise ValueError(
                "head_seed must be provided when n_rules > 1 or "
                f"head_check_count > 0; got None (n_rules={self.n_rules}, "
                f"head_check_count={self.head_check_count})."
            )
        self.head_seed: int = int(head_seed) if head_seed is not None else 0
        self.head_weight: float = float(kwargs.get("head_weight", 0.0))
        if self.head_check_count == 0 and self.head_weight != 0.0:
            raise ValueError(
                "head_weight is non-zero but head_check_count=0; head will "
                "always pass and contribute trivially. Set head_check_count > 0."
            )
        head_bit_range = kwargs.get("head_bit_range", None)
        if head_bit_range is None:
            head_bit_range = (0, B - 1) if self.head_check_count > 0 else (0, -1)
        self.head_bit_range: tuple[int, int] = (
            int(head_bit_range[0]),
            int(head_bit_range[1]),
        )
        if self.head_check_count > 0:
            assert self.head_bit_range[0] <= self.head_bit_range[1], (
                f"head_bit_range={self.head_bit_range} is empty but "
                f"head_check_count={self.head_check_count} > 0."
            )
        self.k_select: int = (
            int(math.ceil(math.log2(self.n_rules))) if self.n_rules > 1 else 0
        )

        # Build selector matrix (shared across rules). Re-seed up to a few
        # times if the random selector accidentally aligns with the trunk
        # (rank deficient combined system). At M*B >= 16 + k_select this is
        # rare in practice, but assert independence loudly if it persists.
        if self.k_select > 0:
            trunk_rank_for_indep = (
                _gf2_rank(self._full_A) if self._full_A.numel() > 0 else 0
            )
            for attempt in range(8):
                S, _ = _gf2_random_fullrank(
                    self.k_select,
                    M * B,
                    self.head_seed + attempt * self._RULE_SEED_STRIDE,
                )
                S_long = S.long()
                if self._full_A.numel() == 0:
                    combined_rank = _gf2_rank(S_long)
                else:
                    combined_rank = _gf2_rank(
                        torch.cat([self._full_A, S_long], dim=0)
                    )
                if combined_rank == trunk_rank_for_indep + self.k_select:
                    break
            else:
                raise ValueError(
                    f"BitwiseXORReward: could not find a selector matrix "
                    f"linearly independent of the trunk after 8 attempts "
                    f"(head_seed={self.head_seed}). Trunk rank={trunk_rank_for_indep}, "
                    f"k_select={self.k_select}, M*B={M*B}."
                )
            self._selector_matrix = S_long
        else:
            self._selector_matrix = torch.zeros(0, M * B, dtype=torch.long)

        # Build head_A (shared across rules) and per-rule head_c. Sharing head_A
        # guarantees uniform per-rule mode count (rank is invariant in c). Per-
        # rule head_c is constructed from a witness b_rule that solves
        # trunk + selector→rule, so the combined system is consistent for every
        # rule by construction.
        lo_h, hi_h = self.head_bit_range
        n_head_bits = max(0, hi_h - lo_h + 1)
        r_head_total = self.head_check_count * n_head_bits
        self._head_c_per_rule = torch.zeros(
            self.n_rules, r_head_total, dtype=torch.long
        )
        if self.head_check_count > 0 and n_head_bits > 0:
            A_r, _ = _gf2_random_fullrank(
                self.head_check_count, M, self.head_seed + self._RULE_SEED_STRIDE
            )
            self._head_A = torch.zeros(r_head_total, M * B, dtype=torch.long)
            row = 0
            for bb in range(lo_h, hi_h + 1):
                for cd in range(M):
                    self._head_A[row : row + self.head_check_count, cd * B + bb] = (
                        A_r[:, cd].long()
                    )
                row += self.head_check_count
            # Per-rule head_c via witness construction.
            for rule in range(self.n_rules):
                sel_target = torch.tensor(
                    [(rule >> i) & 1 for i in range(self.k_select)], dtype=torch.long
                )
                witness_A = torch.cat([self._full_A, self._selector_matrix], dim=0)
                witness_c = torch.cat([self._full_c, sel_target], dim=0)
                b_rule = HyperGrid._solve_gf2_witness(witness_A, witness_c, M * B)
                if b_rule is None:
                    raise ValueError(
                        f"BitwiseXORReward: rule {rule} has no solution under "
                        f"trunk + selector. Cannot construct head_c. "
                        f"Reduce trunk strictness or change head_seed."
                    )
                self._head_c_per_rule[rule] = (self._head_A @ b_rule.long()) & 1
        else:
            self._head_A = torch.zeros(0, M * B, dtype=torch.long)

        # Stack head_A into per-rule shape (n_rules, r_head_total, M*B) for
        # batched gather in __call__. All rules share the same A matrix.
        self._head_A_per_rule = self._head_A.unsqueeze(0).expand(
            self.n_rules, -1, -1
        ).contiguous()

        # Powers of 2 for packing selector bits to rule_idx.
        self._select_powers = (
            torch.tensor([2**i for i in range(self.k_select)], dtype=torch.long)
            if self.k_select > 0
            else torch.zeros(0, dtype=torch.long)
        )

        # Run validator whenever K-rule structure is non-trivial. Covers the
        # K=1-with-head case too (single rule but witness construction must
        # have succeeded for the head_c[0] to be feasible).
        self._uniform_partition: bool = True
        if self.n_rules > 1 or self._head_A_per_rule.shape[1] > 0:
            self._validate_rule_coverage()

    def _validate_rule_coverage(self) -> None:
        """Ensure every rule index has >= 1 mode (trunk + selector→rule + head_rule).

        For each rule k, build the combined GF(2) system:
            [H_trunk; S; H_k] · b = [c_trunk; pack^-1(k); c_k]
        and verify it's consistent via _solve_gf2_has_solution. With k_select
        bits encoding the rule index, the selector contributes k_select scalar
        equations whose RHS is determined by the rule.

        Also tracks _uniform_partition: True iff n_rules is a power of 2 AND
        the combined trunk+selector matrix has rank r_trunk + k_select (i.e.
        the selector is independent of the trunk, so each rule is reachable
        with the same per-rule mode count).
        """
        trunk_A = self._full_A
        trunk_c = self._full_c
        S = self._selector_matrix
        for rule in range(self.n_rules):
            sel_bits = [(rule >> i) & 1 for i in range(self.k_select)]
            sel_c = torch.tensor(sel_bits, dtype=torch.long)
            head_A = self._head_A_per_rule[rule]
            head_c = self._head_c_per_rule[rule]
            full_A = torch.cat([trunk_A, S, head_A], dim=0)
            full_c = torch.cat([trunk_c, sel_c, head_c], dim=0)
            if not HyperGrid._solve_gf2_has_solution(full_A, full_c):
                raise ValueError(
                    f"BitwiseXORReward: rule {rule} has no solution "
                    f"(combined trunk + selector + head is inconsistent over GF(2)). "
                    f"Reduce head_check_count or change head_seed."
                )

        # Uniform partition iff every rule has the same per-rule mode count.
        # This subsumes power-of-2 + selector-rank-independence and also
        # catches incidental rank deficiency in head_k for some rules.
        per_rule = self._per_rule_mode_counts()
        self._uniform_partition = len(set(per_rule)) == 1
        if not self._uniform_partition:
            logger.warning(
                "BitwiseXORReward: K-rule partition is not uniform "
                "(per-rule mode counts: %s). "
                "analytic_mode_count(per_rule=True) will raise.",
                per_rule,
            )

    def analytic_mode_count(self, per_rule: bool = False) -> int:
        """Total mode count via per-rule GF(2) rank summation.

        For each rule k, modes_k = 2^(M·B − rank([trunk; selector; head_k])).
        Total = Σ_k modes_k. With random full-rank head matrices the per-rule
        ranks coincide and total = K · modes_0; for small or degenerate
        configurations they may differ. Multiplied by H^(ndim − M) for
        unconstrained dims.
        """
        M = len(self.dims_constrained)
        B = self._B
        n_vars = M * B
        unconstrained = self.height ** (self.ndim - M)
        per_rule_counts = self._per_rule_mode_counts()
        total = sum(per_rule_counts) * unconstrained

        if per_rule:
            if not self._uniform_partition:
                raise ValueError(
                    "analytic_mode_count(per_rule=True) requires a uniform "
                    "K-partition; per-rule mode counts differ."
                )
            return per_rule_counts[0] * unconstrained
        return total

    def _per_rule_mode_counts(self) -> list[int]:
        """Bit-config mode count per rule (without unconstrained-dim factor).

        For rule k, returns 2^(M·B − rank([trunk; selector; head_k])) when
        the combined system is consistent, else 0. Inconsistency means the
        rule is unreachable — its bit-config mode count is 0.
        """
        M = len(self.dims_constrained)
        B = self._B
        n_vars = M * B
        if self.n_rules == 1 and self._head_A_per_rule.shape[1] == 0:
            if self._full_A.numel() == 0:
                return [2**n_vars]
            if not HyperGrid._solve_gf2_has_solution(self._full_A, self._full_c):
                return [0]
            return [2 ** (n_vars - _gf2_rank(self._full_A))]
        counts = []
        for r in range(self.n_rules):
            sel_target = torch.tensor(
                [(r >> i) & 1 for i in range(self.k_select)], dtype=torch.long
            )
            A = torch.cat(
                [self._full_A, self._selector_matrix, self._head_A_per_rule[r]],
                dim=0,
            )
            c = torch.cat(
                [self._full_c, sel_target, self._head_c_per_rule[r]], dim=0
            )
            if A.numel() == 0:
                counts.append(2**n_vars)
                continue
            if not HyperGrid._solve_gf2_has_solution(A, c):
                counts.append(0)
                continue
            counts.append(2 ** (n_vars - _gf2_rank(A)))
        return counts

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

    def tier_indicators(self, states_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Per-tier independent pass/fail indicators.

        Returns a list of boolean tensors (one per tier), each of shape
        ``states_tensor.shape[:-1]``. ``indicators[t]`` is True for states
        that satisfy tier t's GF(2) parity constraints *independently*
        (not cumulatively).
        """
        dev = states_tensor.device
        dim_idx = self._dim_idx.to(dev)
        bit_positions = self._bit_positions.to(dev)
        full_A = self._full_A.to(dev)
        full_c = self._full_c.to(dev)

        x = states_tensor.index_select(-1, dim_idx)
        all_bits = (x.unsqueeze(-1) >> bit_positions) & 1
        flat_bits = all_bits.reshape(*x.shape[:-1], -1).long()
        prod = (flat_bits @ full_A.t()) & 1

        indicators: list[torch.Tensor] = []
        offset = 0
        for n_chk in self._tier_check_counts:
            if n_chk > 0:
                slice_ok = (
                    prod[..., offset : offset + n_chk] == full_c[offset : offset + n_chk]
                ).all(-1)
                indicators.append(slice_ok)
            else:
                indicators.append(torch.ones(x.shape[:-1], device=dev, dtype=torch.bool))
            offset += n_chk
        return indicators

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        indicators = self.tier_indicators(states_tensor)
        dev = states_tensor.device
        batch_shape = states_tensor.shape[:-1]
        R = torch.full(batch_shape, self.R0, device=dev, dtype=torch.get_default_dtype())
        tier_ok = torch.ones(batch_shape, device=dev, dtype=torch.bool)
        for ind, w in zip(indicators, self.tier_weights):
            tier_ok = tier_ok & ind
            R = R + tier_ok.to(R.dtype) * w

        # K-rule head: selector projects bits to rule_idx (or 0 at K=1);
        # head_{rule_idx} adds an extra parity check that must also pass for
        # head_weight. No-op when head matrices are empty. Trunk-only
        # tier_indicators above leaves head outside CorruptedReward's view.
        if self._head_A_per_rule.shape[1] > 0:
            dim_idx = self._dim_idx.to(dev)
            bit_positions = self._bit_positions.to(dev)
            x = states_tensor.index_select(-1, dim_idx)
            all_bits = (x.unsqueeze(-1) >> bit_positions) & 1
            flat_bits = all_bits.reshape(*x.shape[:-1], -1).long()

            if self.k_select > 0:
                S = self._selector_matrix.to(dev)
                select_powers = self._select_powers.to(dev)
                sel_bits = (flat_bits @ S.t()) & 1
                rule_idx = (sel_bits * select_powers).sum(dim=-1) % self.n_rules
            else:
                rule_idx = torch.zeros(batch_shape, device=dev, dtype=torch.long)

            head_A = self._head_A_per_rule.to(dev)
            head_c = self._head_c_per_rule.to(dev)
            A_per_state = head_A[rule_idx]
            c_per_state = head_c[rule_idx]
            head_prod = (
                torch.einsum("...rj,...j->...r", A_per_state, flat_bits) & 1
            )
            head_ok = (head_prod == c_per_state).all(-1)

            R = R + (tier_ok & head_ok).to(R.dtype) * self.head_weight

        return R


class MultiplicativeCoprimeReward(GridReward):
    """Tiered reward based on prime-support and coprimality/lcm composition.

    Curriculum motivation — knowledge composition:
        This reward tests whether a GFlowNet can learn number-theoretic structure
        progressively: first discovering which coordinates factor over allowed
        primes (tier 0), then learning exponent bounds (tier 1), then cross-
        dimensional coprimality (tier 2), and finally global LCM targets (tier 3).

        Each tier builds on knowledge from prior tiers: learning prime
        factorization at tier 0 is prerequisite for reasoning about exponent caps
        at tier 1, which in turn enables the coprimality reasoning needed at
        tier 2. This tests compositional transfer where each level requires a
        qualitatively different type of constraint, not just more of the same.

        Coordinates are shifted by +1 internally (state 0 -> value 1) so that
        the origin is valid and short trajectories immediately encounter small
        prime-factorable numbers (2, 3, 4, 5, 6, ...), providing early training
        signal for long-horizon credit assignment.

    Each tier progressively adds new constraint types:
        - Tier 0: Prime support — coordinates must factor over allowed primes.
        - Tier 1+: Exponent caps — prime exponents bounded per tier.
        - coprime_start_tier+: Coprime pairs — cross-dimensional coupling.
        - target_lcms: LCM targets — global compositional constraint.

    Reward form:
        R(s) = R0 + Σ_t tier_weights[t] · 1[ constraints_0..t all satisfied ]

    Key kwargs:
        - R0: float, base reward (default 0.0)
        - tier_weights: list[float]
        - primes: list[int], e.g., [2,3,5,7,11]. Primes exceeding height are
          auto-filtered with a warning.
        - exponent_caps: list[int], same length as tier_weights. Cap for every prime
          at tier t (uniform cap across primes for simplicity). Auto-capped to
          floor(log_p(height)) for each prime p.
        - active_dims: Optional[list[int]]; constraints only apply to these dims
          (default: all dims). Other dims are ignored in constraints.
        - coprime_pairs: Optional[list[tuple[int,int]]]; indices relative to
          active_dims.
        - coprime_start_tier: int, first tier at which coprime constraints apply
          (default: 0, preserving backward compatibility).
        - target_lcms: Optional[list[int | None | str]]; per-tier target lcm
          across active dims. Use "auto" to derive from primes and exponent_caps.

    Notes:
    - Coordinates are shifted by +1 internally: state value 0 maps to reward
      value 1, making the origin (0,...,0) trivially valid.
    - Implementation removes primes up to the current tier cap and checks residue == 1.
      Exponent counts are accumulated to evaluate LCM targets.

    Comparison with other compositional rewards:
        - BitwiseXORReward: GF(2) parity checks on bit-planes; rule reuse —
          same parity check type with increasing strictness. Non-local modes.
        - ConditionalMultiScaleReward: base-B digit decomposition with conditional
          constraints across scales; conditional hierarchy — coarse-scale structure
          predicts fine-scale constraints.
        - This class: prime factorization with progressive constraint types
          (support -> caps -> coprimality -> LCM). Knowledge composition —
          each tier requires understanding the prior tier's structure.
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.0))
        tier_weights = kwargs.get("tier_weights", [1.0, 10.0, 100.0])
        assert isinstance(tier_weights, (list, tuple)) and len(tier_weights) > 0
        self.tier_weights: list[float] = [float(w) for w in tier_weights]

        # Max coordinate value after +1 shift is height (state value height-1 + 1).
        max_val = height

        primes = kwargs.get("primes", [2, 3, 5])
        assert isinstance(primes, (list, tuple)) and len(primes) > 0
        raw_primes = [int(p) for p in primes]
        # Auto-filter primes that exceed the representable range.
        self.primes: list[int] = [p for p in raw_primes if p <= max_val]
        if len(self.primes) < len(raw_primes):
            dropped = [p for p in raw_primes if p > max_val]
            logger.warning(
                "MultiplicativeCoprimeReward: primes %s exceed height=%d "
                "(max coord value after +1 shift = %d). Dropped.",
                dropped,
                height,
                max_val,
            )
        if len(self.primes) == 0:
            logger.warning(
                "MultiplicativeCoprimeReward: no usable primes for height=%d. "
                "All states will pass prime-support checks trivially.",
                height,
            )

        exponent_caps = kwargs.get("exponent_caps", [2] * len(self.tier_weights))
        assert len(exponent_caps) == len(self.tier_weights)
        # Auto-cap exponents to what's representable: floor(log_p(max_val)).
        self.exponent_caps: list[int] = []
        for c in exponent_caps:
            c_int = int(c)
            if self.primes:
                # The tightest limit is the smallest prime's max exponent.
                max_achievable = min(
                    int(math.floor(math.log(max_val) / math.log(p))) for p in self.primes
                )
                if c_int > max_achievable:
                    c_int = max_achievable
            self.exponent_caps.append(c_int)

        active_dims = kwargs.get("active_dims", None)
        if active_dims is None:
            active_dims = list(range(ndim))
        self.active_dims: list[int] = list(map(int, active_dims))

        self.coprime_pairs = kwargs.get("coprime_pairs", None)
        self.coprime_start_tier: int = int(kwargs.get("coprime_start_tier", 0))

        # Resolve target_lcms: support "auto" to derive from primes and caps.
        raw_lcms = kwargs.get("target_lcms", [None] * len(self.tier_weights))
        assert isinstance(raw_lcms, (list, tuple)) and len(raw_lcms) == len(
            self.tier_weights
        )
        self.target_lcms: list[int | None] = []
        for t, lcm_val in enumerate(raw_lcms):
            if lcm_val == "auto":
                if self.primes:
                    auto_lcm = 1
                    for p in self.primes:
                        auto_lcm *= p ** self.exponent_caps[t]
                    self.target_lcms.append(auto_lcm)
                else:
                    self.target_lcms.append(None)
            elif lcm_val is not None:
                self.target_lcms.append(int(lcm_val))
            else:
                self.target_lcms.append(None)

        # --- K-rule selector + per-rule top-tier LCM heads ---
        # Trunk = tiers 0..T-2 (existing behavior). Head = tier T-1, replaced
        # per-rule by rule_targets[selector(state)]. Three resolution paths:
        #   1. Explicit `rule_targets` kwarg → use as-is.
        #   2. Explicit `head_seed` kwarg → generate K targets from head_seed
        #      (this is the K-rule path; works for K=1 too).
        #   3. Neither → backward-compat: inherit from target_lcms[T-1] (for
        #      n_rules=1; n_rules>1 without head_seed already raised above).
        self.n_rules: int = int(kwargs.get("n_rules", 1))
        assert self.n_rules >= 1, f"n_rules must be >= 1, got {self.n_rules}"
        head_seed_provided = kwargs.get("head_seed", None) is not None
        if not head_seed_provided and self.n_rules > 1:
            raise ValueError(
                "head_seed must be provided when n_rules > 1; got None."
            )
        self.head_seed: int = (
            int(kwargs["head_seed"]) if head_seed_provided else 0
        )

        rule_targets = kwargs.get("rule_targets", None)
        rule_targets_inherited_from_trunk = False
        if rule_targets is None:
            if head_seed_provided:
                rule_targets = self._generate_rule_targets(
                    self.n_rules, self.head_seed
                )
            else:
                # Legacy K=1 backward-compat: head's single LCM mirrors the
                # trunk's top-tier LCM. Mark as inherited so we don't later
                # clear target_lcms[-1] (the redundancy is harmless and
                # preserving it keeps pre-refactor reward output bit-exact).
                rule_targets = [self.target_lcms[-1]]
                rule_targets_inherited_from_trunk = True
        assert len(rule_targets) == self.n_rules, (
            f"rule_targets has {len(rule_targets)} entries; expected n_rules={self.n_rules}"
        )
        self.rule_targets: list[int | None] = [
            int(t) if t is not None else None for t in rule_targets
        ]

        # Clear the shared trunk top-tier LCM only when the head replaces it
        # with non-trivial per-rule LCMs that *differ* from the trunk's.
        # - Legacy K=1 (no head_seed): rule_targets inherited from trunk —
        #   keep target_lcms[-1] (redundancy is harmless, bit-exact compat).
        # - K-rule with auto-gen or explicit rule_targets: clear to avoid the
        #   trunk LCM conflicting with rule-specific head LCMs (B1 fix).
        # - Selector-only K-rule (rule_targets=[None]*K): keep target_lcms[-1]
        #   so the trunk LCM still constrains modes.
        head_has_lcm = any(t is not None for t in self.rule_targets)
        if head_has_lcm and not rule_targets_inherited_from_trunk:
            self.target_lcms[-1] = None

        # Pre-factor each rule's target_lcm into prime exponents, for fast
        # per-state gather in __call__. Shape: (n_rules, n_primes).
        # Rule with target=None uses sentinel exponents of -1 (always passes).
        n_primes = len(self.primes)
        self._rule_target_exps = torch.full(
            (self.n_rules, n_primes), -1, dtype=torch.long
        )
        for r, T in enumerate(self.rule_targets):
            if T is None:
                continue
            remaining = int(T)
            for j, p in enumerate(self.primes):
                e = 0
                while remaining % p == 0:
                    remaining //= p
                    e += 1
                self._rule_target_exps[r, j] = e
            if remaining != 1:
                raise ValueError(
                    f"rule_targets[{r}]={T} contains primes outside the "
                    f"allowed set {self.primes}."
                )

        if self.n_rules > 1:
            self._validate_rule_coverage()

    def _generate_rule_targets(self, n_rules: int, seed: int) -> list[int]:
        """Generate n_rules distinct LCM targets from a deterministic enum.

        Enumerates cap-tuples (cap_p ∈ {1, ..., top_cap}) over allowed primes —
        cap=0 (prime absent from LCM) is excluded so every rule has a
        non-trivial target. Total tuples = top_cap^n_primes; permutes by seed
        and picks the first n_rules. The prime universe must satisfy
        top_cap^n_primes >= n_rules.

        Note: cap=0 is intentionally excluded. With cap=0, the LCM head
        constraint "no active dim has prime p as factor" combined with
        coprime-pair and exponent-cap constraints typically yields zero or
        near-zero modes — degenerate rules.
        """
        if not self.primes:
            raise ValueError(
                "Cannot generate rule_targets: no allowed primes for height."
            )
        n_primes = len(self.primes)
        top_cap = self.exponent_caps[-1]
        if top_cap < 1:
            raise ValueError(
                f"Cannot generate rule_targets: top exponent_cap={top_cap} < 1."
            )
        n_cap_values = top_cap  # caps in {1, ..., top_cap}
        n_total = n_cap_values**n_primes
        if n_total < n_rules:
            raise ValueError(
                f"Cannot generate {n_rules} distinct LCMs with {n_primes} primes "
                f"and cap_p ∈ {{1..{top_cap}}}: {n_cap_values}^{n_primes}={n_total}."
            )
        all_tuples: list[int] = []
        for idx in range(n_total):
            rem = idx
            caps: list[int] = []
            for _ in range(n_primes):
                caps.append((rem % n_cap_values) + 1)  # shift to {1..top_cap}
                rem //= n_cap_values
            lcm = 1
            for p, c in zip(self.primes, caps):
                lcm *= p**c
            all_tuples.append(lcm)
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=gen).tolist()
        return [all_tuples[perm[i]] for i in range(n_rules)]

    def _validate_rule_coverage(self) -> None:
        """Ensure each rule's LCM target is achievable.

        A target is achievable iff (a) it factors over allowed primes (already
        checked at __init__), and (b) every required exponent is <= top cap.
        Both conditions are necessary; with active_dims >= n_primes_with_nonzero_exp
        and coprime_pairs constraints, sufficiency requires enumeration.
        Here we check only the necessary conditions and verify that the rules
        produce non-degenerate distinct targets.
        """
        top_cap = self.exponent_caps[-1]
        seen = set()
        for r, exps in enumerate(self._rule_target_exps.tolist()):
            if all(e == -1 for e in exps):
                continue
            for j, e in enumerate(exps):
                if e > top_cap:
                    raise ValueError(
                        f"rule_targets[{r}] requires exponent {e} for prime "
                        f"{self.primes[j]}, exceeding top cap {top_cap}."
                    )
            tup = tuple(exps)
            seen.add(tup)
        if len(seen) < self.n_rules:
            logger.warning(
                "MultiplicativeCoprimeReward: only %d distinct rule targets "
                "for n_rules=%d; some rules collide (selector partition "
                "remains valid; mode-set is reduced).",
                len(seen),
                self.n_rules,
            )

    def _selector(self, x_active: torch.Tensor) -> torch.Tensor:
        """Map state's active-dim values to a rule index in [0, n_rules).

        Selector is sum_i x_i mod n_rules (over active dims, shifted values).
        Returns shape (...,) long tensor.
        """
        return x_active.sum(dim=-1) % self.n_rules

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

    def tier_indicators(self, states_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Per-tier independent pass/fail indicators.

        Returns a list of boolean tensors (one per tier), each of shape
        ``states_tensor.shape[:-1]``. ``indicators[t]`` is True for states
        that satisfy tier t's constraints *independently* (not cumulatively).
        """
        x = (
            states_tensor.index_select(
                dim=-1,
                index=torch.tensor(self.active_dims, device=states_tensor.device),
            )
            + 1
        )

        # K-rule selector: rule index per state. At n_rules=1 with a None
        # per-rule target this is trivially 0 and head_lcm_ok is uniformly True.
        rule_idx = self._selector(x)
        rule_target_exps = self._rule_target_exps.to(x.device)
        T_top = len(self.tier_weights) - 1

        indicators: list[torch.Tensor] = []
        for t in range(len(self.tier_weights)):
            cap = self.exponent_caps[t]
            residue, exps = self._factor_exponents_up_to_cap(x.reshape(-1), cap)
            residue = residue.reshape(x.shape)
            exps = exps.reshape((len(self.primes),) + x.shape)

            support_ok = (residue == 1).all(dim=-1)

            if t >= self.coprime_start_tier:
                pairs_ok = self._pairwise_coprime_ok(x)
            else:
                pairs_ok = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)

            lcm_ok = torch.ones_like(pairs_ok)
            lcm_target = self.target_lcms[t]
            if lcm_target is not None:
                lcm_ok = self._lcm_ok(exps, lcm_target)

            # K-rule top-tier per-rule LCM head: active when n_rules > 1, or
            # n_rules=1 with a non-None rule_targets[0]. Folded into lcm_ok so
            # CorruptedReward sees the K-rule-modulated indicator at this tier.
            if t == T_top:
                target_per_state = rule_target_exps[rule_idx]  # (..., n_primes)
                # Sentinel -1 means "no LCM constraint for this rule".
                no_target = (target_per_state == -1).all(dim=-1)
                max_exp = exps.max(dim=-1).values  # (n_primes, ...)
                max_exp_perm = torch.movedim(max_exp, 0, -1)  # (..., n_primes)
                head_match = (max_exp_perm == target_per_state).all(dim=-1)
                head_lcm_ok = no_target | head_match
                lcm_ok = lcm_ok & head_lcm_ok

            indicators.append(support_ok & pairs_ok & lcm_ok)
        return indicators

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        indicators = self.tier_indicators(states_tensor)
        dev = states_tensor.device
        batch_shape = states_tensor.shape[:-1]
        R = torch.full(
            batch_shape,
            self.R0,
            device=dev,
            dtype=torch.get_default_dtype(),
        )
        tier_ok = torch.ones(batch_shape, device=dev, dtype=torch.bool)
        for ind, w in zip(indicators, self.tier_weights):
            tier_ok = tier_ok & ind
            R = R + tier_ok.to(R.dtype) * float(w)
        return R


class ConditionalMultiScaleReward(GridReward):
    """Tiered reward via conditional digit constraints across spatial scales.

    Curriculum motivation — conditional hierarchy:
        This reward tests whether a GFlowNet can learn hierarchical, conditional
        structure: each tier's constraint depends on what was learned at prior
        tiers, creating the strongest form of compositional transfer among the
        three reward types.

        Digit ordering is coarse-to-fine: tier 0 constrains the most significant
        digit (coarsest spatial scale), tier 1 constrains the next digit
        conditioned on the coarse digit, and so on. This creates natural
        distance-correlated difficulty: states near the origin have small
        coordinates (high digits are 0, trivially passing coarse filters), while
        states far from the origin have nonzero high digits that must satisfy the
        filter. Learning coarse-scale structure first provides early training
        signal and directly informs which fine-scale configurations are valid,
        enabling compositional transfer for long-horizon credit assignment.

    Each coordinate is decomposed in base B into L = log_B(H) digits. Tier t
    constrains digit (L-1-t) — the (t+1)-th most significant digit — via a
    shifted filter that depends on all coarser-scale digits already constrained,
    creating a hierarchy where learning coarse-scale structure is prerequisite
    for predicting fine-scale constraints.

    Per-dimension constraint at tier t (0-indexed):
        (d_{L-1-t}(i) + sigma_t(i; r)) mod B < f

    where sigma_t(i; r) = sum_{k=0}^{t-1} a_{t,k}^{(r)} * d_{L-1-k}(i) mod B is a
    linear function of coarser-scale digits with seed-derived coefficients,
    parameterized by the rule index r. Tier 0 has no shift (sigma_0 = 0) and
    is shared across all rules — it forms the trunk.

    K-rule structure (n_rules >= 1):
        - Tier 0 is the shared trunk: constrains the most-significant digit per
          active dim via filter [0, f). Same across all rules.
        - The selector projects each state to a rule index r ∈ [0, n_rules)
          deterministically: r(s) = packed_MSD(s) mod n_rules where
          packed_MSD = sum_i d_{L-1, i} * base^i across active dims.
        - Tiers >= 1 use rule-specific shift coefficients derived from
          (head_seed, r), so each rule has a different head.

        At n_rules=1, the selector always returns 0 and there is one head, with
        coefficients derived from `seed` — bit-exact reproduction of the
        single-rule reward.

    Optional cross-dimensional constraint at tier t (applies to all rules):
        sum_i d_{L-1-t}(i) ≡ 0  (mod m_t)

    Reward form (cumulative — tier t requires all tiers 0..t under the rule):
        R(s) = R0 + sum_t tier_weights[t] * 1[s satisfies tiers 0..t]

    Mode count (closed form, total across all rules):
        Without cross-dim:  modes_T = (f^T)^d * B^{(L-T)*d}
        With cross-dim:     modes_T = (f^T)^d * B^{(L-T)*d} / prod_t m_t

    The total mode count is INVARIANT in n_rules: rules partition the canonical
    mode set. When n_rules divides f^d_active, the partition is uniform and
    modes_per_rule = total / n_rules.

    Partition function (analytic, no enumeration):
        Z = R0 * H^d + sum_t w_t * modes_t

    Key kwargs:
        - R0: float, base reward (default 0.0).
        - tier_weights: list[float], reward weight per tier.
        - base: int, digit base B (default 4). H must be a power of B.
        - filter_width: int, number of passing digit values per tier (default B//2).
          Constant across tiers to avoid mode collapse at deep tiers.
        - seed: int, PRNG seed for generating shift coefficients (default 42).
        - n_rules: int, number of rules K (default 1). Selector partitions
          tier-0-passing states into K buckets; uniform partition requires
          K | f^d_active.
        - head_seed: int, PRNG seed for per-rule head shift coefficients
          (default: same as seed; ensures n_rules=1 reproduces single-rule).
        - cross_dim_mods: Optional[list[int|None]], per-tier modular cross-dim
          constraint. m_t must divide filter_width for exact mode counts.
          Default: no cross-dim constraints.
        - active_dims: Optional[list[int]], subset of dims to constrain
          (default: all dims).

    Comparison with other compositional rewards:
        - BitwiseXORReward: GF(2) parity checks on bit-planes; rule reuse —
          same parity check type with increasing strictness. Non-local modes.
        - MultiplicativeCoprimeReward: prime factorization with progressive
          constraint types; knowledge composition — each tier requires
          understanding the prior tier's structure.
        - This class: conditional hierarchy — each tier introduces a constraint
          whose form depends on what was learned at prior tiers. Coarse-to-fine
          ordering creates distance-correlated difficulty.
    """

    # Stride between per-rule RNG seeds; large prime to decorrelate rules.
    _RULE_SEED_STRIDE: int = 1_000_003

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

        # Per-tier constant filter shift. Default None → 0 at every tier
        # (current behavior: tier-t passing window starts at digit=0). Setting
        # filter_shift[t]=c moves tier t's passing window to start at digit
        # (-c) mod B — i.e., the blind strip cyclically rotates. Useful for
        # spreading mode coverage across the full axis instead of clumping
        # in the lower f/B fraction.
        filter_shift = kwargs.get("filter_shift", None)
        if filter_shift is None:
            filter_shift = [0] * len(self.tier_weights)
        assert len(filter_shift) == len(self.tier_weights), (
            f"filter_shift length {len(filter_shift)} != n_tiers {len(self.tier_weights)}"
        )
        self.filter_shift: list[int] = [int(s) % self.base for s in filter_shift]

        active_dims = kwargs.get("active_dims", None)
        if active_dims is None:
            active_dims = list(range(ndim))
        self.active_dims: list[int] = list(map(int, active_dims))

        # K-rule structure. n_rules=1 reproduces the original single-rule reward
        # bit-exactly: rule 0's RNG is seeded by head_seed (defaulting to seed),
        # which matches the historical generator state.
        self.n_rules: int = int(kwargs.get("n_rules", 1))
        assert self.n_rules >= 1, f"n_rules must be >= 1, got {self.n_rules}"
        head_seed = kwargs.get("head_seed", self.seed)
        if head_seed is None:
            raise ValueError("head_seed=None is not allowed; pass an int or omit.")
        self.head_seed: int = int(head_seed)

        # Generate per-rule shift coefficients.
        # shift_coeffs_per_rule[r][t] has length t (empty list for t=0).
        # Coefficient a_{t,k}^{(r)} references digit (L-1-k) — a coarser digit
        # already constrained by a prior tier — for rule r.
        self.shift_coeffs_per_rule: list[list[list[int]]] = []
        for r in range(self.n_rules):
            rule_seed = (
                self.head_seed if r == 0 else self.head_seed + r * self._RULE_SEED_STRIDE
            )
            rng = torch.Generator().manual_seed(rule_seed)
            rule_shifts: list[list[int]] = []
            for t in range(len(self.tier_weights)):
                if t == 0:
                    rule_shifts.append([])
                else:
                    coeffs = torch.randint(0, self.base, (t,), generator=rng).tolist()
                    rule_shifts.append(coeffs)
            self.shift_coeffs_per_rule.append(rule_shifts)

        # Pre-pack rule shift coefficients as a (n_rules, n_tiers, n_tiers) tensor
        # for fast per-state gather in __call__. Padded with zeros where j >= t.
        n_tiers = len(self.tier_weights)
        shift_tensor = torch.zeros(self.n_rules, n_tiers, n_tiers, dtype=torch.long)
        for r in range(self.n_rules):
            for t in range(n_tiers):
                for j_idx, c in enumerate(self.shift_coeffs_per_rule[r][t]):
                    shift_tensor[r, t, j_idx] = c
        self._shift_coeffs_tensor: torch.Tensor = shift_tensor

        # Cross-dimensional modular constraints (optional, applied to all rules)
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

        # Construction-time check for K-rule structure: every rule must be
        # represented by at least one trunk-passing pattern. Enumerates the
        # f^d_active tier-0-passing patterns (small: e.g. 64 at f=2, d=6).
        self._uniform_partition: bool = True
        if self.n_rules > 1:
            self._validate_rule_coverage()

    def _validate_rule_coverage(self) -> None:
        """Ensure every rule index is hit by at least one trunk-passing state.

        Trunk = tier 0 = each active dim's shifted MSD in [0, filter_width)
        AND (if cross_dim_mods[0] is set) MSD-sum mod m_0 == 0. The selector
        packs MSDs as base-f and mods by n_rules.

        Implementation uses cyclic-uniformity: trunk-passing MSD patterns map
        bijectively to integers [0, f^d_active) via the base-f packing, so
        `pat mod n_rules` distributes uniformly when `n_rules | n_surv`. The
        cross-dim filter at tier 0 thins this set further; for non-trivial
        cross_mod_0 we sample to verify coverage. (Old enumeration over
        f^d_active patterns was intractable at d_active > ~10.)

        Sets `_uniform_partition` based on whether n_rules divides the
        trunk-passing count evenly.
        """
        d_active = len(self.active_dims)
        f = self.filter_width
        cross_mod_0 = self.cross_dim_mods[0]

        n_patterns = f**d_active  # tier-0-filter-passing patterns
        if cross_mod_0 is None:
            n_surv = n_patterns
            uniform_after_cross = True
        else:
            # Each cross_mod_0 is a divisor of f, and the sum-mod-m constraint
            # picks 1/m of the patterns symmetrically. Hence n_surv = n_patterns / m.
            assert n_patterns % cross_mod_0 == 0, (
                f"f^d_active={n_patterns} not divisible by cross_dim_mods[0]={cross_mod_0}"
            )
            n_surv = n_patterns // cross_mod_0
            uniform_after_cross = True

        if self.n_rules > n_surv:
            raise ValueError(
                f"n_rules={self.n_rules} exceeds trunk-passing pattern count "
                f"{n_surv} (f^d_active={f}^{d_active}={n_patterns}, "
                f"cross_dim_mods[0]={cross_mod_0}); some rules cannot be reached."
            )

        # With cyclic uniformity (no cross_dim_mods[0]), or symmetric cross-dim
        # reduction by a divisor of f, the bucket distribution is determined
        # by n_surv mod n_rules. All buckets non-empty iff n_surv >= n_rules,
        # which we already verified. Uniform iff n_surv % n_rules == 0.
        # If cross_mod_0 happens to interact pathologically with the selector,
        # we'd miss it here — fall back to enumeration only at small d_active.
        ENUM_THRESHOLD = 12
        if d_active <= ENUM_THRESHOLD:
            bucket_counts = [0] * self.n_rules
            survivors_iter = range(n_patterns) if cross_mod_0 is None else (
                pat for pat in range(n_patterns)
                if sum((pat // (f**i)) % f for i in range(d_active)) % cross_mod_0 == 0
            )
            for pat in survivors_iter:
                bucket_counts[pat % self.n_rules] += 1
            empty = [k for k, c in enumerate(bucket_counts) if c == 0]
            if empty:
                raise ValueError(
                    f"K-rule selector leaves rules {empty} with no trunk-passing "
                    f"states (n_rules={self.n_rules}, trunk-passing patterns="
                    f"{n_surv}). Loosen cross_dim_mods[0] or reduce n_rules."
                )
            self._uniform_partition: bool = (
                n_surv % self.n_rules == 0 and len(set(bucket_counts)) == 1
            )
        else:
            # Above threshold: rely on cyclic-uniformity. n_rules ≤ n_surv
            # already verified; uniform partition iff n_rules divides n_surv.
            self._uniform_partition = (n_surv % self.n_rules == 0)

        if not self._uniform_partition:
            logger.warning(
                "ConditionalMultiScaleReward: n_rules=%d does not yield a "
                "uniform partition over %d trunk-passing patterns. Rule mode "
                "counts will be imbalanced; analytic_mode_count(per_rule=True) "
                "is no longer well-defined and will raise.",
                self.n_rules,
                n_surv,
            )

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

    def _selector(self, msd_digits: torch.Tensor) -> torch.Tensor:
        """Map state MSDs (..., d_active) -> rule index in [0, n_rules).

        Applies the same shift used by tier 0's filter (sigma_0 = 0 +
        filter_shift[0]) before packing as base-f. For trunk-passing states
        the shifted MSD is in [0, f) so the packing is bijective on the
        trunk-passing set; non-trunk-passing states still get a rule index
        but cannot become modes (their tier-0 check fails).
        """
        d_active = msd_digits.shape[-1]
        f = self.filter_width
        fs0 = self.filter_shift[0]
        shifted = (msd_digits + fs0) % self.base
        powers = f ** torch.arange(d_active, device=msd_digits.device, dtype=torch.long)
        packed = ((shifted % f) * powers).sum(dim=-1)
        return packed % self.n_rules

    def tier_indicators(self, states_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Per-tier independent pass/fail indicators.

        Returns a list of boolean tensors (one per tier), each of shape
        ``states_tensor.shape[:-1]``. ``indicators[t]`` is True for states
        that satisfy tier t's digit constraint *independently*
        (not cumulatively). Note: the shift at tier t still depends on
        coarser digits, so the constraint is state-dependent but evaluated
        per-tier.
        """
        x = states_tensor.index_select(
            dim=-1,
            index=torch.tensor(self.active_dims, device=states_tensor.device),
        ).long()

        L = self.num_levels
        digits = self._extract_digits(x, L)

        # K-rule selector + per-rule shift coefficients. At n_rules=1 with
        # zero filter_shift this collapses to the legacy single-rule behavior.
        rule_idx = self._selector(digits[L - 1])  # shape: x.shape[:-1]
        shift_coeffs_t = self._shift_coeffs_tensor.to(x.device)  # (K, n_tiers, n_tiers)

        indicators: list[torch.Tensor] = []
        for t in range(len(self.tier_weights)):
            # Coarse-to-fine: tier t constrains digit (L-1-t).
            target_digit = digits[L - 1 - t]

            # Tier 0 is the shared trunk (no state-dependent shift). Tiers ≥ 1
            # use rule-specific state-dependent shifts gathered by rule_idx.
            # An optional per-tier constant filter_shift[t] is added on top.
            if t == 0:
                shift = torch.full_like(x, self.filter_shift[t])
            else:
                # tier_coeffs_all: (K, t) — rule k's coefficients a_{t,0..t-1}.
                tier_coeffs_all = shift_coeffs_t[:, t, :t]
                # per_state_coeffs: (..., t) — broadcast-gather over rule_idx.
                per_state_coeffs = tier_coeffs_all[rule_idx]
                shift = torch.full_like(x, self.filter_shift[t])
                for k_idx in range(t):
                    a = per_state_coeffs[..., k_idx].unsqueeze(-1)  # (..., 1)
                    shift = shift + a * digits[L - 1 - k_idx]
                shift = shift % self.base

            shifted = (target_digit + shift) % self.base
            per_dim_ok = (shifted < self.filter_width).all(dim=-1)

            cross_ok = torch.ones_like(per_dim_ok)
            cross_mod = self.cross_dim_mods[t]
            if cross_mod is not None:
                digit_sum = target_digit.sum(dim=-1)
                cross_ok = (digit_sum % int(cross_mod)) == 0

            indicators.append(per_dim_ok & cross_ok)
        return indicators

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        indicators = self.tier_indicators(states_tensor)
        dev = states_tensor.device
        batch_shape = states_tensor.shape[:-1]
        R = torch.full(
            batch_shape,
            self.R0,
            device=dev,
            dtype=torch.get_default_dtype(),
        )
        tier_ok = torch.ones(batch_shape, device=dev, dtype=torch.bool)
        for ind, w in zip(indicators, self.tier_weights):
            tier_ok = tier_ok & ind
            R = R + tier_ok.to(R.dtype) * float(w)
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

    def analytic_mode_count(
        self, tier: int | None = None, per_rule: bool = False
    ) -> int:
        """Compute exact mode count for a given tier (1-indexed) or all tiers.

        Total mode count is invariant in `n_rules` — rules partition the
        canonical mode set rather than multiplying it.

        Args:
            tier: 1-indexed tier number. If None, returns count for the
                  highest tier (most constrained).
            per_rule: If True, returns the per-rule count (total // n_rules).
                Requires uniform partition (n_rules divides the trunk-passing
                pattern count); raises ValueError otherwise.

        Returns:
            Number of states satisfying all constraints up to the given tier
            (over all rules combined if per_rule=False, per rule otherwise).
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
        # extended over inactive dims (which are unconstrained, so B^L each).
        choices_per_coord = (f**T) * (B ** (L - T))
        modes = choices_per_coord**d
        # Inactive dims contribute B^L = self.height each.
        n_inactive = self.ndim - d
        modes *= self.height**n_inactive

        # Cross-dim modular constraints divide out uniformly.
        for t in range(T):
            cross_mod = self.cross_dim_mods[t]
            if cross_mod is not None:
                modes //= cross_mod

        if per_rule:
            if not self._uniform_partition:
                raise ValueError(
                    f"analytic_mode_count(per_rule=True) requires a uniform "
                    f"K-partition; n_rules={self.n_rules} does not partition "
                    f"the trunk-passing set evenly."
                )
            return modes // self.n_rules
        return modes

    def analytic_log_partition(self) -> float:
        """Compute log(Z) analytically.

        Z = R0 * H^ndim + sum_t w_t * modes_t
        """
        Z = self.R0 * (self.height**self.ndim)
        for t in range(len(self.tier_weights)):
            Z += self.tier_weights[t] * self.analytic_mode_count(tier=t + 1)
        return log(Z) if Z > 0 else float("-inf")


# ----------------------------------
# Random / corrupted reward classes
# ----------------------------------


class UniformRandomReward(GridReward):
    """Each state is independently a mode with probability ``mode_prob``.

    Uses a deterministic hash on state coordinates so mode membership is
    reproducible without storing or enumerating all states. There is no
    exploitable spatial or algebraic structure.

    Reward form::

        R(s) = R0 + R_mode   if hash(s, seed) < mode_prob
        R(s) = R0             otherwise

    Key kwargs:
        - R0: float, base reward for non-mode states (default 0.1).
        - R_mode: float, additional reward for mode states (default 2.0).
        - mode_prob: float in (0, 1), probability each state is a mode
          (default 0.01).
        - seed: int, hash seed for reproducibility (default 42).
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.1))
        self.R_mode: float = float(kwargs.get("R_mode", 2.0))
        self.mode_prob: float = float(kwargs.get("mode_prob", 0.01))
        self.seed: int = int(kwargs.get("seed", 42))
        assert (
            0 < self.mode_prob < 1
        ), f"mode_prob must be in (0, 1), got {self.mode_prob}"
        assert self.R_mode > 0, f"R_mode must be positive, got {self.R_mode}"

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        h = _state_hash_uniform(states_tensor, self.seed)
        is_mode = h < self.mode_prob
        base = torch.full(
            states_tensor.shape[:-1],
            self.R0,
            device=states_tensor.device,
            dtype=torch.get_default_dtype(),
        )
        return base + is_mode.to(base.dtype) * self.R_mode


class CorruptedReward(GridReward):
    """Wraps a tiered structured reward and applies per-tier corruption.

    Conceptually, at each tier, a fraction ``corruption_rate`` of states that
    earned that tier's bonus have it "moved" to a random location. This
    degrades the compositional structure at every level proportionally.

    Per-tier corruption logic:
        For each tier *t* and each state *s*:

        1. Compute the base reward's per-tier indicator ``pass_t(s)``.
        2. **Demote**: if ``pass_t(s)`` and ``hash(s, seed + 2*t) < corruption_rate``,
           remove tier *t*'s contribution.
        3. **Promote**: if not ``pass_t(s)`` and
           ``hash(s, seed + 2*t + 1) < replacement_rate_t``, add tier *t*'s
           contribution. ``replacement_rate_t`` is calibrated at init so that
           the expected number of promotions matches demotions.

    Final reward::

        R(s) = R0 + sum_t w_t * corrupted_pass_t(s)

    For non-tiered base rewards, falls back to a single-level binary
    corruption at the mode threshold.

    Key kwargs:
        - base_reward: str, name of the base reward (default
          "conditional_multiscale").
        - base_kwargs: dict, kwargs for the base reward constructor.
        - corruption_rate: float in [0, 1), fraction of tier-passing states
          to demote per tier (default 0.2).
        - seed: int, hash seed (default 137).
    """

    # Mapping from string names to reward classes. Kept in sync with
    # the HyperGrid.reward_functions dict but excludes self-referential
    # entries to prevent recursive wrapping.
    _REWARD_CLASSES: dict[str, type[GridReward]] = {}

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        base_reward_str: str = str(kwargs.get("base_reward", "conditional_multiscale"))
        base_kwargs: dict = dict(kwargs.get("base_kwargs", {}))
        self.corruption_rate: float = float(kwargs.get("corruption_rate", 0.2))
        self.seed: int = int(kwargs.get("seed", 137))

        assert (
            0 <= self.corruption_rate < 1
        ), f"corruption_rate must be in [0, 1), got {self.corruption_rate}"

        # Lazily populate the class-level mapping on first use.
        if not CorruptedReward._REWARD_CLASSES:
            CorruptedReward._REWARD_CLASSES = {
                "original": OriginalReward,
                "cosine": CosineReward,
                "sparse": SparseReward,
                "deceptive": DeceptiveReward,
                "bitwise_xor": BitwiseXORReward,
                "multiplicative_coprime": MultiplicativeCoprimeReward,
                "conditional_multiscale": ConditionalMultiScaleReward,
                "uniform_random": UniformRandomReward,
            }

        assert base_reward_str in self._REWARD_CLASSES, (
            f"Unknown base_reward: {base_reward_str}. "
            f"Must be one of {list(self._REWARD_CLASSES.keys())}"
        )
        self.base_fn: GridReward = self._REWARD_CLASSES[base_reward_str](
            height, ndim, **base_kwargs
        )
        self.base_reward_str = base_reward_str
        self._is_tiered = hasattr(self.base_fn, "tier_indicators")

        # Estimate per-tier replacement rates so E[promotions] ~ E[demotions].
        self._replacement_rates: list[float] = []
        if self._is_tiered and self.corruption_rate > 0:
            self._estimate_replacement_rates()

    def _estimate_replacement_rates(self) -> None:
        """Sample states to estimate per-tier pass fraction, then set
        replacement rates so promotions ~ demotions in expectation."""
        n_samples = min(50_000, max(4096, self.height**self.ndim))
        with torch.no_grad():
            xs = torch.randint(
                0, self.height, (n_samples, self.ndim), device=torch.device("cpu")
            )
            indicators = self.base_fn.tier_indicators(xs)  # type: ignore[attr-defined]
            for ind in indicators:
                frac = float(ind.float().mean().item())
                if frac > 0 and frac < 1:
                    # E[demoted] = frac * corruption_rate * N
                    # E[promoted] = (1 - frac) * replacement_rate * N
                    # Set equal: replacement_rate = corruption_rate * frac / (1 - frac)
                    rate = self.corruption_rate * frac / (1.0 - frac)
                    self._replacement_rates.append(min(rate, 1.0))
                elif frac == 0:
                    # No states pass → nothing to demote → no replacement needed
                    self._replacement_rates.append(0.0)
                else:
                    # All states pass → demotions happen but no non-passing
                    # states to promote. Use corruption_rate as fallback.
                    self._replacement_rates.append(self.corruption_rate)
        assert len(self._replacement_rates) == len(indicators)

    def mode_threshold(self) -> float:
        """Return the mode threshold derived from the base reward."""
        bf = self.base_fn
        if isinstance(bf, ConditionalMultiScaleReward):
            return bf.mode_threshold()
        if hasattr(bf, "tier_weights") and hasattr(bf, "R0"):
            return float(bf.R0) + float(sum(bf.tier_weights))  # type: ignore[attr-defined]
        if isinstance(bf, UniformRandomReward):
            return bf.R0 + bf.R_mode
        # Fallback for simple rewards.
        r0 = float(bf.kwargs.get("R0", 0.0))
        r1 = float(bf.kwargs.get("R1", 0.0))
        r2 = float(bf.kwargs.get("R2", 0.0))
        return r0 + r1 + r2

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        if not self._is_tiered:
            return self._call_simple(states_tensor)

        indicators = self.base_fn.tier_indicators(states_tensor)  # type: ignore[attr-defined]
        dev = states_tensor.device
        batch_shape = states_tensor.shape[:-1]
        r0 = float(getattr(self.base_fn, "R0", self.base_fn.kwargs.get("R0", 0.0)))
        R = torch.full(batch_shape, r0, device=dev, dtype=torch.get_default_dtype())
        tw = getattr(self.base_fn, "tier_weights", [])

        tier_ok = torch.ones(batch_shape, device=dev, dtype=torch.bool)
        for t, (ind, w) in enumerate(zip(indicators, tw)):
            if self.corruption_rate > 0:
                h_demote = _state_hash_uniform(states_tensor, self.seed + 2 * t)
                h_promote = _state_hash_uniform(states_tensor, self.seed + 2 * t + 1)
                demote = ind & (h_demote < self.corruption_rate)
                repl_rate = (
                    self._replacement_rates[t]
                    if t < len(self._replacement_rates)
                    else 0.0
                )
                promote = (~ind) & (h_promote < repl_rate)
                corrupted_ind = (ind & ~demote) | promote
            else:
                corrupted_ind = ind
            tier_ok = tier_ok & corrupted_ind
            R = R + tier_ok.to(R.dtype) * float(w)
        return R

    def _call_simple(self, states_tensor: torch.Tensor) -> torch.Tensor:
        """Fallback for non-tiered base rewards: binary corruption."""
        base_reward = self.base_fn(states_tensor)
        if self.corruption_rate == 0:
            return base_reward

        thr = self.mode_threshold()
        is_base_mode = base_reward >= thr
        h_demote = _state_hash_uniform(states_tensor, self.seed)

        # Estimate replacement rate from mode fraction.
        frac = float(is_base_mode.float().mean().item())
        if 0 < frac < 1:
            repl_rate = min(self.corruption_rate * frac / (1.0 - frac), 1.0)
        else:
            repl_rate = 0.0

        h_promote = _state_hash_uniform(states_tensor, self.seed + 1)
        demote = is_base_mode & (h_demote < self.corruption_rate)
        promote = (~is_base_mode) & (h_promote < repl_rate)

        r0 = float(getattr(self.base_fn, "R0", self.base_fn.kwargs.get("R0", 0.0)))
        mode_r = thr  # mode-level reward for promoted states
        result = base_reward.clone()
        result[demote] = r0
        result[promote] = mode_r
        return result


# -------------------------
# Difficulty preset factories
# -------------------------


def _first_k_dims(k: int, ndim: int) -> list[int]:
    """Return indices [0, 1, ..., min(k, ndim)-1] for the first k dimensions."""
    k = min(max(1, k), ndim)
    return list(range(k))


def _gf2_random_fullrank(
    n_checks: int, n_vars: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a full-rank random binary matrix A and target vector c
    over GF(2).

    Uses a deterministic seed for reproducibility across runs.

    Args:
        n_checks: Number of independent GF(2) equations (rows of A).
        n_vars: Number of binary variables (columns of A).
        seed: Deterministic RNG seed.

    Returns:
        (A, c) where A is (n_checks, n_vars) int tensor and c is
        (n_checks,) int tensor, both with values in {0, 1}. A is
        guaranteed to have full row-rank over GF(2).
    """
    assert 0 < n_checks <= n_vars
    gen = torch.Generator().manual_seed(seed)

    for _ in range(1000):
        A = torch.randint(0, 2, (n_checks, n_vars), generator=gen, dtype=torch.long)
        # Verify full row-rank over GF(2) via row echelon reduction
        if _gf2_rank(A) == n_checks:
            c = torch.randint(0, 2, (n_checks,), generator=gen, dtype=torch.long)
            return A, c

    raise RuntimeError(
        f"Failed to generate full-rank GF(2) matrix "
        f"({n_checks}x{n_vars}) after 1000 attempts"
    )


def _gf2_rank(A: torch.Tensor) -> int:
    """Compute the rank of a binary matrix over GF(2)."""
    A = A.clone().to(dtype=torch.uint8) & 1
    n_rows, n_cols = A.shape
    pivot_row = 0
    for col in range(n_cols):
        found = None
        for r in range(pivot_row, n_rows):
            if A[r, col]:
                found = r
                break
        if found is None:
            continue
        if found != pivot_row:
            A[[pivot_row, found]] = A[[found, pivot_row]]
        for r in range(pivot_row + 1, n_rows):
            if A[r, col]:
                A[r] ^= A[pivot_row]
        pivot_row += 1
        if pivot_row == n_rows:
            break
    return pivot_row


def _preset_seed(name: str) -> int:
    """Deterministic seed from a preset name."""
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**31)


def get_bitwise_xor_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for BitwiseXORReward.

    Difficulty is controlled by the number of constrained dimensions M.
    More constrained dims means more independent GF(2) checks per bit
    position, leading to fewer modes.

    Each preset uses 3 tiers with increasing numbers of GF(2) checks:
      - Tier 0 (curriculum): few checks, many states pass
      - Tier 1 (intermediate): moderate checks
      - Tier 2 (mode): strictest checks, defines the modes

    Mode counts for ndim=10, height=16 (B=4 bit-planes). Per-bit-position
    solutions = 2^(M − cum_top_checks); raised to the B-th power, then
    multiplied by 16^(ndim − M) free-dim configurations:
      - easy (M=3, cum=2):        ~4.3B modes   (16 × 16^7)
      - medium (M=5, cum=4):      ~16.8M modes  (16 × 16^5)
      - hard (M=8, cum=6):        ~65K modes    (256 × 16^2)
      - challenging (M=10, cum=7): ~4K modes    (4096 × 1)
      - impossible (M=12→10, cum=9): 16 modes   (16 × 1; M capped to ndim)

    Notes
    - Uses fixed seeds per preset name for reproducibility.
    - Parity checks are random full-rank GF(2) matrices.
    - Bit ranges are capped to ceil(log2(height)) - 1.
    """
    B = max(1, int(math.ceil(math.log2(max(height, 2)))))
    max_bit = B - 1

    def _make_preset(
        name: str,
        M_target: int,
        checks_per_tier: list[int],
    ) -> dict:
        M = min(M_target, ndim)
        if M < M_target:
            logger.info(
                "BitwiseXOR preset '%s': M_target=%d capped to ndim=%d.",
                name,
                M_target,
                ndim,
            )
        dims = _first_k_dims(M, ndim)
        seed = _preset_seed(name)

        n_tiers = len(checks_per_tier)
        # Geometric tier weights
        tier_weights = [10.0**i for i in range(n_tiers)]
        # All tiers use the full bit range
        bits_per_tier = [(0, max_bit)] * n_tiers

        parity_checks: list[dict | None] = []
        # All tiers' checks apply at the same bit positions, so the combined
        # GF(2) system must be consistent.  A None entry defaults to 1 even-
        # parity check inside BitwiseXORReward, so we must count it.  We cap
        # the cumulative checks (including defaults) to M - 1 so that at
        # least 2 solutions survive per bit position.
        cumulative_checks = 0
        for t in range(n_tiers):
            budget = max(0, M - 1 - cumulative_checks)
            n_chk = min(checks_per_tier[t], M - 1)
            if budget <= 0:
                # No room left — skip this tier entirely (no constraint).
                # Use an explicit zero-row parity check so we don't fall
                # back to the default even-parity (which costs 1 check).
                A_empty = torch.zeros(0, M, dtype=torch.long)
                c_empty = torch.zeros(0, dtype=torch.long)
                parity_checks.append({"A": A_empty, "c": c_empty})
            elif n_chk <= 0:
                # Fall back to default even parity (1 check).
                parity_checks.append(None)
                cumulative_checks += 1
            else:
                n_chk = min(n_chk, budget)
                tier_seed = seed + t * 1000
                A, c = _gf2_random_fullrank(n_chk, M, tier_seed)
                parity_checks.append({"A": A, "c": c})
                cumulative_checks += n_chk

        return dict(
            R0=0.0,
            tier_weights=tier_weights,
            dims_constrained=dims,
            bits_per_tier=bits_per_tier,
            parity_checks=parity_checks,
        )

    presets = {
        "easy": _make_preset("easy", 3, [1, 1, 1]),
        "medium": _make_preset("medium", 5, [1, 2, 2]),
        "hard": _make_preset("hard", 8, [1, 2, 3]),
        "challenging": _make_preset("challenging", 10, [1, 2, 4]),
        "impossible": _make_preset("impossible", 12, [2, 4, 6]),
    }

    # K-rule trunk+heads presets (designed for ndim=10, H=16). Density ~9.5e-7
    # is invariant across K — selector partitions trunk-and-head-passing states.
    # Trunk: M=10 (all dims), checks_per_tier=[1,1,2] -> 4 cumulative trunk
    # checks per bit-plane * 4 bit-planes = 16 trunk checks total.
    # Head: 1 check per bit-plane * 4 bit-planes = 4 head rows per rule.
    # Selector: ceil(log2(K)) bits over the full M*B=40 bit space.
    # Mode count: 2^(40 - 16 - 4) = 2^20 = ~1M total, invariant in K.
    if ndim >= 10:
        k_trunk = _make_preset("Ktrunk", 10, [1, 1, 2])
        for n_rules in (1, 16, 64):
            presets[f"K{n_rules}"] = dict(
                k_trunk,
                n_rules=n_rules,
                head_seed=2025,
                head_weight=1000.0,
                head_check_count=1,
                head_bit_range=(0, max_bit),
            )

        # "Matched" K-presets: production-grade matched-density (~1e-7) with
        # K=64 specialization at ndim=10, h=16. Trunk same as standard K-presets
        # (16 rows); head extended to 2 checks across 3 bit-planes (r_head=6)
        # for richer per-rule heads at slightly tighter density (2.4e-7 vs the
        # standard preset's 9.5e-7). T_max=150, parity-symmetric coverage.
        # Total modes: 2^(40 - 16 - 6) = 2^18 = 262K, density 2.4e-7.
        k_trunk_matched = _make_preset("Ktrunk_matched", 10, [1, 1, 2])
        for n_rules in (1, 16, 64):
            presets[f"K{n_rules}_matched"] = dict(
                k_trunk_matched,
                n_rules=n_rules,
                head_seed=2025,
                head_weight=1000.0,
                head_check_count=2,
                head_bit_range=(0, 2),
            )

    return presets


def get_multiplicative_coprime_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for MultiplicativeCoprimeReward.

    Each preset uses progressive tier structure where each tier adds a new
    constraint type:
      - Tier 0: Prime support only (coords must factor over allowed primes)
      - Tier 1: + Exponent caps (tighten factorization)
      - coprime_start_tier+: + Coprime pairs (cross-dim coupling)
      - Final tier: + LCM target (global compositional constraint)

    Coordinates are shifted +1 internally (origin -> all-ones), so short
    trajectories immediately encounter small prime-factorable numbers.

    Primes exceeding height and exponent caps exceeding log_p(height) are
    auto-filtered/capped in the reward constructor.

    Notes
    - `active_dims` indexes are relative to state dims; we pick first k.
    - `coprime_pairs` are pairs within `active_dims` index space.
    - Tier weights are geometric.
    - Use target_lcms="auto" to derive from primes and exponent_caps.
    """

    def chain_pairs(k: int) -> list[tuple[int, int]]:
        """Return adjacent-index coprime pairs: (0,1), (1,2), ..., (k-2, k-1)."""
        return [(i, i + 1) for i in range(max(0, k - 1))]

    presets = {
        # 3 tiers: support -> caps -> caps (coprime at tier 2)
        "easy": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0],
            primes=[2, 3, 5],
            exponent_caps=[2, 2, 2],
            active_dims=_first_k_dims(3, ndim),
            coprime_pairs=chain_pairs(3),
            coprime_start_tier=2,
            target_lcms=[None, None, None],
        ),
        # 4 tiers: support -> caps -> coprime -> LCM(auto)
        "medium": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7],
            exponent_caps=[2, 2, 2, 2],
            active_dims=_first_k_dims(5, ndim),
            coprime_pairs=chain_pairs(5),
            coprime_start_tier=2,
            target_lcms=[None, None, None, "auto"],
        ),
        # 4 tiers: support -> caps -> coprime -> LCM(auto)
        "hard": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7, 11],
            exponent_caps=[3, 3, 3, 3],
            active_dims=_first_k_dims(8, ndim),
            coprime_pairs=chain_pairs(8),
            coprime_start_tier=2,
            target_lcms=[None, None, None, "auto"],
        ),
        # 4 tiers: support -> caps -> coprime -> LCM(auto)
        "challenging": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7, 11, 13],
            exponent_caps=[3, 3, 4, 4],
            active_dims=_first_k_dims(10, ndim),
            coprime_pairs=chain_pairs(10),
            coprime_start_tier=2,
            target_lcms=[None, None, None, "auto"],
        ),
        # 5 tiers: support -> caps -> coprime -> LCM(auto) -> LCM(auto)
        "impossible": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0, 10000.0],
            primes=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            exponent_caps=[4, 4, 4, 4, 4],
            active_dims=_first_k_dims(12, ndim),
            coprime_pairs=chain_pairs(12),
            coprime_start_tier=2,
            target_lcms=[None, None, None, None, "auto"],
        ),
    }

    # K-rule trunk+heads presets (designed for ndim=6, H=64). Trunk = tiers 0..2
    # (smooth-number + caps + coprime pairs). Head = tier 3, with per-rule LCM
    # target enumerated from cap-tuples. Primes={2,3,5,7} all representable at
    # cap=2 in h=64; cap_p ∈ {1, 2} gives 2^4 = 16 distinct LCMs.
    #
    # K=64 is not supported at h=64 because it would require either (a) a
    # 6-prime universe where 11 and 13 force auto-cap to 1, collapsing the
    # enum, or (b) cap=3 which isn't representable for primes 5 and 7 at h=64.
    # K=64 coprime is left as a TODO; users can pick h=256 with 6 primes.
    # K=1 draws rule 0 from the same enum so density matches K=16.
    if ndim >= 6:
        k_trunk = dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7],
            exponent_caps=[2, 2, 2, 2],
            active_dims=_first_k_dims(6, ndim),
            coprime_pairs=chain_pairs(6),
            coprime_start_tier=2,
            target_lcms=[None, None, None, None],
        )
        for n_rules in (1, 16):
            presets[f"K{n_rules}"] = dict(
                k_trunk,
                n_rules=n_rules,
                head_seed=2025,
            )

    # "Matched" K-presets: production-grade matched-density (~1e-7) with
    # K=64 specialization at ndim=10, h=64. Selector-only K-rule because the
    # cap-tuple LCM enum tops out at K=16 with 4 primes; K=64 needs h>=169 for
    # 6-prime caps, which makes T_max prohibitive. The trunk holds the LCM=auto
    # constraint (44100), and the selector partitions the same canonical mode
    # set into 64 buckets — rules differ in spatial assignment, not per-rule
    # head LCM. T_max=630, smooth-set spans 1–64 per dim (no origin clumping).
    if ndim >= 10:
        k_trunk_matched = dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7],
            exponent_caps=[2, 2, 2, 2],
            active_dims=_first_k_dims(10, ndim),
            coprime_pairs=chain_pairs(10),
            coprime_start_tier=2,
            target_lcms=[None, None, None, "auto"],
        )
        for n_rules in (1, 16, 64):
            presets[f"K{n_rules}_matched"] = dict(
                k_trunk_matched,
                n_rules=n_rules,
                head_seed=2025,
                rule_targets=[None] * n_rules,
            )

    return presets


def get_conditional_multiscale_presets(ndim: int, height: int) -> dict:
    """Return difficulty presets for ConditionalMultiScaleReward.

    All presets use base=4 (requiring H to be a power of 4). The number of
    available digit levels is L = log_4(H). Difficulty is controlled by:
      - Number of tiers (more tiers = deeper compositional hierarchy)
      - Number of active dims (more = exponentially sparser modes)
      - Cross-dim modular constraints (further sparsification)

    Mode counts are computed via:
        modes_T = (f^T * 4^{L-T})^d * H^(ndim-d) / prod_t m_t

    with f = filter_width = 2 (i.e. B//2), so each tier halves modes per coord.

    Digit ordering is coarse-to-fine: tier 0 constrains the most significant
    digit. Near the origin (small coordinates), high digits are 0 and trivially
    pass the filter. Deeper tiers constrain progressively finer digits,
    creating distance-correlated difficulty.

    Two preset families:

    Difficulty presets (single-rule legacy, assuming H=256, i.e. L=4):
      - easy:        2 tiers, 3 active dims -> ~2M modes at tier 2
      - medium:      3 tiers, 4 active dims -> ~1M modes at tier 3
      - hard:        3 tiers, 6 active dims, cross-dim -> ~260K modes at tier 3
      - challenging: 4 tiers, 8 active dims, cross-dim -> ~65K modes at tier 4
      - impossible:  4 tiers, 12 active dims, cross-dim -> ~4K modes at tier 4

    K-rule trunk+heads presets (sparsity matched, K rules sharing tier-0 trunk):
      - K1, K16, K64: 3 tiers, 6 active dims, cross_dim_mods=[2,2,2].
        Total modes invariant in K (~32K total at H=64, density ~5e-7); rules
        partition the canonical mode set. Designed for ndim=6, H=64.

    If height provides fewer digit levels than a preset requires, the preset's
    tier_weights and cross_dim_mods are auto-truncated with a warning.
    """
    base = 4
    # Compute available digit levels for this height.
    num_levels = 0
    h = height
    while h > 1 and h % base == 0:
        h //= base
        num_levels += 1

    def _cap_tiers(preset: dict) -> dict:
        """Truncate tier_weights and cross_dim_mods if they exceed num_levels."""
        tw = preset["tier_weights"]
        if len(tw) > num_levels:
            logger.warning(
                "ConditionalMultiScale preset has %d tiers but height=%d "
                "(base=%d) only provides %d digit levels. "
                "Truncating to %d tiers.",
                len(tw),
                height,
                base,
                num_levels,
                num_levels,
            )
            preset["tier_weights"] = tw[:num_levels]
            cdm = preset.get("cross_dim_mods")
            if cdm is not None:
                preset["cross_dim_mods"] = cdm[:num_levels]
        return preset

    presets = {
        "easy": _cap_tiers(
            dict(
                R0=0.0,
                tier_weights=[1.0, 10.0],
                base=base,
                filter_width=2,
                seed=42,
                active_dims=_first_k_dims(3, ndim),
            )
        ),
        "medium": _cap_tiers(
            dict(
                R0=0.0,
                tier_weights=[1.0, 10.0, 100.0],
                base=base,
                filter_width=2,
                seed=42,
                active_dims=_first_k_dims(4, ndim),
            )
        ),
        "hard": _cap_tiers(
            dict(
                R0=0.0,
                tier_weights=[1.0, 10.0, 100.0],
                base=base,
                filter_width=2,
                seed=42,
                active_dims=_first_k_dims(6, ndim),
                cross_dim_mods=[None, 2, 2],
            )
        ),
        "challenging": _cap_tiers(
            dict(
                R0=0.0,
                tier_weights=[1.0, 10.0, 100.0, 1000.0],
                base=base,
                filter_width=2,
                seed=42,
                active_dims=_first_k_dims(8, ndim),
                cross_dim_mods=[None, None, 2, 2],
            )
        ),
        "impossible": _cap_tiers(
            dict(
                R0=0.0,
                tier_weights=[1.0, 10.0, 100.0, 1000.0],
                base=base,
                filter_width=2,
                seed=42,
                active_dims=_first_k_dims(12, ndim),
                cross_dim_mods=[None, 2, 2, 2],
            )
        ),
    }

    # K-rule trunk+heads presets (designed for ndim=6, H=64).
    # Density ~9.5e-7 (~65K modes) is invariant across K — rules partition the
    # canonical mode set. cross_dim_mods[0] is None: a tier-0 cross-dim filter
    # would shrink the trunk-passing set below n_rules and leave rules empty.
    # head_seed is shared across K presets so K=1's rule 0 == K=16's rule 0
    # == K=64's rule 0 by construction (useful for cross-K diagnostics).
    k_rule_template = dict(
        R0=0.0,
        tier_weights=[1.0, 10.0, 100.0],
        base=base,
        filter_width=2,
        seed=42,
        head_seed=2025,
        active_dims=_first_k_dims(6, ndim),
        cross_dim_mods=[None, 2, 2],
    )
    for n_rules in (1, 16, 64):
        presets[f"K{n_rules}"] = _cap_tiers(dict(k_rule_template, n_rules=n_rules))

    # "Matched" K-presets: production-grade matched-density (~1e-7) with K=64
    # specialization at ndim=24, h=16 (base=4, L=2, f=3). Trades dim count for
    # trajectory length: T_max=360 (vs 882 at ndim=14, h=64). 75% per-axis
    # coverage via filter_width=3 over base=4 — 3-of-4 top digits pass tier 0,
    # so each axis has only a 25% blind strip (cyclic via filter_shift if you
    # want it elsewhere). cross_dim_mods=[3,3] tightens density without
    # collapsing the partition. K_native = 3^24 ≈ 282B; K=64 is non-uniform
    # (3^23 not divisible by 64) but every rule is reachable.
    if ndim >= 24:
        k_rule_matched_template = dict(
            R0=0.0,
            tier_weights=[1.0, 10.0],
            base=base,
            filter_width=3,
            seed=42,
            head_seed=2025,
            active_dims=_first_k_dims(24, ndim),
            cross_dim_mods=[3, 3],
        )
        for n_rules in (1, 16, 64):
            presets[f"K{n_rules}_matched"] = _cap_tiers(
                dict(k_rule_matched_template, n_rules=n_rules)
            )

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


def get_uniform_random_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for UniformRandomReward.

    Difficulty is controlled by ``mode_prob``: lower probability means sparser
    modes, which are harder for GFlowNets to discover.
    """
    presets = {
        "easy": dict(R0=0.1, R_mode=2.0, mode_prob=0.1, seed=42),
        "medium": dict(R0=0.1, R_mode=2.0, mode_prob=0.01, seed=42),
        "hard": dict(R0=0.1, R_mode=2.0, mode_prob=0.001, seed=42),
        "challenging": dict(R0=0.1, R_mode=2.0, mode_prob=0.0001, seed=42),
        "impossible": dict(R0=0.1, R_mode=2.0, mode_prob=0.00001, seed=42),
    }
    return presets


def get_corrupted_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for CorruptedReward.

    Each preset wraps a ``conditional_multiscale`` "medium" base and applies
    increasing corruption. A single ``corruption_rate`` parameter controls the
    fraction of per-tier structure that is randomized.

    Difficulty progression:
      - easy:        10% corruption -> mostly structured
      - medium:      30% corruption -> noticeable randomness
      - hard:        50% corruption -> half structured, half random
      - challenging: 70% corruption -> mostly random
      - impossible:  90% corruption -> near-total randomness

    Note: requires ``height`` to be a power of 4 (same as the base reward).
    """
    cms_presets = get_conditional_multiscale_presets(ndim, height)
    base_kwargs = cms_presets.get("medium", cms_presets.get("easy", {}))

    presets = {
        "easy": dict(
            base_reward="conditional_multiscale",
            base_kwargs=base_kwargs,
            corruption_rate=0.1,
            seed=137,
        ),
        "medium": dict(
            base_reward="conditional_multiscale",
            base_kwargs=base_kwargs,
            corruption_rate=0.3,
            seed=137,
        ),
        "hard": dict(
            base_reward="conditional_multiscale",
            base_kwargs=base_kwargs,
            corruption_rate=0.5,
            seed=137,
        ),
        "challenging": dict(
            base_reward="conditional_multiscale",
            base_kwargs=base_kwargs,
            corruption_rate=0.7,
            seed=137,
        ),
        "impossible": dict(
            base_reward="conditional_multiscale",
            base_kwargs=base_kwargs,
            corruption_rate=0.9,
            seed=137,
        ),
    }
    return presets


def get_reward_presets(reward_fn_str: str, ndim: int, height: int) -> dict:
    """Return presets for a given reward function name.

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
    if reward_fn_str == "uniform_random":
        return get_uniform_random_presets(ndim, height)
    if reward_fn_str == "corrupted":
        return get_corrupted_presets(ndim, height)
    raise ValueError(f"Unknown reward_fn_str for presets: {reward_fn_str}")
