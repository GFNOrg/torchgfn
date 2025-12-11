"""Adapted from https://github.com/Tikquuss/GflowNets_Tutorial"""

import itertools
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
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.utils.common import ensure_same_device

if platform.system() == "Windows":
    multiprocessing.set_start_method("spawn", force=True)
else:
    multiprocessing.set_start_method("fork", force=True)


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
    """

    def __init__(
        self,
        ndim: int = 2,
        height: int = 4,
        reward_fn_str: str = "original",
        reward_fn_kwargs: dict | None = None,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        calculate_partition: bool = False,
        store_all_states: bool = False,
        debug: bool = False,
    ):
        """Initializes the HyperGrid environment.

        Args:
            ndim: The dimension of the grid.
            height: The height of the grid.
            reward_fn_str: The reward function string to use.
            reward_fn_kwargs: The keyword arguments for the reward function.
            device: The device to use.
            calculate_partition: Whether to calculate the log partition function.
            store_all_states: Whether to store all states. If True, the true distribution
                can be accessed via the `true_dist` property.
            debug: If True, emit States with debug guards (not compile-friendly).
        """
        if height <= 4:
            warnings.warn("+ Warning: height <= 4 can lead to unsolvable environments.")

        reward_functions = {
            "original": OriginalReward,
            "cosine": CosineReward,
            "sparse": SparseReward,
            "deceptive": DeceptiveReward,
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
        self.calculate_partition = calculate_partition
        self.store_all_states = store_all_states

        # Pre-computes these values when printing.
        if self.store_all_states:
            self._store_all_states_tensor()
            assert self._all_states_tensor is not None
            print(f"+ Environment has {len(self._all_states_tensor)} states")

        if self.calculate_partition:
            self._calculate_log_partition()
            print(f"+ Environment log partition is {self._log_partition}")

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

    def update_masks(self, states: DiscreteStates) -> None:
        """Updates the masks of the states.

        Args:
            states: The states to update the masks of.
        """
        # Not allowed to take any action beyond the environment height, but
        # allow early termination.
        # TODO: do we need to handle the conditional case here?
        states.set_nonexit_action_masks(
            states.tensor == self.height - 1,
            allow_exit=True,
        )
        states.backward_masks = states.tensor != 0

    def make_random_states(
        self,
        batch_shape: Tuple[int, ...],
        device: torch.device | None = None,
        debug: bool = False,
    ) -> DiscreteStates:
        """Creates a batch of random states.

        Args:
            batch_shape: The shape of the batch.
            device: The device to use.

        Returns:
            A `DiscreteStates` object with random states.
        """
        device = self.device if device is None else device
        tensor = torch.randint(
            0, self.height, batch_shape + self.s0.shape, device=device
        )
        return self.States(tensor, debug=debug)

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Performs a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The next states.
        """
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")
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

        # Other reward schemas are not supported for mode counting via threshold.
        raise NotImplementedError(
            "Mode threshold is only defined for 'original' and 'deceptive' rewards."
        )

    def mode_mask(self, states: DiscreteStates) -> torch.Tensor:
        """Boolean mask indicating which states are in a mode.

        A state is flagged as mode if its reward is greater-or-equal to
        the threshold based on `reward_fn_kwargs` (R0+R1+R2 by default).
        """
        rewards = self.reward(states)
        threshold = self._mode_reward_threshold()
        return rewards >= threshold

    def mode_ids(self, states: DiscreteStates) -> torch.Tensor:
        """Returns an integer id (0..2^ndim-1) for states detected as modes.

        Id is based on which side of the center (> 0.5) each coordinate lies on.
        Non-mode states receive id -1.
        """
        mask = self.mode_mask(states)
        # Determine side relative to the center along each dimension
        pos = (
            states.tensor.to(dtype=torch.get_default_dtype()) / (self.height - 1) > 0.5
        ).long()
        weights = (
            2 ** torch.arange(self.ndim - 1, -1, -1, device=states.tensor.device)
        ).long()
        ids = (pos * weights).sum(dim=-1)
        # Assign -1 to non-mode states
        ids = torch.where(mask, ids, torch.full_like(ids, -1))

        return ids

    def modes_found(self, states: DiscreteStates) -> set[int]:
        """Returns the set of unique mode ids discovered among given states."""
        ids = self.mode_ids(states)
        ids = ids[ids >= 0]
        return set(ids.tolist())

    @property
    def n_modes(self) -> int:
        """Returns the total number of distinct modes for this environment."""
        return 2**self.ndim

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

    # Functions for calculating the true log partition function / state enumeration.
    def _calculate_log_partition(self, batch_size: int = 20_000):
        """Calculates the log partition of the complete hypergrid.

        Args:
            batch_size: The batch size to use for the calculation.
        """

        if self._log_partition is None and self.calculate_partition:
            if self._all_states_tensor is not None:
                self._log_partition = (
                    self.reward_fn(self._all_states_tensor).sum().log().item()
                )
                return

            # The # of possible combinations (with repetition) of
            # numbers, where each
            # number can be any integer from 0 to
            # (inclusive), is given by:
            # n = (k + 1) ** n -- note that k in our case is height-1, as it represents
            # a python index.
            max_height_idx = self.height - 1  # Handles 0 indexing.
            n_expected = (max_height_idx + 1) ** self.ndim
            n_found = 0
            start_time = time()
            total_reward = 0

            for batch in self._generate_combinations_in_batches(
                self.ndim,
                max_height_idx,
                batch_size,
            ):
                batch = torch.LongTensor(list(batch))
                rewards = self.reward_fn(
                    batch
                )  # Operates on raw tensors due to multiprocessing.
                total_reward += rewards.sum().item()  # Accumulate.
                n_found += batch.shape[0]

            assert n_expected == n_found, "failed to compute reward of all indices!"
            end_time = time()
            total_log_reward = log(total_reward)

            print(
                "log_partition = {}, calculated in {} minutes".format(
                    total_log_reward,
                    (end_time - start_time) / 60.0,
                )
            )

            self._log_partition = total_log_reward

    def _store_all_states_tensor(self, batch_size: int = 20_000):
        """Enumerates all states_tensor of the complete hypergrid.

        Args:
            batch_size: The batch size to use for the calculation.
        """
        if self._all_states_tensor is None:
            start_time = time()
            all_states_tensor = []

            for batch in self._generate_combinations_in_batches(
                self.ndim,
                self.height - 1,  # Handles 0 indexing.
                batch_size,
            ):
                all_states_tensor.append(torch.LongTensor(list(batch)))

            all_states_tensor = torch.cat(all_states_tensor, dim=0)
            end_time = time()

            print(
                "calculated tensor of all states in {} minutes".format(
                    (end_time - start_time) / 60.0,
                )
            )

            self._all_states_tensor = all_states_tensor

    @property
    def true_dist(self) -> torch.Tensor | None:
        """Returns the pmf over all states in the hypergrid."""
        if self._true_dist is None and self.all_states is not None:
            assert torch.all(
                self.get_states_indices(self.all_states)
                == torch.arange(self.n_states, device=self.device)
            )
            self._true_dist = self.reward(self.all_states)
            self._true_dist /= self._true_dist.sum()

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

    @property
    def log_partition(self) -> float | None:
        """Returns the log partition of the reward function."""
        return self._log_partition

    @property
    def all_states(self) -> DiscreteStates | None:
        """Returns a tensor of all hypergrid states as a `DiscreteStates` instance."""
        if self._all_states_tensor is None:
            if not self.store_all_states:
                return None
            self._store_all_states_tensor()

        assert self._all_states_tensor is not None
        try:
            ensure_same_device(self._all_states_tensor.device, self.device)
        except ValueError:
            self._all_states_tensor = self._all_states_tensor.to(self.device)

        all_states = self.States(self._all_states_tensor)
        return all_states

    @property
    def terminating_states(self) -> DiscreteStates | None:
        """Returns all terminating states of the environment."""
        return self.all_states

    # Helper methods for enumerating all possible states.
    def _generate_combinations_chunk(self, numbers, n, start, end):
        """Generate combinations with replacement for the specified range."""
        # islice accesses a subset of the full iterator - each job does unique work.
        return itertools.islice(itertools.product(numbers, repeat=n), start, end)

    def _worker(self, task):
        """Executes a single call to `generate_combinations_chunk`."""
        numbers, n, start, end = task
        return self._generate_combinations_chunk(numbers, n, start, end)

    def _generate_combinations_in_batches(self, n, k, batch_size):
        """Uses Pool to collect subsets of the results of itertools.product in parallel."""
        numbers = list(range(k + 1))

        # Number of possible combinations (with repetition) of
        # numbers, where each
        # number can be any integer from 0 to
        # (inclusive).
        total_combinations = (k + 1) ** n
        tasks = [
            (numbers, n, i, min(i + batch_size, total_combinations))
            for i in range(0, total_combinations, batch_size)
        ]

        with multiprocessing.Pool() as pool:
            for result in pool.imap(self._worker, tasks):
                yield result


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
    #       0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 (x-axis)
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
    # y=15: . . . . . . . . . . . . . . . .
    # y=14: . . . . . . . . . . . . . . . .
    # y=13: . . X . . . . . . . . . . X . .
    # y=12: . . . . . . . . . . . . . . . .
    # y=11: . . . . . . . . . . . . . . . .
    # y=10: . . . . . . . . . . . . . . . .
    # y=09: . . . . . . + + + + . . . . . .
    # y=08: . . . . . . + + + + . . . . . .
    # y=07: . . . . . . + + + + . . . . . .
    # y=06: . . . . . . + + + + . . . . . .
    # y=05: . . . . . . . . . . . . . . . .
    # y=04: . . . . . . . . . . . . . . . .
    # y=03: . . . . . . . . . . . . . . . .
    # y=02: . . X . . . . . . . . . . X . .
    # y=01: . . . . . . . . . . . . . . . .
    # y=00: . . . . . . . . . . . . . . . .
    #       0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 (x-axis)

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R0 = self.kwargs.get("R0", 1e-5)
        R1 = self.kwargs.get("R1", 0.1)
        R2 = self.kwargs.get("R2", 2.0)

        ax = abs(states_tensor / (self.height - 1) - 0.5)
        term1 = R0 + R1
        cancel_outer = (0.1 + self._EPS < ax).prod(-1) * R1
        ring_band = ((0.3 + self._EPS < ax) * (ax < 0.4 - self._EPS)).prod(-1) * R2
        return term1 - cancel_outer + ring_band
