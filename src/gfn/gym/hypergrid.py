"""Adapted from https://github.com/Tikquuss/GflowNets_Tutorial"""

import itertools
import multiprocessing
import platform
import warnings
from decimal import Decimal
from functools import reduce
from math import gcd, log
from time import time
from typing import List, Literal, Tuple

import torch

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates

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
        R0: The reward parameter R0.
        R1: The reward parameter R1.
        R2: The reward parameter R2.
        reward_cos: Whether to use the cosine reward.
        calculate_partition: Whether to calculate the log partition function.
        calculate_all_states: Whether to store all states.
        scale_factor: The scale factor for the reward.
    """

    def __init__(
        self,
        ndim: int = 2,
        height: int = 4,
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        reward_cos: bool = False,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        calculate_partition: bool = False,
        calculate_all_states: bool = False,
    ):
        """Initializes the HyperGrid environment.

        Args:
            ndim: The dimension of the grid.
            height: The height of the grid.
            R0: The reward parameter R0.
            R1: The reward parameter R1.
            R2: The reward parameter R2.
            reward_cos: Whether to use the cosine reward.
            device: The device to use.
            calculate_partition: Whether to calculate the log partition function.
            calculate_all_states: Whether to store all states.
        """
        if height <= 4:
            warnings.warn("+ Warning: height <= 4 can lead to unsolvable environments.")

        self.ndim = ndim
        self.height = height
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.reward_cos = reward_cos
        self._all_states = None  # Populated optionally in init.
        self._log_partition = None  # Populated optionally in init.
        self._true_dist_pmf = None  # Populated at first request.
        self.calculate_partition = calculate_partition
        self.calculate_all_states = calculate_all_states

        # Pre-computes these values when printing.
        if self.calculate_all_states:
            self._calculate_all_states_tensor()
            assert self._all_states is not None
            print(f"+ Environment has {len(self._all_states)} states")
        if self.calculate_partition:
            self._calculate_log_partition()
            print(f"+ Environment log partition is {self._log_partition}")

        # This scale is used to stabilize some calculations.
        # self.scale_factor = smallest_multiplier_to_integers([R0, R1, R2])
        self.scale_factor = 1

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
        )
        self.States: type[DiscreteStates] = self.States

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
        self, batch_shape: Tuple[int, ...], device: torch.device | None = None
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
        return self.States(tensor)

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

    def reward(self, final_states: DiscreteStates | torch.Tensor) -> torch.Tensor:
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
        assert isinstance(
            final_states, DiscreteStates | torch.Tensor
        ), f"final_states is {type(final_states)}"
        if isinstance(final_states, DiscreteStates):
            final_states_raw = final_states.tensor
        else:
            final_states_raw = final_states

        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states_raw / (self.height - 1) - 0.5)

        if not self.reward_cos:
            reward = (
                R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
            )
        else:
            pdf_input = ax * 5
            pdf = 1.0 / (2 * torch.pi) ** 0.5 * torch.exp(-(pdf_input**2) / 2)
            reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1

        if isinstance(final_states, DiscreteStates):
            assert (
                reward.shape == final_states.batch_shape
            ), f"reward.shape is {reward.shape} and final_states.batch_shape is {final_states.batch_shape}"
        else:
            n_dims = len(reward.shape)
            assert (
                reward.shape == final_states.shape[:n_dims]
            ), f"reward.shape is {reward.shape} and final_states.shape is {final_states.shape}"
        return reward

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
                rewards = self.reward(
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

    def _calculate_all_states_tensor(self, batch_size: int = 20_000):
        """Enumerates all states of the complete hypergrid.

        Args:
            batch_size: The batch size to use for the calculation.
        """
        if self._all_states is None and self.calculate_all_states:
            start_time = time()
            all_states = []

            for batch in self._generate_combinations_in_batches(
                self.ndim,
                self.height - 1,  # Handles 0 indexing.
                batch_size,
            ):
                all_states.append(torch.LongTensor(list(batch)))

            all_states = torch.cat(all_states, dim=0)
            end_time = time()

            print(
                "calculated tensor of all states in {} minutes".format(
                    (end_time - start_time) / 60.0,
                )
            )

            self._all_states = all_states

    @property
    def true_dist_pmf(self) -> torch.Tensor | None:
        """Returns the pmf over all states in the hypergrid."""
        if self._true_dist_pmf is None and self.calculate_all_states:
            assert torch.all(
                self.get_states_indices(self.all_states)
                == torch.arange(self.n_states, device=self.device)
            )
            self._true_dist_pmf = self.reward(self.all_states)
            self._true_dist_pmf /= self._true_dist_pmf.sum()

        return self._true_dist_pmf

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
    def all_states(self) -> DiscreteStates:
        """Returns a tensor of all hypergrid states as a `DiscreteStates` instance."""
        if self.calculate_all_states and not isinstance(self._all_states, torch.Tensor):
            self._calculate_all_states_tensor()

        assert self._all_states is not None
        all_states = self.States(self._all_states)
        assert isinstance(all_states, DiscreteStates)
        return all_states

    @property
    def terminating_states(self) -> DiscreteStates:
        """Returns all terminating states of the environment."""
        return self.all_states

    # Helper methods for enumerating all possible statxes.
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
