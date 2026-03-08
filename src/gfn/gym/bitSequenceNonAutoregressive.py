"""Non-autoregressive BitSequence environment for GFlowNets.

This environment implements a non-autoregressive version of the bit sequence
generation task, where actions encode both the position and word value to place.
Unlike the standard (autoregressive) BitSequence environment which appends
words left-to-right, this environment allows filling any unfilled position
in any order.

This formulation matches the one used by the GFNX (JAX-based) library,
enabling fair cross-library benchmarking.

Environment details:
    - State: Tensor of shape ``(words_per_seq,)`` with values in
      ``{-1, 0, ..., 2^word_size - 1}``. ``-1`` indicates an unfilled position.
    - Initial state ``s0``: All positions unfilled, ``[-1, -1, ..., -1]``.
    - Terminal states: All positions filled (no ``-1`` values).
    - Forward actions: ``words_per_seq * n_words`` actions, where each action
      ``a`` encodes ``(position, word) = divmod(a, n_words)``. One additional
      exit action (the last action) is only available at terminal states.
    - Backward actions: ``words_per_seq * n_words`` actions. The backward
      action for a forward action ``(pos, word)`` is the same index — it
      clears that position back to ``-1``.
    - Reward: Based on the minimum Hamming distance (at the bit level) between
      the generated sequence and a set of target mode sequences.

Reference:
    Malkin, N., Jain, M., Bengio, E., Sun, C., & Bengio, Y. (2022).
    Trajectory Balance: Improved Credit Assignment in GFlowNets.
    https://arxiv.org/abs/2201.13259
"""

from __future__ import annotations

from typing import ClassVar, List, Literal, Optional, Sequence, Tuple

import torch

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates


class NonAutoregressiveBitSequenceStates(DiscreteStates):
    """States for the non-autoregressive BitSequence environment.

    Each state is a tensor of shape ``(words_per_seq,)`` where each element is
    either ``-1`` (unfilled) or a word value in ``{0, ..., n_words - 1}``.

    Attributes:
        word_size: Number of bits per word.
        words_per_seq: Number of word positions in the sequence.
        n_words: Number of possible word values (``2 ** word_size``).
    """

    word_size: ClassVar[int]
    words_per_seq: ClassVar[int]
    n_words: ClassVar[int]

    def _compute_forward_masks(self) -> torch.Tensor:
        """Compute which forward actions are valid at each state.

        An action ``(pos, word)`` is valid iff position ``pos`` is unfilled
        (value == -1). All ``n_words`` word choices for a given position share
        the same validity. The exit action is only valid when all positions
        are filled.

        Returns:
            Boolean tensor of shape ``(*batch_shape, n_actions)``.
        """
        forward_masks = torch.zeros(
            (*self.batch_shape, self.n_actions),
            dtype=torch.bool,
            device=self.device,
        )
        # pos_unfilled: (*batch_shape, words_per_seq), True where position is empty
        pos_unfilled = self.tensor == -1

        # Repeat for each word choice: shape (*batch_shape, words_per_seq * n_words)
        # Action a = pos * n_words + word, so we need to expand pos_unfilled
        # from (*, W) to (*, W * V) by repeating each position V times.
        forward_masks[..., : self.words_per_seq * self.n_words] = (
            pos_unfilled.unsqueeze(-1)
            .expand(*self.batch_shape, self.words_per_seq, self.n_words)
            .reshape(*self.batch_shape, self.words_per_seq * self.n_words)
        )

        # Exit action: only valid when all positions are filled
        forward_masks[..., -1] = torch.all(~pos_unfilled, dim=-1)
        return forward_masks

    def _compute_backward_masks(self) -> torch.Tensor:
        """Compute which backward actions are valid at each state.

        A backward action ``(pos, word)`` is valid iff position ``pos``
        currently holds that exact word value.

        Returns:
            Boolean tensor of shape ``(*batch_shape, n_actions - 1)``.
        """
        backward_masks = torch.zeros(
            (*self.batch_shape, self.n_actions - 1),
            dtype=torch.bool,
            device=self.device,
        )
        # For each position and each word value, check if tensor[pos] == word
        for word in range(self.n_words):
            # Actions word, n_words + word, 2*n_words + word, ... correspond
            # to positions 0, 1, 2, ... with this word value.
            # Action index = pos * n_words + word
            backward_masks[
                ..., word :: self.n_words
            ] = (self.tensor == word)

        return backward_masks

    def to_str(self) -> List[str]:
        """Convert states to human-readable binary strings.

        Returns:
            List of binary strings, one per state in the flattened batch.
        """
        tensor = self.tensor.view(-1, *self.state_shape)

        def row_to_str(row: torch.Tensor) -> str:
            parts = []
            for val in row:
                v = val.item()
                if v == -1:
                    parts.append("_" * self.word_size)
                else:
                    parts.append(format(v, f"0{self.word_size}b"))
            return "".join(parts)

        return [row_to_str(tensor[i]) for i in range(tensor.shape[0])]


class NonAutoregressiveBitSequence(DiscreteEnv):
    """Non-autoregressive BitSequence environment.

    In this environment, the agent constructs a binary sequence by placing
    words at arbitrary positions. Each action specifies both which position
    to fill and which word value to place there. The episode ends when all
    positions are filled.

    The reward is based on the minimum Hamming distance (computed at the bit
    level) between the completed sequence and a set of target "mode" sequences.

    Args:
        word_size: Number of bits per word (e.g., 1 for single-bit actions).
        seq_size: Total number of bits in the sequence. Must be divisible
            by ``word_size``.
        n_modes: Number of target mode sequences.
        reward_exponent: Controls reward sharpness. Higher values make the
            reward more peaked around the modes.
        H: Optional tensor of shape ``(n_modes, seq_size)`` specifying the
            target modes in binary. If None, modes are generated randomly
            using block patterns.
        device_str: Device to use (``"cpu"`` or ``"cuda"``).
        seed: Random seed for mode generation.
        debug: If True, enable runtime guards (not compile-friendly).

    Attributes:
        word_size: Number of bits per word.
        seq_size: Total number of bits.
        words_per_seq: Number of word positions (``seq_size // word_size``).
        n_words: Number of possible word values (``2 ** word_size``).
        n_modes: Number of target modes.
        reward_exponent: Reward sharpness parameter.
        modes: Target mode sequences as a binary tensor of shape
            ``(n_modes, seq_size)``.

    Example:
        >>> env = NonAutoregressiveBitSequence(word_size=1, seq_size=4, n_modes=2)
        >>> # Action space: 4 positions * 2 word values + 1 exit = 9 actions
        >>> env.n_actions
        9
        >>> # State shape: 4 word positions
        >>> env.s0
        tensor([-1, -1, -1, -1])
    """

    def __init__(
        self,
        word_size: int = 1,
        seq_size: int = 4,
        n_modes: int = 2,
        reward_exponent: float = 2.0,
        H: Optional[torch.Tensor] = None,
        device_str: str = "cpu",
        seed: int = 0,
        debug: bool = False,
    ):
        assert seq_size % word_size == 0, "word_size must divide seq_size."

        self.word_size = word_size
        self.seq_size = seq_size
        self.words_per_seq = seq_size // word_size
        self.n_words = 2**word_size
        self.n_modes_count = n_modes
        self.reward_exponent = reward_exponent

        device = torch.device(device_str)

        s0 = torch.full((self.words_per_seq,), -1, dtype=torch.long, device=device)
        sf = torch.full(
            (self.words_per_seq,), self.n_words, dtype=torch.long, device=device
        )

        # Forward actions: words_per_seq * n_words position-word pairs + 1 exit
        n_actions = self.words_per_seq * self.n_words + 1

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            state_shape=(self.words_per_seq,),
            sf=sf,
            debug=debug,
        )

        # Generate target modes
        self.H = H
        self.modes = self._make_modes(seed, device)
        self.States: type[NonAutoregressiveBitSequenceStates] = self.States

    def make_states_class(self) -> type[NonAutoregressiveBitSequenceStates]:
        """Create the States class with environment-specific constants."""
        env = self

        class StatesImpl(NonAutoregressiveBitSequenceStates):
            state_shape = (env.words_per_seq,)
            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states
            n_actions = env.n_actions
            word_size = env.word_size
            words_per_seq = env.words_per_seq
            n_words = env.n_words

        return StatesImpl

    def make_random_states(
        self,
        batch_shape: Tuple,
        conditions: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        debug: bool = False,
    ) -> NonAutoregressiveBitSequenceStates:
        """Generate random partially-filled states.

        Each position is independently either unfilled (-1) or filled with
        a random word value.

        Args:
            batch_shape: Shape of the batch.
            conditions: Optional conditions tensor.
            device: Device to use.
            debug: If True, enable debug mode.

        Returns:
            Random states.
        """
        device = self.device if device is None else device
        # Random values in {-1, 0, ..., n_words - 1}
        tensor = torch.randint(
            -1, self.n_words, batch_shape + (self.words_per_seq,), device=device
        )
        return self.States(tensor, conditions=conditions, debug=debug)

    def _make_modes(self, seed: int, device: torch.device) -> torch.Tensor:
        """Generate target mode sequences in binary representation.

        If ``H`` is provided, it is used directly as the modes. Otherwise,
        modes are constructed by randomly combining 8-bit block patterns,
        following the procedure from the Trajectory Balance paper.

        Args:
            seed: Random seed.
            device: Device to place the modes tensor on.

        Returns:
            Binary tensor of shape ``(n_modes, seq_size)`` with values in {0, 1}.
        """
        if self.H is not None:
            assert self.H.shape == (self.n_modes_count, self.seq_size)
            return self.H.to(device)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        block_set = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
        )
        block_len = block_set.shape[1]
        n_blocks = self.seq_size // block_len

        if n_blocks == 0:
            # seq_size < block_len: generate random binary modes directly
            return torch.randint(
                0, 2, (self.n_modes_count, self.seq_size), dtype=torch.long, device=device
            )

        num_possible = block_set.shape[0] ** n_blocks
        if self.n_modes_count > num_possible:
            raise ValueError(
                f"Cannot generate {self.n_modes_count} unique modes from "
                f"{num_possible} possible block combinations."
            )

        unique_indices: set = set()
        unique_sequences = []
        while len(unique_sequences) < self.n_modes_count:
            candidate = tuple(
                torch.randint(0, block_set.shape[0], (n_blocks,)).tolist()
            )
            if candidate not in unique_indices:
                unique_indices.add(candidate)
                unique_sequences.append(candidate)

        indices = torch.tensor(unique_sequences)
        modes = block_set[indices].reshape(self.n_modes_count, -1)

        # Truncate to seq_size (in case seq_size is not a multiple of block_len)
        return modes[:, : self.seq_size].to(device)

    def _decode_action(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a flat action index into (position, word) pair.

        Args:
            action: Action tensor of shape ``(*batch_shape, 1)``.

        Returns:
            Tuple of (position, word) tensors, each of shape ``(*batch_shape, 1)``.
        """
        pos = action // self.n_words
        word = action % self.n_words
        return pos, word

    def step(
        self, states: NonAutoregressiveBitSequenceStates, actions: Actions
    ) -> NonAutoregressiveBitSequenceStates:
        """Place a word at the specified position.

        The action encodes ``(position, word)`` as a flat index:
        ``action = position * n_words + word``.

        Args:
            states: Current states.
            actions: Actions encoding (position, word) pairs.

        Returns:
            Next states with the specified positions filled.
        """
        pos, word = self._decode_action(actions.tensor)
        new_tensor = states.tensor.clone()
        # Scatter the word value into the position for each batch element
        new_tensor.scatter_(-1, pos, word)
        return self.States(new_tensor)

    def backward_step(
        self, states: NonAutoregressiveBitSequenceStates, actions: Actions
    ) -> NonAutoregressiveBitSequenceStates:
        """Undo a word placement by clearing the position back to -1.

        The backward action has the same encoding as the forward action:
        ``action = position * n_words + word``. The word component is used
        to identify which position to clear.

        Args:
            states: Current states.
            actions: Backward actions to undo.

        Returns:
            Previous states with the specified positions cleared.
        """
        pos, _ = self._decode_action(actions.tensor)
        new_tensor = states.tensor.clone()
        new_tensor.scatter_(-1, pos, -1)
        return self.States(new_tensor)

    @staticmethod
    def _integers_to_binary(tensor: torch.Tensor, k: int) -> torch.Tensor:
        """Convert a tensor of word integers to their binary representation.

        Args:
            tensor: Integer tensor of shape ``(*batch_shape, words_per_seq)``
                with values in ``{0, ..., 2^k - 1}``.
            k: Number of bits per word.

        Returns:
            Binary tensor of shape ``(*batch_shape, words_per_seq * k)``
            with values in ``{0, 1}``.
        """
        batch_shape = tensor.shape[:-1]
        n_words = tensor.shape[-1]
        bits = (
            tensor.unsqueeze(-1) >> torch.arange(k - 1, -1, -1, device=tensor.device)
        ) & 1
        return bits.reshape(*batch_shape, n_words * k)

    @staticmethod
    def _min_hamming_distance(
        candidates: torch.Tensor, references: torch.Tensor
    ) -> torch.Tensor:
        """Compute minimum Hamming distance from each candidate to any reference.

        Args:
            candidates: Binary tensor of shape ``(*batch_shape, seq_size)``.
            references: Binary tensor of shape ``(n_refs, seq_size)``.

        Returns:
            Tensor of shape ``(*batch_shape,)`` with the minimum distance.
        """
        # candidates: (*batch, seq_size) -> (*batch, 1, seq_size)
        # references: (n_refs, seq_size) -> (1, ..., 1, n_refs, seq_size)
        n_expand = len(candidates.shape) - 1
        refs = references
        for _ in range(n_expand):
            refs = refs.unsqueeze(0)
        dists = (candidates.unsqueeze(-2) != refs).sum(dim=-1)
        return dists.min(dim=-1).values

    def log_reward(
        self, final_states: NonAutoregressiveBitSequenceStates
    ) -> torch.Tensor:
        """Compute log-reward based on Hamming distance to nearest mode.

        The log-reward is:
            ``log R(x) = -reward_exponent * min_d(x, modes) / seq_size``

        where ``min_d`` is the minimum bit-level Hamming distance between the
        completed sequence and any target mode.

        Args:
            final_states: Terminal states with all positions filled.

        Returns:
            Log-reward tensor of shape ``(*batch_shape,)``.
        """
        binary = self._integers_to_binary(final_states.tensor, self.word_size)
        min_dist = self._min_hamming_distance(binary, self.modes)
        return -self.reward_exponent * min_dist.float() / self.seq_size

    def reward(
        self, final_states: NonAutoregressiveBitSequenceStates
    ) -> torch.Tensor:
        """Compute reward as ``exp(log_reward)``.

        Args:
            final_states: Terminal states.

        Returns:
            Reward tensor of shape ``(*batch_shape,)``.
        """
        return torch.exp(self.log_reward(final_states))

    @property
    def n_terminating_states(self) -> int:
        """Total number of possible terminal states."""
        return self.n_words**self.words_per_seq

    @property
    def terminating_states(self) -> NonAutoregressiveBitSequenceStates:
        """Enumerate all terminal states (only feasible for small environments)."""
        digits = torch.arange(self.n_words, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.words_per_seq)
        if self.words_per_seq == 1:
            all_states = all_states.unsqueeze(-1)
        return self.States(all_states)

    def true_dist(self, condition=None) -> torch.Tensor:
        """Compute the true reward distribution over all terminal states."""
        states = self.terminating_states
        rewards = self.reward(states)
        return rewards / rewards.sum()
