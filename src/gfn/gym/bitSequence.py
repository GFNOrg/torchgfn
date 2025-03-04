from __future__ import annotations  # This allows to use the class name in type hints

from typing import ClassVar, List, Optional, Sequence, Tuple, Union, cast

import torch

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import DiscreteEnv
from gfn.preprocessors import Preprocessor
from gfn.states import DiscreteStates

# This environment is the torchgfn implmentation of the bit sequences task presented in :Malkin, Nikolay & Jain, Moksh & Bengio, Emmanuel & Sun, Chen & Bengio, Yoshua. (2022).
# Trajectory Balance: Improved Credit Assignment in GFlowNets. https://arxiv.org/pdf/2201.13259


class BitSequenceStates(DiscreteStates):
    """
    A class representing states of bit sequences in a discrete state space.

    Attributes:
        word_size (ClassVar[int]): The size of each word in the bit sequence.
        tensor (torch.Tensor): The tensor representing the states.
        length (torch.Tensor): The tensor representing the length of each bit sequence.
        forward_masks (Optional[torch.Tensor]): The tensor representing the forward masks.
        backward_masks (Optional[torch.Tensor]): The tensor representing the backward masks.

    Methods:
        __init__(tensor: torch.Tensor, length: Optional[torch.Tensor[int]] = None, forward_masks: Optional[torch.Tensor] = None, backward_masks: Optional[torch.Tensor] = None) -> None:
            Initializes the BitSequencesStates object.

        clone() -> BitSequencesStates:
            Returns a clone of the current BitSequencesStates object.

        __getitem__(index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor) -> BitSequencesStates:
            Returns a subset of the BitSequencesStates object based on the given index.

        __setitem__(index: int | Sequence[int] | Sequence[bool], states: BitSequencesStates) -> None:
            Sets a subset of the BitSequencesStates object based on the given index and states.

        flatten() -> BitSequencesStates:
            Flattens the BitSequencesStates object.

        extend(other: BitSequencesStates) -> None:
            Extends the current BitSequencesStates object with another BitSequencesStates object.

        extend_with_sf(required_first_dim: int) -> None:
            Extends the current BitSequencesStates object with sink states.

        stack_states(cls, states: List[BitSequencesStates]) -> BitSequencesStates:
            Stacks a list of BitSequencesStates objects into a single BitSequencesStates object.

        to_str() -> List[str]:
            Converts the BitSequencesStates object to a list of binary strings.
    """

    word_size: ClassVar[int]

    def __init__(
        self,
        tensor: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        forward_masks: Optional[torch.Tensor] = None,
        backward_masks: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            tensor, forward_masks=forward_masks, backward_masks=backward_masks
        )
        if length is None:
            length = torch.zeros(
                self.batch_shape, dtype=torch.long, device=self.__class__.device
            )
        assert length.dtype == torch.long
        self.length = length

    def clone(self) -> BitSequenceStates:
        return self.__class__(
            self.tensor.detach().clone(),
            self.length,
            self.forward_masks,
            self.backward_masks,
        )

    def _check_both_forward_backward_masks_exist(self):
        assert self.forward_masks is not None and self.backward_masks is not None

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> BitSequenceStates:
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        length = self.length[index]
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        out = self.__class__(states, length, forward_masks, backward_masks)
        return out

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], states: BitSequenceStates
    ) -> None:
        super().__setitem__(index, states)
        self.length[index] = states.length

    def flatten(self) -> BitSequenceStates:
        states = self.tensor.view(-1, *self.state_shape)
        length = self.length.view(-1, self.length.shape[-1])
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks.view(-1, self.forward_masks.shape[-1])
        backward_masks = self.backward_masks.view(-1, self.backward_masks.shape[-1])
        return self.__class__(states, length, forward_masks, backward_masks)

    def extend(self, other: BitSequenceStates) -> None:
        super().extend(other)
        self.length = torch.cat(
            (self.length, other.length), dim=len(self.batch_shape) - 1
        )

    def extend_with_sf(self, required_first_dim: int) -> None:
        super().extend_with_sf(required_first_dim)

        def _extend(masks, first_dim):
            return torch.cat(
                (
                    masks,
                    torch.ones(
                        first_dim - masks.shape[0],
                        *masks.shape[1:],
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ),
                dim=0,
            )

        self.length = _extend(self.length, required_first_dim)

    @classmethod
    def stack_states(cls, states: List[BitSequenceStates]):
        stacked_states = cast(BitSequenceStates, super().stack_states(states))
        stacked_states.length = torch.stack([s.length for s in states], dim=0)
        return stacked_states

    def to_str(self) -> List[str]:
        """
        Converts the tensor to a list of binary strings.

        The tensor is reshaped according to the state shape and then each row is
        converted to a binary string, ignoring entries with a value of -1.

        Returns:
            List[str]: A list of binary strings representing the tensor.
        """
        tensor = self.tensor.view(-1, *self.state_shape)
        mask = tensor != -1

        def row_to_binary_string(row, row_mask):
            valid_entries = row[row_mask]
            return "".join(
                format(x.item(), f"0{self.word_size}b") for x in valid_entries
            )

        return [row_to_binary_string(tensor[i], mask[i]) for i in range(tensor.shape[0])]


class BitSequence(DiscreteEnv):
    """
    Append-only BitSequence is a custom environment that inherits from DiscreteEnv. It represents a sequence of binary words and
    provides methods to manipulate and evaluate these sequences.
    The possible actions are adding binary words at once. Each binary word is represented as its decimal representation in both states and actions.
    If the user wants to use the binary representation during a GFlowNet training, he can create a custom preprocessor.

    Attributes:
        word_size (int): The size of each binary word in the sequence.
        seq_size (int): The total number of digits of the sequence.
        n_modes (int): The number of unique modes in the sequence.
        temperature (float): The temperature parameter for reward calculation.
        H (Optional[torch.Tensor]): A tensor used to create the modes. (For more details, please see make_modes_set method)
        device_str (str): The device to run the computations on ("cpu" or "cuda").
        preprocessor (Optional[Preprocessor]): An optional preprocessor for the environment.
        words_per_seq (int): The number of words per sequence.
        modes (torch.Tensor): The set of modes written as binary.
        s0 (torch.Tensor): The initial state tensor.
        sf (torch.Tensor): The final state tensor.
        n_actions (int): The number of possible actions.
        state_shape (Tuple): The shape of the state tensor.
        action_shape (Tuple): The shape of the action tensor.
        dummy_action (torch.Tensor): A tensor representing a dummy action.
        exit_action (torch.Tensor): A tensor representing the exit action.
    """

    def __init__(
        self,
        word_size: int = 4,
        seq_size: int = 120,
        n_modes: int = 60,
        temperature: float = 1.0,
        H: Optional[torch.Tensor] = None,
        device_str: str = "cpu",
        preprocessor: Optional[Preprocessor] = None,
    ):
        assert seq_size % word_size == 0, "word_size must divide seq_size."
        self.words_per_seq: int = seq_size // word_size
        self.word_size: int = word_size
        self.seq_size: int = seq_size
        self.n_modes: int = n_modes
        n_actions = 2**word_size + 1
        s0 = -torch.ones(
            self.words_per_seq, dtype=torch.long, device=torch.device(device_str)
        )
        state_shape = s0.shape
        action_shape = (1,)
        dummy_action = -torch.ones(1, dtype=torch.long)
        exit_action = (n_actions - 1) * torch.ones(1, dtype=torch.long)
        sf = (n_actions - 1) * torch.ones(
            self.words_per_seq, dtype=torch.long, device=torch.device(device_str)
        )
        super().__init__(
            n_actions,
            s0,
            state_shape,
            action_shape,
            dummy_action,
            exit_action,
            sf,
            device_str,
            preprocessor,
        )
        self.H = H
        self.modes = self.make_modes_set()  # set of modes written as binary
        self.temperature = temperature

    def make_states_class(self) -> type[BitSequenceStates]:
        env = self

        class BitSequenceStatesImplementation(BitSequenceStates):
            state_shape = (env.words_per_seq,)
            s0 = env.s0
            sf = env.sf
            make_random_states_tensor = env.make_random_states_tensor
            n_actions = env.n_actions
            device = env.device
            word_size = env.word_size

        return BitSequenceStatesImplementation

    def states_from_tensor(
        self, tensor: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> BitSequenceStates:
        """Wraps the supplied Tensor in a States instance & updates masks.

        Args:
            tensor: The tensor of shape "state_shape" representing the states.

        Returns:
            States: An instance of States.
        """
        if length is None:
            mask = tensor != -1
            length = mask.int().sum(dim=-1)
        states_instance = self.make_states_class()(tensor, length=length)
        self.update_masks(states_instance)
        return states_instance

    # In some cases overwritten by the user to support specific use-cases.
    def reset(
        self,
        batch_shape: Optional[Union[int, Tuple[int]]] = None,
        sink: bool = False,
    ) -> BitSequenceStates:
        """
        Generates initial or sink states from batch_shape. Doesn't provide random option for now.
        Args:
            batch_shape (Optional[Union[int, Tuple[int]]], optional): The shape of the batch.
                If None, defaults to (1,). If an integer is provided, it is converted to a tuple.
            sink (bool, optional): If True, sink state is created. Defaults to False.
        Returns:
            BitSequenceStates: The initial states of the environment after reset.
        """

        if batch_shape is None:
            batch_shape = (1,)
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        states = self.states_from_batch_shape(
            batch_shape=batch_shape, random=False, sink=sink
        )
        assert isinstance(states, BitSequenceStates)
        self.update_masks(states)

        return states

    def update_masks(self, states: BitSequenceStates) -> None:
        """Updates the forward and backward masks.

        Called automatically after each step.
        """

        is_done = states.length == self.words_per_seq
        states.forward_masks = torch.ones_like(
            states.forward_masks, dtype=torch.bool, device=states.__class__.device
        )
        states.forward_masks[is_done, :-1] = False
        states.forward_masks[~is_done, -1] = False

        is_sink = states.is_sink_state

        last_actions = states.tensor[~is_sink, states[~is_sink].length - 1]
        states.backward_masks = torch.zeros_like(
            states.backward_masks, dtype=torch.bool, device=states.__class__.device
        )
        states.backward_masks[~is_sink, last_actions] = True

    def step(self, states: BitSequenceStates, actions: Actions) -> torch.Tensor:
        """Function that takes a batch of states and actions and returns a batch of next
        states. Does not need to check whether the actions are valid or the states are sink states.

        Args:
            states: A batch of states.
            actions: A batch of actions.

        Returns:
            torch.Tensor: A batch of next states.
        """
        is_exit = actions.is_exit
        old_tensor = states.tensor.clone()
        old_tensor[~is_exit, states.length] = actions.tensor[~is_exit].squeeze()
        old_tensor[is_exit] = torch.full_like(
            old_tensor[is_exit],
            self.n_actions - 1,
            dtype=torch.long,
            device=old_tensor.device,
        )
        return old_tensor

    def backward_step(self, states: BitSequenceStates, actions: Actions) -> torch.Tensor:
        """Function that takes a batch of states and actions and returns a batch of previous
        states. Does not need to check whether the actions are valid or the states are sink states.
        Operates on flattened states only.

        Args:
            states: A batch of states.
            actions: A batch of actions.

        Returns:
            torch.Tensor: A batch of previous states.
        """
        assert (
            actions.tensor.squeeze()
            == states.tensor[
                torch.arange(states.tensor.shape[0]),
                states.length - 1,
            ]
        ).all()
        old_tensor = states.tensor.clone()
        old_tensor[..., states.length - 1] = -1
        return old_tensor

    def _step(self, states: BitSequenceStates, actions: Actions):
        """
        Perform a step in the environment by applying the given actions to the current states.
        Args:
            states (BitSequenceStates): The current states of the environment.
            actions (Actions): The actions to be applied to the current states.
        Returns:
            BitSequenceStates: The new states of the environment after applying the actions.
        """

        new_states = super(DiscreteEnv, self)._step(states, actions)
        assert isinstance(new_states, BitSequenceStates)
        new_states.length = states.length + 1

        self.update_masks(new_states)

        return new_states

    def _backward_step(self, states: BitSequenceStates, actions: Actions):
        """
        Perform a backward step in the environment by undoing the given actions to the current states.

        Args:
            states (BitSequenceStates): The current states of the environment.
            actions (Actions): The actions to be applied to the current states.

        Returns:
            BitSequenceStates: The new states after performing the backward step.
        """
        new_states = super(DiscreteEnv, self)._backward_step(states, actions)
        assert isinstance(new_states, BitSequenceStates)
        new_states.length = states.length - 1
        self.update_masks(new_states)
        return new_states

    def make_modes_set(self, seed: int = 42) -> torch.Tensor:
        """
        Generates a set of unique mode sequences based on the predefined tensor H.

        Args:
            seed (int, optional): The seed for random number generation. Defaults to 42.

        Returns:
            torch.Tensor: A tensor containing the unique mode sequences.

        Raises:
            ValueError: If the number of requested modes exceeds the number of possible unique sequences.

        Notes:
            - The method sets the random seed for both CPU and GPU (if available) to ensure reproducibility.
            - The tensor H is initialized with predefined sequences if it is not already set. This default value is the one chosen in the TB objectie paper.
            - The method ensures that the number of unique sequences generated does not exceed the possible combinations.
            - The method returns a binary sequence, which is useful to easily compute the hamming distance.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.H is None:
            self.H = torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                ]
            )

        m = self.seq_size // self.H.shape[-1]

        num_possible = self.H.shape[0] ** m
        if self.n_modes > num_possible:
            raise ValueError("Not enough unique sequences available.")

        unique_indices = set()
        unique_sequences = []

        while len(unique_sequences) < self.n_modes:
            candidate = tuple(torch.randint(0, self.H.shape[0], (m,)).tolist())
            if candidate not in unique_indices:
                unique_indices.add(candidate)
                unique_sequences.append(candidate)

        indices_iter = torch.tensor(unique_sequences)
        references_iter = self.H[indices_iter].reshape(self.n_modes, -1)

        return references_iter.to(self.device)

    @staticmethod
    def integers_to_binary(tensor: torch.Tensor, k: int) -> torch.Tensor:
        """
        Convert a tensor of integers to their binary representation using k bits.

        Args:
            tensor (torch.Tensor): A tensor containing integers. The tensor must have a dtype of torch.int32 or torch.int64.
            k (int): The number of bits to use for the binary representation of each integer.

        Returns:
            torch.Tensor: A tensor containing the binary representation of the input integers. The shape of the returned tensor
                          will be the same as the input tensor, except the last dimension will be expanded by a factor of k.
                          If an element in the input tensor is -1, the corresponding output will be a vector of k copies of -1.
        """
        assert tensor.dtype in [
            torch.int32,
            torch.int64,
        ], "Tensor must contain integers"

        batch_shape = tensor.shape[:-1]
        length = tensor.shape[-1]

        # Compute binary representation for each integer using k bits.
        binary_tensor = (
            tensor.unsqueeze(-1) >> torch.arange(k - 1, -1, -1, device=tensor.device)
        ) & 1

        # Create a mask for elements that equal -1.
        # For these, we want a vector of k copies of -1.
        mask = (tensor == -1).unsqueeze(-1)  # Shape becomes (..., length, 1)
        binary_tensor = torch.where(
            mask, torch.full_like(binary_tensor, -1), binary_tensor
        )

        # Reshape so that the last dimension has length * k,
        # concatenating the k bits (or -1's) for each integer.
        result = binary_tensor.view(*batch_shape, length * k)
        return result

    @staticmethod
    def binary_to_integers(binary_tensor: torch.Tensor, k: int) -> torch.Tensor:
        """
        Convert a binary tensor to a tensor of integers.

        Args:
            binary_tensor (torch.Tensor): A tensor containing binary values. The tensor must be of type int64.
            k (int): The number of bits in each integer.

        Returns:
            torch.Tensor: A tensor of integers obtained from the binary tensor.

        Raises:
            AssertionError: If the binary_tensor is not of type int64.

        Example:
            >>> binary_tensor = torch.tensor([[1, 0, 1, 1], [0, 1, -1, -1]], dtype=torch.int64)
            >>> k = 2
            >>> binary_to_integers(binary_tensor, k)
            tensor([[3, 1],
                [2, -1]])
        """
        assert binary_tensor.dtype == torch.int64, "Binary tensor must be of type int64"
        batch_shape = binary_tensor.shape[:-1]
        length = binary_tensor.shape[-1] // k

        binary_tensor = binary_tensor.view(*batch_shape, length, k)
        powers = 2 ** torch.arange(k - 1, -1, -1, device=binary_tensor.device)
        result = (binary_tensor * powers).sum(dim=-1)
        result = torch.where(result < 0, torch.tensor(-1, device=result.device), result)
        return result

    @staticmethod
    def hamming_distance(
        candidates: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the smallest edit distance from each candidate row to any reference row.

        Parameters:
        - candidates: Tensor of shape (*batch_shape, length)
        - reference: Tensor of shape (n_ref, length)

        Returns:
        - Tensor of shape (*batch_shape) containing the smallest edit distance for each candidate row.
        """
        candidates_exp = candidates.unsqueeze(-2)
        reference_exp = reference.unsqueeze(0)
        distances = (candidates_exp != reference_exp).sum(dim=-1)
        min_distances = distances.min(dim=-1).values

        return min_distances

    def reward(self, final_states: BitSequenceStates):
        """
        Calculate the reward for the given final states.

        The reward is computed based on the Hamming distance between the binary
        representation of the final states and the predefined modes. The reward
        is then scaled using an exponential function with a temperature parameter.

        Args:
            final_states (BitSequenceStates): The final states for which the reward
                                              is to be calculated.

        Returns:
            torch.Tensor: The calculated reward for the given final states.
        """

        binary_final_states = self.integers_to_binary(
            final_states.tensor, self.word_size
        )
        edit_distance = self.hamming_distance(binary_final_states, self.modes)
        return torch.exp(-edit_distance / self.temperature)

    def create_test_set(self, k: int, seed: int = 42) -> BitSequenceStates:
        """
        Create a test set by altering k times each mode a random number of bits selected randomly.
        Test set of size n_modes * k.

        Args:
            k (int): Number of variations per mode.
            seed (int, optional): Seed for reproducibility. If None, randomness is not fixed.

        Returns:
            torch.Tensor: The generated test set in the decimal representation.
        """
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        K = torch.randint(0, self.seq_size, (self.n_modes, k), generator=g)
        test_set = self.modes.repeat_interleave(k, dim=0)
        for i in range(self.n_modes * k):
            n_changes = K.view(-1)[i]
            indices = torch.randperm(self.seq_size, generator=g)[:n_changes]
            test_set[i, indices] = 1 - test_set[i, indices]
        return self.states_from_tensor(
            self.binary_to_integers(test_set, k=self.word_size)
        )

    def trajectory_from_terminating_states(
        self, terminating_states_tensor: torch.Tensor
    ) -> Trajectories:
        """
        Generate trajectories from terminating states. This works because the DAG is a tree in the append-only version of BitSequence.

        Args:
            terminating_states_tensor (torch.Tensor): A tensor containing the terminating
                states from which to generate the trajectories. The shape of the tensor
                should be (n_trajectories, words_per_seq).

        Returns:
            Trajectories: An object containing the generated trajectories, including the
                states, actions, and other relevant information.
        """
        n_trajectories = terminating_states_tensor.shape[0]
        list_of_states = []
        list_of_actions = []

        for i in range(self.words_per_seq + 1):
            if i > 0:
                prefix = terminating_states_tensor[:, :i].to(self.device)
            else:
                prefix = torch.empty(
                    (n_trajectories, 0),
                    dtype=terminating_states_tensor.dtype,
                    device=self.device,
                )

            suffix = torch.full(
                (n_trajectories, self.words_per_seq - i),
                -1,
                dtype=terminating_states_tensor.dtype,
                device=self.device,
            )

            new_tensor = torch.cat((prefix, suffix), dim=1)

            list_of_states.append(self.states_from_tensor(new_tensor))

        list_of_states.append(self.reset((n_trajectories,), sink=True))
        states = BitSequenceStates.stack_states(list_of_states)

        for i in range(self.words_per_seq):
            word_tensor = terminating_states_tensor[:, i].to(self.device)
            list_of_actions.append(self.actions_from_tensor(word_tensor.unsqueeze(-1)))

        list_of_actions.append(self.Actions.make_exit_actions((n_trajectories,)))
        actions = self.Actions.stack(list_of_actions)

        traj: Trajectories = Trajectories(
            self,
            states,
            actions=actions,
            when_is_done=(self.words_per_seq + 1)
            * torch.ones(n_trajectories, dtype=torch.long, device=self.device),
            log_rewards=torch.zeros(n_trajectories, device=self.device),
        )

        return traj


class BitSequencePlus(BitSequence):
    """
    Prepend-Append version of BitSequence env.
    """

    def __init__(
        self,
        word_size: int = 4,
        seq_size: int = 120,
        n_modes: int = 60,
        temperature: float = 1.0,
        H: Optional[torch.Tensor] = None,
        device_str: str = "cpu",
        preprocessor: Optional[Preprocessor] = None,
    ):
        assert seq_size % word_size == 0, "word_size must divide seq_size."
        self.words_per_seq: int = seq_size // word_size
        self.word_size: int = word_size
        self.seq_size: int = seq_size
        self.n_modes: int = n_modes
        n_actions = 2 ** (word_size + 1) + 1
        s0 = -torch.ones(
            self.words_per_seq, dtype=torch.long, device=torch.device(device_str)
        )
        state_shape = s0.shape
        action_shape = (1,)
        dummy_action = -torch.ones(1, dtype=torch.long)
        exit_action = (n_actions - 1) * torch.ones(1, dtype=torch.long)
        sf = ((n_actions - 1) // 2) * torch.ones(
            self.words_per_seq, dtype=torch.long, device=torch.device(device_str)
        )
        DiscreteEnv.__init__(
            self,
            n_actions,
            s0,
            state_shape,
            action_shape,
            dummy_action,
            exit_action,
            sf,
            device_str,
            preprocessor,
        )
        self.H = H
        self.modes = self.make_modes_set()  # set of modes written as binary
        self.temperature = temperature

    def update_masks(self, states: BitSequenceStates) -> None:

        is_done = states.length == self.words_per_seq
        states.forward_masks = torch.ones_like(
            states.forward_masks, dtype=torch.bool, device=states.__class__.device
        )
        states.forward_masks[is_done, :-1] = False
        states.forward_masks[~is_done, -1] = False

        is_sink = states.is_sink_state

        last_actions = states.tensor[~is_sink, states[~is_sink].length - 1]
        first_actions = states.tensor[~is_sink, 0]
        states.backward_masks = torch.zeros_like(
            states.backward_masks, dtype=torch.bool, device=states.__class__.device
        )
        states.backward_masks[~is_sink, last_actions] = True
        states.backward_masks[~is_sink, first_actions + (self.n_actions - 1) // 2] = True

    def step(self, states: BitSequenceStates, actions: Actions) -> torch.Tensor:
        is_exit = actions.is_exit
        old_tensor = states.tensor.clone()
        append_mask = (actions.tensor < (self.n_actions - 1) // 2).squeeze()
        prepend_mask = ~append_mask
        assert states.length
        old_tensor[append_mask & ~is_exit, states.length[append_mask & ~is_exit]] = (
            actions.tensor[append_mask & ~is_exit].squeeze()
        )

        old_tensor[prepend_mask & ~is_exit, 1:] = old_tensor[
            prepend_mask & ~is_exit, :-1
        ]
        old_tensor[prepend_mask & ~is_exit, 0] = (
            actions.tensor[prepend_mask & ~is_exit].squeeze() - (self.n_actions - 1) // 2
        )

        old_tensor[is_exit] = torch.full_like(
            old_tensor[is_exit],
            self.n_actions - 1,
            dtype=torch.long,
            device=old_tensor.device,
        )
        return old_tensor

    def backward_step(self, states: BitSequenceStates, actions: Actions) -> torch.Tensor:
        old_tensor = states.tensor.clone()
        remove_end_mask = (actions.tensor < (self.n_actions - 1) // 2).squeeze()
        remove_front_mask = ~remove_end_mask
        assert states.length
        old_tensor[remove_end_mask, states.length[remove_end_mask] - 1] = -1

        old_tensor[remove_front_mask, :-1] = old_tensor[remove_front_mask, 1:]

        old_tensor[remove_front_mask, -1] = -1
        return old_tensor

    def trajectory_from_terminating_states(
        self, terminating_states_tensor: torch.Tensor
    ) -> Trajectories:
        raise NotImplementedError("Only available for append-only BitSequence.")
