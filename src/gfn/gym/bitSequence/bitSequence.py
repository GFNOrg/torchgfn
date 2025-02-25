from __future__ import annotations  # This allows to use the class name in type hints

import torch
from typing import Optional, Union, Tuple

from gfn.env import DiscreteEnv, Env
from gfn.preprocessors import Preprocessor
from gfn.actions import Actions
from gfn.containers import Trajectories

from .bitSequenceStates import BitSequenceStates

# This environment is the torchgfn implmentation of the bit sequences task presented in :Malkin, Nikolay & Jain, Moksh & Bengio, Emmanuel & Sun, Chen & Bengio, Yoshua. (2022).
# Trajectory Balance: Improved Credit Assignment in GFlowNets. https://arxiv.org/pdf/2201.13259


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
        self.update_masks(states)  # pyright:ignore

        return states  # pyright:ignore

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

        last_actions = states.tensor[
            ~is_sink, states[~is_sink].length - 1  # pyright:ignore
        ]
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

    def backward_step(
        self, states: BitSequenceStates, actions: Actions
    ) -> torch.Tensor:
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
                states.length - 1,  # pyright:ignore
            ]
        ).all()
        old_tensor = states.tensor.clone()
        old_tensor[..., states.length - 1] = -1  # pyright:ignore
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

        new_states: BitSequenceStates = Env._step(
            self, states, actions
        )  # pyright:ignore
        new_states.length = states.length + 1  # pyright:ignore

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
        new_states: BitSequenceStates = Env._backward_step(
            self, states, actions
        )  # pyright:ignore
        new_states.length = states.length - 1  # pyright:ignore
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
