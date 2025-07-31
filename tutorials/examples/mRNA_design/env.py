r"""Custom mRNA codon design environment using torchgfn to generate mRNA sequences encoding a given protein. It supports a multi-objective optimization over biological properties of mRNA sequences. Implemented using the DiscreteEnv class.
Each timestep corresponds to choosing a synonymous codon for the next amino acid in the sequence.
**Action Space**:
Number of CODONS + 1 possible actions (all codons + 1 exit action)
**State Representation**:
A vector of length = protein length, initialized to -1. Codons are filled in step-by-step.
Masking (Action Constraints)**:
At each position t, only codons that correspond to the t-th amino acid are allowed to ensure biological correctness.
**Reward function**:
A combination of multiple biological properties to evaluate the mRNA sequence. Weights of these objectives can be updated dynamically to reflect different reward configurations. Rewards and constraints are modular and can be extended to incorporate new objectives.
The environment is customizable for different organisms by using species-specific codon tables preferences, and could serve as a benchmark environment in computational biology. This enables exploration of codon space, which is a large search space given a protein sequence, to optimize for mRNA design.  Applicable to mRNA vaccines, protein therapeutics, and gene expression optimization.
"""

from typing import Union

import torch
from torch import Tensor
from utils import (
    ALL_CODONS,
    IDX_TO_CODON,
    N_CODONS,
    compute_cai,
    compute_gc_content_vectorized,
    compute_mfe_energy,
    get_synonymous_indices,
)

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates


# --- mRNA Design Environment ---
class CodonDesignEnv(DiscreteEnv):
    """
    Environment for designing mRNA codon sequences for a given protein.
    Action space is the global codon set (size N_CODONS) plus an exit action.
    Dynamic masks restrict actions at each step:
    - At step t < seq_length: only synonymous codons for protein_seq[t] are allowed.
    - At step t == seq_length: only the exit action is allowed.

    """

    def __init__(
        self,
        protein_seq: str,
        device: torch.device,
        sf=None,
    ):

        self._device = device
        self.protein_seq = protein_seq
        self.seq_length = len(protein_seq)

        # Total possible actions = N_CODONS + 1 (for exit action)
        self.n_actions = N_CODONS + 1
        self.exit_action_index = N_CODONS  # Index for the exit action

        self.syn_indices = [get_synonymous_indices(aa) for aa in protein_seq]

        # Precompute GC counts for all codons
        self.codon_gc_counts = torch.tensor(
            [codon.count("G") + codon.count("C") for codon in ALL_CODONS],
            device=self._device,
            dtype=torch.float,
        )

        s0 = torch.full(
            (self.seq_length,), fill_value=-1, dtype=torch.long, device=self._device
        )
        sf = torch.full(
            (self.seq_length,), fill_value=0, dtype=torch.long, device=self._device
        )

        self.weights = torch.tensor([0.3, 0.3, 0.4]).to(device=self._device)

        super().__init__(
            n_actions=self.n_actions,
            s0=s0,
            state_shape=(self.seq_length,),
            action_shape=(1,),  # Each action is a single index
            sf=sf,
        )

        self.idx_to_codon = IDX_TO_CODON
        self.States: type[DiscreteStates] = self.States

    def set_weights(self, w: Union[list[float], Tensor]):
        """
        Store the current preference weights (w) for conditional reward.
        """
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32)

        self.weights = w

    def step(
        self,
        states: DiscreteStates,
        actions: Actions,
    ) -> DiscreteStates:

        states_tensor = states.tensor.to(self._device)
        batch_size = states_tensor.shape[0]
        current_length = (states_tensor != -1).sum(dim=1)

        max_length = states_tensor.shape[1]
        new_states = states_tensor.clone()
        valid_actions = actions.tensor.squeeze(-1)

        for i in range(batch_size):

            if (
                current_length[i].item() < max_length
                and valid_actions[i].item() != self.exit_action_index
            ):
                new_states[i, int(current_length[i].item())] = int(
                    valid_actions[i].item()
                )

        return self.States(new_states)

    def backward_step(
        self,
        states: DiscreteStates,
        actions: Actions,
    ) -> torch.Tensor:

        states_tensor = states.tensor
        batch_size, seq_len = states_tensor.shape
        current_length = (states_tensor != -1).sum(dim=1)
        new_states = states_tensor.clone()

        for i in range(batch_size):
            if current_length[i] > 0:
                new_states[i, current_length[i] - 1] = -1

        return new_states

    def update_masks(self, states: DiscreteStates) -> None:

        states_tensor = states.tensor
        batch_size = states_tensor.shape[0]
        current_length = (states_tensor != -1).sum(dim=1)

        forward_masks = torch.zeros(
            (batch_size, self.n_actions), dtype=torch.bool, device=self._device
        )
        backward_masks = torch.zeros(
            (batch_size, self.n_actions - 1), dtype=torch.bool, device=self._device
        )

        for i in range(batch_size):

            cl = int(current_length[i].item())

            if cl < self.seq_length:

                # Allow synonymous codons
                syns = self.syn_indices[cl]
                forward_masks[i, syns] = True

            elif cl == self.seq_length:
                # Allow only exit action
                forward_masks[i, self.exit_action_index] = True

            # Backward masks
            if cl > 0:
                last_codon = int(states_tensor[i, cl - 1].item())
                if last_codon >= 0:
                    backward_masks[i, last_codon] = True

        states.forward_masks = forward_masks
        states.backward_masks = backward_masks

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:

        states_tensor = final_states.tensor
        batch_size = states_tensor.shape[0]

        gc_percents = []
        mfe_energies = []
        cai_scores = []

        # Process each sequence individually
        for i in range(batch_size):

            seq_indices = states_tensor[i].long()

            # Compute GC content
            gc_percent = compute_gc_content_vectorized(
                seq_indices, codon_gc_counts=self.codon_gc_counts
            )
            gc_percents.append(gc_percent)

            # Compute MFE
            mfe_energy = compute_mfe_energy(seq_indices)
            mfe_energies.append(mfe_energy)

            # Compute CAI
            cai_score = compute_cai(seq_indices)
            cai_scores.append(cai_score)

        device = states_tensor.device
        gc_percent = torch.tensor(gc_percents, device=device, dtype=torch.float32)
        mfe_energy = torch.tensor(mfe_energies, device=device, dtype=torch.float32)
        cai_score = torch.tensor(cai_scores, device=device, dtype=torch.float32)

        # Calculate weighted reward
        reward_components = torch.stack([gc_percent, -mfe_energy, cai_score], dim=-1)
        reward = (reward_components * self.weights.to(device)).sum(dim=-1)

        return reward

    def is_terminal(self, states: DiscreteStates) -> torch.BoolTensor:
        states_tensor = states
        current_length = (states_tensor != -1).sum(dim=1)
        return current_length >= self.seq_length

    @staticmethod
    def make_sink_states_tensor(shape, device=None):
        return torch.zeros(shape, dtype=torch.long, device=device)
