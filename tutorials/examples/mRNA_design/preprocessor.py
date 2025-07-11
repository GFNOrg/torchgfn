import torch
import torch.nn as nn
from utils import N_CODONS

from gfn.preprocessors import Preprocessor
from gfn.states import States


class CodonSequencePreprocessor(Preprocessor):
    """Preprocessor for codon sequence states"""

    def __init__(self, seq_length: int, embedding_dim: int, device: torch.device):
        super().__init__(output_dim=seq_length * embedding_dim)
        self.device = device
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            N_CODONS + 1, embedding_dim, padding_idx=N_CODONS
        ).to(device)

    def preprocess(self, states: States) -> torch.Tensor:

        states_tensor = states.tensor.long().clone()
        states_tensor[states_tensor == -1] = N_CODONS
        states_tensor = states_tensor.to(self.device)
        embedded = self.embedding(states_tensor)

        out = embedded.view(states_tensor.shape[0], -1)
        return out
