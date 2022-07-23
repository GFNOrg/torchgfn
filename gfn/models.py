import torch.nn as nn
import torch
from torch import Tensor
from typing import Union


class NeuralNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 architecture: Union[None, nn.Module] = None,
                 hidden_dim: Union[None, int] = None):
        super().__init__()
        self.input_dim = input_dim
        if architecture is None:
            assert hidden_dim is not None
            architecture = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, output_dim))
        self.logit_maker = architecture

    def forward(self, preprocessed_states: Tensor):
        logits = self.logit_maker(preprocessed_states)
        return logits


class Uniform(nn.Module):
    def __init__(self, output_dim):
        """
        :param n_actions: the number of all possible actions
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(self, preprocessed_states):
        logits = torch.zeros(
            *preprocessed_states.shape[:-1], self.output_dim).to(preprocessed_states.device)
        return logits
