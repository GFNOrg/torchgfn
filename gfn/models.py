import torch.nn as nn
import torch
from torch import Tensor
from torchtyping import TensorType
from typing import Union
from gfn.estimators import GFNModule

# Typing
batch_shape = None
input_dim = None
output_dim = None
InputTensor = TensorType['batch_shape', 'input_dim', float]
OutputTensor = TensorType['batch_shape', 'output_dim', float]


class NeuralNet(nn.Module, GFNModule):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 torso: Union[None, nn.Module] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if torso is None:
            self.torso = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()
                                       )
        else:
            self.torso = torso
        self.last_layer = nn.Linear(hidden_dim, output_dim)
            
    def forward(self, preprocessed_states: Tensor):
        logits = self.torso(preprocessed_states)
        logits = self.last_layer(logits)
        return logits

class Uniform(GFNModule):
    def __init__(self, output_dim):
        """
        :param n_actions: the number of all possible actions
        """
        self.input_dim = None
        self.output_dim = output_dim

    def __call__(self, preprocessed_states: Tensor):
        logits = torch.zeros(
            *preprocessed_states.shape[:-1], self.output_dim).to(preprocessed_states.device)
        return logits


if __name__ == '__main__':
    print('PF weights')
    pf = NeuralNet(input_dim=3, hidden_dim=4, output_dim=5)
    print(list(pf.named_parameters()))

    print('\n PB_tied weights')
    pb_tied = NeuralNet(input_dim=3, hidden_dim=4,
                        output_dim=5, torso=pf.torso)
    print(list(pb_tied.named_parameters()))

    print('\n PB_free weights')
    pb_free = NeuralNet(input_dim=3, hidden_dim=4, output_dim=5)
    print(list(pb_free.named_parameters()))


    from torch.optim import Adam

    optimizer_tied = Adam(pf.parameters(), lr=0.01)
    optimizer_free = Adam(pf.parameters(), lr=0.01)

    optimizer_tied.add_param_group({'params': pb_tied.last_layer.parameters(), 'lr': 0.01})
    optimizer_free.add_param_group({'params': pb_free.parameters(), 'lr': 0.01})

    print(optimizer_tied)
    print(optimizer_free)
    