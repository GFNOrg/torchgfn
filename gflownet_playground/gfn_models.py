import torch.nn as nn
import torch


class PF(nn.Module):
    """
    Module representing the forward transition probabilities function P_F.
    It outputs a distribution over "actions", rather than next states.
    """
    def __init__(self, input_dim, n_actions, preprocessor, architecture=None, h=None):
        """
        :param architecture: nn.Module with input layer of size input_dim and output layer of size n_actions
        :param input_dim: input size
        :param preprocessor: object of type gflownet_playground.utils.Preprocessor. The preprocessor's output size should be input_size
        :param n_actions: output size
        :param h: if architecture is not specified then it's the number of hidden units for each of the 2 hidden layers
        """
        super().__init__()
        self.input_dim = input_dim
        if architecture is None:
            assert h is not None
            architecture = nn.Sequential(nn.Linear(input_dim, h), nn.ReLU(),
             nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions))
        self.logit_maker = architecture
        self.preprocessor = preprocessor


    def forward(self, states, masks=None):
        """
        :param states: (batched) states of the GFN. tensor of size k x state_dim
        :param masks: optional BoolTensor of size k x n_actions, value True if impossible action
        :return: logits corresponding to P_F(. | s) represented as a tensor of size k x n_actions.
        """
        assert states.ndim == 2            
        preprocessed_states = self.preprocessor.preprocess(states)
        logits = self.logit_maker(preprocessed_states)
        if masks is not None:
            logits[masks] = - float('inf')
        return logits

class UniformPB(nn.Module):
    """
    Module representing the backward transition probabilities function P_B.
    It outputs a distribution over "actions", rather than previous states.
    """
    def __init__(self, n_actions):
        """
        :param n_actions: the number of all possible actions
        """
        super().__init__()
        self.n_actions = n_actions
    
    def forward(self, states, masks=None):
        """
        :param states: (batched) states of the GFN. tensor of size k x state_dim    
        :param masks: optional BoolTensor of size k x n_actions-1, value True if impossible action
        :return: logits corresponding to P_B(. | s) represented as a tensor of size k x n_actions-1.
        """
        logits = torch.zeros(states.shape[0], self.n_actions - 1)
        if masks is not None:
            logits[masks] = - float('inf')
        return logits


class UniformPF(nn.Module):
    """
    Module representing a uniform forward transition probabilities function P_F.
    It outputs a distribution over "actions", rather than next states.
    """
    def __init__(self, n_actions):
        """
        :param n_actions: the number of all possible actions
        """
        super().__init__()
        self.n_actions = n_actions
    
    def forward(self, states, masks=None):
        """
        :param states: (batched) states of the GFN. tensor of size k x state_dim    
        :param masks: optional BoolTensor of size k x n_actions, value True if impossible action
        :return: logits corresponding to P_F(. | s) represented as a tensor of size k x n_actions.
        """
        logits = torch.zeros(states.shape[0], self.n_actions)
        if masks is not None:
            logits[masks] = - float('inf')
        return logits