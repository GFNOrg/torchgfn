from abc import ABC, abstractmethod
import torch
from torchtyping import TensorType

from gfn.envs.env import AbstractStatesBatch
from gfn.preprocessors.base import Preprocessor
from torch.distributions import Categorical


class ActionSampler(ABC):
    "Implements a method that samples actions from any given batch of states."

    def __init__(self, temperature: float = 1., sf_temperature: float = 0.) -> None:
        # sf_temperature is a quantity to SUBTRACT from the logits of the final action.
        self.temperature = temperature
        self.sf_temperature = sf_temperature

    @abstractmethod
    def get_raw_logits(self, states: AbstractStatesBatch) -> TensorType['batch_size', 'n_actions']:
        pass

    def get_logits(self, states: AbstractStatesBatch) -> TensorType['batch_size', 'n_actions']:
        logits = self.get_raw_logits(states)
        logits[~states.masks] = -float('inf')
        return logits

    def get_probs(self, states: AbstractStatesBatch) -> TensorType['batch_size', 'n_actions']:
        logits = self.get_logits(states)
        logits[..., -1] -= self.sf_temperature
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return logits, probs

    def sample(self, states: AbstractStatesBatch) -> TensorType['batch_size', torch.long]:
        logits, probs = self.get_probs(states)
        return logits, Categorical(probs).sample()


class BackwardsActionSampler(ActionSampler):
    def get_logits(self, states: AbstractStatesBatch) -> TensorType['batch_size', 'n_actions']:
        logits = self.get_raw_logits(states)
        logits[~states.backward_masks] = -float('inf')
        return logits

    def get_probs(self, states: AbstractStatesBatch) -> TensorType['batch_size', 'n_actions-1']:
        logits = self.get_logits(states)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        # The following line is hack that works: when probs are nan, it means
        # that the state is already done (usually during backwards sampling).
        # In which case, any action can be passed to the backward_step function
        # making the state stay at s_0
        probs = probs.nan_to_num(nan=1./probs.shape[-1])
        return logits, probs


class FixedActions(ActionSampler):
    # Should be used for debugging and testing purposes.
    def __init__(self, actions: TensorType['batch_size', 'n_steps'], **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions = actions
        self.total_steps = actions.shape[1]
        self.step = 0

    def get_raw_logits(self, states: AbstractStatesBatch) -> TensorType['batch_size', 'n_actions']:
        logits = torch.ones_like(
            states.masks, dtype=torch.float) * (-float('inf'))

        logits.scatter_(1, self.actions[:, self.step].unsqueeze(-1), 0.)
        self.step += 1
        return logits


class UniformActionSampler(ActionSampler):
    def get_raw_logits(self, states):
        return torch.zeros_like(states.masks, dtype=torch.float)


class UniformBackwardsActionSampler(BackwardsActionSampler):
    def get_raw_logits(self, states):
        return torch.zeros_like(states.backward_masks, dtype=torch.float)


class ModuleActionSampler(ActionSampler):
    def __init__(self, preprocessor: Preprocessor, module: torch.nn.Module,
                 temperature: float = 1., sf_temperature: float = 0.) -> None:
        # module needs to return logits ! not probs
        super().__init__(temperature, sf_temperature)
        self.preprocessor = preprocessor
        self.module = module

    def get_raw_logits(self, states):
        preprocessed_states = self.preprocessor.preprocess(states)
        logits = self.module(preprocessed_states)
        return logits


class BackwardsModuleActionSampler(BackwardsActionSampler, ModuleActionSampler):
    pass
