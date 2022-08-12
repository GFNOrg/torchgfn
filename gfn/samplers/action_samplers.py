from abc import ABC, abstractmethod
import torch
from torchtyping import TensorType  # type: ignore

from gfn.envs.env import AbstractStatesBatch
from gfn.preprocessors.base import Preprocessor
from torch.distributions import Categorical

# Typing
batch_size = None
n_actions = None
n_steps = None
Tensor2D = TensorType['batch_size', 'n_actions']
Tensor2D2 = TensorType['batch_size', 'n_steps']
Tensor1D = TensorType['batch_size', torch.long]


class ActionSampler(ABC):
    "Implements a method that samples actions from any given batch of states."

    def __init__(self, temperature: float = 1., sf_temperature: float = 0.) -> None:
        # sf_temperature is a quantity to SUBTRACT from the logits of the final action.
        self.temperature = temperature
        self.sf_temperature = sf_temperature

    @abstractmethod
    def get_raw_logits(self, states: AbstractStatesBatch) -> Tensor2D:
        pass

    def get_logits(self, states: AbstractStatesBatch) -> Tensor2D:
        logits = self.get_raw_logits(states)
        logits[~states.masks] = -float('inf')
        return logits

    def get_probs(self, states: AbstractStatesBatch) -> Tensor2D:
        logits = self.get_logits(states)
        logits[..., -1] -= self.sf_temperature
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return logits, probs

    def sample(self, states: AbstractStatesBatch) -> Tensor1D:
        logits, probs = self.get_probs(states)
        return logits, Categorical(probs).sample()


class BackwardsActionSampler(ActionSampler):
    def get_logits(self, states: AbstractStatesBatch) -> Tensor2D:
        logits = self.get_raw_logits(states)
        logits[~states.backward_masks] = -float('inf')
        return logits

    def get_probs(self, states: AbstractStatesBatch) -> Tensor2D:
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
    def __init__(self, actions: Tensor2D2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions = actions
        self.total_steps = actions.shape[1]
        self.step = 0

    def get_raw_logits(self, states: AbstractStatesBatch) -> Tensor2D:
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
