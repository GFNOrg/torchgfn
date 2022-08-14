from abc import ABC, abstractmethod
import torch
from torchtyping import TensorType
from typing import Tuple

from gfn.envs.env import AbstractStatesBatch
from gfn.estimators import LogEdgeFlowEstimator, LogitPBEstimator, LogitPFEstimator
from torch.distributions import Categorical

# Typing
batch_size = None
n_actions = None
n_steps = None
Tensor2D = TensorType["batch_size", "n_actions"]
Tensor2D2 = TensorType["batch_size", "n_steps"]
Tensor1D = TensorType["batch_size", float]


class ActionSampler(ABC):
    "Implements a method that samples actions from any given batch of states."

    def __init__(self, temperature: float = 1.0, sf_temperature: float = 0.0) -> None:
        # sf_temperature is a quantity to SUBTRACT from the logits of the final action.
        self.temperature = temperature
        self.sf_temperature = sf_temperature

    @abstractmethod
    def get_raw_logits(self, states: AbstractStatesBatch) -> Tensor2D:
        pass

    def get_logits(self, states: AbstractStatesBatch) -> Tensor2D:
        logits = self.get_raw_logits(states)
        logits[~states.masks] = -float("inf")
        return logits

    def get_probs(self, states: AbstractStatesBatch) -> Tuple[Tensor2D, Tensor2D]:
        logits = self.get_logits(states)
        logits[..., -1] -= self.sf_temperature
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return logits, probs

    def sample(self, states: AbstractStatesBatch) -> Tuple[Tensor2D, Tensor1D]:
        logits, probs = self.get_probs(states)
        return logits, Categorical(probs).sample()


class BackwardsActionSampler(ActionSampler):
    def get_logits(self, states: AbstractStatesBatch) -> Tensor2D:
        logits = self.get_raw_logits(states)
        logits[~states.backward_masks] = -float("inf")
        return logits

    def get_probs(self, states: AbstractStatesBatch) -> Tensor2D:
        logits = self.get_logits(states)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        # The following line is hack that works: when probs are nan, it means
        # that the state is already done (usually during backwards sampling).
        # In which case, any action can be passed to the backward_step function
        # making the state stay at s_0
        probs = probs.nan_to_num(nan=1.0 / probs.shape[-1])
        return logits, probs


class FixedActions(ActionSampler):
    # Should be used for debugging and testing purposes.
    def __init__(self, actions: Tensor2D2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions = actions
        self.total_steps = actions.shape[1]
        self.step = 0

    def get_raw_logits(self, states: AbstractStatesBatch) -> Tensor2D:
        logits = torch.ones_like(states.masks, dtype=torch.float) * (-float("inf"))

        logits.scatter_(1, self.actions[:, self.step].unsqueeze(-1), 0.0)
        self.step += 1
        return logits


class UniformActionSampler(ActionSampler):
    def get_raw_logits(self, states):
        return torch.zeros_like(states.masks, dtype=torch.float)


class UniformBackwardsActionSampler(BackwardsActionSampler):
    def get_raw_logits(self, states):
        return torch.zeros_like(states.backward_masks, dtype=torch.float)


class LogitPFActionSampler(ActionSampler):
    def __init__(self, logit_PF: LogitPFEstimator, **kwargs):
        super().__init__(**kwargs)
        self.logit_PF = logit_PF

    def get_raw_logits(self, states):
        return self.logit_PF(states)


class LogitPBActionSampler(BackwardsActionSampler):
    def __init__(self, logit_PB: LogitPBEstimator, **kwargs):
        super().__init__(**kwargs)
        self.logit_PB = logit_PB

    def get_raw_logits(self, states):
        return self.logit_PB(states)


class LogEdgeFlowsActionSampler(ActionSampler):
    def __init__(self, log_edge_flow_estimator: LogEdgeFlowEstimator, **kwargs):
        super().__init__(**kwargs)
        self.log_edge_flow_estimator = log_edge_flow_estimator

    def get_raw_logits(self, states):
        logits = self.log_edge_flow_estimator(states)
        env_rewards = self.log_edge_flow_estimator.env.reward(states)
        env_log_rewards = torch.log(env_rewards).unsqueeze(-1)
        all_logits = torch.cat([logits, env_log_rewards], dim=-1)
        return all_logits
