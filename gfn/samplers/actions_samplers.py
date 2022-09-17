from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from gfn.containers import States
from gfn.estimators import LogEdgeFlowEstimator, LogitPBEstimator, LogitPFEstimator

# Typing
Tensor2D = TensorType["batch_size", "n_actions"]
Tensor2D2 = TensorType["batch_size", "n_steps"]
Tensor1D = TensorType["batch_size", torch.long]


class ActionsSampler(ABC):
    "Implements a method that samples actions from any given batch of states."

    def __init__(
        self,
        temperature: float = 1.0,
        sf_temperature: float = 0.0,
        epsilon: float = 0.0,
    ) -> None:
        # sf_temperature is a quantity to SUBTRACT from the logits of the final action.
        # with probability epsilon, an action is sampled uniformly at random.
        self.temperature = temperature
        self.sf_temperature = sf_temperature
        self.epsilon = epsilon

    @abstractmethod
    def get_raw_logits(self, states: States) -> Tensor2D:
        pass

    def get_logits(self, states: States) -> Tensor2D:
        logits = self.get_raw_logits(states)
        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        logits[~states.forward_masks] = -float("inf")
        return logits

    def get_probs(
        self,
        states: States,
    ) -> Tuple[Tensor2D, Tensor2D]:
        logits = self.get_logits(states)
        logits[..., -1] -= self.sf_temperature
        temperature = self.temperature
        probs = torch.softmax(logits / temperature, dim=-1)
        return logits, probs

    def sample(self, states: States) -> Tuple[Tensor2D, Tensor1D]:
        with torch.no_grad():
            logits, probs = self.get_probs(states)
        if self.epsilon > 0:
            probs = (
                1 - self.epsilon
            ) * probs + self.epsilon * states.forward_masks.float() / states.forward_masks.sum(
                dim=-1, keepdim=True
            ).float()

        return logits, Categorical(probs).sample()


class BackwardActionsSampler(ActionsSampler):
    def get_logits(self, states: States) -> Tensor2D:
        logits = self.get_raw_logits(states)
        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        logits[~states.backward_masks] = -float("inf")
        return logits

    def get_probs(
        self, states: States, temperature: Optional[float] = None
    ) -> Tensor2D:
        logits = self.get_logits(states)
        temperature = temperature if temperature is not None else self.temperature
        probs = torch.softmax(logits / temperature, dim=-1)
        # The following line is hack that works: when probs are nan, it means
        # that the state is already done (usually during backward sampling).
        # In which case, any action can be passed to the backward_step function
        # making the state stay at s_0
        probs = probs.nan_to_num(nan=1.0 / probs.shape[-1])
        return logits, probs


class LogitPFActionsSampler(ActionsSampler):
    def __init__(self, estimator: LogitPFEstimator, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator

    def get_raw_logits(self, states):
        return self.estimator(states)


class LogitPBActionsSampler(BackwardActionsSampler):
    def __init__(self, estimator: LogitPBEstimator, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator

    def get_raw_logits(self, states):
        return self.estimator(states)


class LogEdgeFlowsActionsSampler(ActionsSampler):
    def __init__(self, estimator: LogEdgeFlowEstimator, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator

    def get_raw_logits(self, states):
        logits = self.estimator(states)
        # env_rewards = self.estimator.preprocessor.env.reward(states)
        # env_log_rewards = torch.log(env_rewards).unsqueeze(-1)
        # all_logits = torch.cat([logits, env_log_rewards], dim=-1)
        return logits
