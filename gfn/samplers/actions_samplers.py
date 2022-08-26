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
        scheduler_gamma: Optional[float] = None,
        scheduler_milestones: Optional[List[int]] = None,
    ) -> None:
        # sf_temperature is a quantity to SUBTRACT from the logits of the final action.
        # This is useful for preventing the final action from being chosen too often.
        # scheduler_milestones is a list of milestones at which to adjust the temperature and sf_temperature.,
        # and scheduler_gamma is the factor by which to multiply the temperatures at each milestone
        # the step is incremented by 1 after each call to sample(), regardless of the batch size.
        self.temperature = temperature
        self.sf_temperature = sf_temperature
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_milestones = scheduler_milestones
        self.step = 0
        self.current_milestone_index = 0

    @abstractmethod
    def get_raw_logits(self, states: States) -> Tensor2D:
        pass

    def get_logits(self, states: States) -> Tensor2D:
        logits = self.get_raw_logits(states)
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
        logits, probs = self.get_probs(states)
        self.update_state()

        return logits, Categorical(probs).sample()

    def update_state(self) -> None:
        if self.scheduler_gamma is not None and self.scheduler_milestones is not None:
            if self.scheduler_milestones[self.current_milestone_index] == self.step:
                self.temperature *= self.scheduler_gamma
                self.sf_temperature *= self.scheduler_gamma
                self.current_milestone_index += 1
        self.step += 1


class BackwardActionsSampler(ActionsSampler):
    def get_logits(self, states: States) -> Tensor2D:
        logits = self.get_raw_logits(states)
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


class FixedActionsSampler(ActionsSampler):
    # Should be used for debugging and testing purposes.
    def __init__(self, actions: Tensor2D2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions = actions
        self.total_steps = actions.shape[1]

    def get_raw_logits(self, states: States) -> Tensor2D:
        logits = torch.ones_like(states.forward_masks, dtype=torch.float) * (
            -float("inf")
        )

        logits.scatter_(1, self.actions[:, self.step].unsqueeze(-1), 0.0)
        return logits


class UniformActionsSampler(ActionsSampler):
    def get_raw_logits(self, states):
        return torch.zeros_like(states.forward_masks, dtype=torch.float)


class UniformBackwardActionsSampler(BackwardActionsSampler):
    def get_raw_logits(self, states):
        return torch.zeros_like(states.backward_masks, dtype=torch.float)


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
        env_rewards = self.estimator.preprocessor.env.reward(states)
        env_log_rewards = torch.log(env_rewards).unsqueeze(-1)
        all_logits = torch.cat([logits, env_log_rewards], dim=-1)
        return all_logits
