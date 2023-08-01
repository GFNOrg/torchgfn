from typing import Tuple

import torch
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.modules import GFNModule
from gfn.states import States


# TODO: Environment should not live inside the estimator and here... needs refactor.
class ActionsSampler:
    """`ActionSampler`s are containers for `PolicyEstimator`s.

    Attributes:
        estimator: the submitted PolicyEstimator.
        env: the Environment instance inside the PolicyEstimator.
    """

    def __init__(self, estimator: GFNModule) -> None:
        self.estimator = estimator
        self.env = estimator.env

    def sample(self, states: States) -> Tuple[Actions, TT["batch_shape", torch.float]]:
        """Samples actions from the given states.

        Args:
            states (States): A batch of states.

        Returns:
            A tuple of tensors containing:
             - An Actions object containing the sampled actions.
             - A tensor of shape (*batch_shape,) containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
        """
        module_output = self.estimator(states)
        dist = self.estimator.to_probability_distribution(states, module_output)

        with torch.no_grad():
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if torch.any(torch.isinf(log_probs)):
            raise RuntimeError("Log probabilities are inf. This should not happen.")

        return self.env.Actions(actions), log_probs


# TODO: Remove rest of file.

# class ActionsSampler2(ABC):
#     """
#     Base class for action sampling methods.
#     """

#     @abstractmethod
#     def sample(self, states: States) -> Tuple[TT["batch_shape", torch.long], TT["batch_shape", torch.long]]:
#         """
#         Args:
#             states (States): A batch of states.

#         Returns:
#             Tuple[Tensor[batch_size], Tensor[batch_size]]: A tuple of tensors containing the log probabilities of the sampled actions, and the sampled actions.
#         """
#         pass


# class BackwardActionsSampler(ActionsSampler):
#     """
#     Base class for backward action sampling methods.
#     """

#     pass


# class DiscreteActionsSampler(ActionsSampler):
#     """
#     For Discrete environments.
#     """

#     def __init__(
#         self,
#         estimator: DiscretePFEstimator | LogEdgeFlowEstimator,
#         temperature: float = 1.0,
#         sf_bias: float = 0.0,
#         epsilon: float = 0.0,
#     ) -> None:
#         """Implements a method that samples actions from any given batch of states.

#         Args:
#             temperature (float, optional): scalar to divide the logits by before softmax. Defaults to 1.0.
#             sf_bias (float, optional): scalar to subtract from the exit action logit before dividing by temperature. Defaults to 0.0.
#             epsilon (float, optional): with probability epsilon, a random action is chosen. Defaults to 0.0.
#         """
#         self.estimator = estimator
#         self.temperature = temperature
#         self.sf_bias = sf_bias
#         self.epsilon = epsilon

#     def get_raw_logits(self, states: States) -> TensorBatchActions:
#         """
#         This is before illegal actions are masked out and the exit action is biased.
#         Should be used for Discrete action spaces only.

#         Returns:
#             A 2D tensor of shape (batch_size, n_actions) containing the logits for each action in each state in the batch.
#         """
#         logits = self.estimator(states)
#         return logits

#     def get_logits(self, states: States) -> TT["batch_shape", "action_shape"]:
#         """Transforms the raw logits by masking illegal actions.

#         Raises:
#             ValueError: if one of the resulting logits is NaN.

#         Returns:
#             A 2D tensor of shape (batch_size, n_actions) containing the transformed logits.
#         """
#         logits = self.get_raw_logits(states)

#         if torch.any(torch.all(torch.isnan(logits), 1)):
#             raise ValueError("NaNs in estimator")
#         states.forward_masks, _ = correct_cast(
#             states.forward_masks, states.backward_masks
#         )
#         logits[~states.forward_masks] = -float("inf")
#         return logits

#     def get_probs(
#         self,
#         states: States,
#     ) -> TT["batch_shape", "action_shape"]:
#         """
#         Returns:
#             The probabilities of each action in each state in the batch.
#         """
#         logits = self.get_logits(states)
#         logits[..., -1] -= self.sf_bias
#         probs = torch.softmax(logits / self.temperature, dim=-1)
#         return probs

#     def sample(self, states: States) -> Tuple[TT["batch_shape", torch.long], TT["batch_shape", torch.long]]:
#         probs = self.get_probs(states)
#         states.forward_masks, _ = correct_cast(
#             states.forward_masks, states.backward_masks
#         )
#         if self.epsilon > 0:
#             uniform_dist = (
#                 states.forward_masks.float()
#                 / states.forward_masks.sum(dim=-1, keepdim=True).float()
#             )
#             probs = (1 - self.epsilon) * probs + self.epsilon * uniform_dist
#         dist = Categorical(probs=probs)
#         with torch.no_grad():
#             actions = dist.sample()
#         actions_log_probs = dist.log_prob(actions)

#         return actions_log_probs, actions


# class BackwardDiscreteActionsSampler(DiscreteActionsSampler, BackwardActionsSampler):
#     """
#     For sampling backward actions in discrete environments.
#     """

#     def __init__(
#         self,
#         estimator: DiscretePBEstimator,
#         temperature: float = 1.0,
#         epsilon: float = 0.0,
#     ) -> None:
#         """s_f is not biased in the backward sampler."""
#         super().__init__(
#             estimator, temperature=temperature, sf_bias=0.0, epsilon=epsilon
#         )

#     def get_logits(self, states: States) -> TT["batch_shape", "action_shape"]:
#         logits = self.get_raw_logits(states)
#         if torch.any(torch.all(torch.isnan(logits), 1)):
#             raise ValueError("NaNs in estimator")
#         _, states.backward_masks = correct_cast(
#             states.forward_masks, states.backward_masks
#         )
#         logits[~states.backward_masks] = -float("inf")
#         return logits

#     def get_probs(self, states: States) -> TT["batch_shape", "action_shape"]:
#         logits = self.get_logits(states)
#         probs = torch.softmax(logits / self.temperature, dim=-1)
#         # The following line is hack that works: when probs are nan, it means
#         # that the state is already done (usually during backward sampling).
#         # In which case, any action can be passed to the backward_step function
#         # making the state stay at s_0
#         probs = probs.nan_to_num(nan=1.0 / probs.shape[-1])
#         return probs
