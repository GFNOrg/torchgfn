import torch
from torch.distributions import Distribution
import torch.nn as nn
from torchtyping import TensorType as TType

from gfn.envs import DiscreteEnv, Env
from gfn.estimators import ProbabilityEstimator
from gfn.states import DiscreteStates
from gfn.utils.distributions import UnsqueezedCategorical


class DiscretePFEstimator(ProbabilityEstimator):
    r"""Container for estimators $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$.

    Note that while this class resembles LogEdgeFlowProbabilityEstimator, they have different semantic meaning.
    With LogEdgeFlowEstimator, the module output is the log of the flow from the parent to the child,
    while with DiscretePFEstimator, the module output is arbitrary.
    """

    def __init__(
        self,
        env: Env,
        module: nn.Module,
        temperature: float = 1.0,
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
    ):
        """Initializes a estimator for P_F for discrete environments.

        Args:
            temperature (float, optional): scalar to divide the logits by before softmax. Defaults to 1.0.
            sf_bias (float, optional): scalar to subtract from the exit action logit before dividing by temperature. Defaults to 0.0.
            epsilon (float, optional): with probability epsilon, a random action is chosen. Defaults to 0.0.
        """
        super().__init__(env, module)
        self.temperature = temperature
        self.sf_bias = sf_bias
        self.epsilon = epsilon

    def check_output_dim(
        self, module_output: TType["batch_shape", "output_dim", float]
    ):
        if not isinstance(self.env, DiscreteEnv):
            raise ValueError("DiscretePFEstimator only supports discrete environments.")
        if module_output.shape[-1] != self.env.n_actions:
            raise ValueError(
                f"DiscretePFEstimator output dimension should be {self.env.n_actions}, but is {module_output.shape[-1]}."
            )

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: TType["batch_shape", "output_dim", float],
    ) -> Distribution:
        logits = module_output
        logits[~states.forward_masks] = -float("inf")
        logits[:, -1] -= self.sf_bias
        probs = torch.softmax(logits / self.temperature, dim=-1)

        uniform_dist_probs = states.forward_masks.float() / states.forward_masks.sum(
            dim=-1, keepdim=True
        )
        probs = (1 - self.epsilon) * probs + self.epsilon * uniform_dist_probs

        return UnsqueezedCategorical(probs=probs)


class DiscretePBEstimator(ProbabilityEstimator):
    r"""Container for estimators $s \mapsto (P_B(s' \mid s))_{s' \in Parents(s)}$"""

    def check_output_dim(
        self, module_output: TType["batch_shape", "output_dim", float]
    ):
        if not isinstance(self.env, DiscreteEnv):
            raise ValueError("DiscretePBEstimator only supports discrete environments.")
        if module_output.shape[-1] != self.env.n_actions - 1:
            raise ValueError(
                f"DiscretePBEstimator output dimension should be {self.env.n_actions - 1}, but is {module_output.shape[-1]}."
            )

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: TType["batch_shape", "output_dim", float],
    ) -> Distribution:
        logits = module_output
        logits[~states.backward_masks] = -float("inf")

        return UnsqueezedCategorical(logits=logits)