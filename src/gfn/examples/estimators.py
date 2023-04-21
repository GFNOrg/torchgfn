from gfn.estimators import ProbabilityEstimator

from torch.distributions import Distribution, Categorical
from torchtyping import TensorType

from gfn.envs import DiscreteEnv
from gfn.states import DiscreteStates

# Typing
OutputTensor = TensorType["batch_shape", "output_dim", float]


class DiscretePFEstimator(ProbabilityEstimator):
    r"""Container for estimators $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$.

    Note that while this class resembles LogEdgeFlowProbabilityEstimator, they have different semantic meaning.
    With LogEdgeFlowEstimator, the module output is the log of the flow from the parent to the child,
    while with DiscretePFEstimator, the module output is arbitrary.
    """

    def check_output_dim(self, module_output: OutputTensor):
        if not isinstance(self.env, DiscreteEnv):
            raise ValueError("DiscretePFEstimator only supports discrete environments.")
        if module_output.shape[-1] != self.env.n_actions:
            raise ValueError(
                f"DiscretePFEstimator output dimension should be {self.env.n_actions}, but is {module_output.shape[-1]}."
            )

    def to_probability_distribution(
        self, states: DiscreteStates, module_output: OutputTensor
    ) -> Distribution:
        logits = module_output
        logits[~states.forward_masks] = -float("inf")

        return Categorical(logits=logits)


class DiscretePBEstimator(ProbabilityEstimator):
    r"""Container for estimators $s \mapsto (P_B(s' \mid s))_{s' \in Parents(s)}$"""

    def check_output_dim(self, module_output: OutputTensor):
        if not isinstance(self.env, DiscreteEnv):
            raise ValueError("DiscretePBEstimator only supports discrete environments.")
        if module_output.shape[-1] != self.env.n_actions - 1:
            raise ValueError(
                f"DiscretePBEstimator output dimension should be {self.env.n_actions - 1}, but is {module_output.shape[-1]}."
            )

    def to_probability_distribution(
        self, states: DiscreteStates, module_output: OutputTensor
    ) -> Distribution:
        logits = module_output
        logits[~states.backward_masks] = -float("inf")

        return Categorical(logits=logits)
