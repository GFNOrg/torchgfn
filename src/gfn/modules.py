from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution
from torchtyping import TensorType as TT

from gfn.env import DiscreteEnv, Env
from gfn.states import DiscreteStates, States
from gfn.utils.distributions import UnsqueezedCategorical


class GFNModule(ABC, nn.Module):
    r"""Base class for modules mapping states distributions.

    Training a GFlowNet requires parameterizing one or more of the following functions:
    - $s \mapsto (\log F(s \rightarrow s'))_{s' \in Children(s)}$
    - $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$
    - $s' \mapsto (P_B(s \mid s'))_{s \in Parents(s')}$
    - $s \mapsto (\log F(s))_{s \in States}$

    This class is the base class for all such function estimators. The estimators need
    to encapsulate a `nn.Module`, which takes a a batch of preprocessed states as input
    and outputs a batch of outputs of the desired shape. When the goal is to represent
    a probability distribution, the outputs would correspond to the parameters of the
    distribution, e.g. logits for a categorical distribution for discrete environments.

    The call method is used to output logits, or the parameters to distributions.
    Otherwise, one can overwrite and use the to_probability_distribution() method to
    directly output a probability distribution.

    The preprocessor is also encapsulated in the estimator via the environment.
    These function estimators implement the `__call__` method, which takes `States`
    objects as inputs and calls the module on the preprocessed states.

    Attributes:
        env: the environment.
        module: The module to use. If the module is a Tabular module (from
            `gfn.utils.modules`), then the environment preprocessor needs to be an
            `EnumPreprocessor`.
        preprocessor: Preprocessor from the environment.
        _output_dim_is_checked: Flag for tracking whether the output dimenions of
            the states (after being preprocessed and transformed by the modules) have
            been verified.
    """

    def __init__(self, env: Env, module: nn.Module) -> None:
        """Initalize the FunctionEstimator with an environment and a module.
        Args:
            env: the environment.
            module: The module to use. If the module is a Tabular module (from
                `gfn.utils.modules`), then the environment preprocessor needs to be an
                `EnumPreprocessor`.
        """
        nn.Module.__init__(self)
        self.env = env
        self.module = module
        self.preprocessor = env.preprocessor  # TODO: passed explicitly?
        self._output_dim_is_checked = False

    def forward(self, states: States) -> TT["batch_shape", "output_dim", float]:
        out = self.module(self.preprocessor(states))
        if not self._output_dim_is_checked:
            self.check_output_dim(out)
            self._output_dim_is_checked = True

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.env})"

    @property
    @abstractmethod
    def expected_output_dim(self) -> int:
        """Expected output dimension of the module."""
        pass

    def check_output_dim(
        self, module_output: TT["batch_shape", "output_dim", float]
    ) -> None:
        """Check that the output of the module has the correct shape. Raises an error if not."""
        if module_output.shape[-1] != self.expected_output_dim():
            raise ValueError(
                f"{self.__class__.__name__} output dimension should be {self.expected_output_dim()}"
                + f" but is {module_output.shape[-1]}."
            )

    def to_probability_distribution(
        self,
        states: States,
        module_output: TT["batch_shape", "output_dim", float],
    ) -> Distribution:
        """Transform the output of the module into a probability distribution.

        Not all modules must implement this method, but it is required to define a
        policy from a module's outputs. See `DiscretePolicyEstimator` for an example
        using a categorical distribution, but note this can be done for all continuous
        distributions as well.
        """
        raise NotImplementedError


class ScalarEstimator(GFNModule):
    def expected_output_dim(self) -> int:
        return 1


class DiscretePolicyEstimator(GFNModule):
    r"""Container for forward and backward policy estimators.

    $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$.

    or

    $s \mapsto (P_B(s' \mid s))_{s' \in Parents(s)}$.

    Note that while this class resembles LogEdgeFlowProbabilityEstimator, they have
    different semantic meaning. With LogEdgeFlowEstimator, the module output is the log
    of the flow from the parent to the child, while with DiscretePFEstimator, the
    module output is arbitrary.

    Attributes:
        temperature: scalar to divide the logits by before softmax.
        sf_bias: scalar to subtract from the exit action logit before dividing by
            temperature.
        epsilon: with probability epsilon, a random action is chosen.
    """

    def __init__(
        self,
        env: Env,
        module: nn.Module,
        forward: bool,
        greedy_eps: float = 0.0,
        temperature: float = 1.0,
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
    ):
        """Initializes a estimator for P_F for discrete environments.

        Args:
            forward: if True, then this is a forward policy, else backward policy.
            greedy_eps: if > 0 , then we go off policy using greedy epsilon exploration.
            temperature: scalar to divide the logits by before softmax. Does nothing
                if greedy_eps is 0.
            sf_bias: scalar to subtract from the exit action logit before dividing by
                temperature. Does nothing if greedy_eps is 0.
            epsilon: with probability epsilon, a random action is chosen. Does nothing
                if greedy_eps is 0.
        """
        super().__init__(env, module)
        assert greedy_eps >= 0
        self._forward = forward
        self._greedy_eps = greedy_eps
        self.temperature = temperature
        self.sf_bias = sf_bias
        self.epsilon = epsilon

    @property
    def greedy_eps(self):
        return self._greedy_eps

    def expected_output_dim(self) -> int:
        if self._forward:
            return self.env.n_actions
        else:
            return self.env.n_actions - 1

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: TT["batch_shape", "output_dim", float],
    ) -> Categorical:
        """Returns a probability distribution given a batch of states and module output."""
        masks = states.forward_masks if self._forward else states.backward_masks
        logits = module_output
        logits[~masks] = -float("inf")

        # Forward policy supports exploration in many implementations.
        if self._greedy_eps:
            logits[:, -1] -= self.sf_bias
            probs = torch.softmax(logits / self.temperature, dim=-1)
            uniform_dist_probs = masks.float() / masks.sum(dim=-1, keepdim=True)
            probs = (1 - self.epsilon) * probs + self.epsilon * uniform_dist_probs

            return UnsqueezedCategorical(probs=probs)

        # LogEdgeFlows are greedy, as are more P_B.
        else:
            return UnsqueezedCategorical(logits=logits)
