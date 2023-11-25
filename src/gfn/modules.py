from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution
from torchtyping import TensorType as TT

from gfn.preprocessors import IdentityPreprocessor, Preprocessor
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

    The preprocessor is also encapsulated in the estimator.
    These function estimators implement the `__call__` method, which takes `States`
    objects as inputs and calls the module on the preprocessed states.

    Attributes:
        preprocessor: Preprocessor object that transforms raw States objects to tensors
            that can be used as input to the module. Optional, defaults to
            `IdentityPreprocessor`.
        module: The module to use. If the module is a Tabular module (from
            `gfn.utils.modules`), then the environment preprocessor needs to be an
            `EnumPreprocessor`.
        preprocessor: Preprocessor from the environment.
        _output_dim_is_checked: Flag for tracking whether the output dimenions of
            the states (after being preprocessed and transformed by the modules) have
            been verified.
    """

    def __init__(
        self,
        module: nn.Module,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
    ) -> None:
        """Initalize the FunctionEstimator with an environment and a module.
        Args:
            module: The module to use. If the module is a Tabular module (from
                `gfn.utils.modules`), then the environment preprocessor needs to be an
                `EnumPreprocessor`.
            preprocessor: Preprocessor object.
            is_backward: Flags estimators of probability distributions over parents.
        """
        nn.Module.__init__(self)
        self.module = module
        if preprocessor is None:
            assert hasattr(module, "input_dim"), (
                "Module needs to have an attribute `input_dim` specifying the input "
                + "dimension, in order to use the default IdentityPreprocessor."
            )
            preprocessor = IdentityPreprocessor(module.input_dim)
        self.preprocessor = preprocessor
        self._output_dim_is_checked = False
        self.is_backward = is_backward

    def forward(self, states: States) -> TT["batch_shape", "output_dim", float]:
        out = self.module(self.preprocessor(states))
        if not self._output_dim_is_checked:
            self.check_output_dim(out)
            self._output_dim_is_checked = True

        return out

    def __repr__(self):
        return f"{self.__class__.__name__} module"

    @property
    @abstractmethod
    def expected_output_dim(self) -> int:
        """Expected output dimension of the module."""

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
        *args,
    ) -> Distribution:
        """Transform the output of the module into a probability distribution.

        The kwargs modify a base distribution, for example to encourage exploration.

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
    r"""Container for forward and backward policy estimators for discrete environments.

    $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$.

    or

    $s \mapsto (P_B(s' \mid s))_{s' \in Parents(s)}$.

    Attributes:
        temperature: scalar to divide the logits by before softmax.
        sf_bias: scalar to subtract from the exit action logit before dividing by
            temperature.
        epsilon: with probability epsilon, a random action is chosen.
    """

    def __init__(
        self,
        module: nn.Module,
        n_actions: int,
        preprocessor: Preprocessor | None,
        is_backward: bool = False,
    ):
        """Initializes a estimator for P_F for discrete environments.

        Args:
            n_actions: Total number of actions in the Discrete Environment.
            is_backward: if False, then this is a forward policy, else backward policy.
        """
        super().__init__(module, preprocessor, is_backward=is_backward)
        self.n_actions = n_actions

    def expected_output_dim(self) -> int:
        if self.is_backward:
            return self.n_actions - 1
        else:
            return self.n_actions

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: TT["batch_shape", "output_dim", float],
        temperature: float = 1.0,
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
    ) -> Categorical:
        """Returns a probability distribution given a batch of states and module output.

        Args:
            temperature: scalar to divide the logits by before softmax. Does nothing
                if set to 1.0 (default), in which case it's on policy.
            sf_bias: scalar to subtract from the exit action logit before dividing by
                temperature. Does nothing if set to 0.0 (default), in which case it's
                on policy.
            epsilon: with probability epsilon, a random action is chosen. Does nothing
                if set to 0.0 (default), in which case it's on policy."""
        masks = states.backward_masks if self.is_backward else states.forward_masks
        logits = module_output
        logits[~masks] = -float("inf")

        # Forward policy supports exploration in many implementations.
        if temperature != 1.0 or sf_bias != 0.0 or epsilon != 0.0:
            logits[:, -1] -= sf_bias
            probs = torch.softmax(logits / temperature, dim=-1)
            uniform_dist_probs = masks.float() / masks.sum(dim=-1, keepdim=True)
            probs = (1 - epsilon) * probs + epsilon * uniform_dist_probs

            return UnsqueezedCategorical(probs=probs)

        # LogEdgeFlows are greedy, as are more P_B.
        else:
            return UnsqueezedCategorical(logits=logits)
