from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

from gfn.preprocessors import IdentityPreprocessor, Preprocessor
from gfn.states import DiscreteStates, States
from gfn.utils.distributions import UnsqueezedCategorical

REDUCTION_FXNS = {
    "mean": torch.mean,
    "sum": torch.sum,
    "prod": torch.prod,
}


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
        _output_dim_is_checked: Flag for tracking whether the output dimensions of
            the states (after being preprocessed and transformed by the modules) have
            been verified.
        _is_backward: Flag for tracking whether this estimator is used for predicting
            probability distributions over parents.
    """

    def __init__(
        self,
        module: nn.Module,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
    ) -> None:
        """Initialize the GFNModule with nn.Module and a preprocessor.
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

    def forward(self, input: States | torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            input: The input to the module, as states or a tensor.

        Returns the output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        if isinstance(input, States):
            input = self.preprocessor(input)

        out = self.module(input)

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

    def check_output_dim(self, module_output: torch.Tensor) -> None:
        """Check that the output of the module has the correct shape. Raises an error if not."""
        assert module_output.dtype == torch.float
        if module_output.shape[-1] != self.expected_output_dim:
            raise ValueError(
                f"{self.__class__.__name__} output dimension should be {self.expected_output_dim}"
                + f" but is {module_output.shape[-1]}."
            )

    def to_probability_distribution(
        self,
        states: States,
        module_output: torch.Tensor,
        **policy_kwargs: Any,
    ) -> Distribution:
        """Transform the output of the module into a probability distribution.

        The kwargs modify a base distribution, for example to encourage exploration.

        Not all modules must implement this method, but it is required to define a
        policy from a module's outputs. See `DiscretePolicyEstimator` for an example
        using a categorical distribution, but note this can be done for all continuous
        distributions as well.

        Args:
            states: The states to use.
            module_output: The output of the module as a tensor of shape (*batch_shape, output_dim).
            **policy_kwargs: Keyword arguments to modify the distribution.

        Returns a distribution object.
        """
        raise NotImplementedError


class ScalarEstimator(GFNModule):
    r"""Class for estimating scalars such as LogZ or state flow functions of DB/SubTB.

    Training a GFlowNet requires sometimes requires the estimation of precise scalar
    values, such as the partition function of flows on the DAG. This Estimator is
    designed for those cases.

    The function approximator used for `module` need not directly output a scalar. If
    it does not, `reduction` will be used to aggregate the outputs of the module into
    a single scalar.

    Attributes:
        preprocessor: Preprocessor object that transforms raw States objects to tensors
            that can be used as input to the module. Optional, defaults to
            `IdentityPreprocessor`.
        module: The module to use. If the module is a Tabular module (from
            `gfn.utils.modules`), then the environment preprocessor needs to be an
            `EnumPreprocessor`.
        preprocessor: Preprocessor from the environment.
        _output_dim_is_checked: Flag for tracking whether the output dimensions of
            the states (after being preprocessed and transformed by the modules) have
            been verified.
        _is_backward: Flag for tracking whether this estimator is used for predicting
            probability distributions over parents.
            reduction_function: String denoting the
    """

    def __init__(
        self,
        module: nn.Module,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
        reduction: str = "mean",
    ):
        """Initialize the GFNModule with a scalar output.
        Args:
            module: The module to use. If the module is a Tabular module (from
                `gfn.utils.modules`), then the environment preprocessor needs to be an
                `EnumPreprocessor`.
            preprocessor: Preprocessor object.
            is_backward: Flags estimators of probability distributions over parents.
            reduction: str name of the one of the REDUCTION_FXNS keys: {}
        """.format(
            list(REDUCTION_FXNS.keys())
        )
        super().__init__(module, preprocessor, is_backward)
        assert reduction in REDUCTION_FXNS, "reduction function not one of {}".format(
            REDUCTION_FXNS.keys()
        )
        self.reduction_fxn = REDUCTION_FXNS[reduction]

    @property
    def expected_output_dim(self) -> int:
        return 1

    def forward(self, input: States | torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            input: The input to the module, as states or a tensor.

        Returns the output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        if isinstance(input, States):
            input = self.preprocessor(input)

        out = self.module(input)

        # Ensures estimator outputs are always scalar.
        if out.shape[-1] != 1:
            out = self.reduction_fxn(out, -1)

        if not self._output_dim_is_checked:
            # self.check_output_dim(out)
            self._output_dim_is_checked = True

        return out


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
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
    ):
        """Initializes a estimator for P_F for discrete environments.

        Args:
            n_actions: Total number of actions in the Discrete Environment.
            is_backward: if False, then this is a forward policy, else backward policy.
        """
        super().__init__(module, preprocessor, is_backward=is_backward)
        self.n_actions = n_actions

    @property
    def expected_output_dim(self) -> int:
        if self.is_backward:
            return self.n_actions - 1
        else:
            return self.n_actions

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: torch.Tensor,
        temperature: float = 1.0,
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
    ) -> Categorical:
        """Returns a probability distribution given a batch of states and module output.

        We handle off-policyness using these kwargs.

        Args:
            states: The states to use.
            module_output: The output of the module as a tensor of shape (*batch_shape, output_dim).
            temperature: scalar to divide the logits by before softmax. Does nothing
                if set to 1.0 (default), in which case it's on policy.
            sf_bias: scalar to subtract from the exit action logit before dividing by
                temperature. Does nothing if set to 0.0 (default), in which case it's
                on policy.
            epsilon: with probability epsilon, a random action is chosen. Does nothing
                if set to 0.0 (default), in which case it's on policy."""
        # self.check_output_dim(module_output)

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

        # LogEdgeFlows are greedy, as are most P_B.
        else:
            return UnsqueezedCategorical(logits=logits)


class ConditionalDiscretePolicyEstimator(DiscretePolicyEstimator):
    r"""Container for forward and backward policy estimators for discrete environments.

    $s \mapsto (P_F(s' \mid s, c))_{s' \in Children(s)}$.

    or

    $s \mapsto (P_B(s' \mid s, c))_{s' \in Parents(s)}$.

    Attributes:
        temperature: scalar to divide the logits by before softmax.
        sf_bias: scalar to subtract from the exit action logit before dividing by
            temperature.
        epsilon: with probability epsilon, a random action is chosen.
    """

    def __init__(
        self,
        state_module: nn.Module,
        conditioning_module: nn.Module,
        final_module: nn.Module,
        n_actions: int,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
    ):
        """Initializes a estimator for P_F for discrete environments.

        Args:
            n_actions: Total number of actions in the Discrete Environment.
            is_backward: if False, then this is a forward policy, else backward policy.
        """
        super().__init__(state_module, n_actions, preprocessor, is_backward)
        self.n_actions = n_actions
        self.conditioning_module = conditioning_module
        self.final_module = final_module

    def _forward_trunk(self, states: States, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass of the trunk of the module.

        Args:
            states: The input states.
            conditioning: The conditioning input.

        Returns the output of the trunk of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        state_out = self.module(self.preprocessor(states))
        conditioning_out = self.conditioning_module(conditioning)
        out = self.final_module(torch.cat((state_out, conditioning_out), -1))

        return out

    def forward(self, states: States, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            states: The input states.
            conditioning: The conditioning input.

        Returns the output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        out = self._forward_trunk(states, conditioning)

        if not self._output_dim_is_checked:
            # self.check_output_dim(out)
            self._output_dim_is_checked = True

        return out


class ConditionalScalarEstimator(ConditionalDiscretePolicyEstimator):
    r"""Class for conditionally estimating scalars (LogZ,  DB/SubTB state logF).

    Training a GFlowNet requires sometimes requires the estimation of precise scalar
    values, such as the partition function of flows on the DAG. In the case of a
    conditional GFN, the logZ or logF estimate is also conditional. This Estimator is
    designed for those cases.

    The function approximator used for `final_module` need not directly output a scalar.
    If it does not, `reduction` will be used to aggregate the outputs of the module into
    a single scalar.

    Attributes:
        preprocessor: Preprocessor object that transforms raw States objects to tensors
            that can be used as input to the module. Optional, defaults to
            `IdentityPreprocessor`.
        module: The module to use. If the module is a Tabular module (from
            `gfn.utils.modules`), then the environment preprocessor needs to be an
            `EnumPreprocessor`.
        preprocessor: Preprocessor from the environment.
        reduction_fxn: the selected torch reduction operation.
        _output_dim_is_checked: Flag for tracking whether the output dimensions of
            the states (after being preprocessed and transformed by the modules) have
            been verified.
        _is_backward: Flag for tracking whether this estimator is used for predicting
            probability distributions over parents.
            reduction_function: String denoting the
    """

    def __init__(
        self,
        state_module: nn.Module,
        conditioning_module: nn.Module,
        final_module: nn.Module,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
        reduction: str = "mean",
    ):
        """Initialize a conditional GFNModule with a scalar output.
        Args:
            state_module: The module to use for state representations. If the module is
                a Tabular module (from `gfn.utils.modules`), then the environment
                preprocessor needs to be an `EnumPreprocessor`.
            conditioning_module: The module to use for conditioning representations.
            final_module: The module to use for computing the final output.
            preprocessor: Preprocessor object.
            is_backward: Flags estimators of probability distributions over parents.
            reduction: str name of the one of the REDUCTION_FXNS keys: {}
        """.format(
            list(REDUCTION_FXNS.keys())
        )

        super().__init__(
            state_module,
            conditioning_module,
            final_module,
            n_actions=1,
            preprocessor=preprocessor,
            is_backward=is_backward,
        )
        assert reduction in REDUCTION_FXNS, "reduction function not one of {}".format(
            REDUCTION_FXNS.keys()
        )
        self.reduction_fxn = REDUCTION_FXNS[reduction]

    def forward(self, states: States, conditioning: torch.tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            states: The input states.
            conditioning: The tensor for conditioning.

        Returns the output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        out = self._forward_trunk(states, conditioning)

        # Ensures estimator outputs are always scalar.
        if out.shape[-1] != 1:
            out = self.reduction_fxn(out, -1)

        if not self._output_dim_is_checked:
            # self.check_output_dim(out)
            self._output_dim_is_checked = True

        return out

    @property
    def expected_output_dim(self) -> int:
        return 1

    def to_probability_distribution(
        self,
        states: States,
        module_output: torch.Tensor,
        **policy_kwargs: Any,
    ) -> Distribution:
        raise NotImplementedError
