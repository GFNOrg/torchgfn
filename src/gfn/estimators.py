from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Categorical, Distribution

from gfn.actions import GraphActions, GraphActionType
from gfn.preprocessors import IdentityPreprocessor, Preprocessor
from gfn.states import DiscreteStates, States
from gfn.utils.distributions import GraphActionDistribution, UnsqueezedCategorical

REDUCTION_FUNCTIONS = {
    "mean": torch.mean,
    "sum": torch.sum,
    "prod": torch.prod,
}


class Estimator(ABC, nn.Module):
    r"""Base class for modules mapping states to distributions or scalar values.

    Training a GFlowNet requires parameterizing one or more of the following functions:
    - $s \mapsto (\log F(s \rightarrow s'))_{s' \in Children(s)}$
    - $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$
    - $s' \mapsto (P_B(s \mid s'))_{s \in Parents(s')}$
    - $s \mapsto (\log F(s))_{s \in States}$

    This class is the base class for all such function estimators. The estimators need
    to encapsulate a `nn.Module`, which takes a batch of preprocessed states as input
    and outputs a batch of outputs of the desired shape. When the goal is to represent
    a probability distribution, the outputs would correspond to the parameters of the
    distribution, e.g. logits for a categorical distribution for discrete environments.

    The call method is used to output logits, or the parameters to distributions.
    Otherwise, one can overwrite and use the `to_probability_distribution()` method to
    directly output a probability distribution.

    The preprocessor is also encapsulated in the estimator.
    These function estimators implement the `__call__` method, which takes `States`
    objects as inputs and calls the module on the preprocessed states.

    Attributes:
        module: The neural network module to use. If it is a Tabular module (from
            `gfn.utils.modules`), then the environment preprocessor needs to be an
            `EnumPreprocessor`.
        preprocessor: Preprocessor object that transforms raw States objects to tensors
            that can be used as input to the module. Optional, defaults to
            `IdentityPreprocessor`.
        is_backward: Flag indicating whether this estimator is for backward policy,
            i.e., is used for predicting probability distributions over parents.
    """

    def __init__(
        self,
        module: nn.Module,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
    ) -> None:
        """Initializes an Estimator with a neural network module and a preprocessor.

        Args:
            module: The neural network module to use.
            preprocessor: Preprocessor object that transforms states to tensors. If None,
                uses `IdentityPreprocessor` with the module's input_dim.
            is_backward: Flag indicating whether this estimator is for backward policy,
                i.e., is used for predicting probability distributions over parents.
        """
        nn.Module.__init__(self)
        self.module = module
        if preprocessor is None:
            assert hasattr(module, "input_dim") and isinstance(module.input_dim, int), (
                "Module needs to have an attribute `input_dim` specifying the input "
                + "dimension, in order to use the default IdentityPreprocessor."
            )
            preprocessor = IdentityPreprocessor(module.input_dim)
        self.preprocessor = preprocessor
        self.is_backward = is_backward

    def forward(self, input: States) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            input: The input to the module as states.

        Returns:
            The output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        out = self.module(self.preprocessor(input))
        if self.expected_output_dim is not None:
            assert out.shape[-1] == self.expected_output_dim, (
                f"Module output shape {out.shape} does not match expected output "
                f"dimension {self.expected_output_dim}"
            )
        return out

    def __repr__(self):
        """Returns a string representation of the Estimator.

        Returns:
            A string summary of the Estimator.
        """
        return f"{self.__class__.__name__} module"

    @property
    @abstractmethod
    def expected_output_dim(self) -> Optional[int]:
        """Expected output dimension of the module.

        Returns:
            The expected output dimension of the module, or None if the output dimension
            is not well-defined (e.g., when the output is a TensorDict for GraphActions).
        """

    def to_probability_distribution(
        self,
        states: States,
        module_output: torch.Tensor,
        **policy_kwargs: Any,
    ) -> Distribution:
        """Transforms the output of the module into a probability distribution.

        The kwargs may contain parameters to modify a base distribution, for example to
        encourage exploration.

        This method is optional for modules that don't need to output probability
        distributions (e.g., when estimating logF for flow matching). However, it is
        required for modules that define policies, as it converts raw module outputs into
        probability distributions over actions. See `DiscretePolicyEstimator` for an
        example using categorical distributions for discrete actions, or `BoxPFEstimator`
        for examples using continuous distributions like Beta mixtures.

        Args:
            states: The states to use.
            module_output: The output of the module as a tensor of shape
                (*batch_shape, output_dim).
            **policy_kwargs: Keyword arguments to modify the distribution.

        Returns:
            A distribution object.
        """
        raise NotImplementedError


class ScalarEstimator(Estimator):
    r"""Class for estimating scalars such as logZ of TB or state/edge flows of DB/SubTB.

    Training a GFlowNet sometimes requires the estimation of precise scalar values,
    such as the partition function (for TB) or state/edge flows (for DB/SubTB).
    This Estimator is designed for those cases.

    Attributes:
        module: The neural network module to use. This doesn't have to directly output a
            scalar. If it does not, `reduction` will be used to aggregate the outputs of
            the module into a single scalar.
        preprocessor: Preprocessor object that transforms raw States objects to tensors
            that can be used as input to the module.
        is_backward: Always False for ScalarEstimator (since it's direction-agnostic).
        reduction_function: Function used to reduce multi-dimensional outputs to scalars.
    """

    def __init__(
        self,
        module: nn.Module,
        preprocessor: Preprocessor | None = None,
        reduction: str = "mean",
    ):
        """Initializes a ScalarEstimator.

        Args:
            module: The neural network module to use.
            preprocessor: Preprocessor object that transforms states to tensors. If None,
                uses `IdentityPreprocessor` with the module's input_dim.
            reduction: String name of one of the REDUCTION_FUNCTIONS keys.
        """
        super().__init__(module, preprocessor, False)
        assert (
            reduction in REDUCTION_FUNCTIONS
        ), f"reduction function not one of {REDUCTION_FUNCTIONS.keys()}"
        self.reduction_function = REDUCTION_FUNCTIONS[reduction]

    @property
    def expected_output_dim(self) -> int:
        """Expected output dimension of the module.

        Returns:
            Always 1, as this estimator outputs scalar values.
        """
        return 1

    def forward(self, input: States) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            input: The input to the module as states.

        Returns:
            The output of the module, as a tensor of shape (*batch_shape, 1).
        """
        out = self.module(self.preprocessor(input))

        # Ensures estimator outputs are always scalar.
        if out.shape[-1] != 1:
            out = self.reduction_function(out, -1)

        assert out.shape[-1] == 1

        return out


class DiscretePolicyEstimator(Estimator):
    r"""Forward or backward policy estimators for discrete environments.

    Estimates either:
    - $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$ (forward policy)
    - $s' \mapsto (P_B(s \mid s'))_{s \in Parents(s')}$ (backward policy)

    This estimator is designed for discrete environments where actions are represented
    by integer indices and states have forward/backward masks indicating valid actions.

    Attributes:
        module: The neural network module to use.
        n_actions: Total number of actions in the discrete environment.
        preprocessor: Preprocessor object that transforms raw States objects to tensors.
        is_backward: Flag indicating whether this estimator is for backward policy,
            i.e., is used for predicting probability distributions over parents.
    """

    def __init__(
        self,
        module: nn.Module,
        n_actions: int,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
    ):
        """Initializes a DiscretePolicyEstimator.

        Args:
            module: The neural network module to use.
            n_actions: Total number of actions in the discrete environment.
            preprocessor: Preprocessor object that transforms states to tensors.
            is_backward: Flag indicating whether this estimator is for backward policy,
                i.e., is used for predicting probability distributions over parents.
        """
        super().__init__(module, preprocessor, is_backward=is_backward)
        self.n_actions = n_actions

    @property
    def expected_output_dim(self) -> int:
        """Expected output dimension of the module.

        Returns:
            n_actions for forward policies, n_actions - 1 for backward policies.
        """
        if self.is_backward:
            return self.n_actions - 1
        else:
            return self.n_actions

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: torch.Tensor,
        sf_bias: float = 0.0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> Categorical:
        """Returns a probability distribution given a batch of states and module output.

        The kwargs may contain parameters to modify a base distribution, for example to
        encourage exploration.

        Args:
            states: The discrete states where the policy is evaluated.
            module_output: The output of the module as a tensor of shape
                (*batch_shape, output_dim).
            sf_bias: Scalar to subtract from the exit action logit before dividing by
                temperature. Does nothing if set to 0.0 (default), in which case it's
                on policy.
            temperature: Scalar to divide the logits by before softmax. Does nothing
                if set to 1.0 (default), in which case it's on policy.
            epsilon: With probability epsilon, a random action is chosen. Does nothing
                if set to 0.0 (default), in which case it's on policy.

        Returns:
            A Categorical distribution over the actions.
        """
        assert module_output.shape[-1] == self.expected_output_dim, (
            f"Module output shape {module_output.shape} does not match "
            f"expected output dimension {self.expected_output_dim}"
        )
        assert temperature > 0.0
        assert 0.0 <= epsilon <= 1.0

        masks = states.backward_masks if self.is_backward else states.forward_masks
        logits = module_output
        logits[~masks] = -float("inf")

        if sf_bias != 0.0:
            logits[:, -1] -= sf_bias

        if temperature != 1.0:
            logits /= temperature

        probs = torch.softmax(logits, dim=-1)

        if epsilon != 0.0:
            uniform_dist_probs = torch.where(
                masks.sum(dim=-1, keepdim=True) == 0,
                torch.zeros_like(masks),
                masks.to(torch.get_default_dtype()) / masks.sum(dim=-1, keepdim=True),
            )
            probs = (1 - epsilon) * probs + epsilon * uniform_dist_probs

        return UnsqueezedCategorical(probs=probs)


class ConditionalDiscretePolicyEstimator(DiscretePolicyEstimator):
    r"""Conditional forward or backward policy estimators for discrete environments.

    Estimates either, with conditioning $c$:
    - $s \mapsto (P_F(s' \mid s, c))_{s' \in Children(s)}$ (conditional forward policy)
    - $s' \mapsto (P_B(s \mid s', c))_{s \in Parents(s')}$ (conditional backward policy)

    This estimator is designed for discrete environments where the policy depends on
    both the state and some conditioning information. It uses a multi-module architecture
    where state and conditioning are processed separately before being combined.

    Attributes:
        module: The neural network module for state processing.
        conditioning_module: The neural network module for conditioning processing.
        final_module: The neural network module that combines state and conditioning.
        n_actions: Total number of actions in the discrete environment.
        preprocessor: Preprocessor object that transforms raw States objects to tensors.
        is_backward: Flag indicating whether this estimator is for backward policy,
            i.e., is used for predicting probability distributions over parents.
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
        """Initializes a ConditionalDiscretePolicyEstimator.

        Args:
            state_module: The neural network module for state processing.
            conditioning_module: The neural network module for conditioning processing.
            final_module: The neural network module that combines state and conditioning.
            n_actions: Total number of actions in the discrete environment.
            preprocessor: Preprocessor object that transforms states to tensors.
            is_backward: Flag indicating whether this estimator is for backward policy,
                i.e., is used for predicting probability distributions over parents.
        """
        super().__init__(state_module, n_actions, preprocessor, is_backward)
        self.n_actions = n_actions
        self.conditioning_module = conditioning_module
        self.final_module = final_module

    def _forward_trunk(self, states: States, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass of the trunk of the module.

        This method processes the state and conditioning inputs separately, then
        combines them through the final module.

        Args:
            states: The input states.
            conditioning: The conditioning tensor.

        Returns:
            The output of the trunk of the module, as a tensor of shape
                (*batch_shape, output_dim).
        """
        state_out = self.module(self.preprocessor(states))
        conditioning_out = self.conditioning_module(conditioning)
        out = self.final_module(torch.cat((state_out, conditioning_out), -1))

        return out

    def forward(self, states: States, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            states: The input states.
            conditioning: The conditioning tensor.

        Returns:
            The output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        out = self._forward_trunk(states, conditioning)
        assert out.shape[-1] == self.expected_output_dim, (
            f"Module output shape {out.shape} does not match expected output "
            f"dimension {self.expected_output_dim}"
        )
        return out


class ConditionalScalarEstimator(ConditionalDiscretePolicyEstimator):
    r"""Class for conditionally estimating scalars (logZ, DB/SubTB state logF).

    Similar to `ScalarEstimator`, the function approximator used for `final_module` need
    not directly output a scalar. If it does not, `reduction` will be used to aggregate
    the outputs of the module into a single scalar.

    Attributes:
        module: The neural network module for state processing.
        conditioning_module: The neural network module for conditioning processing.
        final_module: The neural network module that combines state and conditioning.
        preprocessor: Preprocessor object that transforms raw States objects to tensors.
        is_backward: Always False for ConditionalScalarEstimator (since it's
            direction-agnostic).
        reduction_function: Function used to reduce multi-dimensional outputs to scalars.
    """

    def __init__(
        self,
        state_module: nn.Module,
        conditioning_module: nn.Module,
        final_module: nn.Module,
        preprocessor: Preprocessor | None = None,
        reduction: str = "mean",
    ):
        """Initializes a ConditionalScalarEstimator.

        Args:
            state_module: The neural network module for state processing.
            conditioning_module: The neural network module for conditioning processing.
            final_module: The neural network module that combines state and conditioning.
            preprocessor: Preprocessor object that transforms states to tensors.
            reduction: String name of one of the REDUCTION_FUNCTIONS keys.
        """

        super().__init__(
            state_module,
            conditioning_module,
            final_module,
            n_actions=1,
            preprocessor=preprocessor,
            is_backward=False,
        )
        assert (
            reduction in REDUCTION_FUNCTIONS
        ), "reduction function not one of {}".format(REDUCTION_FUNCTIONS.keys())
        self.reduction_function = REDUCTION_FUNCTIONS[reduction]

    def forward(self, states: States, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            states: The input states.
            conditioning: The tensor for conditioning.

        Returns:
            The output of the module, as a tensor of shape (*batch_shape, 1).
        """
        out = self._forward_trunk(states, conditioning)

        # Ensures estimator outputs are always scalar.
        if out.shape[-1] != 1:
            out = self.reduction_function(out, -1)

        assert out.shape[-1] == self.expected_output_dim, (
            f"Module output shape {out.shape} does not match expected output "
            f"dimension {self.expected_output_dim}"
        )
        return out

    @property
    def expected_output_dim(self) -> int:
        """Expected output dimension of the module.

        Returns:
            Always 1, as this estimator outputs scalar values.
        """
        return 1

    def to_probability_distribution(
        self,
        states: States,
        module_output: torch.Tensor,
        **policy_kwargs: Any,
    ) -> Distribution:
        """Transforms the output of the module into a probability distribution.

        This method should not be called for ConditionalScalarEstimator as it outputs
        scalar values, not probability distributions.

        Raises:
            NotImplementedError: This method is not implemented for scalar estimators.
        """
        raise NotImplementedError


class DiscreteGraphPolicyEstimator(Estimator):
    r"""Forward or backward policy estimators for graph-based environments.

    Estimates either, where $s$ and $s'$ are graph states:
    - $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$ (forward policy)
    - $s' \mapsto (P_B(s \mid s'))_{s \in Parents(s')}$ (backward policy)

    This estimator is designed for graph-based environments where actions modify graphs
    and states are represented as graphs. The output is a TensorDict containing logits
    for different action components (action type, node class, edge class, edge index).

    Attributes:
        module: The neural network module to use.
        preprocessor: Preprocessor object that transforms GraphStates objects to tensors.
        is_backward: Flag indicating whether this estimator is for backward policy,
            i.e., is used for predicting probability distributions over parents.
    """

    def to_probability_distribution(
        self,
        states: States,
        module_output: TensorDict,
        sf_bias: float = 0.0,
        temperature: dict[str, float] = defaultdict(lambda: 1.0),
        epsilon: dict[str, float] = defaultdict(lambda: 0.0),
    ) -> Distribution:
        """Returns a probability distribution given a batch of states and module output.

        Similar to `DiscretePolicyEstimator.to_probability_distribution()`, but handles
        the complex structure of graph actions through a TensorDict. The method applies
        masks, biases, temperature scaling, and epsilon-greedy exploration to each
        action component separately.

        Args:
            states: The graph states where the policy is evaluated.
            module_output: The output of the module as a TensorDict containing logits
                for different action components.
            sf_bias: Scalar to subtract from the exit action logit before dividing by
                temperature.
            temperature: Dictionary mapping action component keys to temperature values
                for scaling logits.
            epsilon: Dictionary mapping action component keys to epsilon values for
                exploration.

        Returns:
            A GraphActionDistribution over the graph actions.
        """
        masks = states.backward_masks if self.is_backward else states.forward_masks
        logits = module_output
        logits[GraphActions.ACTION_TYPE_KEY][~masks[GraphActions.ACTION_TYPE_KEY]] = (
            -float("inf")
        )
        logits[GraphActions.NODE_CLASS_KEY][~masks[GraphActions.NODE_CLASS_KEY]] = (
            -float("inf")
        )
        logits[GraphActions.NODE_INDEX_KEY][~masks[GraphActions.NODE_INDEX_KEY]] = (
            -float("inf")
        )
        logits[GraphActions.EDGE_CLASS_KEY][~masks[GraphActions.EDGE_CLASS_KEY]] = (
            -float("inf")
        )
        logits[GraphActions.EDGE_INDEX_KEY][~masks[GraphActions.EDGE_INDEX_KEY]] = (
            -float("inf")
        )

        # Check if no possible edge can be added,
        # and assert that action type cannot be ADD_EDGE
        no_possible_edge_index = torch.isneginf(logits[GraphActions.EDGE_INDEX_KEY]).all(
            -1
        )
        assert torch.isneginf(
            logits[GraphActions.ACTION_TYPE_KEY][
                no_possible_edge_index, GraphActionType.ADD_EDGE
            ]
        ).all()
        logits[GraphActions.EDGE_INDEX_KEY][no_possible_edge_index] = 0.0

        # Check if no possible edge class can be added,
        # and assert that action type cannot be ADD_EDGE
        no_possible_edge_class = torch.isneginf(logits[GraphActions.EDGE_CLASS_KEY]).all(
            -1
        )
        assert torch.isneginf(
            logits[GraphActions.ACTION_TYPE_KEY][
                no_possible_edge_class, GraphActionType.ADD_EDGE
            ]
        ).all()
        logits[GraphActions.EDGE_CLASS_KEY][no_possible_edge_class] = 0.0

        # Check if no possible node can be added,
        # and assert that action type cannot be ADD_NODE
        no_possible_node = torch.isneginf(logits[GraphActions.NODE_CLASS_KEY]).all(-1)
        no_possible_node &= torch.isneginf(logits[GraphActions.NODE_INDEX_KEY]).all(-1)
        assert torch.isneginf(
            logits[GraphActions.ACTION_TYPE_KEY][
                no_possible_node, GraphActionType.ADD_NODE
            ]
        ).all()
        logits[GraphActions.NODE_CLASS_KEY][no_possible_node] = 0.0
        logits[GraphActions.NODE_INDEX_KEY][no_possible_node] = 0.0

        probs = {}
        for key in logits.keys():
            probs[key] = self.logits_to_probs(
                logits[key],
                masks[key],
                sf_bias=sf_bias if key == GraphActions.ACTION_TYPE_KEY else 0.0,
                temperature=temperature[key],
                epsilon=epsilon[key],
            )

        return GraphActionDistribution(probs=TensorDict(probs), is_backward=self.is_backward)

    @staticmethod
    def logits_to_probs(
        logits: torch.Tensor,
        masks: torch.Tensor,
        sf_bias: float = 0.0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Convert logits to probabilities with optional bias, temperature, and epsilon.

        This static method implements the same logic as `DiscretePolicyEstimator`'s
        probability conversion, but is separated to handle each action component
        independently in graph environments.

        Args:
            logits: The logits tensor.
            masks: The masks tensor indicating valid actions.
            sf_bias: Scalar to subtract from the exit action logit.
            temperature: Scalar to divide the logits by before softmax.
            epsilon: Probability of choosing a random action.

        Returns:
            A tensor of probabilities.
        """
        assert temperature > 0.0
        assert 0.0 <= epsilon <= 1.0

        if sf_bias != 0.0:
            logits[..., GraphActionType.EXIT] = (
                logits[..., GraphActionType.EXIT] - sf_bias
            )

        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        if epsilon != 0.0:
            masks_sum = masks.sum(dim=-1, keepdim=True)
            probs = torch.where(
                masks_sum == 0,
                probs,
                (1 - epsilon) * probs + epsilon * masks.to(logits.dtype) / masks_sum,
            )

        return probs

    @property
    def expected_output_dim(self) -> Optional[int]:
        """Expected output dimension of the module.

        Returns:
            None, as the output_dim of a TensorDict is not well-defined.
        """
        return None
