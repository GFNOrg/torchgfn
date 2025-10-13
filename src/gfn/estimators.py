from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Optional, cast

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Categorical, Distribution

from gfn.actions import GraphActions, GraphActionType
from gfn.adapters import (
    DefaultEstimatorAdapter,
    EstimatorAdapter,
    RecurrentEstimatorAdapter,
)
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
        _default_adapter_class: The default adapter class for this estimator.
    """

    _default_adapter_class = DefaultEstimatorAdapter

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
                "dimension, in order to use the default IdentityPreprocessor."
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
        preprocessed_input = self.preprocessor(input)
        out = self.module(preprocessed_input)
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

    @property
    def default_adapter_class(self) -> type[EstimatorAdapter] | None:
        """The default adapter class for this estimator."""
        return self._default_adapter_class

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
        _default_adapter_class: There is no default adapter class for this estimator.
    """

    _default_adapter_class = None

    def __init__(
        self,
        module: nn.Module,
        preprocessor: Preprocessor | None = None,
        reduction: str = "mean",
    ):
        """Initializes a ScalarEstimator.

        Args:
            module: The neural network module to use.
            preprocessor: Preprocessor object that transforms states to tensors. If
            None, uses `IdentityPreprocessor` with the module's input_dim.
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

    def _calculate_module_output(self, input: States) -> torch.Tensor:
        return self.module(self.preprocessor(input))

    def forward(self, input: States) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            input: The input to the module as states.

        Returns:
            The output of the module, as a tensor of shape (*batch_shape, 1).
        """
        out = self._calculate_module_output(input)

        # Ensures estimator outputs are always scalar.
        if out.shape[-1] != 1:
            out = self.reduction_function(out, -1)

        assert out.shape[-1] == 1

        return out


class LogitBasedEstimator(Estimator):
    r"""Base class for estimators that output logits.

    This class is used to define estimators that output logits, which can be used to
    construct probability distributions.

    Attributes:
        module: The neural network module to use.
        preprocessor: Preprocessor object that transforms raw States objects to tensors.
        is_backward: Flag indicating whether this estimator is for backward policy,
            i.e., is used for predicting probability distributions over parents.
        _default_adapter_class: The default adapter class for this estimator.
    """

    @staticmethod
    def _prepare_logits(
        logits: torch.Tensor,
        masks: torch.Tensor,
        sf_index: int | None,
        sf_bias: float,
        temperature: float,
    ) -> torch.Tensor:
        """Clone and apply mask, bias and temperature to logits."""
        assert temperature > 0.0

        x = logits.clone()
        x[~masks] = -float("inf")

        # Ensure at least one finite per row if requested
        if sf_index is not None:
            no_valid = masks.sum(dim=-1) == 0
            if no_valid.any():
                x[no_valid] = -float("inf")
                x[no_valid, sf_index] = 0.0  # degenerate one-hot
        else:
            # If entire row is masked (e.g., unused component like EDGE_INDEX), place a
            # harmless finite fallback to avoid all -inf rows through the pipeline, but
            # only when there is at least one column.
            if x.shape[-1] > 0:
                no_valid = masks.sum(dim=-1) == 0
                if no_valid.any():
                    x[no_valid] = -float("inf")
                    # set the first column to 0.0 as a dummy finite value
                    x[no_valid, 0] = 0.0

        # Assert that each row has at least one finite entry.
        assert torch.isfinite(x).any(dim=-1).all(), "All -inf row after masking"

        if sf_index is not None and sf_bias != 0.0:
            x[..., sf_index] = x[..., sf_index] - sf_bias

        if temperature != 1.0:
            x = x / temperature

        return x

    @staticmethod
    def _uniform_log_probs(masks: torch.Tensor) -> torch.Tensor:
        """Uniform log-probs over valid actions; -inf for invalid."""
        masks_sum = masks.sum(dim=-1, keepdim=True)
        log_uniform = torch.full_like(
            masks, fill_value=-float("inf"), dtype=torch.get_default_dtype()
        )
        valid_rows = (masks_sum > 0).squeeze(-1)
        if valid_rows.any():
            log_uniform[valid_rows] = torch.where(
                masks[valid_rows],
                -torch.log(masks_sum[valid_rows].to(torch.get_default_dtype())),
                torch.full_like(
                    masks[valid_rows],
                    fill_value=-float("inf"),
                    dtype=torch.get_default_dtype(),
                ),
            )
        return log_uniform

    @staticmethod
    def _mix_with_uniform_in_log_space(
        lsm: torch.Tensor, masks: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        """Compute log((1-eps) p + eps u) in log space."""
        assert 0.0 <= epsilon <= 1.0

        if epsilon == 0.0:
            return lsm

        log_uniform = LogitBasedEstimator._uniform_log_probs(masks).to(lsm.dtype)
        eps_tensor = torch.as_tensor(epsilon, dtype=lsm.dtype, device=lsm.device)

        # Row-wise safe mixing: compute only on valid entries; for rows with no valid
        # actions, keep the original lsm row to avoid all -inf.
        out = torch.full_like(lsm, -float("inf"))
        has_any_valid = masks.any(dim=-1)

        if masks.any():
            mixed_valid_outputs = torch.logaddexp(
                lsm[masks] + torch.log1p(-eps_tensor),
                log_uniform[masks] + torch.log(eps_tensor),
            )
            out[masks] = mixed_valid_outputs

        # For rows with no valid entries (should only happen for unused components,
        # e.g., EDGE_INDEX when no edges are able to be added), retain the original
        # normalized log-probs to avoid all -inf rows.
        if (~has_any_valid).any():
            out[~has_any_valid] = lsm[~has_any_valid]

        return out

    @staticmethod
    def _compute_logits_for_distribution(
        logits: torch.Tensor,
        masks: torch.Tensor,
        sf_index: int | None,
        sf_bias: float,
        temperature: float,
        epsilon: float,
    ) -> torch.Tensor:
        """Return logits to feed a Categorical:
        - If epsilon == 0: masked, biased, temperature-scaled logits.
        - Else: normalized log-probs of the epsilon-greedy mixture (valid as logits).
        """
        assert not torch.isnan(logits).any(), "Module output logits contain NaNs"

        # Prepare logits first (masking, bias, temperature) in the existing dtype
        x = LogitBasedEstimator._prepare_logits(
            logits, masks, sf_index, sf_bias, temperature
        )

        assert not torch.isnan(x).any(), "Prepared logits contain NaNs"

        # Perform numerically sensitive ops in float32 when inputs are low-precision
        orig_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if orig_dtype in (torch.float16, torch.bfloat16)
            else orig_dtype
        )

        assert torch.isfinite(x).any(dim=-1).all(), "All -inf row before log-softmax"

        lsm = torch.log_softmax(x.to(compute_dtype), dim=-1)
        assert (
            torch.isfinite(lsm).any(dim=-1).all()
        ), "Invalid log-probs after log_softmax"

        if epsilon == 0.0:
            return lsm.to(orig_dtype) if lsm.dtype != orig_dtype else lsm

        mixed = LogitBasedEstimator._mix_with_uniform_in_log_space(lsm, masks, epsilon)
        assert torch.isfinite(mixed).any(dim=-1).all(), "Invalid log-probs after mixing"

        return mixed.to(orig_dtype) if mixed.dtype != orig_dtype else mixed


class ConditionalLogZEstimator(ScalarEstimator):
    """Conditional logZ estimator.

    This estimator is used to estimate the logZ of a GFlowNet from a conditioning
    tensor. Since conditioning is a tensor, it does not have a preprocessor. Reduction is used to aggregate the outputs of the module into a single scalar.

    Attributes:
        module: The neural network module to use.
        reduction: String name of one of the REDUCTION_FUNCTIONS keys.
        _default_adapter_class: There is no default adapter class for this estimator.
    """

    _default_adapter_class = None

    def __init__(self, module: nn.Module, reduction: str = "mean"):
        super().__init__(module, preprocessor=None, reduction=reduction)

    def _calculate_module_output(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input)


class DiscretePolicyEstimator(LogitBasedEstimator):
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
        _default_adapter_class: The default adapter class for this estimator.
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
        """Returns a Categorical distribution given a batch of states and module output.

        This implementation stays in logit/log-prob space for numerical stability.
        When epsilon > 0, we construct the epsilon-greedy distribution by mixing the
        original distribution with a uniform distribution and pass the resuling
        normalized log-probs as logits.

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

        logits = LogitBasedEstimator._compute_logits_for_distribution(
            module_output,
            states.backward_masks if self.is_backward else states.forward_masks,
            sf_index=-1,
            sf_bias=sf_bias,
            temperature=temperature,
            epsilon=epsilon,
        )

        return UnsqueezedCategorical(logits=logits)


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
        _default_adapter_class: The default adapter class for this estimator.
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
        _default_adapter_class: There is no default adapter class for this estimator.
    """

    _default_adapter_class = None

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


class DiscreteGraphPolicyEstimator(LogitBasedEstimator):
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
        _default_adapter_class: The default adapter class for this estimator.
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
        Ga = GraphActions  # Shorthand for GraphActions.
        GaType = GraphActionType  # Shorthand for GraphActionType.

        logits = module_output
        logits[Ga.ACTION_TYPE_KEY][~masks[Ga.ACTION_TYPE_KEY]] = -float("inf")
        logits[Ga.NODE_CLASS_KEY][~masks[Ga.NODE_CLASS_KEY]] = -float("inf")
        logits[Ga.NODE_INDEX_KEY][~masks[Ga.NODE_INDEX_KEY]] = -float("inf")
        logits[Ga.EDGE_CLASS_KEY][~masks[Ga.EDGE_CLASS_KEY]] = -float("inf")
        logits[Ga.EDGE_INDEX_KEY][~masks[Ga.EDGE_INDEX_KEY]] = -float("inf")

        # The following operations are to ensure that the logits are valid, i.e.,
        # contain at least one finite entry.

        # Check if no possible edge can be added,
        # and assert that action type cannot be ADD_EDGE
        no_possible_edge_index = torch.isneginf(logits[Ga.EDGE_INDEX_KEY]).all(-1)
        assert torch.isneginf(
            logits[Ga.ACTION_TYPE_KEY][no_possible_edge_index, GaType.ADD_EDGE]
        ).all()
        logits[Ga.EDGE_INDEX_KEY][no_possible_edge_index] = 0.0

        # Check if no possible edge class can be added,
        # and assert that action type cannot be ADD_EDGE
        no_possible_edge_class = torch.isneginf(logits[Ga.EDGE_CLASS_KEY]).all(-1)
        assert torch.isneginf(
            logits[Ga.ACTION_TYPE_KEY][no_possible_edge_class, GaType.ADD_EDGE]
        ).all()
        logits[Ga.EDGE_CLASS_KEY][no_possible_edge_class] = 0.0

        # Check if no possible node can be added; if either class OR index has no
        # valid options, disallow ADD_NODE in action type and set harmless finite
        # fallbacks for the unused components.
        no_possible_node_class = torch.isneginf(logits[Ga.NODE_CLASS_KEY]).all(-1)
        no_possible_node_index = torch.isneginf(logits[Ga.NODE_INDEX_KEY]).all(-1)

        # If backward, we only need to check if the node index is possible to remove.
        # If forward, we only need to check if the node class is possible to add.
        no_possible_add_node = (
            no_possible_node_index if self.is_backward else no_possible_node_class
        )
        logits[Ga.ACTION_TYPE_KEY][no_possible_add_node, GaType.ADD_NODE] = -float("inf")
        logits[Ga.NODE_CLASS_KEY][no_possible_node_class] = 0.0
        logits[Ga.NODE_INDEX_KEY][no_possible_node_index] = 0.0

        transformed_logits = {}
        for key in logits.keys():
            assert isinstance(key, str)
            assert not torch.isnan(logits[key]).any(), f"logits[{key}] contains NaNs"

            # Pad zero-length components to length 1 with an invalid mask so downstream
            # operations have at least one column and distributions can be constructed.
            local_logits = logits[key]
            local_masks = masks[key]
            if local_logits.shape[-1] == 0:
                local_logits = torch.zeros(
                    *local_logits.shape[:-1],
                    1,
                    dtype=local_logits.dtype,
                    device=local_logits.device,
                )
                local_masks = torch.zeros(
                    *local_masks.shape[:-1],
                    1,
                    dtype=torch.bool,
                    device=local_masks.device,
                )

            # Logit transformations allow for off-policy exploration.
            transformed_logits[key] = (
                LogitBasedEstimator._compute_logits_for_distribution(
                    logits=local_logits,
                    masks=local_masks,
                    # ACTION_TYPE_KEY contains the exit action logit.
                    sf_index=GaType.EXIT if key == Ga.ACTION_TYPE_KEY else None,
                    sf_bias=sf_bias if key == Ga.ACTION_TYPE_KEY else 0.0,
                    temperature=temperature[key],
                    epsilon=epsilon[key],
                )
            )

        return GraphActionDistribution(
            logits=TensorDict(transformed_logits), is_backward=self.is_backward
        )

    @property
    def expected_output_dim(self) -> Optional[int]:
        """Expected output dimension of the module.

        Returns:
            None, as the output_dim of a TensorDict is not well-defined.
        """
        return None


class RecurrentDiscretePolicyEstimator(DiscretePolicyEstimator):
    """Discrete policy estimator for recurrent architectures with explicit carry.

    Many sequence models (e.g., RNN/LSTM/GRU/Transformer in autoregressive mode)
    maintain a recurrent hidden state ("carry") that must be threaded through
    successive calls during sampling. This class formalizes that pattern for
    GFlowNet policies by:

    - Exposing a forward signature ``forward(states, carry) -> (logits, carry)``
      so the policy can update and return the next carry at each step.
    - Requiring an ``init_carry(batch_size, device)`` method to allocate the
      initial hidden state for a rollout.
    - Ensuring the per-step output (``logits`` over actions) is derived from the
      latest token/time step while the internal model may process sequences.

    Interaction with the sampler/adapters
    -------------------------------------
    The sampler uses a ``RecurrentEstimatorAdapter`` which calls this estimator
    with the current carry, updates the carry on every step, and records
    per-step artifacts. Non-recurrent estimators should use the default adapter
    and the standard ``DiscretePolicyEstimator`` base class instead.

    Notes
    -----
    - Forward is intended for on-policy generation; off-policy evaluation over
      entire trajectories typically requires different batching and masking.
    - ``init_carry`` is a hard requirement for compatibility with the recurrent
      adapter.

    Attributes:
        module: The neural network module to use.
        n_actions: Total number of actions in the discrete environment.
        preprocessor: Preprocessor object that transforms states to tensors.
        is_backward: Flag indicating whether this estimator is for backward policy,
            i.e., is used for predicting probability distributions over parents.
        _default_adapter_class: The default adapter class for this estimator.
    """

    _default_adapter_class = RecurrentEstimatorAdapter

    def __init__(
        self,
        module: nn.Module,
        n_actions: int,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
    ):
        """Initializes a RecurrentDiscretePolicyEstimator.

        Args:
            module: The neural network module to use.
            n_actions: Total number of actions in the discrete environment.
            preprocessor: Preprocessor object that transforms states to tensors.
        """
        if preprocessor is None:
            preprocessor = IdentityPreprocessor(output_dim=None)

        super().__init__(
            module=module,
            n_actions=n_actions,
            preprocessor=preprocessor,
            is_backward=is_backward,
        )

    def forward(
        self,
        states: States,
        carry: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass of the module.

        Args:
            states: The input states.
            carry: The carry from the previous step.

        Returns:
            The output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        # Prepare integer token sequences without -1 padding and use a BOS index.
        # We infer the active sequence length per row from (token != -1).
        tokens = states.tensor
        if not torch.is_floating_point(tokens):
            tokens = tokens.long()
        else:
            tokens = tokens.to(dtype=torch.long)

        # Replace padding (-1) with BOS index expected by the sequence model.
        # RecurrentDiscreteSequenceModel reserves index == vocab_size for BOS.
        bos_index = getattr(self.module, "vocab_size", self.n_actions - 1)
        tokens = torch.where(
            tokens < 0, torch.as_tensor(bos_index, device=tokens.device), tokens
        )

        # Determine a common prefix length across the (active) batch.
        # Active rows in a rollout step share the same length; use max for safety.
        # We still derive length from original states.tensor where -1 marks padding.
        original = states.tensor
        valid_mask = original >= 0
        if valid_mask.ndim == 1:
            max_len = int(valid_mask.sum().item())
        else:
            max_len = int(valid_mask.sum(dim=-1).max().item())
        if max_len <= 0:
            max_len = 1  # Ensure at least BOS is processed

        # Trim to the common active prefix length and run the sequence model.
        seq_input = tokens[..., :max_len]
        logits, carry = self.module(seq_input, carry)

        # Use the logits corresponding to the last processed token.
        logits = logits[:, -1, :]  # (b, n_actions)

        if self.expected_output_dim is not None:
            assert logits.shape[-1] == self.expected_output_dim, (
                f"Module output shape {logits.shape} does not match expected output "
                f"dimension {self.expected_output_dim}"
            )

        return logits, carry

    def init_carry(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        init_carry = getattr(self.module, "init_carry", None)
        if not callable(init_carry):
            raise NotImplementedError(
                "Module does not implement init_carry(batch_size, device)."
            )
        init_carry_fn = cast(Callable[[int, torch.device], Any], init_carry)

        return init_carry_fn(batch_size, device)
