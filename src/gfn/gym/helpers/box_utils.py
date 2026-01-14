"""This file contains utilitary functions for the Box environment."""

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Beta, Categorical, Distribution, MixtureSameFamily

from gfn.estimators import Estimator, PolicyMixin
from gfn.gym import Box
from gfn.states import States
from gfn.utils.modules import MLP

PI_2_INV: float = 2.0 / torch.pi
PI_2: float = torch.pi / 2.0
CLAMP: float = torch.finfo(torch.get_default_dtype()).eps


# =============================================================================
# Cartesian Increment Approach (simplified, faster)
# Inspired by gflownet's ContinuousCube implementation.
# =============================================================================


class BoxCartesianDistribution(Distribution):
    """Cartesian increment distribution for Box environment.

    Uses MixtureSameFamily(Categorical, Beta) per dimension for sampling increments.
    Much simpler than polar coordinates - samples relative increments per dimension
    and converts to absolute using: action = min_incr + r * (max_range).

    Attributes:
        delta: Minimum step size.
        epsilon: Small value for numerical stability.
    """

    arg_constraints = {}  # No constraints for custom distribution

    def __init__(
        self,
        states: States,
        exit_logits: Tensor,
        mixture_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
        delta: float,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize the distribution.

        Args:
            states: Current states, shape (batch, n_dim).
            exit_logits: Logits for exit probability, shape (batch,).
            mixture_logits: Mixture weights, shape (batch, n_dim, n_components).
            alpha: Beta alpha params, shape (batch, n_dim, n_components).
            beta: Beta beta params, shape (batch, n_dim, n_components).
            delta: Minimum step size.
            epsilon: Numerical stability constant.
        """
        super().__init__()
        self.delta = delta
        self.epsilon = epsilon
        self.states = states
        self.n_dim = states.tensor.shape[-1]

        # Exit distribution
        from torch.distributions import Bernoulli

        self.exit_dist = Bernoulli(logits=exit_logits)

        # Increment distribution per dimension
        mix = Categorical(logits=mixture_logits)
        components = Beta(alpha, beta)
        self.increment_dist = MixtureSameFamily(mix, components)

        # Compute valid ranges for each state
        # s0: can go anywhere in [0, delta], so min_incr=0, max_range=delta
        # non-s0: must step at least delta, max is 1-state
        is_s0 = torch.all(states.tensor == 0, dim=-1, keepdim=True)
        self.is_s0 = is_s0.squeeze(-1)
        self.min_incr = torch.where(is_s0, 0.0, delta)
        # For s0: max_range = delta (action in [0, delta])
        # For non-s0: max_range = 1 - state - delta (action in [delta, 1-state])
        self.max_range = torch.where(
            is_s0,
            torch.full_like(states.tensor, delta),
            (1.0 - states.tensor - delta),
        ).clamp(min=epsilon)

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Sample actions using Cartesian per-dimension increments."""
        # Sample exit decisions
        exit_mask = self.exit_dist.sample().bool()

        # Force exit if at boundary (any dim >= 1 - delta)
        at_boundary = torch.any(
            self.states.tensor >= 1 - self.delta - self.epsilon, dim=-1
        )
        exit_mask = exit_mask | at_boundary

        # Can't exit from s0
        exit_mask = exit_mask & ~self.is_s0

        # Sample relative increments r ∈ [0, 1] per dimension
        r = self.increment_dist.sample().clamp(0.0, 1.0)  # (batch, n_dim)

        # Convert relative to absolute: action = min_incr + r * max_range
        actions = self.min_incr + r * self.max_range

        # Clamp to ensure actions stay in valid range
        # For s0: action in [0, delta], for non-s0: action in [delta, 1-state]
        is_s0_expanded = self.is_s0.unsqueeze(-1)
        max_action = torch.where(
            is_s0_expanded,
            torch.full_like(actions, self.delta),
            1.0 - self.states.tensor,
        )
        actions = torch.clamp(actions, min=0.0)
        actions = torch.minimum(actions, max_action)

        # Set exit actions
        actions[exit_mask] = float("-inf")

        return actions

    def log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability using Cartesian per-dimension approach."""
        device = actions.device

        # Identify exit actions
        is_exit = torch.all(actions == float("-inf"), dim=-1)

        # At boundary, exit is forced (log_prob = 0)
        at_boundary = torch.any(
            self.states.tensor >= 1 - self.delta - self.epsilon, dim=-1
        )

        # For non-exit: replace -inf with valid placeholder to avoid NaN in computation
        safe_actions = torch.where(
            is_exit.unsqueeze(-1),
            self.min_incr + 0.5 * self.max_range,  # placeholder for exit actions
            actions,
        )

        # Convert absolute to relative: r = (action - min_incr) / max_range
        r = (safe_actions - self.min_incr) / self.max_range
        r = r.clamp(self.epsilon, 1 - self.epsilon)

        # Get log prob from Beta mixture (sum over dimensions)
        log_p_beta = self.increment_dist.log_prob(r).sum(dim=-1)

        # Jacobian: dr/da = 1/max_range, so log|da/dr| = log(max_range)
        log_jacobian = torch.log(self.max_range).sum(dim=-1)

        # Add log(1 - exit_prob) for choosing not to exit
        log_no_exit = torch.log1p(-self.exit_dist.probs)

        # Non-exit log prob
        log_probs = log_p_beta + log_jacobian + log_no_exit

        # For exit actions: log P(exit)
        log_p_exit = self.exit_dist.log_prob(torch.ones(1, device=device))
        log_probs = torch.where(is_exit, log_p_exit.expand_as(log_probs), log_probs)

        # Forced exits at boundary have log_prob = 0
        log_probs = torch.where(
            at_boundary & is_exit, torch.zeros_like(log_probs), log_probs
        )

        # Exit from s0 is not allowed
        log_probs = torch.where(
            self.is_s0 & is_exit, torch.full_like(log_probs, float("-inf")), log_probs
        )

        return log_probs


class BoxCartesianPFEstimator(Estimator, PolicyMixin):
    """Simplified PF estimator using Cartesian increments.

    Much simpler than BoxPFEstimator - uses a single MLP and BoxCartesianDistribution.
    """

    def __init__(
        self,
        env: Box,
        module: nn.Module,
        n_components: int,
        min_concentration: float = 0.1,
        max_concentration: float = 5.0,
    ) -> None:
        """Initialize the estimator.

        Args:
            env: The Box environment.
            module: The neural network module.
            n_components: Number of mixture components.
            min_concentration: Minimum Beta concentration parameter.
            max_concentration: Maximum Beta concentration parameter.
        """
        super().__init__(module)
        self.n_components = n_components
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.delta = env.delta
        self.epsilon = env.epsilon
        self.n_dim = 2

    @property
    def expected_output_dim(self) -> int:
        """Expected output dimension: exit_logit + (weights + alpha + beta) * n_dim * n_comp."""
        return 1 + 3 * self.n_dim * self.n_components

    def to_probability_distribution(
        self, states: States, module_output: Tensor
    ) -> Distribution:
        """Convert module output to a probability distribution.

        Args:
            states: The states.
            module_output: Output from the module, shape (batch, expected_output_dim).

        Returns:
            BoxCartesianDistribution instance.
        """
        batch_size = states.tensor.shape[0]
        n_comp = self.n_components
        n_dim = self.n_dim

        # Parse module output
        # Format: [exit_logit, weights..., alpha..., beta...]
        exit_logits = module_output[:, 0]

        # Reshape parameters to (batch, n_dim, n_comp)
        offset = 1
        mixture_logits = module_output[:, offset : offset + n_dim * n_comp].reshape(
            batch_size, n_dim, n_comp
        )
        offset += n_dim * n_comp
        alpha_raw = module_output[:, offset : offset + n_dim * n_comp].reshape(
            batch_size, n_dim, n_comp
        )
        offset += n_dim * n_comp
        beta_raw = module_output[:, offset : offset + n_dim * n_comp].reshape(
            batch_size, n_dim, n_comp
        )

        # Normalize concentration parameters
        alpha = self.min_concentration + (
            self.max_concentration - self.min_concentration
        ) * torch.sigmoid(alpha_raw)
        beta = self.min_concentration + (
            self.max_concentration - self.min_concentration
        ) * torch.sigmoid(beta_raw)

        return BoxCartesianDistribution(
            states=states,
            exit_logits=exit_logits,
            mixture_logits=mixture_logits,
            alpha=alpha,
            beta=beta,
            delta=self.delta,
            epsilon=self.epsilon,
        )


class BoxCartesianPBDistribution(Distribution):
    """Backward Cartesian distribution for Box environment.

    Similar to forward but ranges are [min_incr, state] to go backwards.
    States near origin (norm < delta) must go directly to s0.
    """

    arg_constraints = {}  # No constraints for custom distribution

    def __init__(
        self,
        states: States,
        mixture_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
        delta: float,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize the backward distribution."""
        super().__init__()
        self.delta = delta
        self.epsilon = epsilon
        self.states = states
        self.n_dim = states.tensor.shape[-1]

        # Per-dimension: if state[d] < delta, must go directly to 0 in that dim
        # (batch, n_dim) boolean mask
        self.dim_near_origin = states.tensor < delta

        # States where ALL dimensions are near origin (fully deterministic)
        self.fully_near_origin = torch.all(self.dim_near_origin, dim=-1)

        # Increment distribution per dimension
        mix = Categorical(logits=mixture_logits)
        components = Beta(alpha, beta)
        self.increment_dist = MixtureSameFamily(mix, components)

        # For backward: action in [delta, state] for dims where state >= delta
        # For dims where state < delta: action = state (deterministic)
        self.min_incr = delta
        # max_range = state - delta, but only meaningful where state >= delta
        self.max_range = (states.tensor - self.min_incr).clamp(min=epsilon)

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Sample backward actions."""
        # Start with action = state for all (handles near-origin dims correctly)
        actions = self.states.tensor.clone()

        # For dimensions where state >= delta, sample from [delta, state]
        if not self.fully_near_origin.all():
            r = self.increment_dist.sample().clamp(0.0, 1.0)
            sampled_actions = self.min_incr + r * self.max_range
            # Clamp to ensure action <= state
            sampled_actions = torch.min(sampled_actions, self.states.tensor)

            # Only use sampled actions for dims where state >= delta
            actions = torch.where(self.dim_near_origin, actions, sampled_actions)

        return actions

    def log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability of backward actions.

        For each dimension:
        - If state[d] < delta: action[d] must equal state[d] (deterministic, log_prob = 0)
        - If state[d] >= delta: action[d] sampled from Beta mixture in [delta, state[d]]
        """
        actions.device

        # Check deterministic constraints: for near-origin dims, action must equal state.
        # A deterministic constraint violation (resulting in -inf log_prob) can occur when:
        #   1. Trajectory mismatch: The action was sampled by a different policy (e.g., forward)
        #      that doesn't respect backward semantics for near-origin states.
        #   2. Manual trajectory construction: Actions were manually specified without
        #      ensuring action[d] = state[d] for dimensions where state[d] < delta.
        #   3. Numerical precision: Floating-point differences between action and state
        #      exceed self.epsilon. Consider increasing epsilon if this occurs frequently.
        #   4. Policy bug: The forward policy sampled actions that are inconsistent with
        #      the environment's state transition rules.
        action_matches_state = torch.abs(actions - self.states.tensor) < self.epsilon
        deterministic_ok = torch.where(
            self.dim_near_origin,
            action_matches_state,
            torch.ones_like(
                action_matches_state
            ),  # non-deterministic dims always OK here
        )
        # If any deterministic constraint is violated, log_prob = -inf
        all_deterministic_ok = torch.all(deterministic_ok, dim=-1)

        # For non-deterministic dimensions, compute Beta log_prob
        # Convert absolute to relative: r = (action - delta) / max_range
        r = (actions - self.min_incr) / self.max_range
        r = r.clamp(self.epsilon, 1 - self.epsilon)

        # Get log prob from Beta mixture per dimension
        log_p_per_dim = self.increment_dist.log_prob(r)

        # Jacobian per dimension: log(max_range)
        log_jacobian_per_dim = torch.log(self.max_range)

        # Only sum over non-deterministic dimensions
        log_p_stochastic = torch.where(
            self.dim_near_origin,
            torch.zeros_like(log_p_per_dim),  # deterministic dims contribute 0
            log_p_per_dim + log_jacobian_per_dim,
        ).sum(dim=-1)

        # Combine: if deterministic constraints violated, -inf; otherwise, sum of stochastic
        log_probs = torch.where(
            all_deterministic_ok,
            log_p_stochastic,
            torch.full_like(log_p_stochastic, float("-inf")),
        )

        return log_probs


class BoxCartesianPBEstimator(Estimator, PolicyMixin):
    """Simplified PB estimator using Cartesian increments."""

    def __init__(
        self,
        env: Box,
        module: nn.Module,
        n_components: int,
        min_concentration: float = 0.1,
        max_concentration: float = 5.0,
    ) -> None:
        """Initialize the estimator."""
        super().__init__(module, is_backward=True)
        self.n_components = n_components
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.delta = env.delta
        self.epsilon = env.epsilon
        self.n_dim = 2

    @property
    def expected_output_dim(self) -> int:
        """Expected output dimension: (weights + alpha + beta) * n_dim * n_comp."""
        return 3 * self.n_dim * self.n_components

    def to_probability_distribution(
        self, states: States, module_output: Tensor
    ) -> Distribution:
        """Convert module output to backward probability distribution."""
        batch_size = states.tensor.shape[0]
        n_comp = self.n_components
        n_dim = self.n_dim

        # Parse module output (no exit logit for backward)
        offset = 0
        mixture_logits = module_output[:, offset : offset + n_dim * n_comp].reshape(
            batch_size, n_dim, n_comp
        )
        offset += n_dim * n_comp
        alpha_raw = module_output[:, offset : offset + n_dim * n_comp].reshape(
            batch_size, n_dim, n_comp
        )
        offset += n_dim * n_comp
        beta_raw = module_output[:, offset : offset + n_dim * n_comp].reshape(
            batch_size, n_dim, n_comp
        )

        # Normalize concentration parameters
        conc_range = self.max_concentration - self.min_concentration
        alpha = self.min_concentration + conc_range * torch.sigmoid(alpha_raw)
        beta = self.min_concentration + conc_range * torch.sigmoid(beta_raw)

        return BoxCartesianPBDistribution(
            states=states,
            mixture_logits=mixture_logits,
            alpha=alpha,
            beta=beta,
            delta=self.delta,
            epsilon=self.epsilon,
        )


class BoxCartesianPFMLP(MLP):
    """Simplified MLP for Box forward policy using Cartesian increments.

    Output format: [exit_logit, mixture_logits..., alpha..., beta...]
    where mixture_logits, alpha, beta each have shape n_dim * n_components.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_hidden_layers: int,
        n_components: int,
        n_dim: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize the MLP.

        Args:
            hidden_dim: Hidden layer dimension.
            n_hidden_layers: Number of hidden layers.
            n_components: Number of mixture components.
            n_dim: Number of dimensions (default 2 for Box).
            **kwargs: Additional arguments for MLP.
        """
        self.n_components = n_components
        self.n_dim = n_dim

        # Output: exit_logit + (weights + alpha + beta) * n_dim * n_comp
        output_dim = 1 + 3 * n_dim * n_components

        super().__init__(
            input_dim=n_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            activation_fn="elu",
            **kwargs,
        )

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Forward pass."""
        return super().forward(preprocessed_states)


class BoxCartesianPBMLP(MLP):
    """Simplified MLP for Box backward policy using Cartesian increments."""

    def __init__(
        self,
        hidden_dim: int,
        n_hidden_layers: int,
        n_components: int,
        n_dim: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize the MLP.

        Args:
            hidden_dim: Hidden layer dimension.
            n_hidden_layers: Number of hidden layers.
            n_components: Number of mixture components.
            n_dim: Number of dimensions (default 2 for Box).
            **kwargs: Additional arguments for MLP.
        """
        self.n_components = n_components
        self.n_dim = n_dim

        # Output: (weights + alpha + beta) * n_dim * n_comp (no exit for backward)
        output_dim = 3 * n_dim * n_components

        super().__init__(
            input_dim=n_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            activation_fn="elu",
            **kwargs,
        )

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Forward pass."""
        return super().forward(preprocessed_states)


# =============================================================================
# Legacy Polar Coordinate Approach (original implementation)
# =============================================================================


class QuarterCircle(Distribution):
    """Represents distributions on quarter circles.

    The distributions are Mixture of Beta distributions on the possible angle range.

    When a state is of norm <= delta, and northeastern=False, then the distribution
    is a Dirac at the state (i.e. the only possible parent is s_0).

    Adapted from https://github.com/saleml/continuous-gfn/blob/master/sampling.py

    This is useful for the `Box` environment.

    Attributes:
        delta: The radius of the quarter disk.
        northeastern: Whether the quarter disk is northeastern or southwestern.
        n_states: The number of states.
        n_components: The number of components in the mixture.
        centers: The centers of the distribution.
        base_dist: The base distribution.
        min_angles: The minimum angles.
        max_angles: The maximum angles.
    """

    delta: float
    northeastern: bool
    n_states: int
    n_components: int
    centers: States
    base_dist: MixtureSameFamily
    min_angles: Tensor
    max_angles: Tensor

    def __init__(
        self,
        delta: float,
        northeastern: bool,
        centers: States,
        mixture_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> None:
        """Initializes the distribution.

        Args:
            delta: the radius of the quarter disk.
            northeastern: whether the quarter disk is northeastern or southwestern.
            centers: the centers of the distribution with shape `(n_states, 2)`.
            mixture_logits: Tensor of shape `(n_states, n_components)` containing the
                logits of the mixture of Beta distributions.
            alpha: Tensor of shape `(n_states, n_components)` containing the alpha
                parameters of the Beta distributions.
            beta: Tensor of shape `(n_states, n_components)` containing the beta
                parameters of the Beta distributions.
        """
        self.delta = delta
        self.northeastern = northeastern
        self.n_states, self.n_components = mixture_logits.shape

        assert centers.tensor.shape == (self.n_states, 2)
        self.centers = centers

        assert alpha.shape == (self.n_states, self.n_components)
        assert beta.shape == (self.n_states, self.n_components)
        self.base_dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha, beta),
        )

        self.min_angles, self.max_angles = self.get_min_and_max_angles()

    def get_min_and_max_angles(self) -> Tuple[Tensor, Tensor]:
        """Computes the minimum and maximum angles for the distribution.

        Returns:
            A tuple of two tensors of shape `(n_states,)` containing the minimum and
            maximum angles, respectively.
        """
        if self.northeastern:
            min_angles = torch.where(
                self.centers.tensor[..., 0] <= 1 - self.delta,
                0.0,
                PI_2_INV * torch.arccos((1 - self.centers.tensor[..., 0]) / self.delta),
            )
            max_angles = torch.where(
                self.centers.tensor[..., 1] <= 1 - self.delta,
                1.0,
                PI_2_INV * torch.arcsin((1 - self.centers.tensor[..., 1]) / self.delta),
            )
        else:
            min_angles = torch.where(
                self.centers.tensor[..., 0] >= self.delta,
                0.0,
                PI_2_INV * torch.arccos((self.centers.tensor[..., 0]) / self.delta),
            )
            max_angles = torch.where(
                self.centers.tensor[..., 1] >= self.delta,
                1.0,
                PI_2_INV * torch.arcsin((self.centers.tensor[..., 1]) / self.delta),
            )

        assert min_angles.shape == (self.n_states,)
        assert max_angles.shape == (self.n_states,)
        return min_angles, max_angles

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Samples from the distribution.

        Args:
            sample_shape: the shape of the samples to generate.

        Returns:
            The sampled actions of shape `(sample_shape, n_states, 2)`.
        """
        base_01_samples = self.base_dist.sample(sample_shape=sample_shape)

        sampled_angles = (
            self.min_angles + (self.max_angles - self.min_angles) * base_01_samples
        )
        sampled_angles = PI_2 * sampled_angles

        sampled_actions = self.delta * torch.stack(
            [torch.cos(sampled_angles), torch.sin(sampled_angles)],
            dim=-1,
        )

        if not self.northeastern:
            # when centers are of norm <= delta, the distribution is a Dirac at the center
            centers_in_quarter_disk = (
                torch.norm(self.centers.tensor, dim=-1) <= self.delta
            )
            # repeat the centers_in_quarter_disk tensor to be of shape (*centers.batch_shape, 2)
            centers_in_quarter_disk = centers_in_quarter_disk.unsqueeze(-1).repeat(
                *([1] * len(self.centers.batch_shape)), 2
            )
            sampled_actions = torch.where(
                centers_in_quarter_disk,
                self.centers.tensor,
                sampled_actions,
            )

            # Sometimes, when a point is at the border of the square (e.g. (1e-8, something) or (something, 1e-9))
            # Then the approximation errors lead to the sampled_actions being slightly larger than the state or slightly
            # negative at the low coordinate. So what we do is we set the sampled_action to be half that coordinate

            sampled_actions = torch.where(
                sampled_actions > self.centers.tensor,
                self.centers.tensor / 2,
                sampled_actions,
            )

            sampled_actions = torch.where(
                sampled_actions < 0,
                self.centers.tensor / 2,
                sampled_actions,
            )
        else:
            # When at the border of the square,
            # the approximation errors might lead to the sampled_actions being slightly negative at the high coordinate
            # We set the sampled_action to be 0
            # This is actually of no impact, given that the true action that would be sampled is the exit action
            sampled_actions = torch.where(
                sampled_actions < 0,
                0,
                sampled_actions,
            )
            if torch.any(
                torch.abs(torch.norm(sampled_actions, dim=-1) - self.delta) > 1e-5
            ):
                raise ValueError("Sampled actions should be of norm delta ish")

        assert sampled_actions.shape == sample_shape + (self.n_states, 2)
        return sampled_actions

    def log_prob(self, sampled_actions: Tensor) -> Tensor:
        """Computes the log probability of the sampled actions.

        Args:
            sampled_actions: Tensor of shape `(*batch_shape, 2)` with the actions to
                compute the log probability of.

        Returns:
            The log probability of the sampled actions as a tensor of shape `batch_shape`.
        """
        assert sampled_actions.shape[-1] == 2
        batch_shape = sampled_actions.shape[:-1]

        # Clamp actions to valid range (no double precision needed with atan2)
        sampled_actions = sampled_actions.clamp(min=0.0, max=self.delta)

        # Use atan2 instead of arccos - numerically stable without double precision
        sampled_angles = (
            torch.atan2(sampled_actions[..., 1], sampled_actions[..., 0]) / PI_2
        )

        base_01_samples = (sampled_angles - self.min_angles) / (
            self.max_angles - self.min_angles
        ).clamp(min=CLAMP, max=1 - CLAMP)

        if not self.northeastern:
            # Clamp to avoid numerical issues at boundaries
            base_01_samples = base_01_samples.clamp(1e-4, 1 - 1e-4)

        # Handle exit actions (marked as -inf) which produce nan angles
        base_01_samples = torch.where(
            torch.isnan(base_01_samples),
            torch.ones_like(base_01_samples) * 0.5,
            base_01_samples,
        ).clamp(min=CLAMP, max=1 - CLAMP)

        # Handle backward case where sampled_actions equal centers (Dirac distribution)
        if not self.northeastern:
            base_01_samples = torch.where(
                torch.norm(self.centers.tensor, dim=-1) <= self.delta,
                torch.ones_like(base_01_samples) * CLAMP,
                base_01_samples,
            ).clamp(min=CLAMP, max=1 - CLAMP)

        base_01_logprobs = self.base_dist.log_prob(base_01_samples)

        if not self.northeastern:
            base_01_logprobs = base_01_logprobs.clamp_max(100)

        logprobs = (
            base_01_logprobs
            - np.log(self.delta)
            - np.log(np.pi / 2)
            - torch.log((self.max_angles - self.min_angles).clamp_(min=CLAMP))
            # The clamp doesn't really matter, because if we need to clamp, it means the actual action is exit action
        )

        if not self.northeastern:
            # when centers are of norm <= delta, the distribution is a Dirac at the center
            logprobs = torch.where(
                torch.norm(self.centers.tensor, dim=-1) <= self.delta,
                torch.zeros_like(logprobs),
                logprobs,
            )

        if torch.any(torch.isinf(logprobs)) or torch.any(torch.isnan(logprobs)):
            raise ValueError("logprobs contains inf or nan")

        assert logprobs.shape == batch_shape
        return logprobs


class QuarterDisk(Distribution):
    """Represents a distribution on the northeastern quarter disk.

    The radius and the angle follow Mixture of Betas distributions.

    Adapted from https://github.com/saleml/continuous-gfn/blob/master/sampling.py

    This is useful for the `Box` environment.

    Attributes:
        delta: The radius of the quarter disk.
        mixture_logits: The logits of the mixture of Beta distributions.
        base_r_dist: The base distribution for the radius.
        base_theta_dist: The base distribution for the angle.
        n_components: The number of components in the mixture.
    """

    delta: float
    mixture_logits: Tensor
    base_r_dist: MixtureSameFamily
    base_theta_dist: MixtureSameFamily
    n_components: int

    def __init__(
        self,
        delta: float,
        mixture_logits: Tensor,
        alpha_r: Tensor,
        beta_r: Tensor,
        alpha_theta: Tensor,
        beta_theta: Tensor,
    ) -> None:
        """Initializes the distribution.

        Args:
            delta: the radius of the quarter disk.
            mixture_logits: Tensor of shape `(n_components,)` containing the logits of
                the mixture of Beta distributions.
            alpha_r: Tensor of shape `(n_components,)` containing the alpha parameters
                of the Beta distributions for the radius.
            beta_r: Tensor of shape `(n_components,)` containing the beta parameters of
                the Beta distributions for the radius.
            alpha_theta: Tensor of shape `(n_components,)` containing the alpha
                parameters of the Beta distributions for the angle.
            beta_theta: Tensor of shape `(n_components,)` containing the beta
                parameters of the Beta distributions for the angle.
        """
        self.delta = delta
        self.mixture_logits = mixture_logits
        (self.n_components,) = mixture_logits.shape

        assert alpha_r.shape == (self.n_components,)
        assert beta_r.shape == (self.n_components,)
        assert alpha_theta.shape == (self.n_components,)
        assert beta_theta.shape == (self.n_components,)

        self.base_r_dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha_r, beta_r),
        )

        self.base_theta_dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha_theta, beta_theta),
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Samples from the distribution.

        Args:
            sample_shape: the shape of the samples to generate.

        Returns:
            The sampled actions of shape `(sample_shape, 2)`.
        """
        base_r_01_samples = self.base_r_dist.sample(sample_shape=sample_shape)
        base_theta_01_samples = self.base_theta_dist.sample(sample_shape=sample_shape)

        sampled_actions = self.delta * (
            torch.stack(
                [
                    base_r_01_samples * torch.cos(PI_2 * base_theta_01_samples),
                    base_r_01_samples * torch.sin(PI_2 * base_theta_01_samples),
                ],
                dim=-1,
            )
        )

        assert sampled_actions.shape == sample_shape + (2,)
        return sampled_actions

    def log_prob(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the sampled actions.

        Args:
            sampled_actions: Tensor of shape `(*batch_shape, 2)` with the actions to
                compute the log probability of.

        Returns:
            The log probability of the sampled actions as a tensor of shape `batch_shape`.
        """
        assert sampled_actions.shape[-1] == 2
        batch_shape = sampled_actions.shape[:-1]

        # Compute radius (no double precision needed with atan2)
        base_r_01_samples = (
            torch.sqrt(torch.sum(sampled_actions**2, dim=-1)) / self.delta
        ).clamp(CLAMP, 1.0)

        # Use atan2 instead of arccos - numerically stable without double precision
        base_theta_01_samples = (
            torch.atan2(sampled_actions[..., 1], sampled_actions[..., 0]) / PI_2
        ).clamp(CLAMP, 1 - CLAMP)

        logprobs = (
            self.base_r_dist.log_prob(base_r_01_samples)
            + self.base_theta_dist.log_prob(base_theta_01_samples)
            - np.log(self.delta)
            - np.log(PI_2)
            - torch.log(base_r_01_samples * self.delta)
        )

        if torch.any(torch.isinf(logprobs)):
            raise ValueError("logprobs contains inf")

        assert logprobs.shape == batch_shape
        return logprobs


class QuarterCircleWithExit(Distribution):
    """Extends `QuarterCircle` with an exit action.

    When sampling, with probability `exit_probability`, the `exit_action`
    `[-inf, -inf]` is sampled. The `log_prob` function is adjusted accordingly.

    Attributes:
        delta: The radius of the quarter disk.
        epsilon: The epsilon value to consider the state as being at the
            border of the square.
        centers: The centers of the distribution.
        dist_without_exit: The distribution without the exit action.
        exit_probability: The probability of exiting.
        exit_action: The exit action.
        n_states: The number of states.
    """

    delta: float
    epsilon: float
    centers: States
    dist_without_exit: QuarterCircle
    exit_probability: Tensor
    exit_action: Tensor
    n_states: int

    def __init__(
        self,
        delta: float,
        centers: States,
        exit_probability: Tensor,
        mixture_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
        epsilon: float = 1e-4,
    ) -> None:
        """Initializes the distribution.

        Args:
            delta: the radius of the quarter disk.
            centers: the centers of the distribution with shape `(n_states, 2)`.
            exit_probability: Tensor of shape `(n_states,)` containing the
                probability of exiting the quarter disk.
            mixture_logits: Tensor of shape `(n_states, n_components)` containing the
                logits of the mixture of Beta distributions.
            alpha: Tensor of shape `(n_states, n_components)` containing the alpha
                parameters of the Beta distributions.
            beta: Tensor of shape `(n_states, n_components)` containing the beta
                parameters of the Beta distributions.
            epsilon: the epsilon value to consider the state as being at the border
                of the square.
        """
        self.n_states, n_components = mixture_logits.shape
        assert centers.tensor.shape == (self.n_states, 2)
        assert exit_probability.shape == (self.n_states,)
        assert alpha.shape == (self.n_states, n_components)
        assert beta.shape == (self.n_states, n_components)

        self.delta = delta
        self.epsilon = epsilon
        self.centers = centers
        self.dist_without_exit = QuarterCircle(
            delta=delta,
            northeastern=True,
            centers=centers,
            mixture_logits=mixture_logits,
            alpha=alpha,
            beta=beta,
        )
        self.exit_probability = exit_probability
        self.exit_action = torch.FloatTensor([-float("inf"), -float("inf")]).to(
            centers.device
        )

    def sample(self) -> Tensor:
        """Samples from the distribution.

        Returns:
            The sampled actions with shape `(n_states, 2)`.
        """
        actions = self.dist_without_exit.sample()
        repeated_exit_probability = self.exit_probability.repeat((1,))
        exit_mask = torch.bernoulli(repeated_exit_probability).bool()

        # When torch.norm(1 - states, dim=-1) <= env.delta or
        # torch.any(self.centers.tensor >= 1 - self.epsilon, dim=-1), we have to exit
        exit_mask[torch.norm(1 - self.centers.tensor, dim=-1) <= self.delta] = True
        exit_mask[torch.any(self.centers.tensor >= 1 - self.epsilon, dim=-1)] = True
        actions[exit_mask] = self.exit_action

        assert actions.shape == self.centers.tensor.shape == (self.n_states, 2)

        return actions

    def log_prob(self, sampled_actions: Tensor) -> Tensor:
        """Computes the log probability of the sampled actions.

        Args:
            sampled_actions: Tensor of shape `(*batch_shape, 2)` with the actions to
                compute the log probability of.

        Returns:
            The log probability of the sampled actions as a tensor of shape `batch_shape`.
        """
        exit = torch.all(
            sampled_actions == torch.full_like(sampled_actions[0], -float("inf")), 1
        )
        logprobs = torch.full_like(self.exit_probability, fill_value=-float("inf"))
        logprobs[~exit] = self.dist_without_exit.log_prob(sampled_actions)[~exit]
        logprobs[~exit] = logprobs[~exit] + torch.log(1 - self.exit_probability)[~exit]
        logprobs[exit] = torch.log(self.exit_probability[exit])
        # When torch.norm(1 - states, dim=-1) <= env.delta, logprobs should be 0
        # When torch.any(self.centers.tensor >= 1 - self.epsilon, dim=-1), logprobs should be 0
        logprobs[torch.norm(1 - self.centers.tensor, dim=-1) <= self.delta] = 0.0
        logprobs[torch.any(self.centers.tensor >= 1 - self.epsilon, dim=-1)] = 0.0
        return logprobs


class DistributionWrapper(Distribution):
    """A wrapper that combines `QuarterDisk` and `QuarterCircleWithExit`.

    Attributes:
        idx_is_initial: The indices of the initial states.
        idx_not_initial: The indices of the non-initial states.
        quarter_disk: The `QuarterDisk` distribution.
        quarter_circ: The `QuarterCircleWithExit`
            distribution.
    """

    idx_is_initial: Tensor
    idx_not_initial: Tensor
    _output_shape: tuple[int, ...]
    quarter_disk: Optional[QuarterDisk]
    quarter_circ: Optional[QuarterCircleWithExit]

    def __init__(
        self,
        states: States,
        delta: float,
        epsilon: float,
        mixture_logits: Tensor,
        alpha_r: Tensor,
        beta_r: Tensor,
        alpha_theta: Tensor,
        beta_theta: Tensor,
        exit_probability: Tensor,
        n_components: int,
        n_components_s0: int,
    ) -> None:
        """Initializes the distribution.

        Args:
            states: The states.
            delta: The radius of the quarter disk.
            epsilon: The epsilon value.
            mixture_logits: The logits of the mixture of Beta distributions.
            alpha_r: The alpha parameters of the Beta distributions for the radius.
            beta_r: The beta parameters of the Beta distributions for the radius.
            alpha_theta: The alpha parameters of the Beta distributions for the angle.
            beta_theta: The beta parameters of the Beta distributions for the angle.
            exit_probability: The probability of exiting.
            n_components: The number of components in the mixture.
            n_components_s0: The number of components in the mixture for s0.
        """
        self.idx_is_initial = torch.where(torch.all(states.tensor == 0, 1))[0]
        self.idx_not_initial = torch.where(torch.any(states.tensor != 0, 1))[0]
        self._output_shape = states.tensor.shape
        self.quarter_disk = None
        if len(self.idx_is_initial) > 0:
            self.quarter_disk = QuarterDisk(
                delta=delta,
                mixture_logits=mixture_logits[self.idx_is_initial[0], :n_components_s0],
                alpha_r=alpha_r[self.idx_is_initial[0], :n_components_s0],
                beta_r=beta_r[self.idx_is_initial[0], :n_components_s0],
                alpha_theta=alpha_theta[self.idx_is_initial[0], :n_components_s0],
                beta_theta=beta_theta[self.idx_is_initial[0], :n_components_s0],
            )
        self.quarter_circ = None
        if len(self.idx_not_initial) > 0:
            self.quarter_circ = QuarterCircleWithExit(
                delta=delta,
                centers=states[self.idx_not_initial],  # Remove initial states.
                exit_probability=exit_probability[self.idx_not_initial],
                mixture_logits=mixture_logits[self.idx_not_initial, :n_components],
                alpha=alpha_theta[self.idx_not_initial, :n_components],
                beta=beta_theta[self.idx_not_initial, :n_components],
                epsilon=epsilon,
            )

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Samples from the distribution.

        Args:
            sample_shape: the shape of the samples to generate.

        Returns:
            A tensor of shape `(sample_shape + self._output_shape)` containing the
            sampled actions.
        """
        output = torch.zeros(sample_shape + self._output_shape).to(
            self.idx_is_initial.device
        )

        n_disk_samples = len(self.idx_is_initial)
        if n_disk_samples > 0:
            assert self.quarter_disk is not None
            sample_disk = self.quarter_disk.sample(
                sample_shape=torch.Size(sample_shape + (n_disk_samples,))
            )
            output[self.idx_is_initial] = sample_disk
        if len(self.idx_not_initial) > 0:
            assert self.quarter_circ is not None
            sample_circ = self.quarter_circ.sample()
            output[self.idx_not_initial] = sample_circ

        return output

    def log_prob(self, sampled_actions: Tensor) -> Tensor:
        """Computes the log probability of the sampled actions.

        Args:
            sampled_actions: Tensor of shape `(*batch_shape, 2)` with the actions to
                compute the log probability of.

        Returns:
            A tensor of shape `(*batch_shape)` containing the log probabilities.
        """
        log_prob = torch.zeros(sampled_actions.shape[:-1]).to(self.idx_is_initial.device)
        n_disk_samples = len(self.idx_is_initial)
        if n_disk_samples > 0:
            assert self.quarter_disk is not None
            log_prob[self.idx_is_initial] = self.quarter_disk.log_prob(
                sampled_actions[self.idx_is_initial]
            )
        if len(self.idx_not_initial) > 0:
            assert self.quarter_circ is not None
            log_prob[self.idx_not_initial] = self.quarter_circ.log_prob(
                sampled_actions[self.idx_not_initial]
            )
        if torch.any(torch.isinf(log_prob)):
            raise ValueError("log_prob contains inf")
        return log_prob


class BoxPFMLP(MLP):
    """A MLP for the forward policy of the Box environment.

    Attributes:
        n_components_s0 (int): The number of components for s0.
        n_components (int): The number of components for non-s0 states.
        PFs0 (nn.Parameter): The parameters for the s0 distribution.
    """

    n_components_s0: int
    n_components: int
    _n_comp_max: int
    _input_dim: int
    PFs0: nn.Parameter

    def __init__(
        self,
        hidden_dim: int,
        n_hidden_layers: int,
        n_components_s0: int,
        n_components: int,
        **kwargs: Any,
    ) -> None:
        """Initializes the MLP.

        Args:
            hidden_dim: The hidden dimension.
            n_hidden_layers: The number of hidden layers.
            n_components_s0: The number of components for s0.
            n_components: The number of components for non-s0 states.
            kwargs: Other arguments for the MLP.
        """
        self._n_comp_max = max(n_components_s0, n_components)
        self.n_components_s0 = n_components_s0
        self.n_components = n_components

        input_dim = 2
        self._input_dim = input_dim

        output_dim = 1 + 3 * self.n_components

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            activation_fn="elu",
            **kwargs,
        )
        # Does not include the + 1 to handle the exit probability (which is
        # impossible at t=0).
        self.PFs0 = nn.Parameter(torch.zeros(1, 5 * self.n_components_s0))

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Computes the forward pass of the neural network.

        Args:
            preprocessed_states: The tensor states of shape `(*batch_shape, 2)`.

        Returns:
            A tensor of shape `(*batch_shape, 1 + 5 * max_n_components)`.
        """
        assert preprocessed_states.shape[-1] == 2
        batch_shape = preprocessed_states.shape[:-1]

        if preprocessed_states.ndim != 2:
            raise ValueError(
                f"preprocessed_states should be of shape [B, 2], got {preprocessed_states.shape}"
            )
        B = preprocessed_states.shape[0]
        device = preprocessed_states.device
        output_dim = 1 + 5 * self._n_comp_max

        # Run network for all states (needed for non-s0 states)
        net_out = super().forward(preprocessed_states)  # [B, 1 + 3 * n_components]

        # Detect s0 states
        is_s0 = torch.all(preprocessed_states == 0.0, dim=1, keepdim=True)  # [B, 1]

        # Build output tensor for non-s0 case (most common path)
        # Map network output [exit, weights, alpha, beta] to full output format
        desired_out = torch.zeros(B, output_dim, device=device)

        # For non-s0: copy network output to appropriate positions
        # Network outputs: [exit_logit, mixture_logits..., alpha..., beta...]
        # Format: 1 exit + n_comp mixture + n_comp alpha + n_comp beta = 1 + 3*n_comp
        n_comp = self.n_components
        desired_out[:, 0] = net_out[:, 0]  # exit logit
        desired_out[:, 1 : 1 + n_comp] = net_out[:, 1 : 1 + n_comp]  # mixture logits
        desired_out[:, 1 + self._n_comp_max : 1 + self._n_comp_max + n_comp] = net_out[
            :, 1 + n_comp : 1 + 2 * n_comp
        ]  # alpha
        desired_out[:, 1 + 2 * self._n_comp_max : 1 + 2 * self._n_comp_max + n_comp] = (
            net_out[:, 1 + 2 * n_comp :]
        )  # beta

        # For s0 states: override with PFs0 parameters using torch.where
        # PFs0 has shape [1, 5 * n_components_s0] containing [weights, alpha_r, beta_r, alpha_theta, beta_theta]
        if is_s0.any():
            # Expand PFs0 to batch size and use where to select
            pfs0_expanded = self.PFs0.expand(B, -1)  # [B, 5 * n_components_s0]
            n_s0 = self.n_components_s0

            # Map PFs0 to output positions (s0 doesn't have exit logit in PFs0)
            s0_out = torch.zeros(B, output_dim, device=device)
            s0_out[:, 1 : 1 + n_s0] = pfs0_expanded[:, :n_s0]  # mixture logits
            s0_out[:, 1 + self._n_comp_max : 1 + self._n_comp_max + n_s0] = (
                pfs0_expanded[:, n_s0 : 2 * n_s0]
            )  # alpha_r
            s0_out[:, 1 + 2 * self._n_comp_max : 1 + 2 * self._n_comp_max + n_s0] = (
                pfs0_expanded[:, 2 * n_s0 : 3 * n_s0]
            )  # beta_r
            s0_out[:, 1 + 3 * self._n_comp_max : 1 + 3 * self._n_comp_max + n_s0] = (
                pfs0_expanded[:, 3 * n_s0 : 4 * n_s0]
            )  # alpha_theta
            s0_out[:, 1 + 4 * self._n_comp_max : 1 + 4 * self._n_comp_max + n_s0] = (
                pfs0_expanded[:, 4 * n_s0 :]
            )  # beta_theta

            # Use torch.where to select between s0 and non-s0 outputs
            desired_out = torch.where(is_s0.expand(-1, output_dim), s0_out, desired_out)

        # Apply sigmoid to exit probability and concentration parameters
        # Exit probability at position 0
        desired_out[:, 0] = torch.sigmoid(desired_out[:, 0])
        # Concentration params are at positions [1+n_comp_max, end]
        desired_out[:, 1 + self._n_comp_max :] = torch.sigmoid(
            desired_out[:, 1 + self._n_comp_max :]
        )

        assert desired_out.shape == batch_shape + (1 + 5 * self._n_comp_max,)
        return desired_out


class BoxPBMLP(MLP):
    """A MLP for the backward policy of the Box environment.

    Attributes:
        n_components (int): The number of components for each distribution parameter.
    """

    n_components: int
    _input_dim: int

    def __init__(
        self,
        hidden_dim: int,
        n_hidden_layers: int,
        n_components: int,
        **kwargs: Any,
    ) -> None:
        """Initializes the MLP.

        Args:
            hidden_dim: The hidden dimension.
            n_hidden_layers: The number of hidden layers.
            n_components: The number of components for each distribution parameter.
            kwargs: Other arguments for the MLP.
        """
        input_dim = 2
        self._input_dim = input_dim
        output_dim = 3 * n_components

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            activation_fn="elu",
            **kwargs,
        )

        self.n_components = n_components

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Computes the forward pass of the neural network.

        Args:
            preprocessed_states: The tensor states of shape `(*batch_shape, 2)`.

        Returns:
            A tensor of shape `(*batch_shape, 3 * n_components)`.
        """
        assert preprocessed_states.shape[-1] == 2
        batch_shape = preprocessed_states.shape[:-1]

        out = super().forward(preprocessed_states)

        # Apply sigmoid to all except the dimensions between 0 and self.n_components.
        out[..., self.n_components :] = torch.sigmoid(out[..., self.n_components :])

        assert out.shape == batch_shape + (3 * self.n_components,)
        return out


class BoxStateFlowModule(MLP):
    """A MLP for the state flow function of the Box environment.

    Attributes:
        logZ_value (nn.Parameter): The log partition function value.
    """

    logZ_value: nn.Parameter

    def __init__(self, logZ_value: Tensor, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            logZ_value: The log partition function value.
            kwargs: Other arguments for the MLP.
        """
        super().__init__(**kwargs)
        self.logZ_value = nn.Parameter(logZ_value)

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Computes the forward pass of the neural network.

        Args:
            preprocessed_states: The tensor states of shape `(*batch_shape, input_dim)`.

        Returns:
            A tensor of shape `(*batch_shape, output_dim)`.
        """
        out = super().forward(preprocessed_states)
        idx_s0 = torch.all(preprocessed_states == 0.0, 1)
        out[idx_s0] = self.logZ_value

        return out


class BoxPBUniform(nn.Module):
    """A uniform backward policy for the Box environment.

    This module returns `(1, 1, 1)` for all states. Used with `QuarterCircle`,
    it leads to a uniform distribution over parents in the south-western part of
    the circle.
    """

    input_dim: int = 2

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Computes the forward pass of the neural network.

        Args:
            preprocessed_states: The tensor states of shape `(*batch_shape, 2)`.

        Returns:
            A tensor of shape `(*batch_shape, 3)` filled with ones.
        """
        # return (1, 1, 1) for all states, thus the "+ (3,)".
        assert preprocessed_states.shape[-1] == 2
        batch_shape = preprocessed_states.shape[:-1]
        return torch.ones(batch_shape + (3,), device=preprocessed_states.device)


def split_PF_module_output(
    output: Tensor, n_comp_max: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Splits the module output into the expected parameter sets.

    Args:
        output: the module_output from the P_F model as a tensor of shape
            `(*batch_shape, output_dim)`.
        n_comp_max: the larger number of the two `n_components` and `n_components_s0`.

    Returns:
        A tuple containing:
            - `exit_probability`: A probability unique to `QuarterCircleWithExit`.
            - `mixture_logits`: Parameters shared by `QuarterDisk` and
                `QuarterCircleWithExit`.
            - `alpha_r`: Parameters shared by `QuarterDisk` and `QuarterCircleWithExit`.
            - `beta_r`: Parameters shared by `QuarterDisk` and `QuarterCircleWithExit`.
            - `alpha_theta`: Parameters unique to `QuarterDisk`.
            - `beta_theta`: Parameters unique to `QuarterDisk`.
    """
    (
        exit_probability,  # Unique to QuarterCircleWithExit.
        mixture_logits,  # Shared by QuarterDisk and QuarterCircleWithExit.
        alpha_theta,  # Shared by QuarterDisk and QuarterCircleWithExit.
        beta_theta,  # Shared by QuarterDisk and QuarterCircleWithExit.
        alpha_r,  # Unique to QuarterDisk.
        beta_r,  # Unique to QuarterDisk.
    ) = torch.split(
        output,
        [
            1,  # Unique to QuarterCircleWithExit.
            n_comp_max,  # Shared by QuarterDisk and QuarterCircleWithExit.
            n_comp_max,  # Shared by QuarterDisk and QuarterCircleWithExit.
            n_comp_max,  # Shared by QuarterDisk and QuarterCircleWithExit.
            n_comp_max,  # Unique to QuarterDisk.
            n_comp_max,  # Unique to QuarterDisk.
        ],
        dim=-1,
    )

    return (exit_probability, mixture_logits, alpha_theta, beta_theta, alpha_r, beta_r)


class BoxPFEstimator(Estimator, PolicyMixin):
    r"""Estimator for `P_F` for the Box environment.

    This estimator uses the `DistributionWrapper` distribution.

    Attributes:
        n_components_s0: The number of components for s0.
        n_components: The number of components for non-s0 states.
        min_concentration: The minimum concentration for the Beta distributions.
        max_concentration: The maximum concentration for the Beta distributions.
        delta: The radius of the quarter disk.
        epsilon: The epsilon value.
    """

    _n_comp_max: int
    n_components_s0: int
    n_components: int
    min_concentration: float
    max_concentration: float
    delta: float
    epsilon: float

    def __init__(
        self,
        env: Box,
        module: nn.Module,
        n_components_s0: int,
        n_components: int,
        min_concentration: float = 0.1,
        max_concentration: float = 2.0,
    ) -> None:
        """Initializes the estimator.

        Args:
            env: The environment.
            module: The module to use.
            n_components_s0: The number of components for s0.
            n_components: The number of components for non-s0 states.
            min_concentration: The minimum concentration for the Beta distributions.
            max_concentration: The maximum concentration for the Beta distributions.
        """
        super().__init__(module)
        self._n_comp_max = max(n_components_s0, n_components)
        self.n_components_s0 = n_components_s0
        self.n_components = n_components

        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.delta = env.delta
        self.epsilon = env.epsilon

    @property
    def expected_output_dim(self) -> int:
        """Returns the expected output dimension of the module."""
        return 1 + 5 * self._n_comp_max

    def to_probability_distribution(
        self, states: States, module_output: Tensor
    ) -> Distribution:
        """Converts the module output to a probability distribution.

        Args:
            states: the states for which to convert the module output to a
                probability distribution.
            module_output: the output of the module for the states as a tensor of
                shape `(*batch_shape, output_dim)`.

        Returns:
            The probability distribution for the states.
        """
        # First, we verify that the batch shape of states is 1
        assert len(states.batch_shape) == 1

        # The module_output is of shape (*batch_shape, 1 + 5 * max_n_components), why:
        # We need:
        #   + one scalar for the exit probability,
        #   + self.n_components for the alpha_theta
        #   + self.n_components for the betas_theta
        #   + self.n_components for the mixture logits
        # but we also need compatibility with the s0 state, which has two additional
        # parameters:
        #   + self.n_s0_components for the alpha_r
        #   + self.n_s0_components for the beta_r
        # and finally, we want to be able to give a different number of parameters to
        # s0 and st. So we need to use self._n_comp_max to split on the larger size, and
        # then index to slice out the smaller size when appropriate.
        assert module_output.shape == states.batch_shape + (1 + 5 * self._n_comp_max,)

        (
            exit_probability,
            mixture_logits,
            alpha_theta,
            beta_theta,
            alpha_r,
            beta_r,
        ) = split_PF_module_output(module_output, self._n_comp_max)
        mixture_logits = mixture_logits  # .contiguous().view(-1)

        def _normalize(x: Tensor) -> Tensor:
            return (
                self.min_concentration
                + (self.max_concentration - self.min_concentration) * x
            )  # .contiguous().view(-1)

        alpha_r = _normalize(alpha_r)
        beta_r = _normalize(beta_r)
        alpha_theta = _normalize(alpha_theta)
        beta_theta = _normalize(beta_theta)

        return DistributionWrapper(
            states,
            self.delta,
            self.epsilon,
            mixture_logits,
            alpha_r,
            beta_r,
            alpha_theta,
            beta_theta,
            exit_probability.squeeze(-1),
            self.n_components,
            self.n_components_s0,
        )


class BoxPBEstimator(Estimator, PolicyMixin):
    r"""Estimator for `P_B` for the Box environment.

    This estimator uses the `QuarterCircle(northeastern=False)` distribution.

    Attributes:
        n_components: The number of components for the mixture.
        min_concentration: The minimum concentration for the Beta distributions.
        max_concentration: The maximum concentration for the Beta distributions.
        delta: The radius of the quarter disk.
    """

    n_components: int
    min_concentration: float
    max_concentration: float
    delta: float

    def __init__(
        self,
        env: Box,
        module: nn.Module,
        n_components: int,
        min_concentration: float = 0.1,
        max_concentration: float = 2.0,
    ) -> None:
        """Initializes the estimator.

        Args:
            env: The environment.
            module: The module to use.
            n_components: The number of components for the mixture.
            min_concentration: The minimum concentration for the Beta distributions.
            max_concentration: The maximum concentration for the Beta distributions.
        """
        super().__init__(module, is_backward=True)
        self.module = module
        self.n_components = n_components

        self.min_concentration = min_concentration
        self.max_concentration = max_concentration

        self.delta = env.delta

    @property
    def expected_output_dim(self) -> int:
        """Returns the expected output dimension of the module."""
        return 3 * self.n_components

    def to_probability_distribution(
        self, states: States, module_output: Tensor
    ) -> Distribution:
        """Converts the module output to a probability distribution.

        Args:
            states: the states for which to convert the module output to a
                probability distribution.
            module_output: the output of the module for the states as a tensor of
                shape `(*batch_shape, output_dim)`.

        Returns:
            The probability distribution for the states.
        """
        # First, we verify that the batch shape of states is 1
        assert len(states.batch_shape) == 1
        mixture_logits, alpha, beta = torch.split(
            module_output, self.n_components, dim=-1
        )

        def _normalize(x: Tensor) -> Tensor:
            return (
                self.min_concentration
                + (self.max_concentration - self.min_concentration) * x
            )  # .contiguous().view(-1)

        if not isinstance(self.module, BoxPBUniform):
            alpha = _normalize(alpha)
            beta = _normalize(beta)
        return QuarterCircle(
            delta=self.delta,
            northeastern=False,
            centers=states,
            mixture_logits=mixture_logits,
            alpha=alpha,
            beta=beta,
        )
