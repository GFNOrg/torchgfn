"""Cartesian increment estimators and distributions for the Box environment."""

from typing import Any

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Beta, Categorical, Distribution, MixtureSameFamily

from gfn.estimators import Estimator, PolicyMixin
from gfn.gym.box import BoxPolar
from gfn.states import States
from gfn.utils.modules import MLP


class BoxCartesianDistribution(Distribution):
    """Cartesian increment distribution for Box environment.

    Uses MixtureSameFamily(Categorical, Beta) per dimension for sampling increments.
    Much simpler than polar coordinates - samples relative increments per dimension
    and converts to absolute using: action = min_incr + r * (max_range).

    Attributes:
        delta: Minimum step size.
        epsilon: Small value for numerical stability.
    """

    arg_constraints: dict = {}  # No constraints for custom distribution

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
        # s0: can go anywhere in [0, 1] (like gflownet's first step from source)
        # non-s0: must step at least delta, max is 1-state
        is_s0 = torch.all(states.tensor == 0, dim=-1, keepdim=True)
        self.is_s0 = is_s0.squeeze(-1)
        self.min_incr = torch.where(is_s0, 0.0, delta)
        # For s0: max_range = 1.0 (action in [0, 1], full coverage of state space)
        # For non-s0: max_range = 1 - state - delta (action in [delta, 1-state])
        self.max_range = torch.where(
            is_s0,
            torch.ones_like(states.tensor),
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
        # For s0: action in [0, 1] (full space coverage), for non-s0: action in [delta, 1-state]
        is_s0_expanded = self.is_s0.unsqueeze(-1)
        max_action = torch.where(
            is_s0_expanded,
            torch.ones_like(actions),  # s0 can reach anywhere in [0, 1]
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

        # At boundary, exit is forced (log_prob = 0 for exit, -inf for non-exit)
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
        safe_max_range = self.max_range.clamp(min=self.epsilon)
        r = (safe_actions - self.min_incr) / safe_max_range
        r = r.clamp(self.epsilon, 1 - self.epsilon)

        # Get log prob from Beta mixture (sum over dimensions)
        log_p_beta_per_dim = self.increment_dist.log_prob(r)
        log_p_beta = log_p_beta_per_dim.sum(dim=-1)

        # Jacobian correction for change of variables:
        # action = min_incr + r * max_range, so dr/daction = 1/max_range
        # log p(action) = log p(r) + log|dr/daction| = log p(r) - log(max_range)
        log_jacobian_per_dim = -torch.log(safe_max_range)
        log_jacobian = log_jacobian_per_dim.sum(dim=-1)

        # Add log(1 - exit_prob) for choosing not to exit.
        # Clamp exit probability to avoid log(0) when exit_prob = 1.
        # From s0, exit is forbidden during sampling so the exit logit is meaningless;
        # zero out log_no_exit there so log P_F(a | s0) = log_p_beta + log_jacobian only.
        exit_probs_clamped = self.exit_dist.probs.clamp(
            min=self.epsilon, max=1 - self.epsilon
        )
        log_no_exit = torch.log1p(-exit_probs_clamped)
        log_no_exit = torch.where(self.is_s0, torch.zeros_like(log_no_exit), log_no_exit)

        # Non-exit log prob
        log_probs = log_p_beta + log_jacobian + log_no_exit

        # For exit actions: log P(exit)
        log_p_exit = self.exit_dist.log_prob(torch.ones(1, device=device))
        log_probs = torch.where(is_exit, log_p_exit.expand_as(log_probs), log_probs)

        # Handle boundary states specially:
        # - If at boundary and exiting: use learned exit log_prob (same as non-boundary).
        #   Do NOT force log_prob = 0; the exit Bernoulli still contributes to the TB loss
        #   so the policy learns to exit at the right place (reward ring vs hard boundary).
        # - If at boundary and not exiting: log_prob = -inf (impossible)
        log_probs = torch.where(
            at_boundary & ~is_exit, torch.full_like(log_probs, float("-inf")), log_probs
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
        env: BoxPolar,
        module: nn.Module,
        n_components: int,
        min_concentration: float = 0.1,
        max_concentration: float = 5.0,
        numerical_epsilon: float = 1e-6,
    ) -> None:
        """Initialize the estimator.

        Args:
            env: The Box environment.
            module: The neural network module.
            n_components: Number of mixture components.
            min_concentration: Minimum Beta concentration parameter.
            max_concentration: Maximum Beta concentration parameter.
            numerical_epsilon: Small constant for clamping max_range before log (Jacobian
                stability). Kept separate from env.epsilon (geometric tolerance) to avoid
                log-Jacobian explosions when env.epsilon is very small (e.g. 1e-10).
        """
        super().__init__(module)
        self.n_components = n_components
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.delta = env.delta
        self.epsilon = env.epsilon
        self.numerical_epsilon = numerical_epsilon
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
            epsilon=self.numerical_epsilon,
        )


class BoxCartesianPBDistribution(Distribution):
    """Backward Cartesian distribution for Box environment.

    In torchgfn's design, the source state is the origin [0, 0]. The BTS
    (back-to-source) action is deterministic: when action = state, we go
    directly to the origin. This is always the case for the s1 -> s0 transition.

    Unlike gflownet's ContinuousCube (where source is abstract and BTS is
    stochastic), here BTS is always forced/deterministic with log_prob = 0.
    """

    arg_constraints: dict = {}  # No constraints for custom distribution

    def __init__(
        self,
        states: States,
        mixture_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
        delta: float,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize the backward distribution.

        Args:
            states: Current states.
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

        # Per-dimension: if state[d] <= delta, BTS is forced for that trajectory
        # (when state = delta, the only valid backward action is BTS since action range is empty)
        self.dim_near_origin = states.tensor <= delta
        self.any_dim_near_origin = torch.any(self.dim_near_origin, dim=-1)

        # Increment distribution per dimension
        mix = Categorical(logits=mixture_logits)
        components = Beta(alpha, beta)
        self.increment_dist = MixtureSameFamily(mix, components)

        # For backward: action in [delta, state] for dims where state >= delta
        self.min_incr = delta
        self.max_range = (states.tensor - self.min_incr).clamp(min=epsilon)

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Sample backward actions.

        BTS (action = state) only happens when forced (near origin).
        Otherwise, sample decrements from Beta mixture.
        """
        # BTS is only taken when forced (near origin) - no stochastic BTS
        is_bts = self.any_dim_near_origin

        # Start with BTS actions (action = state, goes to origin)
        actions = self.states.tensor.clone()

        # For non-BTS, sample decrements from Beta mixture
        if not is_bts.all():
            r = self.increment_dist.sample().clamp(0.0, 1.0)
            sampled_actions = self.min_incr + r * self.max_range
            # Clamp to ensure action <= state
            sampled_actions = torch.min(sampled_actions, self.states.tensor)

            # Use sampled actions only for non-BTS trajectories
            is_bts_expanded = is_bts.unsqueeze(-1)
            actions = torch.where(is_bts_expanded, actions, sampled_actions)

        return actions

    def log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability of backward actions.

        BTS actions (action = state) always have log_prob = 0 (deterministic).
        Non-BTS actions have log_prob = log p_beta(r) + log_jacobian.

        When near origin (any dim <= delta), BTS is forced and log_prob = 0.
        """
        # Check if action is BTS (action equals state)
        # Use a tolerance based on delta for numerical robustness
        bts_tolerance = self.delta * 0.1  # 10% of delta
        is_bts = torch.all(
            torch.abs(actions - self.states.tensor) < bts_tolerance, dim=-1
        )

        # When near origin, BTS is forced - treat any action as BTS
        # (the only valid action from near-origin states is BTS anyway)
        is_bts = is_bts | self.any_dim_near_origin

        # For non-BTS actions, compute Beta mixture log_prob
        # Use safe placeholders for BTS actions to avoid NaN
        safe_actions = torch.where(
            is_bts.unsqueeze(-1),
            self.min_incr + 0.5 * self.max_range.clamp(min=self.epsilon),  # placeholder
            actions,
        )

        # Convert to relative: r = (action - delta) / max_range
        # Use a reasonable minimum for max_range to avoid extreme Jacobians
        safe_max_range = self.max_range.clamp(min=self.epsilon)
        r = (safe_actions - self.min_incr) / safe_max_range
        r = r.clamp(self.epsilon, 1 - self.epsilon)

        # Get log prob from Beta mixture per dimension
        log_p_per_dim = self.increment_dist.log_prob(r)

        # Jacobian: dr/daction = 1/max_range
        log_jacobian_per_dim = -torch.log(safe_max_range)

        # Sum over dimensions and replace NaN with large negative (invalid)
        log_p_beta = (log_p_per_dim + log_jacobian_per_dim).sum(dim=-1)
        log_p_beta = torch.nan_to_num(log_p_beta, nan=-1e6, posinf=1e6, neginf=-1e6)

        # BTS actions have log_prob = 0 (deterministic)
        # Non-BTS actions have log_prob from Beta + Jacobian
        log_probs = torch.where(
            is_bts,
            torch.zeros_like(log_p_beta),  # BTS: always 0
            log_p_beta,  # Non-BTS: Beta + Jacobian
        )

        return log_probs


class BoxCartesianPBEstimator(Estimator, PolicyMixin):
    """Simplified PB estimator using Cartesian increments with back-to-source."""

    def __init__(
        self,
        env: BoxPolar,
        module: nn.Module,
        n_components: int,
        min_concentration: float = 0.1,
        max_concentration: float = 5.0,
        numerical_epsilon: float = 1e-6,
    ) -> None:
        """Initialize the estimator."""
        super().__init__(module, is_backward=True)
        self.n_components = n_components
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.delta = env.delta
        self.epsilon = env.epsilon
        self.numerical_epsilon = numerical_epsilon
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

        # Parse module output: [mixture_logits..., alpha..., beta...]
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
            epsilon=self.numerical_epsilon,
        )


class BoxCartesianPFMLP(MLP):
    """Simplified MLP for Box forward policy using Cartesian increments.

    Output format: [exit_logit, mixture_logits..., alpha..., beta...]
    where mixture_logits, alpha, beta each have shape n_dim * n_components.

    States are normalized from [0, 1] to [-1, 1] before the forward pass to
    match the gflownet reference (states2policy normalization) and to provide
    symmetric, zero-centred inputs to the network.
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
        """Forward pass. Normalizes [0, 1] states to [-1, 1] before the MLP."""
        return super().forward(2.0 * preprocessed_states - 1.0)


class BoxCartesianPBMLP(MLP):
    """Simplified MLP for Box backward policy using Cartesian increments.

    Output format: [mixture_logits..., alpha..., beta...]
    where mixture_logits, alpha, beta each have shape n_dim * n_components.

    States are normalized from [0, 1] to [-1, 1] before the forward pass,
    matching the gflownet reference normalization.
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

        # Output: (weights + alpha + beta) * n_dim * n_comp
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
        """Forward pass. Normalizes [0, 1] states to [-1, 1] before the MLP."""
        return super().forward(2.0 * preprocessed_states - 1.0)
