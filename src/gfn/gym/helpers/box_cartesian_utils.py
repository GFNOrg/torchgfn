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
        temperature: float = 1.0,
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
            temperature: Softmax temperature > 1 increases entropy of discrete
                choices (exit Bernoulli, mixture component selection), providing
                exploration. T=1 is the default (no smoothing). Applied only to
                logits; Beta alpha/beta parameters are left unchanged.
        """
        super().__init__()
        self.delta = delta
        self.epsilon = epsilon
        self.states = states
        self.n_dim = states.tensor.shape[-1]

        # Exit distribution — divided by temperature for entropy control
        from torch.distributions import Bernoulli

        self.exit_dist = Bernoulli(logits=exit_logits / temperature)

        # Increment distribution per dimension — mixture logits divided by temperature
        mix = Categorical(logits=mixture_logits / temperature)
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
        max_concentration: float = 100.0,
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
        self.temperature: float = 1.0  # set externally to anneal exploration

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
            temperature=self.temperature,
        )


class BoxCartesianPBDistribution(Distribution):
    """Backward Cartesian distribution for Box environment.

    In torchgfn's design, the source state is the origin [0, 0]. The BTS
    (back-to-source) action moves directly to s0 by setting action = state.

    WHY THE LEARNED BTS BERNOULLI IS CRITICAL
    ------------------------------------------
    The Trajectory Balance (TB) loss is:

        L = (log P_F(τ) + log Z - log P_B(τ) - log R(x_T))^2

    The BTS step (x_1 → s0) is the *last* step of every backward trajectory.
    Before this fix, BTS was always deterministic (forced): log P_B(BTS | x_1) = 0.
    A constant zero drops out of the gradient, so P_B received no gradient
    from the BTS step — regardless of trajectory length.

    For 1-step trajectories (s0 → x_T, then immediately BTS back to s0), P_B
    was *entirely* gradient-free: log P_B(τ) = 0 for every such trajectory,
    making P_B invisible to the TB loss. This is particularly harmful because
    the reward landscape is dominated by states reachable from s0 in one step
    (the high-reward ring at |x - 0.5| ∈ (0.3, 0.4)).

    With a learned Bernoulli P(BTS | x):
    - log P_B(BTS | x_1)  = log P(BTS=1 | x_1)  — gradient flows into P_B
    - log P_B(~BTS | x_1) = log P(BTS=0 | x_1)  — also receives gradient
    This closes the TB loop fully: P_B now has an incentive to assign higher
    probability to BTS from states that are indeed close to the reward modes,
    and lower probability from states that are far away.

    FORCED vs. STOCHASTIC BTS
    -------------------------
    When any dimension of the state is <= delta, the valid backward increment
    range [delta, state[d]] is empty for that dimension. BTS is the *only*
    valid action, so it is forced (log_prob = 0, deterministic). For all other
    states, BTS is an optional choice sampled from the learned Bernoulli.
    """

    arg_constraints: dict = {}  # No constraints for custom distribution

    def __init__(
        self,
        states: States,
        bts_logits: Tensor,
        mixture_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
        delta: float,
        epsilon: float = 1e-6,
        temperature: float = 1.0,
    ) -> None:
        """Initialize the backward distribution.

        Args:
            states: Current states.
            bts_logits: Logits for the BTS Bernoulli, shape (batch,).
            mixture_logits: Mixture weights, shape (batch, n_dim, n_components).
            alpha: Beta alpha params, shape (batch, n_dim, n_components).
            beta: Beta beta params, shape (batch, n_dim, n_components).
            delta: Minimum step size.
            epsilon: Numerical stability constant.
            temperature: Softmax temperature; see BoxCartesianDistribution.
        """
        super().__init__()
        self.delta = delta
        self.epsilon = epsilon
        self.states = states
        self.n_dim = states.tensor.shape[-1]

        # BTS Bernoulli: learned probability of going back to source
        from torch.distributions import Bernoulli

        self.bts_dist = Bernoulli(logits=bts_logits / temperature)

        # BTS is forced when any dimension is within delta of origin
        self.dim_near_origin = states.tensor <= delta
        self.any_dim_near_origin = torch.any(self.dim_near_origin, dim=-1)

        # Increment distribution per dimension
        mix = Categorical(logits=mixture_logits / temperature)
        components = Beta(alpha, beta)
        self.increment_dist = MixtureSameFamily(mix, components)

        # For backward: action in [delta, state] for dims where state >= delta
        self.min_incr = delta
        self.max_range = (states.tensor - self.min_incr).clamp(min=epsilon)

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Sample backward actions.

        BTS (action = state) is forced when near origin; otherwise sampled
        from the learned Bernoulli. Non-BTS samples come from Beta mixture.
        """
        # Forced BTS for states with any dim <= delta
        is_bts = self.any_dim_near_origin.clone()

        # Stochastic BTS for non-forced states via learned Bernoulli
        can_choose_bts = ~self.any_dim_near_origin
        if can_choose_bts.any():
            sampled_bts = self.bts_dist.sample().bool() & can_choose_bts
            is_bts = is_bts | sampled_bts

        # Start with BTS actions (action = state → next state = s0)
        actions = self.states.tensor.clone()

        # For non-BTS, sample decrements from Beta mixture
        if not is_bts.all():
            r = self.increment_dist.sample().clamp(0.0, 1.0)
            sampled_actions = self.min_incr + r * self.max_range
            sampled_actions = torch.min(sampled_actions, self.states.tensor)
            is_bts_expanded = is_bts.unsqueeze(-1)
            actions = torch.where(is_bts_expanded, actions, sampled_actions)

        return actions

    def log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability of backward actions.

        - Forced BTS (any_dim_near_origin): log_prob = 0 (deterministic).
        - Stochastic BTS: log_prob = log P(BTS=1 | s) from Bernoulli.
        - Non-BTS: log_prob = log P(BTS=0 | s) + log_p_beta + log_jacobian.
        """
        # Detect BTS: action ≈ state within delta/2 tolerance
        bts_tolerance = self.delta * 0.5
        is_bts = torch.all(
            torch.abs(actions - self.states.tensor) < bts_tolerance, dim=-1
        )

        # Safe placeholder for BTS slots to avoid NaN in Beta log_prob
        safe_actions = torch.where(
            is_bts.unsqueeze(-1),
            self.min_incr + 0.5 * self.max_range.clamp(min=self.epsilon),
            actions,
        )

        # Convert absolute to relative: r = (action - delta) / max_range
        safe_max_range = self.max_range.clamp(min=self.epsilon)
        r = (safe_actions - self.min_incr) / safe_max_range
        r = r.clamp(self.epsilon, 1 - self.epsilon)

        log_p_per_dim = self.increment_dist.log_prob(r)
        log_jacobian_per_dim = -torch.log(safe_max_range)
        log_p_beta = (log_p_per_dim + log_jacobian_per_dim).sum(dim=-1)
        log_p_beta = torch.nan_to_num(log_p_beta, nan=-1e6, posinf=1e6, neginf=-1e6)

        # Bernoulli log_prob for each element choosing BTS vs. not
        log_p_bts = self.bts_dist.log_prob(torch.ones_like(log_p_beta))
        log_p_no_bts = self.bts_dist.log_prob(torch.zeros_like(log_p_beta))

        log_probs = torch.where(
            self.any_dim_near_origin,
            torch.zeros_like(log_p_beta),  # forced BTS: deterministic
            torch.where(
                is_bts,
                log_p_bts,  # stochastic BTS: Bernoulli log P(BTS=1)
                log_p_no_bts + log_p_beta,  # non-BTS: Bernoulli + Beta
            ),
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
        max_concentration: float = 100.0,
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
        self.temperature: float = 1.0  # set externally to anneal exploration

    @property
    def expected_output_dim(self) -> int:
        """Expected output dimension: bts_logit + (weights + alpha + beta) * n_dim * n_comp."""
        return 1 + 3 * self.n_dim * self.n_components

    def to_probability_distribution(
        self, states: States, module_output: Tensor
    ) -> Distribution:
        """Convert module output to backward probability distribution."""
        batch_size = states.tensor.shape[0]
        n_comp = self.n_components
        n_dim = self.n_dim

        # Parse module output: [bts_logit, mixture_logits..., alpha..., beta...]
        bts_logits = module_output[:, 0]

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
        conc_range = self.max_concentration - self.min_concentration
        alpha = self.min_concentration + conc_range * torch.sigmoid(alpha_raw)
        beta = self.min_concentration + conc_range * torch.sigmoid(beta_raw)

        return BoxCartesianPBDistribution(
            states=states,
            bts_logits=bts_logits,
            mixture_logits=mixture_logits,
            alpha=alpha,
            beta=beta,
            delta=self.delta,
            epsilon=self.numerical_epsilon,
            temperature=self.temperature,
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

        # Output: bts_logit + (weights + alpha + beta) * n_dim * n_comp
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
