"""GFNX library (JAX-based) runner for benchmarking.

This runner adapts the gfnx library from benchmark/gfnx/.
The library uses JAX/Equinox, so proper JIT warmup and synchronization
via jax.block_until_ready() is critical for accurate timing.

Supports environments: hypergrid, ising, bitseq.

Key JAX/Equinox concepts used here:
- eqx.filter_jit: JIT-compiles functions while handling non-array leaves (like functions)
- eqx.partition/combine: Splits pytrees into array/non-array parts for gradient computation
- jax.vmap: Vectorizes functions over batch dimensions
- optax.multi_transform: Applies different optimizers to different parameter groups
"""

import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional

from benchmark.lib_runners.base import BenchmarkConfig, LibraryRunner

# Add gfnx to path so we can import the library
GFNX_PATH = Path(__file__).parent.parent / "gfnx"
if str(GFNX_PATH / "src") not in sys.path:
    sys.path.insert(0, str(GFNX_PATH / "src"))


class GFNXRunner(LibraryRunner):
    """Benchmark runner for the gfnx JAX-based library.

    Critical timing considerations for JAX:
    - JIT compilation happens on first call, so warmup is essential
    - Use jax.block_until_ready() to ensure async operations complete

    Supports environments:
    - hypergrid: Discrete grid navigation
    - ising: Discrete Ising model environment
    - bitseq: Bit sequence generation
    """

    name = "gfnx"

    def __init__(self):
        # Training state contains all JAX arrays and pytrees needed for training
        self.train_state = None
        # JIT-compiled training step function (created per environment type)
        self.train_step_fn = None
        self.config = None
        self._iteration = 0
        self._env_type = None

    def setup(self, config: BenchmarkConfig, seed: int) -> None:
        """Initialize environment, model, and optimizer based on env_name.

        Dispatches to environment-specific setup methods which create:
        1. The gfnx environment and its parameters
        2. The policy network (MLPPolicy)
        3. The optimizer with separate learning rates for network and logZ
        4. The TrainState NamedTuple containing all training state
        5. The JIT-compiled training step function
        """
        self.config = config
        self._env_type = config.env_name

        if config.env_name == "hypergrid":
            self._setup_hypergrid(config, seed)
        elif config.env_name == "ising":
            self._setup_ising(config, seed)
        elif config.env_name == "bitseq":
            self._setup_bitseq(config, seed)
        else:
            raise ValueError(f"Unknown environment: {config.env_name}")

        self._iteration = 0

    def _setup_hypergrid(self, config: BenchmarkConfig, seed: int) -> None:
        """Setup HyperGrid environment with Trajectory Balance loss.

        HyperGrid is a discrete navigation environment where the agent moves
        through an N-dimensional grid from origin to any terminal state.
        """
        import equinox as eqx
        import gfnx
        import jax
        import jax.numpy as jnp
        import optax

        ndim = config.env_kwargs["ndim"]
        height = config.env_kwargs["height"]

        # JAX uses explicit PRNG keys for reproducibility
        rng_key = jax.random.PRNGKey(seed)
        env_init_key = jax.random.PRNGKey(seed + 1)

        # Create gfnx environment with reward module
        # EasyHypergridRewardModule provides a simple reward structure
        reward_module = gfnx.EasyHypergridRewardModule()
        env = gfnx.environment.HypergridEnvironment(reward_module, dim=ndim, side=height)
        # env.init() creates the environment parameters (reward params, etc.)
        env_params = env.init(env_init_key)

        # Create policy network that outputs both forward and backward logits
        rng_key, net_init_key = jax.random.split(rng_key)
        model = MLPPolicy(
            input_size=env.observation_space.shape[0],
            n_fwd_actions=env.action_space.n,
            n_bwd_actions=env.backward_action_space.n,
            hidden_size=config.hidden_dim,
            train_backward_policy=True,  # Learn P_B for TB loss
            depth=config.n_layers,
            rng_key=net_init_key,
        )

        # logZ is a learnable scalar for the partition function estimate
        logZ = jnp.array(0.0)

        # Setup optimizer with separate learning rates for network and logZ
        # This is common in GFlowNet training where logZ needs higher LR
        model_params_init = eqx.filter(model, eqx.is_array)
        initial_optax_params = {"model_params": model_params_init, "logZ": logZ}

        # param_labels maps each parameter to its optimizer group
        param_labels = {
            "model_params": jax.tree.map(lambda _: "network_lr", model_params_init),
            "logZ": "logZ_lr",
        }

        optimizer_defs = {
            "network_lr": optax.adam(learning_rate=config.lr),
            "logZ_lr": optax.adam(learning_rate=config.lr_logz),
        }
        optimizer = optax.multi_transform(optimizer_defs, param_labels)  # type: ignore
        opt_state = optimizer.init(initial_optax_params)

        # No exploration during benchmark for consistency
        exploration_schedule = optax.constant_schedule(0.0)

        # Pack everything into a NamedTuple for easy passing through JAX transforms
        self.train_state = HypergridTrainState(
            rng_key=rng_key,
            env=env,
            env_params=env_params,
            model=model,
            logZ=logZ,
            optimizer=optimizer,
            opt_state=opt_state,
            exploration_schedule=exploration_schedule,
            num_envs=config.batch_size,
        )

        # Create the JIT-compiled training step
        self.train_step_fn = create_hypergrid_train_step()

    def _setup_ising(self, config: BenchmarkConfig, seed: int) -> None:
        """Setup Ising environment with Trajectory Balance loss.

        The Ising environment models spin configurations on a lattice.
        Each state is a binary assignment of spins, built incrementally.
        """
        import equinox as eqx
        import gfnx
        import jax
        import jax.numpy as jnp
        import optax

        L = config.env_kwargs["L"]  # Lattice side length
        N = L**2  # Total number of spins

        rng_key = jax.random.PRNGKey(seed)
        env_init_key = jax.random.PRNGKey(seed + 1)

        # IsingRewardModule computes rewards based on spin configurations
        reward_module = gfnx.IsingRewardModule()
        env = gfnx.environment.IsingEnvironment(reward_module, dim=N)
        env_params = env.init(env_init_key)

        rng_key, net_init_key = jax.random.split(rng_key)
        model = MLPPolicy(
            input_size=env.observation_space.shape[0],
            n_fwd_actions=env.action_space.n,
            n_bwd_actions=env.backward_action_space.n,
            hidden_size=config.hidden_dim,
            train_backward_policy=True,
            depth=config.n_layers,
            rng_key=net_init_key,
        )

        logZ = jnp.array(0.0)

        model_params_init = eqx.filter(model, eqx.is_array)
        initial_optax_params = {"model_params": model_params_init, "logZ": logZ}

        param_labels = {
            "model_params": jax.tree.map(lambda _: "network_lr", model_params_init),
            "logZ": "logZ_lr",
        }

        optimizer_defs = {
            "network_lr": optax.adam(learning_rate=config.lr),
            "logZ_lr": optax.adam(learning_rate=config.lr_logz),
        }
        optimizer = optax.multi_transform(optimizer_defs, param_labels)  # type: ignore
        opt_state = optimizer.init(initial_optax_params)

        exploration_schedule = optax.constant_schedule(0.0)

        self.train_state = IsingTrainState(
            rng_key=rng_key,
            env=env,
            env_params=env_params,
            model=model,
            logZ=logZ,
            optimizer=optimizer,
            opt_state=opt_state,
            exploration_schedule=exploration_schedule,
            num_envs=config.batch_size,
        )

        self.train_step_fn = create_ising_train_step()

    def _setup_bitseq(self, config: BenchmarkConfig, seed: int) -> None:
        """Setup BitSequence environment with Trajectory Balance loss.

        BitSequence generates sequences of bits/tokens, with rewards based
        on proximity to a set of target "mode" sequences.
        """
        import equinox as eqx
        import gfnx
        import jax
        import jax.numpy as jnp
        import optax

        word_size = config.env_kwargs.get("word_size", 1)  # Bits per word
        seq_size = config.env_kwargs.get("seq_size", 4)  # Words per sequence
        n_modes = config.env_kwargs.get("n_modes", 2)  # Number of target modes

        rng_key = jax.random.PRNGKey(seed)
        env_init_key = jax.random.PRNGKey(seed + 1)

        # BitseqRewardModule rewards sequences close to the mode set
        reward_module = gfnx.BitseqRewardModule(
            sentence_len=seq_size,
            k=word_size,
            mode_set_size=n_modes,
            reward_exponent=2.0,  # Sharpness of reward around modes
        )
        env = gfnx.BitseqEnvironment(reward_module, n=seq_size, k=word_size)
        env_params = env.init(env_init_key)

        rng_key, net_init_key = jax.random.split(rng_key)
        model = MLPPolicy(
            input_size=env.observation_space.shape[0],
            n_fwd_actions=env.action_space.n,
            n_bwd_actions=env.backward_action_space.n,
            hidden_size=config.hidden_dim,
            train_backward_policy=True,
            depth=config.n_layers,
            rng_key=net_init_key,
        )

        logZ = jnp.array(0.0)

        model_params_init = eqx.filter(model, eqx.is_array)
        initial_optax_params = {"model_params": model_params_init, "logZ": logZ}

        param_labels = {
            "model_params": jax.tree.map(lambda _: "network_lr", model_params_init),
            "logZ": "logZ_lr",
        }

        optimizer_defs = {
            "network_lr": optax.adam(learning_rate=config.lr),
            "logZ_lr": optax.adam(learning_rate=config.lr_logz),
        }
        optimizer = optax.multi_transform(optimizer_defs, param_labels)  # type: ignore
        opt_state = optimizer.init(initial_optax_params)

        exploration_schedule = optax.constant_schedule(0.0)

        self.train_state = BitseqTrainState(
            rng_key=rng_key,
            env=env,
            env_params=env_params,
            model=model,
            logZ=logZ,
            optimizer=optimizer,
            opt_state=opt_state,
            exploration_schedule=exploration_schedule,
            num_envs=config.batch_size,
        )

        self.train_step_fn = create_bitseq_train_step()

    def warmup(self, n_iters: int) -> None:
        """Run warmup iterations to trigger JIT compilation.

        JAX compiles functions on first call. Running warmup iterations
        ensures compilation overhead isn't counted in benchmark timing.
        """
        for i in range(n_iters):
            self.train_state = self.train_step_fn(i, self.train_state)

        # Block until all async operations complete
        self.synchronize()
        self._iteration = 0

    def run_iteration(self) -> None:
        """Run a single training iteration.

        The train_step_fn is already JIT-compiled, so this is just
        a function call that updates the train_state in place.
        """
        self.train_state = self.train_step_fn(self._iteration, self.train_state)
        self._iteration += 1

    def synchronize(self) -> None:
        """Ensure all JAX operations are complete.

        JAX operations are asynchronous - they return immediately while
        computation continues on device. block_until_ready() forces
        synchronization for accurate timing measurements.
        """
        import equinox as eqx
        import jax

        if self.train_state is not None:
            # Extract all arrays from train_state and wait for them
            params, _ = eqx.partition(self.train_state, eqx.is_array)
            jax.block_until_ready(params)

    def get_peak_memory(self) -> Optional[int]:
        """Return peak memory usage in bytes.

        JAX memory stats are device-dependent and may not be available
        on all backends (e.g., CPU backend doesn't track memory).
        """
        import jax

        try:
            devices = jax.local_devices()
            if devices and hasattr(devices[0], "memory_stats"):
                stats = devices[0].memory_stats()
                if stats and "peak_bytes_in_use" in stats:
                    return stats["peak_bytes_in_use"]
        except Exception:
            pass
        return None

    def cleanup(self) -> None:
        """Release resources and clear JAX caches."""
        import jax

        self.train_state = None
        self.train_step_fn = None

        # Clear compiled function caches to free memory
        jax.clear_caches()


# ============================================================================
# Train State definitions for each environment
# ============================================================================
# Using NamedTuples allows these to be passed through JAX transforms.
# Each field is typed as Any because JAX arrays and pytrees don't have
# static types that work well with Python's type system.


class HypergridTrainState(NamedTuple):
    """Training state for Hypergrid environment."""

    rng_key: Any  # JAX PRNG key for stochastic operations
    env: Any  # gfnx.HypergridEnvironment instance
    env_params: Any  # Environment parameters (reward params, etc.)
    model: Any  # MLPPolicy instance
    logZ: Any  # Learnable log partition function estimate
    optimizer: Any  # optax optimizer (GradientTransformation)
    opt_state: Any  # Optimizer state (momentum, etc.)
    exploration_schedule: Any  # Epsilon schedule for exploration
    num_envs: int  # Batch size for parallel trajectory sampling


class IsingTrainState(NamedTuple):
    """Training state for Ising environment."""

    rng_key: Any
    env: Any  # gfnx.IsingEnvironment
    env_params: Any
    model: Any
    logZ: Any
    optimizer: Any
    opt_state: Any
    exploration_schedule: Any
    num_envs: int


class BitseqTrainState(NamedTuple):
    """Training state for BitSequence environment."""

    rng_key: Any
    env: Any  # gfnx.BitseqEnvironment
    env_params: Any
    model: Any
    logZ: Any
    optimizer: Any
    opt_state: Any
    exploration_schedule: Any
    num_envs: int


# ============================================================================
# MLP Policy (shared across environments)
# ============================================================================


class MLPPolicy:
    """MLP policy network for forward and backward actions.

    Outputs both forward and backward action logits from a shared network.
    The network has a single output head that's split into forward and
    backward parts, encouraging parameter sharing.

    In GFlowNets:
    - Forward logits: Used to sample actions when building trajectories
    - Backward logits: Used to compute P_B for the TB loss
    """

    def __init__(
        self,
        input_size: int,
        n_fwd_actions: int,
        n_bwd_actions: int,
        hidden_size: int,
        train_backward_policy: bool,
        depth: int,
        rng_key,
    ):
        import equinox as eqx

        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions

        # Output size includes both forward and backward logits
        output_size = n_fwd_actions
        if train_backward_policy:
            output_size += n_bwd_actions

        # Equinox MLP with specified depth and width
        self.network = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=hidden_size,
            depth=depth,
            key=rng_key,
        )

    def __call__(self, x):
        """Forward pass returning both forward and backward logits.

        Args:
            x: Observation tensor of shape (input_size,)

        Returns:
            Dict with 'forward_logits' and 'backward_logits' tensors
        """
        import jax.numpy as jnp

        x = self.network(x)
        if self.train_backward_policy:
            # Split output into forward and backward parts
            forward_logits, backward_logits = jnp.split(x, [self.n_fwd_actions], axis=-1)
        else:
            forward_logits = x
            backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
        }


# ============================================================================
# Train Step functions for each environment
# ============================================================================
# Each create_*_train_step() returns a JIT-compiled function that:
# 1. Samples a batch of trajectories using the forward policy
# 2. Computes the Trajectory Balance loss
# 3. Updates model parameters and logZ using gradients
#
# The train step functions are nearly identical across environments, but
# are kept separate because:
# - Different TrainState types (for type safety in the JIT)
# - Different environment-specific details in the future
# - Clearer debugging when issues arise


def create_hypergrid_train_step():
    """Create a JIT-compiled training step for Hypergrid.

    Returns a function: (iteration, train_state) -> train_state
    """
    import equinox as eqx
    import gfnx
    import jax
    import jax.numpy as jnp
    import optax

    @eqx.filter_jit
    def train_step(idx: int, train_state: HypergridTrainState) -> HypergridTrainState:
        """Single training step for Hypergrid environment.

        Steps:
        1. Sample trajectories using forward policy
        2. Compute TB loss: (log P_F + log Z) vs (log P_B + log R)
        3. Compute gradients and update parameters
        """
        rng_key = train_state.rng_key
        num_envs = train_state.num_envs
        env = train_state.env
        env_params = train_state.env_params

        # Split model into learnable params and static structure
        # This is needed for gradient computation in Equinox
        policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

        # Split RNG key for trajectory sampling
        rng_key, sample_traj_key = jax.random.split(rng_key)

        # Get current exploration epsilon (0.0 for benchmark)
        cur_eps = train_state.exploration_schedule(idx)

        # Forward policy function for trajectory rollout
        def fwd_policy_fn(rng_key, env_obs, policy_params):
            """Compute forward action logits for a batch of observations."""
            # Reconstruct the model from params and static parts
            current_model = eqx.combine(policy_params, policy_static)
            # vmap applies the model to each observation in the batch
            policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
            fwd_logits = policy_outputs["forward_logits"]

            # Apply epsilon-greedy exploration (uniform random with prob epsilon)
            rng_key, exploration_key = jax.random.split(rng_key)
            batch_size, _ = fwd_logits.shape
            exploration_mask = jax.random.bernoulli(
                exploration_key, cur_eps, (batch_size,)
            )
            fwd_logits = jnp.where(exploration_mask[..., None], 0, fwd_logits)
            return fwd_logits, policy_outputs

        # Sample complete trajectories from initial to terminal states
        traj_data, aux_info = gfnx.utils.forward_rollout(
            rng_key=sample_traj_key,
            num_envs=num_envs,
            policy_fn=fwd_policy_fn,
            policy_params=policy_params,
            env=env,
            env_params=env_params,
        )

        # Loss function for gradient computation
        def loss_fn(
            current_all_params,
            static_model_parts,
            current_traj_data,
            current_env,
            current_env_params,
        ):
            """Compute Trajectory Balance loss.

            TB Loss = E[(log P_F(τ) + log Z - log P_B(τ) - log R(x))^2]

            Where:
            - P_F(τ): Forward probability of trajectory τ
            - Z: Partition function estimate (learned)
            - P_B(τ): Backward probability of trajectory τ
            - R(x): Reward at terminal state x
            """
            model_learnable_params = current_all_params["model_params"]
            logZ_val = current_all_params["logZ"]

            # Reconstruct model and get policy outputs for entire trajectory
            model_to_call = eqx.combine(model_learnable_params, static_model_parts)
            # Double vmap: over batch and over time steps
            policy_outputs_traj = jax.vmap(jax.vmap(model_to_call))(
                current_traj_data.obs
            )

            # ========== Compute Forward Log Probabilities ==========
            fwd_logits_traj = policy_outputs_traj["forward_logits"]

            # Mask invalid actions (e.g., moving outside grid)
            invalid_fwd_mask = jax.vmap(
                current_env.get_invalid_mask, in_axes=(1, None), out_axes=1
            )(current_traj_data.state, current_env_params)
            masked_fwd_logits_traj = gfnx.utils.mask_logits(
                fwd_logits_traj, invalid_fwd_mask
            )

            # Convert to log probabilities
            fwd_all_log_probs_traj = jax.nn.log_softmax(masked_fwd_logits_traj, axis=-1)

            # Select log prob of the action that was actually taken
            fwd_logprobs_traj = jnp.take_along_axis(
                fwd_all_log_probs_traj,
                jnp.expand_dims(current_traj_data.action, axis=-1),
                axis=-1,
            ).squeeze(-1)

            # Zero out padded time steps (trajectories may have different lengths)
            fwd_logprobs_traj = jnp.where(current_traj_data.pad, 0.0, fwd_logprobs_traj)

            # Sum log probs along trajectory: log P_F(τ) = Σ log P_F(a_t|s_t)
            sum_log_pf_along_traj = fwd_logprobs_traj.sum(axis=1)
            log_pf_traj = logZ_val + sum_log_pf_along_traj

            # ========== Compute Backward Log Probabilities ==========
            # Get state transitions for computing backward actions
            prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)
            fwd_actions = current_traj_data.action[:, :-1]
            curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)

            # Get the backward action that would undo each forward action
            bwd_actions_traj = jax.vmap(
                current_env.get_backward_action,
                in_axes=(1, 1, 1, None),
                out_axes=1,
            )(prev_states, fwd_actions, curr_states, current_env_params)

            bwd_logits_traj = policy_outputs_traj["backward_logits"]
            # Shift by 1: backward logits at state s_t predict action to reach s_{t-1}
            bwd_logits_for_pb = bwd_logits_traj[:, 1:]

            # Mask invalid backward actions
            invalid_bwd_mask = jax.vmap(
                current_env.get_invalid_backward_mask,
                in_axes=(1, None),
                out_axes=1,
            )(curr_states, current_env_params)

            masked_bwd_logits_traj = gfnx.utils.mask_logits(
                bwd_logits_for_pb, invalid_bwd_mask
            )
            bwd_all_log_probs_traj = jax.nn.log_softmax(masked_bwd_logits_traj, axis=-1)

            # Select log prob of the backward action
            log_pb_selected = jnp.take_along_axis(
                bwd_all_log_probs_traj,
                jnp.expand_dims(bwd_actions_traj, axis=-1),
                axis=-1,
            ).squeeze(-1)

            # Zero out padded steps
            pad_mask_for_bwd = current_traj_data.pad[:, :-1]
            log_pb_selected = jnp.where(pad_mask_for_bwd, 0.0, log_pb_selected)

            # ========== Compute Target: log P_B(τ) + log R(x) ==========
            log_rewards_at_steps = current_traj_data.log_gfn_reward[:, :-1]
            masked_log_rewards_at_steps = jnp.where(
                pad_mask_for_bwd, 0.0, log_rewards_at_steps
            )

            log_pb_plus_rewards_along_traj = (
                log_pb_selected + masked_log_rewards_at_steps
            )
            target = jnp.sum(log_pb_plus_rewards_along_traj, axis=1)

            # ========== TB Loss: MSE between log P_F + log Z and log P_B + log R ==========
            loss = optax.losses.squared_error(log_pf_traj, target).mean()
            return loss

        # Compute loss and gradients
        params_for_loss = {"model_params": policy_params, "logZ": train_state.logZ}
        mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
            params_for_loss, policy_static, traj_data, env, env_params
        )

        # Apply optimizer updates
        optax_params_for_update = {
            "model_params": policy_params,
            "logZ": train_state.logZ,
        }
        updates, new_opt_state = train_state.optimizer.update(
            grads, train_state.opt_state, optax_params_for_update
        )

        # Apply updates to get new parameters
        new_model = eqx.apply_updates(train_state.model, updates["model_params"])
        new_logZ = eqx.apply_updates(train_state.logZ, updates["logZ"])

        # Return updated train state
        return train_state._replace(
            rng_key=rng_key,
            model=new_model,
            logZ=new_logZ,
            opt_state=new_opt_state,
        )

    return train_step


def create_ising_train_step():
    """Create a JIT-compiled training step for Ising environment.

    The training logic is identical to Hypergrid - only the environment
    and state types differ. See create_hypergrid_train_step for detailed comments.
    """
    import equinox as eqx
    import gfnx
    import jax
    import jax.numpy as jnp
    import optax

    @eqx.filter_jit
    def train_step(idx: int, train_state: IsingTrainState) -> IsingTrainState:
        rng_key = train_state.rng_key
        num_envs = train_state.num_envs
        env = train_state.env
        env_params = train_state.env_params

        policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

        rng_key, sample_traj_key = jax.random.split(rng_key)
        cur_eps = train_state.exploration_schedule(idx)

        def fwd_policy_fn(rng_key, env_obs, policy_params):
            current_model = eqx.combine(policy_params, policy_static)
            policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
            fwd_logits = policy_outputs["forward_logits"]

            rng_key, exploration_key = jax.random.split(rng_key)
            batch_size, _ = fwd_logits.shape
            exploration_mask = jax.random.bernoulli(
                exploration_key, cur_eps, (batch_size,)
            )
            fwd_logits = jnp.where(exploration_mask[..., None], 0, fwd_logits)
            return fwd_logits, policy_outputs

        traj_data, aux_info = gfnx.utils.forward_rollout(
            rng_key=sample_traj_key,
            num_envs=num_envs,
            policy_fn=fwd_policy_fn,
            policy_params=policy_params,
            env=env,
            env_params=env_params,
        )

        def loss_fn(
            current_all_params,
            static_model_parts,
            current_traj_data,
            current_env,
            current_env_params,
        ):
            model_learnable_params = current_all_params["model_params"]
            logZ_val = current_all_params["logZ"]

            model_to_call = eqx.combine(model_learnable_params, static_model_parts)
            policy_outputs_traj = jax.vmap(jax.vmap(model_to_call))(
                current_traj_data.obs
            )

            # Forward log probabilities
            fwd_logits_traj = policy_outputs_traj["forward_logits"]
            invalid_fwd_mask = jax.vmap(
                current_env.get_invalid_mask, in_axes=(1, None), out_axes=1
            )(current_traj_data.state, current_env_params)
            masked_fwd_logits_traj = gfnx.utils.mask_logits(
                fwd_logits_traj, invalid_fwd_mask
            )
            fwd_all_log_probs_traj = jax.nn.log_softmax(masked_fwd_logits_traj, axis=-1)
            fwd_logprobs_traj = jnp.take_along_axis(
                fwd_all_log_probs_traj,
                jnp.expand_dims(current_traj_data.action, axis=-1),
                axis=-1,
            ).squeeze(-1)
            fwd_logprobs_traj = jnp.where(current_traj_data.pad, 0.0, fwd_logprobs_traj)
            sum_log_pf_along_traj = fwd_logprobs_traj.sum(axis=1)
            log_pf_traj = logZ_val + sum_log_pf_along_traj

            # Backward log probabilities
            prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)
            fwd_actions = current_traj_data.action[:, :-1]
            curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)

            bwd_actions_traj = jax.vmap(
                current_env.get_backward_action,
                in_axes=(1, 1, 1, None),
                out_axes=1,
            )(prev_states, fwd_actions, curr_states, current_env_params)

            bwd_logits_traj = policy_outputs_traj["backward_logits"]
            bwd_logits_for_pb = bwd_logits_traj[:, 1:]
            invalid_bwd_mask = jax.vmap(
                current_env.get_invalid_backward_mask,
                in_axes=(1, None),
                out_axes=1,
            )(curr_states, current_env_params)

            masked_bwd_logits_traj = gfnx.utils.mask_logits(
                bwd_logits_for_pb, invalid_bwd_mask
            )
            bwd_all_log_probs_traj = jax.nn.log_softmax(masked_bwd_logits_traj, axis=-1)
            log_pb_selected = jnp.take_along_axis(
                bwd_all_log_probs_traj,
                jnp.expand_dims(bwd_actions_traj, axis=-1),
                axis=-1,
            ).squeeze(-1)

            pad_mask_for_bwd = current_traj_data.pad[:, :-1]
            log_pb_selected = jnp.where(pad_mask_for_bwd, 0.0, log_pb_selected)

            log_rewards_at_steps = current_traj_data.log_gfn_reward[:, :-1]
            masked_log_rewards_at_steps = jnp.where(
                pad_mask_for_bwd, 0.0, log_rewards_at_steps
            )

            log_pb_plus_rewards_along_traj = (
                log_pb_selected + masked_log_rewards_at_steps
            )
            target = jnp.sum(log_pb_plus_rewards_along_traj, axis=1)

            loss = optax.losses.squared_error(log_pf_traj, target).mean()
            return loss

        params_for_loss = {"model_params": policy_params, "logZ": train_state.logZ}
        mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
            params_for_loss, policy_static, traj_data, env, env_params
        )

        optax_params_for_update = {
            "model_params": policy_params,
            "logZ": train_state.logZ,
        }
        updates, new_opt_state = train_state.optimizer.update(
            grads, train_state.opt_state, optax_params_for_update
        )

        new_model = eqx.apply_updates(train_state.model, updates["model_params"])
        new_logZ = eqx.apply_updates(train_state.logZ, updates["logZ"])

        return train_state._replace(
            rng_key=rng_key,
            model=new_model,
            logZ=new_logZ,
            opt_state=new_opt_state,
        )

    return train_step


def create_bitseq_train_step():
    """Create a JIT-compiled training step for BitSequence environment.

    The training logic is identical to Hypergrid - only the environment
    and state types differ. See create_hypergrid_train_step for detailed comments.
    """
    import equinox as eqx
    import gfnx
    import jax
    import jax.numpy as jnp
    import optax

    @eqx.filter_jit
    def train_step(idx: int, train_state: BitseqTrainState) -> BitseqTrainState:
        rng_key = train_state.rng_key
        num_envs = train_state.num_envs
        env = train_state.env
        env_params = train_state.env_params

        policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

        rng_key, sample_traj_key = jax.random.split(rng_key)
        cur_eps = train_state.exploration_schedule(idx)

        def fwd_policy_fn(rng_key, env_obs, policy_params):
            current_model = eqx.combine(policy_params, policy_static)
            policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
            fwd_logits = policy_outputs["forward_logits"]

            rng_key, exploration_key = jax.random.split(rng_key)
            batch_size, _ = fwd_logits.shape
            exploration_mask = jax.random.bernoulli(
                exploration_key, cur_eps, (batch_size,)
            )
            fwd_logits = jnp.where(exploration_mask[..., None], 0, fwd_logits)
            return fwd_logits, policy_outputs

        traj_data, aux_info = gfnx.utils.forward_rollout(
            rng_key=sample_traj_key,
            num_envs=num_envs,
            policy_fn=fwd_policy_fn,
            policy_params=policy_params,
            env=env,
            env_params=env_params,
        )

        def loss_fn(
            current_all_params,
            static_model_parts,
            current_traj_data,
            current_env,
            current_env_params,
        ):
            model_learnable_params = current_all_params["model_params"]
            logZ_val = current_all_params["logZ"]

            model_to_call = eqx.combine(model_learnable_params, static_model_parts)
            policy_outputs_traj = jax.vmap(jax.vmap(model_to_call))(
                current_traj_data.obs
            )

            # Forward log probabilities
            fwd_logits_traj = policy_outputs_traj["forward_logits"]
            invalid_fwd_mask = jax.vmap(
                current_env.get_invalid_mask, in_axes=(1, None), out_axes=1
            )(current_traj_data.state, current_env_params)
            masked_fwd_logits_traj = gfnx.utils.mask_logits(
                fwd_logits_traj, invalid_fwd_mask
            )
            fwd_all_log_probs_traj = jax.nn.log_softmax(masked_fwd_logits_traj, axis=-1)
            fwd_logprobs_traj = jnp.take_along_axis(
                fwd_all_log_probs_traj,
                jnp.expand_dims(current_traj_data.action, axis=-1),
                axis=-1,
            ).squeeze(-1)
            fwd_logprobs_traj = jnp.where(current_traj_data.pad, 0.0, fwd_logprobs_traj)
            sum_log_pf_along_traj = fwd_logprobs_traj.sum(axis=1)
            log_pf_traj = logZ_val + sum_log_pf_along_traj

            # Backward log probabilities
            prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)
            fwd_actions = current_traj_data.action[:, :-1]
            curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)

            bwd_actions_traj = jax.vmap(
                current_env.get_backward_action,
                in_axes=(1, 1, 1, None),
                out_axes=1,
            )(prev_states, fwd_actions, curr_states, current_env_params)

            bwd_logits_traj = policy_outputs_traj["backward_logits"]
            bwd_logits_for_pb = bwd_logits_traj[:, 1:]
            invalid_bwd_mask = jax.vmap(
                current_env.get_invalid_backward_mask,
                in_axes=(1, None),
                out_axes=1,
            )(curr_states, current_env_params)

            masked_bwd_logits_traj = gfnx.utils.mask_logits(
                bwd_logits_for_pb, invalid_bwd_mask
            )
            bwd_all_log_probs_traj = jax.nn.log_softmax(masked_bwd_logits_traj, axis=-1)
            log_pb_selected = jnp.take_along_axis(
                bwd_all_log_probs_traj,
                jnp.expand_dims(bwd_actions_traj, axis=-1),
                axis=-1,
            ).squeeze(-1)

            pad_mask_for_bwd = current_traj_data.pad[:, :-1]
            log_pb_selected = jnp.where(pad_mask_for_bwd, 0.0, log_pb_selected)

            log_rewards_at_steps = current_traj_data.log_gfn_reward[:, :-1]
            masked_log_rewards_at_steps = jnp.where(
                pad_mask_for_bwd, 0.0, log_rewards_at_steps
            )

            log_pb_plus_rewards_along_traj = (
                log_pb_selected + masked_log_rewards_at_steps
            )
            target = jnp.sum(log_pb_plus_rewards_along_traj, axis=1)

            loss = optax.losses.squared_error(log_pf_traj, target).mean()
            return loss

        params_for_loss = {"model_params": policy_params, "logZ": train_state.logZ}
        mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
            params_for_loss, policy_static, traj_data, env, env_params
        )

        optax_params_for_update = {
            "model_params": policy_params,
            "logZ": train_state.logZ,
        }
        updates, new_opt_state = train_state.optimizer.update(
            grads, train_state.opt_state, optax_params_for_update
        )

        new_model = eqx.apply_updates(train_state.model, updates["model_params"])
        new_logZ = eqx.apply_updates(train_state.logZ, updates["logZ"])

        return train_state._replace(
            rng_key=rng_key,
            model=new_model,
            logZ=new_logZ,
            opt_state=new_opt_state,
        )

    return train_step
