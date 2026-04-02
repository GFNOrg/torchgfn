# Exploration Strategies

GFlowNets must discover diverse modes in the reward landscape. Without exploration, the policy can collapse to a small number of high-reward modes while missing others. `torchgfn` provides several complementary exploration mechanisms.

## Epsilon-Greedy

Add uniform random actions with probability `epsilon` during sampling:

```python
trajectories = gflownet.sample_trajectories(env, n=batch_size, epsilon=0.1)
```

This mixes the learned policy with a uniform distribution. Higher epsilon means more exploration but noisier training signal.

For graph environments, epsilon can be specified per action component using a dictionary:

```python
from collections import defaultdict
epsilon = defaultdict(float)
epsilon[GraphActions.ACTION_TYPE_KEY] = 0.1
epsilon[GraphActions.EDGE_INDEX_KEY] = 0.05

trajectories = gflownet.sample_trajectories(env, n=batch_size, epsilon=epsilon)
```

**Note:** Epsilon-greedy makes training off-policy — you must set `recalculate_all_logprobs=True` when computing the loss.

**See:** `train_hypergrid_simple.py` (`--epsilon`), `train_with_example_modes.py` (per-component epsilon).

## Temperature Scaling

Soften or sharpen the policy distribution by scaling logits before sampling:

```python
trajectories = gflownet.sample_trajectories(env, n=batch_size, temperature=2.0)
```

- `temperature > 1`: Flatter distribution, more exploration
- `temperature < 1`: Sharper distribution, more exploitation
- `temperature = 1`: No change (on-policy)

Like epsilon-greedy, temperature scaling makes training off-policy.

**See:** `train_hypergrid_exploration_examples.py` (systematic comparison of temperature values).

## Temperature Annealing

For continuous environments, a common pattern is to start with high temperature (exploration) and decay to 1.0 (exploitation) over training:

```python
for iteration in range(n_iterations):
    progress = iteration / (n_iterations // 2)  # Anneal over first half
    temperature = max(1.0, initial_temp * (1 - progress))
    trajectories = gflownet.sample_trajectories(env, n=batch_size, temperature=temperature)
```

**See:** `train_box.py` (`BoxCartesianPFEstimator.temperature` attribute, linearly decayed).

## Exploration Variance (Continuous Environments)

For continuous action spaces, inject additional variance into the sampling distribution via `scale_factor` or `exploration_std`:

```python
trajectories = gflownet.sample_trajectories(env, n=batch_size, scale_factor=1.5)
```

This widens the action distribution without changing its center. A schedule that decays from high to zero works well:

```python
scale_schedule = torch.linspace(2.0, 0.0, n_iterations)
```

**See:** `train_line.py` (decaying `scale_factor` schedule), `train_diffusion_rtb.py` (`exploration_std` warm-down).

## Noisy Layers

Add learnable noise to network weights for state-dependent exploration:

```python
from gfn.utils.modules import MLP
module = MLP(input_dim, output_dim, n_noisy_layers=2)
```

Noisy layers inject parametric noise into the final layers of the policy network. Unlike epsilon-greedy (which is state-independent), noisy layers enable the network to learn where to explore.

**See:** `train_hypergrid_exploration_examples.py` (comparison with other strategies).

## Local Search Sampling

The `LocalSearchSampler` implements the back-and-forth heuristic: from a terminal state, sample backward K steps to a junction state, then sample forward to a new terminal state. This refines existing trajectories rather than generating from scratch.

```python
from gfn.samplers import LocalSearchSampler

sampler = LocalSearchSampler(pf_estimator, pb_estimator)
trajectories = sampler.sample_trajectories(
    env,
    n=batch_size,
    n_local_search_loops=2,
    back_ratio=0.5,
    use_metropolis_hastings=True,
)
```

- `n_local_search_loops`: Number of refinement rounds per trajectory
- `back_ratio`: Fraction of trajectory length to walk backward (controls search depth)
- `use_metropolis_hastings`: Accept/reject refined trajectories based on a probabilistic criterion

**See:** `train_hypergrid_local_search.py`, `train_box.py` (with `--sampler local_search`).

## Replay Buffers as Exploration

Replay buffers provide implicit exploration by reusing diverse past experience. See the [Off-Policy Training guide](off_policy_training.md) for details on `ReplayBuffer`, `TerminatingStateBuffer`, and expert data warm-starting.

## Comparing Strategies

`train_hypergrid_exploration_examples.py` provides a systematic comparison framework that runs 9 configurations (on-policy, replay buffer, epsilon variants, noisy layers, temperature, and combinations) across multiple seeds and plots mode discovery, L1 distance, and logZ error. This is the best starting point for understanding which strategies work for your problem.

## Summary

| Strategy | Off-policy? | State-dependent? | Best for |
|----------|------------|-------------------|----------|
| Epsilon-greedy | Yes | No | Simple baseline, discrete environments |
| Temperature | Yes | No | Softening/sharpening action selection |
| Noisy layers | Yes | Yes | Learnable, adaptive exploration |
| Scale factor | Yes | No | Continuous action spaces |
| Local search | No (refines) | Yes | Improving existing trajectories |
| Replay buffer | Yes | N/A | Reusing diverse past experience |
