# Off-Policy Training and Replay Buffers

By default, GFlowNet training is **on-policy**: trajectories are sampled from the current forward policy and used immediately to compute the loss. Off-policy training decouples sampling from learning, allowing you to reuse past experience via replay buffers.

## On-Policy vs Off-Policy

The key distinction is whether log-probabilities need to be recomputed at training time.

**On-policy** (no replay buffer, no exploration noise):
```python
trajectories = gflownet.sample_trajectories(env, n=batch_size, save_logprobs=True)
training_samples = gflownet.to_training_samples(trajectories)
loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=False)
```

Here, `save_logprobs=True` caches the log-probabilities computed during sampling, and `recalculate_all_logprobs=False` reuses them during loss computation. This avoids redundant forward passes.

**Off-policy** (replay buffer or exploration noise):
```python
trajectories = gflownet.sample_trajectories(env, n=batch_size, save_logprobs=False)
# ... store in buffer, retrieve later ...
training_samples = gflownet.to_training_samples(trajectories)
loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
```

Since the policy has changed since the trajectories were collected, the cached log-probabilities are stale and must be recalculated via `recalculate_all_logprobs=True`.

**When is training off-policy?** Any of these make it off-policy:
- Using a replay buffer
- Epsilon-greedy exploration (`epsilon > 0`)
- Temperature scaling (`temperature != 1.0`)
- Noisy layers in the policy network

**See:** `train_hypergrid_buffer.py` for a direct comparison of on-policy and off-policy losses.

## Replay Buffers

### ReplayBuffer

**Class:** `ReplayBuffer`

A standard experience replay buffer that stores `Trajectories` objects and samples uniformly.

```python
from gfn.containers import ReplayBuffer

replay_buffer = ReplayBuffer(capacity=1000)

# During training:
trajectories = gflownet.sample_trajectories(env, n=batch_size)
replay_buffer.add(trajectories)
buffer_trajectories = replay_buffer.sample(n=batch_size)
```

Use `prioritized_sampling=True` for reward-proportional sampling and `prioritized_capacity=True` to keep higher-reward trajectories when evicting.

**See:** `train_hypergrid_buffer.py`, `train_graph_ring.py`.

### TerminatingStateBuffer

**Class:** `TerminatingStateBuffer`

An alternative that stores only terminal states rather than full trajectories. At training time, backward trajectories are sampled from the stored terminal states and reversed to create forward training data.

```python
# Sample backward from terminal states, then reverse
backward_sampler = Sampler(gflownet.pb)
backward_trajectories = backward_sampler.sample_trajectories(
    env, states=terminal_states, is_backward=True
)
training_trajectories = backward_trajectories.reverse_backward_trajectories()
```

This is more memory-efficient than storing full trajectories, but requires a trained backward policy to reconstruct trajectories.

**See:** `train_hypergrid_buffer.py` (with `--buffer_type terminating_state`).

### NormBasedDiversePrioritizedReplayBuffer

**Class:** `NormBasedDiversePrioritizedReplayBuffer`

A diversity-aware replay buffer that scores stored trajectories using multiple components: retention priority, novelty (via pairwise distance), reward magnitude, and mode bonuses. Used in distributed training setups for improved mode coverage.

**See:** `train_hypergrid.py`, `train_hypergrid_ddp.py`.

## Buffer Prefilling

Filling the buffer before training begins can stabilize early training. This is done by sampling trajectories without computing gradients:

```python
with torch.no_grad():
    for _ in range(prefill_steps):
        trajectories = gflownet.sample_trajectories(env, n=batch_size)
        replay_buffer.add(trajectories)
```

**See:** `train_hypergrid_buffer.py` (prefill loop before the main training loop).

## Expert Data Warm-Starting

You can pre-fill a replay buffer with trajectories from known high-reward terminal states by sampling backward from them and reversing:

```python
backward_sampler = Sampler(gflownet.pb)
backward_trajectories = backward_sampler.sample_trajectories(
    env, states=expert_states, is_backward=True
)
forward_trajectories = backward_trajectories.reverse_backward_trajectories()
replay_buffer.add(forward_trajectories)
```

This technique helps with exploration in environments with many modes that are hard to discover from scratch. The `reverse_backward_trajectories()` method on `Trajectories` handles flipping the trajectory direction.

**See:** `train_with_example_modes.py` (pre-fills with half the known ring graph modes).

## Mixing On-Policy and Off-Policy Data

A common pattern is to mix freshly sampled on-policy trajectories with replayed off-policy data in each training step:

```python
# Fresh on-policy batch
online_trajectories = gflownet.sample_trajectories(env, n=batch_size // 2)
replay_buffer.add(online_trajectories)

# Off-policy batch from buffer
buffer_trajectories = replay_buffer.sample(n=batch_size // 2)

# Compute loss on both (must recalculate since buffer data is off-policy)
all_trajectories = online_trajectories.extend(buffer_trajectories)
loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
```

**See:** `train_with_example_modes.py` (50/50 online + buffer mixing).
