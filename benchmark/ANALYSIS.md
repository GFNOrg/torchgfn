# GFlowNet Library Benchmark Analysis

## Overview

This document compares the per-iteration training performance of three GFlowNet libraries:

- **torchgfn** (this repository) — PyTorch-based, object-oriented API with `States`/`Actions` classes
- **gflownet** ([alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet)) — PyTorch-based, Hydra-configured, uses raw tensor batches
- **gfnx** ([jax-based](https://github.com/GFNOrg/gfnx)) — JAX/Equinox-based, JIT-compiled train steps

Environments benchmarked:
- **Hypergrid** (2D, height=8): Discrete grid navigation with corner rewards. Supported by all three libraries.
- **Box/CCube** (2D, delta=0.1): Continuous environment with Beta mixture policies. Supported by torchgfn and gflownet only.
- **Ising** (6x6, 10x10): Discrete Ising model with periodic BCs. Supported by all three libraries.
- **BitSequence** (non-autoregressive): Binary sequence generation with Hamming-distance reward. Supported by torchgfn and gfnx only.

All benchmarks use **Trajectory Balance (TB) loss** with hidden_dim=256, 2-layer MLPs, lr=1e-3 / lr_logz=0.1, and batch sizes 32, 256, and 2048.

## Configuration Parity

| Parameter | torchgfn | gflownet | gfnx |
|-----------|----------|----------|------|
| Batch size | 32/256/2048 | 32/256/2048 | 32/256/2048 |
| Hidden dim | 256 | 256 | 256 |
| N layers | 2 | 2 | 2 |
| Parameters (hypergrid) | 71,430 | 70,915 | — |
| Parameters (box) | 74,528 | 74,784 | — |
| LR (policy) | 1e-3 | 1e-3 | 1e-3 |
| LR (logZ) | 0.1 | 0.1 | 0.1 |
| Grad clip | None | None | None |
| Loss | TB | TB | TB |

### Known Differences

| Aspect | torchgfn | gflownet | gfnx |
|--------|----------|----------|------|
| **Backward policy (hypergrid)** | Learned (shared trunk) | Learned (shared trunk) | Learned (shared head) |
| **Backward policy (box)** | Configurable (uniform or learned) | Uniform (fixed) | N/A |
| **PF log-prob caching (box)** | Yes (`save_logprobs=True`) | No (recomputed in loss) | N/A |
| **PF log-prob recomputation (hypergrid)** | Yes (off-policy sampling) | Yes | Yes |
| **LR scheduler** | None | StepLR (every iter) | None |
| **Loss validation** | None | Checks `torch.isfinite` | None |
| **State representation** | `States` objects | Raw tensors | JAX arrays |
| **Input normalization (box)** | `2*x - 1` | `2*x - 1` | N/A |

### gfnx Rollout Overhead (JAX)

The gfnx library performs extra computation during trajectory rollout compared to
torchgfn and gflownet. This is a library design decision, not a benchmark
configuration issue, but it affects wall-clock comparisons and should be noted.

**Extra work per rollout step (inside `jax.lax.scan`)**:

1. **Entropy computation**: gfnx computes per-step entropy
   (`-sum(p * log p)` over valid actions) at every environment step during
   rollout. torchgfn and gflownet do not compute entropy during sampling.

2. **Redundant softmax**: gfnx computes both `softmax` and `log_softmax` on the
   policy logits at each step (for entropy and action sampling respectively).
   torchgfn computes only `log_softmax`.

3. **Double forward pass**: After rollout, gfnx recomputes the full policy
   network output over the entire trajectory in the loss function
   (`jax.vmap(jax.vmap(model))(traj_data.obs)`). This is the same approach
   torchgfn uses when `recalculate_all_logprobs=True`, but in gfnx the first
   forward pass (during rollout) also stores the outputs in `traj_data.info`,
   meaning the rollout-phase forward pass is wasted compute.

4. **Trajectory transpose**: After `jax.lax.scan` produces trajectories in
   `(T, B, ...)` layout, gfnx transposes the entire pytree to `(B, T, ...)`
   via `jax.tree.map(lambda x: jnp.transpose(x, ...))`.

**Impact on batch size scaling**:

The `jax.lax.scan` loop has fixed per-step overhead regardless of batch size.
For small batches (e.g., 32), this overhead dominates iteration time and makes
gfnx appear slower than PyTorch-based libraries. At larger batch sizes (256+),
the per-step compute grows and amortizes the scan overhead, allowing JAX's
whole-program JIT compilation to potentially close or reverse the gap.

**Quantifying the overhead**:

For HyperGrid (2D, height=8), `max_steps_in_episode = dim * side = 16`, so
the scan runs 17 steps (16 + 1 padding). Each step involves a full policy
forward pass, mask computation, softmax, log_softmax, entropy, action sampling,
and environment step — all of which are executed sequentially within the scan.
By contrast, torchgfn's eager-mode sampling loop has lower fixed overhead per
step but cannot benefit from cross-step fusion.

## Results

*5000 iterations, 3 seeds, CPU (macOS). Parameter counts match between libraries.*

### Hypergrid (2D, height=8)

| Library | Params | Iter Time (ms) | sample (ms) | loss (ms) | backward (ms) | optimizer (ms) |
|---------|--------|---------------|-------------|-----------|---------------|----------------|
| torchgfn | 71,430 | 9.81 ± 1.10 | 6.72 (69%) | 1.27 (13%) | 1.02 (10%) | 0.76 (8%) |
| gflownet | 70,915 | 12.77 ± 2.57 | 7.13 (57%) | 1.58 (13%) | 3.06 (24%) | 0.75 (6%) |

**torchgfn is 1.3x faster on hypergrid.**

### Box 2D (uniform backward policy)

| Library | Params | Iter Time (ms) | sample (ms) | loss (ms) | backward (ms) | optimizer (ms) |
|---------|--------|---------------|-------------|-----------|---------------|----------------|
| torchgfn | 74,528 | 5.17 ± 0.75 | 2.51 (49%) | 0.73 (14%) | 1.18 (23%) | 0.70 (14%) |
| gflownet | 74,784 | 4.59 ± 0.56 | 2.12 (47%) | 1.02 (23%) | 0.64 (14%) | 0.76 (17%) |

**gflownet is 1.13x faster on box (uniform PB).**

### Box 2D (learned backward policy)

*Not re-run with updated MLP. Previous results (with architecture mismatch):*

| Library | Iter Time (ms) | sample (ms) | loss (ms) | backward (ms) | optimizer (ms) |
|---------|---------------|-------------|-----------|---------------|----------------|
| torchgfn | 6.94 ± 2.40 | 2.80 (40%) | 0.96 (14%) | 2.01 (29%) | 1.16 (17%) |
| gflownet | 4.39 ± 0.37 | 2.14 (49%) | 0.98 (22%) | 0.62 (14%) | 0.66 (15%) |

## Analysis

### Hypergrid: torchgfn is faster (1.3x)

torchgfn is **1.3x faster** on hypergrid (9.81 vs 12.77ms). With matched architectures (~71K params), the phase breakdown shows:

1. **Sampling**: Nearly equal (6.72ms vs 7.13ms). gflownet's batch construction overhead (creates empty batch, samples sub-batches, merges) slightly outweighs torchgfn's `States`/`Actions` dispatch overhead.

2. **Backward pass**: torchgfn is **3x faster** (1.02ms vs 3.06ms). gflownet's backward pass is slower likely due to:
   - Additional computational graph complexity from loss validation (`torch.isfinite` checks)
   - LR scheduler step inside the timed loop
   - Batch merging creating more complex computation graphs

3. **Optimizer**: Nearly identical (0.76ms vs 0.75ms), confirming architecture parity.

4. **Loss**: Nearly identical (1.27ms vs 1.58ms). Both recompute all log-probs.

### Box: gflownet is faster (1.13x)

With matched architectures, the gap narrowed from 1.24x to **1.13x**. The remaining difference:

1. **Sampling overhead** (+0.39ms): torchgfn's `States`/`Actions` object construction per step. With only ~3 steps per box trajectory (vs ~14 for hypergrid), this fixed overhead is proportionally larger.

2. **Backward pass** (+0.54ms): torchgfn uses `save_logprobs=True` to cache forward policy log-probs during sampling. This makes the loss phase faster (0.73 vs 1.02ms, saving 0.29ms) but retains the sampling computation graph, making backward more expensive. Net cost: +0.25ms.

3. **Loss phase** (-0.29ms): torchgfn wins here because cached log-probs avoid recomputation.

4. **Optimizer** (-0.06ms): Now nearly equal with matched architectures. Previously torchgfn was 1.6x slower due to the MLP off-by-one bug creating ~140K params vs ~75K.

### MLP n_hidden_layers fix

During benchmarking, we discovered an off-by-one bug in torchgfn's `MLP` class: `n_hidden_layers=2` created 4 Linear layers (~140K params) instead of the expected 3 (~75K params). This was fixed so that `n_hidden_layers` counts total hidden layers including the input projection, matching the standard convention used by gflownet:

| n_hidden_layers | Before fix | After fix |
|-----------------|-----------|-----------|
| 1 | 3 Linears | 2 Linears (input→H→output) |
| 2 | 4 Linears | 3 Linears (input→H→H→output) |
| 3 | 5 Linears | 4 Linears |

This fix improved the box benchmark by eliminating the optimizer time gap (1.06ms → 0.70ms) and reducing backward time.

### Cross-environment scaling

The performance gap between libraries varies with trajectory length:

```
Short trajectories (box, ~3 steps):
  Fixed overhead dominates → gflownet's raw-tensor approach wins (1.13x)

Long trajectories (hypergrid, ~14 steps):
  Per-step cost dominates → torchgfn's simpler backward path wins (1.3x)
```

torchgfn **scales better with trajectory length** but has higher fixed overhead per iteration. The crossover point is between 3 and 14 steps.

### gflownet batch size scaling issue

The gflownet library exhibits poor batch size scaling. On HyperGrid (2D,
height=8), iteration times scale super-linearly with batch size:

| Batch Size | torchgfn (ms) | gflownet (ms) | gfnx (ms) |
|------------|---------------|----------------|-----------|
| 32         | 31.6          | 51.5           | TBD       |
| 256        | 41.2          | 300.3          | TBD       |
| 2048       | 105.7         | 2351.4         | TBD       |

torchgfn scales ~3.3x for a 64x batch increase. gflownet scales **45.7x** for
the same increase. This is inherent to the library's design, not a benchmark
configuration issue.

**Root cause: multiple O(batch × action_space) and O(batch) Python-level operations**

The following bottlenecks were identified in the gflownet library source code
(all paths relative to `benchmark/gflownet/`):

#### 1. `actions2indices` — O(batch × action_space) tensor expansion

**File:** `gflownet/envs/base.py`, lines 191–206

```python
def actions2indices(self, actions):
    action_space = torch.unsqueeze(self.action_space_torch, 0).expand(
        actions.shape[0], -1, -1
    )
    actions = torch.unsqueeze(actions, 1).expand(-1, self.action_space_dim, -1)
    return torch.where(torch.all(actions == action_space, dim=2))[1]
```

This compares every action in the batch against every entry in the action space
by expanding both tensors to shape `[batch_size, action_space_dim, action_dim]`
and doing an element-wise comparison. For batch_size=2048 and
action_space_dim=33, this creates and compares tensors of shape
`[2048, 33, 2]`. The `torch.where` call then searches the full product for
matches.

A vectorized replacement would use a hash map or `torch.searchsorted` to look up
action indices in O(batch) time instead of O(batch × action_space).

**Called from:** `get_logprobs()` at `gflownet/envs/base.py:684`, which is
called during every training step to compute the log-probability of sampled
actions.

#### 2. `get_logprobs` — per-element dictionary lookup fallback

**File:** `gflownet/envs/base.py`, lines 680–695

When actions are passed as a Python list (which happens from `sample_batch`),
the code falls back to per-element dictionary lookups:

```python
action_indices = tlong(
    [self.action2index(action) for action in actions],
    device=self.device,
)
```

This is O(batch_size) in Python with per-element tuple conversion and dictionary
lookup overhead. At batch_size=2048, this list comprehension dominates.

#### 3. `states2policy` — dense one-hot tensor allocation

**File:** `gflownet/envs/grid.py`, lines 153–187

```python
states_policy = torch.zeros(
    (n_states, self.length * self.n_dim), dtype=self.float, device=self.device
)
states_policy[rows, cols.flatten()] = 1.0
```

Creates a dense `(batch_size, length × n_dim)` tensor of zeros and then sets
scattered indices to 1.0. For batch_size=2048 with a 4D grid of height=32, this
allocates `2048 × 128 = 262,144` floats per call. This is called from
`states2proxy` (line 147) during reward computation.

#### 4. Per-sample equality checks in batch construction

**File:** `gflownet/envs/base.py`, lines 800 and 891

```python
if any([self.equal(state, s) for s in states]):
    add = False
```

When building batches with uniqueness constraints, the code iterates over all
previously collected states and does per-element equality checks. This is
O(batch_size²) in the worst case when checking each new sample against all
previously accepted samples.

#### 5. Python-level batch iteration in `sample_batch`

**File:** `gflownet/gflownet.py`, lines 565–750

The trajectory sampling loop operates at the Python level with per-sample
processing, list appends, and repeated tensor-to-list conversions. This
prevents PyTorch from batching GPU operations efficiently and adds per-sample
Python interpreter overhead that scales linearly with batch size.

**Conclusion:** These patterns are fundamental to gflownet's architecture
(Python-level sample iteration, expand-and-compare action indexing, per-element
lookups). They would require significant refactoring to fix and are not
introduced by the benchmark. The benchmark accurately reflects the library's
real-world performance at different batch sizes.

## Optimizations Applied

The following optimizations were applied to torchgfn during this benchmarking effort:

### Distribution-level (box only)
1. **Replaced `Bernoulli` with `F.logsigmoid`**: Avoids creating a distribution object for exit probability log-prob computation.
2. **Replaced `MixtureSameFamily`** with inline `logsumexp + Beta`: Avoids the overhead of PyTorch's mixture distribution wrapper.
3. **Pre-computed `at_boundary`** mask: Avoids redundant comparison in hot loop.

### Framework-level (all environments)
4. **`States._make_view()`**: Lightweight constructor for internal slicing that bypasses `__init__` validation (device resolution, `.to()` dispatch, conditions checking).
5. **Lazy `is_sink_state` / `is_initial_state` caching**: These boolean masks are computed once and cached, with invalidation on `__setitem__`.
6. **Pre-allocated dummy `Actions` tensor**: Uses `.clone()` instead of `.repeat()` each step.
7. **Direct `_conditions` assignment**: Bypasses the conditions setter's shape/dtype validation for internal operations.

### Bug fix
8. **MLP `n_hidden_layers` off-by-one**: Fixed so `n_hidden_layers=2` creates 3 Linear layers (was 4). This is a library-wide fix affecting all environments.

### Impact
- Box (uniform PB): 8.2ms → 5.17ms (37% reduction, from 2.1x to 1.13x gap vs gflownet)
- Hypergrid: 10.98ms → 9.81ms (11% reduction, 1.3x faster than gflownet)

## Reproducing

Run benchmarks with per-phase timing:

```bash
# Hypergrid (all libraries)
python -m benchmark.benchmark_libraries --scenario tb_hypergrid_small --seeds 0 1 2 --n-iterations 5000

# Box with uniform backward policy (torchgfn vs gflownet)
python -m benchmark.benchmark_libraries --scenario tb_box_2d_uniform_pb --seeds 0 1 2 --n-iterations 5000

# Box with learned backward policy
python -m benchmark.benchmark_libraries --scenario tb_box_2d --seeds 0 1 2 --n-iterations 5000
```

Results are saved as JSON in `benchmark/outputs/` with full per-phase timing data.
