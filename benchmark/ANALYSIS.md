# GFlowNet Library Benchmark Analysis

## Overview

This document compares the per-iteration training performance of three GFlowNet libraries:

- **torchgfn** (this repository) — PyTorch-based, object-oriented API with `States`/`Actions` classes
- **gflownet** ([alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet)) — PyTorch-based, Hydra-configured, uses raw tensor batches
- **gfnx** ([jax-based](https://github.com/GFNOrg/gfnx)) — JAX/Equinox-based, JIT-compiled train steps

Environments benchmarked:
- **Hypergrid** (2D, height=8): Discrete grid navigation with corner rewards. Supported by all three libraries.
- **Box/CCube** (2D, delta=0.25): Continuous environment with Beta mixture policies. Supported by torchgfn and gflownet only.
- **Ising**: TODO

All benchmarks use **Trajectory Balance (TB) loss** with batch_size=16, hidden_dim=256, 2-layer MLPs, and lr=1e-3 / lr_logz=0.1.

## Configuration Parity

| Parameter | torchgfn | gflownet | gfnx |
|-----------|----------|----------|------|
| Batch size | 16 | 16 | 16 |
| Hidden dim | 256 | 256 | 256 |
| N layers | 2 | 2 | 2 |
| Parameters (hypergrid) | 71,430 | 70,915 | — |
| Parameters (box) | 74,528 | 74,784 | — |
| LR (policy) | 1e-3 | 1e-3 | 1e-3 |
| LR (logZ) | 0.1 | 0.1 | 0.1 |
| Grad clip | 1.0 | 1.0 | N/A (JAX) |
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
