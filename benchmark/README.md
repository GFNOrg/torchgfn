# GFlowNet Library Benchmarks

This benchmark compares three GFlowNet libraries across multiple environments:
- **torchgfn** - PyTorch-based (this repository)
- **gflownet** - PyTorch-based, Hydra configuration system
- **gfnx** - JAX/Equinox-based

## Setup

Initialize the benchmark dependencies:

```bash
git submodule update --init --recursive
```

## Environment Support Matrix

Not all libraries support all environments:

| Environment | torchgfn | gflownet | gfnx | Description |
|-------------|:--------:|:--------:|:----:|-------------|
| hypergrid   | ✓        | ✓        | ✓    | Discrete grid navigation |
| ising       | ✓        | ✓        | ✓    | Discrete Ising model |
| box         | ✓        | ✓        | -    | Continuous 2D box |
| bitseq      | ✓        | -        | ✓    | Bit sequence generation |

## Available Scenarios

### Hypergrid (all libraries)
- `tb_hypergrid_small` - 2D grid, height 8, 1000 iterations (quick test)
- `tb_hypergrid_medium` - 4D grid, height 16, 2000 iterations
- `tb_hypergrid_large` - 4D grid, height 32, 5000 iterations

### Ising (all libraries)
- `tb_ising_6x6` - 6x6 lattice (36 spins), 1000 iterations
- `tb_ising_10x10` - 10x10 lattice (100 spins), 2000 iterations

### Box/CCube (torchgfn, gflownet)
- `tb_box_2d` - 2D continuous box, delta=0.25, 1000 iterations

### BitSequence (torchgfn, gfnx)
- `tb_bitseq_small` - word_size=1, seq_size=4, 2 modes, 1000 iterations
- `tb_bitseq_medium` - word_size=2, seq_size=8, 4 modes, 2000 iterations

## Usage

### Quick Test

```bash
# Test with just torchgfn first (least dependencies)
python benchmark/benchmark_libraries.py --scenario tb_hypergrid_small --libraries torchgfn --seeds 0
```

### Running Benchmarks

**Recommended approach (run each library separately to avoid OpenMP conflicts):**

```bash
# Hypergrid - all libraries
python benchmark/benchmark_libraries.py --scenario tb_hypergrid_small --libraries torchgfn --seeds 0 1 2
python benchmark/benchmark_libraries.py --scenario tb_hypergrid_small --libraries gflownet --seeds 0 1 2
python benchmark/benchmark_libraries.py --scenario tb_hypergrid_small --libraries gfnx --seeds 0 1 2

# Ising - all libraries
python benchmark/benchmark_libraries.py --scenario tb_ising_6x6 --libraries torchgfn --seeds 0 1 2
python benchmark/benchmark_libraries.py --scenario tb_ising_6x6 --libraries gflownet --seeds 0 1 2
python benchmark/benchmark_libraries.py --scenario tb_ising_6x6 --libraries gfnx --seeds 0 1 2

# Box - torchgfn and gflownet only
python benchmark/benchmark_libraries.py --scenario tb_box_2d --libraries torchgfn --seeds 0 1 2
python benchmark/benchmark_libraries.py --scenario tb_box_2d --libraries gflownet --seeds 0 1 2

# BitSequence - torchgfn and gfnx only
python benchmark/benchmark_libraries.py --scenario tb_bitseq_small --libraries torchgfn --seeds 0 1 2
python benchmark/benchmark_libraries.py --scenario tb_bitseq_small --libraries gfnx --seeds 0 1 2
```

**Run all supported libraries for an environment (automatic filtering):**

```bash
# The script automatically selects supported libraries if --libraries is omitted
python benchmark/benchmark_libraries.py --scenario tb_box_2d --seeds 0 1 2
# Will run torchgfn and gflownet only (gfnx doesn't support box)
```

Results are saved with library names in the filename (e.g., `benchmark_tb_ising_6x6_torchgfn_20231218_143052.json`).

## Important Implementation Differences

### Ising Environment

| Library | Environment Class | Loss Type | Notes |
|---------|-------------------|-----------|-------|
| torchgfn | `DiscreteEBM` + `IsingModel` | Flow Matching | Uses coupling matrix J with periodic boundary conditions |
| gflownet | `ising` env | Trajectory Balance | Configured via Hydra with uniform proxy |
| gfnx | `IsingEnvironment` | Trajectory Balance | Uses `IsingRewardModule` |

### Box/CCube Environment

| Library | Environment Class | Policy Type | Notes |
|---------|-------------------|-------------|-------|
| torchgfn | `Box` | `BoxPFEstimator`/`BoxPBEstimator` | Continuous Beta mixture policies |
| gflownet | `ccube` | MLP with continuous actions | Configured via Hydra with corners proxy |

### BitSequence Environment

| Library | Environment Class | Loss Type | Notes |
|---------|-------------------|-----------|-------|
| torchgfn | `BitSequence` | Trajectory Balance | Uses shared trunk for PF/PB |
| gfnx | `BitseqEnvironment` | Trajectory Balance | Uses `BitseqRewardModule` with configurable modes |

### General Differences

- **torchgfn**: Pure PyTorch, imperative style, flexible API
- **gflownet**: PyTorch with Hydra configuration, more complex setup but highly configurable
- **gfnx**: JAX/Equinox, functional style, JIT compilation for performance

## macOS OpenMP Conflict

On macOS, conda/pip environments often have multiple copies of `libomp.dylib` from different sources (e.g., `llvm-openmp` from conda, plus bundled copies in PyTorch, scikit-learn, etc.). This causes an error:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
...
Abort trap: 6
```

**Cause:** Mixed conda/pip installations result in multiple OpenMP runtime libraries. Common sources include:
- `llvm-openmp` or `libopenblas` from conda
- PyTorch's bundled `libomp.dylib`
- scikit-learn's bundled `libomp.dylib`

**Solution:** The benchmark script automatically sets `KMP_DUPLICATE_LIB_OK=TRUE` at startup to work around this conflict.

**Note on benchmark accuracy:** While `KMP_DUPLICATE_LIB_OK=TRUE` can theoretically cause issues, in practice:
- All libraries use the same workaround, ensuring fair comparison
- The conflict is between *identical* OpenMP implementations (just different copies)
- For relative performance comparisons between libraries, the results remain valid

**Alternative (clean environment):** For maximum confidence, create a fresh conda environment using only pip packages:

```bash
conda create -n benchmark python=3.10
conda activate benchmark
pip install torch torchgfn scikit-learn jax jaxlib equinox  # all from pip
```

## Output Format

Results are saved as JSON files in `benchmark/outputs/` with the following structure:

```json
{
  "scenario": "tb_hypergrid_small",
  "timestamp": "20231218_143052",
  "config": {
    "env_name": "hypergrid",
    "env_kwargs": {"ndim": 2, "height": 8},
    "n_iterations": 1000,
    "batch_size": 16,
    ...
  },
  "results": [...],
  "summary": {
    "torchgfn": {
      "n_runs": 3,
      "mean_iter_time_ms": 5.23,
      "std_iter_time_ms": 0.42,
      "mean_throughput_iters_per_sec": 191.2,
      "mean_peak_memory_mb": 512.3
    }
  }
}
```
