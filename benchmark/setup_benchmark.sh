#!/bin/bash
# Setup script for the torchgfn benchmark suite.
# Installs all dependencies needed to run benchmark_libraries.py with all three
# libraries (torchgfn, gflownet, gfnx) on a Linux/CUDA system.
#
# Usage:
#   cd /path/to/torchgfn
#   bash benchmark/setup_benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1/4  Initializing git submodules ==="
git submodule update --init --recursive

echo ""
echo "=== 2/4  Detecting CUDA version ==="
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || true)
if [ -z "$CUDA_VER" ]; then
    echo "ERROR: Could not detect CUDA version from PyTorch. Is PyTorch installed?"
    exit 1
fi
echo "Detected PyTorch CUDA: $CUDA_VER"

# Determine JAX CUDA extra based on major CUDA version
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
if [ "$CUDA_MAJOR" -ge 12 ]; then
    JAX_CUDA_EXTRA="cuda12"
elif [ "$CUDA_MAJOR" -ge 11 ]; then
    JAX_CUDA_EXTRA="cuda11_cudnn86"
else
    echo "ERROR: Unsupported CUDA version $CUDA_VER (need >= 11)"
    exit 1
fi
echo "Will install JAX with [$JAX_CUDA_EXTRA] support"

echo ""
echo "=== 3/4  Installing JAX ecosystem (for gfnx) ==="
# JAX >=0.5 uses plugin-based CUDA; the [cuda12] extra handles this
# automatically. Reinstall to ensure CUDA plugin is present even if
# jax was previously installed without it.
pip install --force-reinstall "jax[${JAX_CUDA_EXTRA}]>=0.4.27" \
    equinox \
    optax \
    chex \
    flashbax \
    jax_tqdm \
    jaxtyping

echo ""
echo "=== 4/4  Installing gflownet dependencies ==="
pip install \
    hydra-core \
    torchtyping \
    botorch \
    plotly

echo ""
echo "=== Verification ==="
echo -n "JAX version:     " && python -c "import jax; print(jax.__version__)"
echo -n "JAX devices:     " && python -c "import jax; print(jax.devices())"
echo -n "Equinox version: " && python -c "import equinox; print(equinox.__version__)"
echo -n "Optax version:   " && python -c "import optax; print(optax.__version__)"
echo -n "Hydra version:   " && python -c "import hydra; print(hydra.__version__)"
echo -n "gfnx importable: " && python -c "import sys; sys.path.insert(0, 'benchmark/gfnx/src'); import gfnx; print('yes')"
echo -n "gflownet importable: " && python -c "import sys; sys.path.insert(0, 'benchmark/gflownet'); from gflownet.utils.common import gflownet_from_config; print('yes')"

# Verify JAX can see a GPU
JAX_GPU=$(python -c "import jax; devs = jax.devices(); print('yes' if any(d.platform == 'gpu' for d in devs) else 'no')" 2>/dev/null)
if [ "$JAX_GPU" != "yes" ]; then
    echo ""
    echo "WARNING: JAX does not see a GPU! gfnx benchmarks will run on CPU."
    echo "  Ensure CUDA drivers are accessible and try re-running this script."
    echo "  JAX devices: $(python -c 'import jax; print(jax.devices())')"
fi

echo ""
echo "Setup complete! See benchmark help:"
echo "  python benchmark/benchmark_libraries.py --scenario <SCENARIO> --seeds 0 1 2"
echo ""
