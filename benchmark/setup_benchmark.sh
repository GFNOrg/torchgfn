#!/bin/bash
# Setup script for the torchgfn benchmark suite.
# Installs all dependencies needed to run benchmark_libraries.py with all three
# libraries (torchgfn, gflownet, gfnx).
#
# Supports both Linux/CUDA and macOS/Metal platforms.
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
echo "=== 2/4  Detecting platform ==="

if [ "$(uname)" = "Darwin" ]; then
    echo "Detected macOS."
    echo ""
    echo "This will install JAX with Metal (Apple GPU) support."
    echo "If you intended to install the CUDA version, run this on a Linux machine instead."
    echo ""
    read -rp "Continue with macOS/Metal install? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    PLATFORM="mac"
else
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
    PLATFORM="cuda"
fi

echo ""
echo "=== 3/4  Installing JAX ecosystem (for gfnx) ==="

if [ "$PLATFORM" = "mac" ]; then
    pip install \
        jax==0.4.35 \
        jaxlib==0.4.35 \
        jax-metal==0.1.1 \
        equinox \
        optax \
        chex \
        flashbax \
        jax_tqdm \
        jaxtyping
else
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
fi

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

# Verify JAX can see an accelerator
if [ "$PLATFORM" = "mac" ]; then
    JAX_ACCEL=$(python -c "import jax; devs = jax.devices(); print('yes' if any(d.platform == 'metal' for d in devs) else 'no')" 2>/dev/null)
    if [ "$JAX_ACCEL" != "yes" ]; then
        echo ""
        echo "WARNING: JAX does not see a Metal GPU! gfnx benchmarks will run on CPU."
        echo "  JAX devices: $(python -c 'import jax; print(jax.devices())')"
    fi
else
    JAX_ACCEL=$(python -c "import jax; devs = jax.devices(); print('yes' if any(d.platform == 'gpu' for d in devs) else 'no')" 2>/dev/null)
    if [ "$JAX_ACCEL" != "yes" ]; then
        echo ""
        echo "WARNING: JAX does not see a GPU! gfnx benchmarks will run on CPU."
        echo "  Ensure CUDA drivers are accessible and try re-running this script."
        echo "  JAX devices: $(python -c 'import jax; print(jax.devices())')"
    fi
fi

echo ""
echo "Setup complete! See benchmark help:"
echo "  python benchmark/benchmark_libraries.py --scenario <SCENARIO> --seeds 0 1 2"
echo ""
