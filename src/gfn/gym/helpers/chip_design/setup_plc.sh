#!/bin/bash
# Downloads plc_wrapper_main and optionally builds an Apptainer container.
#
# Usage:
#   ./setup_plc.sh                  # Download native binary (Linux x86-64)
#   ./setup_plc.sh --container      # Also build Apptainer .sif image
#   ./setup_plc.sh --container-only # Only build Apptainer .sif (HPC)
#   ./setup_plc.sh --version 0.0.3  # Use a specific version
#
# The binary is downloaded from Google Cloud Storage:
#   https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CT_VERSION="0.0.4"
BUILD_CONTAINER=false
CONTAINER_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            CT_VERSION="$2"
            shift 2
            ;;
        --container)
            BUILD_CONTAINER=true
            shift
            ;;
        --container-only)
            BUILD_CONTAINER=true
            CONTAINER_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--version VERSION] [--container] [--container-only]"
            exit 1
            ;;
    esac
done

DOWNLOAD_URL="https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main_${CT_VERSION}"
BINARY_PATH="${SCRIPT_DIR}/plc_wrapper_main"

if [ "$CONTAINER_ONLY" = false ]; then
    echo "Downloading plc_wrapper_main version ${CT_VERSION}..."
    curl -fSL -o "${BINARY_PATH}" "${DOWNLOAD_URL}"
    chmod 555 "${BINARY_PATH}"
    echo "Binary saved to ${BINARY_PATH}"
fi

if [ "$BUILD_CONTAINER" = true ]; then
    if ! command -v apptainer &>/dev/null && ! command -v singularity &>/dev/null; then
        echo "Error: apptainer or singularity not found."
        echo "  Install: https://apptainer.org/docs/admin/main/installation.html"
        exit 1
    fi

    APPTAINER_CMD="apptainer"
    if ! command -v apptainer &>/dev/null; then
        APPTAINER_CMD="singularity"
    fi

    SIF_PATH="${SCRIPT_DIR}/plc.sif"
    DEF_PATH="${SCRIPT_DIR}/plc.def"

    if [ ! -f "${DEF_PATH}" ]; then
        echo "Error: ${DEF_PATH} not found."
        exit 1
    fi

    echo "Building Apptainer image (this may take a minute)..."
    ${APPTAINER_CMD} build "${SIF_PATH}" "${DEF_PATH}"
    echo "Container saved to ${SIF_PATH}"
fi

echo "Done. Run the tests to verify:"
echo "  python -c \"from gfn.gym.chip_design import ChipDesign; env = ChipDesign(); print('OK:', env.n_macros, 'macros')\""
