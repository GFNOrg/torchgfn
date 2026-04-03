#!/bin/bash
# Wrapper that runs plc_wrapper_main inside an Apptainer/Singularity container.
# Accepts the same arguments as the native binary.
# The container shares the host filesystem, so Unix socket IPC works transparently.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF_PATH="${SCRIPT_DIR}/plc.sif"

if [ ! -f "${SIF_PATH}" ]; then
    echo "Error: ${SIF_PATH} not found. Run setup_plc.sh --container first." >&2
    exit 1
fi

APPTAINER_CMD="apptainer"
if ! command -v apptainer &>/dev/null; then
    if command -v singularity &>/dev/null; then
        APPTAINER_CMD="singularity"
    else
        echo "Error: apptainer or singularity not found." >&2
        exit 1
    fi
fi

exec ${APPTAINER_CMD} exec "${SIF_PATH}" plc_wrapper_main "$@"
