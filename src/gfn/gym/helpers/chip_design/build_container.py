"""Build the Singularity/Apptainer container for plc_wrapper_main.

Usage:
    python -m gfn.gym.helpers.chip_design.build_container

On systems with fakeroot support:
    python -m gfn.gym.helpers.chip_design.build_container --from-def

This creates ``plc_wrapper.sif`` next to this file.
"""

import argparse
import os
import shutil
import subprocess
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEF_FILE = os.path.join(_DIR, "plc_wrapper.def")
_SIF_FILE = os.path.join(_DIR, "plc_wrapper.sif")
_DOCKER_URI = "docker://ubuntu:20.04"


def _find_singularity() -> str:
    for name in ("singularity", "apptainer"):
        path = shutil.which(name)
        if path is not None:
            return path
    print(
        "ERROR: Neither 'singularity' nor 'apptainer' found on PATH.\n"
        "On HPC clusters try: module load singularity",
        file=sys.stderr,
    )
    sys.exit(1)


def build(from_def: bool = False) -> None:
    singularity = _find_singularity()

    if os.path.isfile(_SIF_FILE):
        print(f"Removing existing image: {_SIF_FILE}")
        os.remove(_SIF_FILE)

    if from_def:
        cmd = [singularity, "build", "--fakeroot", _SIF_FILE, _DEF_FILE]
    else:
        cmd = [singularity, "pull", _SIF_FILE, _DOCKER_URI]

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"Image built: {_SIF_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-def",
        action="store_true",
        help="Build from plc_wrapper.def (requires fakeroot). "
        "Default: pull docker://ubuntu:20.04.",
    )
    args = parser.parse_args()
    build(from_def=args.from_def)
