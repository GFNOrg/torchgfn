# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PlacementCost client class."""

import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
from typing import Any, Text

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(__file__)


def _resolve_plc_binary() -> str:
    """Resolves the plc_wrapper_main binary location.

    Resolution order:
      1. PLC_WRAPPER_MAIN environment variable
      2. Native binary in package directory
      3. Apptainer wrapper script in package directory
    """
    # 1. Explicit env var
    env_path = os.environ.get("PLC_WRAPPER_MAIN")
    if env_path:
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path
        raise FileNotFoundError(
            f"PLC_WRAPPER_MAIN={env_path} is not an executable file."
        )

    # 2. Native binary in package dir
    native = os.path.join(_PKG_DIR, "plc_wrapper_main")
    if os.path.isfile(native) and os.access(native, os.X_OK):
        return native

    # 3. Apptainer wrapper script + .sif image
    wrapper = os.path.join(_PKG_DIR, "run_plc_apptainer.sh")
    sif = os.path.join(_PKG_DIR, "plc.sif")
    if os.path.isfile(wrapper) and os.access(wrapper, os.X_OK) and os.path.isfile(sif):
        # Verify apptainer/singularity is available
        if shutil.which("apptainer") or shutil.which("singularity"):
            return wrapper

    raise FileNotFoundError(
        "plc_wrapper_main not found. Requires Linux x86-64. Set up with one of:\n"
        "  Native:    cd src/gfn/gym/helpers/chip_design && ./setup_plc.sh\n"
        "  HPC:       cd src/gfn/gym/helpers/chip_design && ./setup_plc.sh --container-only\n"
        "  Explicit:  export PLC_WRAPPER_MAIN=/path/to/plc_wrapper_main"
    )


class PlacementCost:
    """PlacementCost object wrapper."""

    BUFFER_LEN = 1024 * 1024
    MAX_RETRY = 256

    def __init__(
        self,
        netlist_file: Text,
        plc_wrapper_main: str = "",
        macro_macro_x_spacing: float = 0.0,
        macro_macro_y_spacing: float = 0.0,
    ) -> None:
        """Creates a PlacementCost client object.

        It creates a subprocess by calling plc_wrapper_main and communicate with
        it over an `AF_UNIX` channel.

        Args:
          netlist_file: Path to the netlist proto text file.
          plc_wrapper_main: Path to the plc_wrapper_main binary or wrapper script.
              If empty, auto-detects via _resolve_plc_binary().
          macro_macro_x_spacing: Macro-to-macro x spacing in microns.
          macro_macro_y_spacing: Macro-to-macro y spacing in microns.
        """
        if not plc_wrapper_main:
            plc_wrapper_main = _resolve_plc_binary()

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = tempfile.NamedTemporaryFile().name
        self.sock.bind(address)
        self.sock.listen(1)
        args = [
            plc_wrapper_main,
            "--uid=0",
            "--gid=0",
            f"--pipe_address={address}",
            f"--netlist_file={netlist_file}",
            f"--macro_macro_x_spacing={macro_macro_x_spacing}",
            f"--macro_macro_y_spacing={macro_macro_y_spacing}",
        ]
        self.process = subprocess.Popen([str(a) for a in args])
        self.conn, _ = self.sock.accept()

    # See circuit_training/environment/plc_client_test.py for the supported APIs.
    def __getattr__(self, name) -> Any:
        # snake_case to PascalCase.
        name = name.replace("_", " ").title().replace(" ", "")

        def f(*args) -> Any:
            json_args = json.dumps({"name": name, "args": args})
            self.conn.send(json_args.encode("utf-8"))
            json_ret = b""
            retry = 0
            # The stream from the unix socket can be incomplete after a single call
            # to `recv` for large (200kb+) return values, e.g. GetMacroAdjacency. The
            # loop retries until the returned value is valid json. When the host is
            # under load ~10 retries have been needed. Adding a sleep did not seem to
            # make a difference only added latency. b/210838186
            while True:
                part = self.conn.recv(PlacementCost.BUFFER_LEN)
                json_ret += part
                if len(part) < PlacementCost.BUFFER_LEN:
                    json_str = json_ret.decode("utf-8")
                    try:
                        output = json.loads(json_str)
                        break
                    except json.decoder.JSONDecodeError as e:
                        logger.warning("JSONDecode Error for %s \n %s", name, e)
                        if retry < PlacementCost.MAX_RETRY:
                            logger.info(
                                "Looking for more data for %s on connection:%s/%s",
                                name,
                                retry,
                                PlacementCost.MAX_RETRY,
                            )
                            retry += 1
                        else:
                            raise e
            if isinstance(output, dict):
                if "ok" in output and not output["ok"]:  # Status::NotOk
                    raise ValueError(
                        f"Error in calling {name} with {args}: {output['message']}."
                    )
                elif "__tuple__" in output:  # Tuple
                    output = tuple(output["items"])
            elif isinstance(output, list):
                if (
                    len(output) > 0
                    and isinstance(output[0], dict)
                    and "__tuple__" in output[0]
                ):  # List of tuples
                    output = [tuple(o["items"]) for o in output]
            return output

        return f

    def close(self) -> None:
        self.conn.close()
        self.process.kill()
        self.process.wait()
        self.sock.close()
