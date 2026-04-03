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
from typing import Any, Optional, Text

logger = logging.getLogger(__name__)

DEFAULT_PLC_WRAPPER = os.path.join(os.path.dirname(__file__), "plc_wrapper_main")
DEFAULT_SIF_IMAGE = os.path.join(os.path.dirname(__file__), "plc_wrapper.sif")


def _find_singularity() -> Optional[str]:
    """Returns the path to singularity/apptainer, or None."""
    for name in ("singularity", "apptainer"):
        path = shutil.which(name)
        if path is not None:
            return path
    return None


class PlacementCost:
    """PlacementCost object wrapper."""

    BUFFER_LEN = 1024 * 1024
    MAX_RETRY = 256

    def __init__(
        self,
        netlist_file: Text,
        plc_wrapper_main: str = DEFAULT_PLC_WRAPPER,
        macro_macro_x_spacing: float = 0.0,
        macro_macro_y_spacing: float = 0.0,
        singularity_image: Optional[str] = None,
    ) -> None:
        """Creates a PlacementCost client object.

        It creates a subprocess by calling plc_wrapper_main and communicate with
        it over an `AF_UNIX` channel.

        Args:
          netlist_file: Path to the netlist proto text file.
          plc_wrapper_main: Path to the plc_wrapper_main binary.
          macro_macro_x_spacing: Macro-to-macro x spacing in microns.
          macro_macro_y_spacing: Macro-to-macro y spacing in microns.
          singularity_image: Path to a Singularity/Apptainer .sif image.
            If provided, the plc_wrapper_main binary is executed inside the
            container. If ``"auto"``, uses the default .sif image bundled
            with this package (if it exists).
        """
        if not plc_wrapper_main:
            raise ValueError("plc_wrapper_main should be specified.")

        if singularity_image == "auto":
            if os.path.isfile(DEFAULT_SIF_IMAGE):
                singularity_image = DEFAULT_SIF_IMAGE
            else:
                logger.info(
                    "No .sif image found at %s, running binary directly.",
                    DEFAULT_SIF_IMAGE,
                )
                singularity_image = None

        plc_wrapper_main = os.path.abspath(plc_wrapper_main)
        netlist_file = os.path.abspath(netlist_file)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = tempfile.NamedTemporaryFile().name
        self.sock.bind(address)
        self.sock.listen(1)

        plc_args = [
            plc_wrapper_main,
            "--uid=0",
            "--gid=0",
            f"--pipe_address={address}",
            f"--netlist_file={netlist_file}",
            f"--macro_macro_x_spacing={macro_macro_x_spacing}",
            f"--macro_macro_y_spacing={macro_macro_y_spacing}",
        ]

        if singularity_image is not None:
            singularity_image = os.path.abspath(singularity_image)
            if not os.path.isfile(singularity_image):
                raise FileNotFoundError(
                    f"Singularity image not found: {singularity_image}. "
                    f"Build it with: singularity build plc_wrapper.sif plc_wrapper.def"
                )
            singularity_bin = _find_singularity()
            if singularity_bin is None:
                raise RuntimeError(
                    "singularity_image was specified but neither "
                    "'singularity' nor 'apptainer' is on PATH."
                )
            bind_paths = {
                os.path.dirname(plc_wrapper_main),
                os.path.dirname(netlist_file),
                os.path.dirname(address),
            }
            bind_arg = ",".join(sorted(bind_paths))
            args = [
                singularity_bin,
                "exec",
                "--bind",
                bind_arg,
                singularity_image,
            ] + plc_args
        else:
            args = plc_args

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
