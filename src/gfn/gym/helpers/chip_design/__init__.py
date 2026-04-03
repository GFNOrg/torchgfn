# Code copied from https://github.com/google-research/circuit_training
# @article{mirhoseini2021graph,
#   title={A graph placement methodology for fast chip design},
#   author={Mirhoseini*, Azalia and Goldie*, Anna and Yazgan, Mustafa and Jiang, Joe
#   Wenjie and Songhori, Ebrahim and Wang, Shen and Lee, Young-Joon and Johnson,
#   Eric and Pathak, Omkar and Nazi, Azade and Pak, Jiwoo and Tong, Andy and
#   Srinivasa, Kavya and Hang, William and Tuncer, Emre and V. Le, Quoc and
#   Laudon, James and Ho, Richard and Carpenter, Roger and Dean, Jeff},
#   journal={Nature},
#   volume={594},
#   number={7862},
#   pages={207--212},
#   year={2021},
#   publisher={Nature Publishing Group}
# }
#
# Requires Linux x86-64. plc_wrapper_main is a closed-source binary from Google.
#
# Setup:
#   Linux x86-64 (native):
#     cd src/gfn/gym/helpers/chip_design && ./setup_plc.sh
#
#   HPC (via Apptainer/Singularity):
#     cd src/gfn/gym/helpers/chip_design && ./setup_plc.sh --container-only
#
#   Explicit path:
#     export PLC_WRAPPER_MAIN=/path/to/plc_wrapper_main

import os

SAMPLE_NETLIST_FILE = os.path.join(
    os.path.dirname(__file__), "test_data", "netlist.pb.txt"
)
SAMPLE_INIT_PLACEMENT = os.path.join(
    os.path.dirname(__file__), "test_data", "initial.plc"
)

MEDIUM_NETLIST_FILE = os.path.join(
    os.path.dirname(__file__), "test_data", "netlist_medium.pb.txt"
)
MEDIUM_INIT_PLACEMENT = os.path.join(
    os.path.dirname(__file__), "test_data", "initial_medium.plc"
)

PLC_DOWNLOAD_URL = (
    "https://storage.googleapis.com/rl-infra-public/"
    "circuit-training/placement_cost/plc_wrapper_main_{version}"
)
