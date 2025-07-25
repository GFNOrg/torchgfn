from types import SimpleNamespace
from typing import Optional

import yaml
from CAI import CAI


def load_config(path: str):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)


# References to compute CAI
config = load_config("config.yaml")
def_refs = config.def_refs


class CAICalculator:

    def __init__(
        self,
        target_rna: str,
        reference_seqs: Optional[list[str]] = None,
        ref_file: Optional[str] = None,
    ):
        self.target_rna = target_rna
        self.target_dna = target_rna.replace("U", "T")

        if reference_seqs is not None:
            self.reference_seqs = reference_seqs

        else:
            self.reference_seqs = def_refs

    def _load_reference_sequences_from_file(self, filepath: str) -> list[str]:
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def compute_cai(self) -> float:
        cai_obj = CAI(self.target_dna, reference=self.reference_seqs)
        return cai_obj
