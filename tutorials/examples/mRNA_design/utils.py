from types import SimpleNamespace
from typing import Dict, List

import torch
import yaml
from CAI_calculator import CAICalculator
from MFE_calculator import RNAFolder

# --- Biological Constants ---

# Stop codons
STOP_CODONS: List[str] = ["UAA", "UAG", "UGA"]

# Codon table mapping amino acids (or stop *) to codons, example of protein sequence : ACDEFGHIKLMNPQ
CODON_TABLE: Dict[str, List[str]] = {
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "C": ["UGU", "UGC"],
    "D": ["GAU", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["UUU", "UUC"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
    "H": ["CAU", "CAC"],
    "I": ["AUU", "AUC", "AUA"],
    "K": ["AAA", "AAG"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "M": ["AUG"],
    "N": ["AAU", "AAC"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
    "W": ["UGG"],
    "Y": ["UAU", "UAC"],
    "*": ["UAA", "UAG", "UGA"],  # Stop codons
}


# Amino acid list
AA_LIST: List[str] = list(CODON_TABLE.keys())
ALL_CODONS: List[str] = sorted(
    list(set(c for codons in CODON_TABLE.values() for c in codons))
)
N_CODONS: int = len(ALL_CODONS)
CODON_TO_IDX: Dict[str, int] = {codon: idx for idx, codon in enumerate(ALL_CODONS)}
IDX_TO_CODON: Dict[int, str] = {idx: codon for codon, idx in CODON_TO_IDX.items()}


codon_gc_counts = torch.tensor(
    [codon.count("G") + codon.count("C") for codon in ALL_CODONS], dtype=torch.float
)


# -- Helper: Tokenize codon string to LongTensor
def tokenize_sequence_to_tensor(seq):
    codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
    indices = [CODON_TO_IDX[c] for c in codons if c in CODON_TO_IDX]
    return torch.tensor(indices, dtype=torch.long)


def decode_sequence(tensor_seq):
    return "".join([IDX_TO_CODON[int(i)] for i in tensor_seq])


# --- Utility Functions ---
def get_synonymous_indices(amino_acid: str) -> List[int]:
    """
    Return the list of global codon indices that encode the given amino acid.
    Handles standard amino acids and '*'.
    """
    codons = CODON_TABLE.get(amino_acid, [])
    return [CODON_TO_IDX[c] for c in codons]


def mRNA_string_to_tensor(rna: str):

    rna_index = []
    for i in range(0, len(rna) - 3, 3):
        index = CODON_TO_IDX[rna[i : i + 3]]
        rna_index.append(index)

    rna_tensor = torch.tensor(rna_index)
    return rna_tensor


def to_mRNA_string(rna_tensor: torch.Tensor):

    rna_string = ""
    for i in range(0, len(rna_tensor)):
        cd = IDX_TO_CODON[int(rna_tensor[i].item())]
        rna_string += cd

    return rna_string


def load_config(path: str):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)


def compute_gc_content_vectorized(
    indices: torch.Tensor, codon_gc_counts: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized GC content calculation using precomputed codon GC counts
    """
    device = indices.device
    gc_counts = codon_gc_counts[indices].sum(dim=0)
    total_nucleotides = indices.shape[0] * 3
    gc_content = gc_counts / total_nucleotides * 100

    return gc_content.to(device)


def compute_mfe_energy(indices: torch.Tensor, energies=None, loop_min=4) -> torch.Tensor:
    """
    Compute the minimum free energy (MFE) of an RNA sequence using Zucker Algorithm.
    """
    device = indices.device
    mfe_energies = []
    rna_str = to_mRNA_string(indices)
    try:
        sol = RNAFolder(energies=energies, loop_min=loop_min)
        s = sol.solve(rna_str)
        energy = s.energy()
    except Exception as e:
        print(f"Energy computation failed for: {rna_str}, error: {e}")
        energy = float("inf")

    mfe_energies.append(energy)
    return torch.tensor(mfe_energies, dtype=torch.float32).to(device)


def compute_cai(indices: torch.Tensor, energies=None, loop_min=4) -> torch.Tensor:

    device = indices.device
    cai_scores = []
    rna_str = to_mRNA_string(indices)
    try:
        calc = CAICalculator(rna_str)
        score = calc.compute_cai()
    except Exception as e:
        print(f"CAI computation failed for: {rna_str}, error: {e}")
        score = float("inf")
    cai_scores.append(score)
    return torch.tensor(cai_scores, dtype=torch.float32).to(device)


def compute_reward_components(state, codon_gc_counts):
    gc_content = compute_gc_content_vectorized(state, codon_gc_counts).item()
    mfe_energy = compute_mfe_energy(state).item()
    cai_score = compute_cai(state).item()
    return gc_content, mfe_energy, cai_score


def compute_reward(state, codon_gc_counts, weights):
    gc, mfe, cai = compute_reward_components(state, codon_gc_counts)
    reward = sum(w * r for w, r in zip(weights, [gc, -mfe, cai]))  # weighted sum
    return reward, (gc, mfe, cai)
