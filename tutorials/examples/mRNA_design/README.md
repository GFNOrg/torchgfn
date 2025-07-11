# Context:
An mRNA sequence is a string of letters (A, U, C, G) that carries the instructions from DNA to make a specific protein in the cell. It is transcribed from DNA and serves as a template for protein synthesis during translation. The sequence is organized into triplets called codons, each coding for a specific amino acid in the resulting protein.

# Environment:
Custom mRNA codon design environment using torchgfn to generate mRNA sequences encoding a given protein. It supports a multi-objective optimization over biological properties of mRNA sequences. Implemented using the DiscreteEnv class.
Each timestep corresponds to choosing a synonymous codon for the next amino acid in the sequence.

## Action Space: 
Number of CODONS + 1 possible actions (all codons + 1 exit action)
## State Representation: 
A vector of length = protein length, initialized to -1. Codons are filled in step-by-step.
## Masking (Action Constraints): 
At each position t, only codons that correspond to the t-th amino acid are allowed to ensure biological correctness.
## Reward function: 
A combination of multiple biological properties to evaluate the mRNA sequence. Weights of these objectives can be updated dynamically to reflect different reward configurations. Rewards and constraints are modular and can be extended to incorporate new objectives.

The environment is customizable for different organisms by using species-specific codon tables preferences, and could serve as a benchmark environment in computational biology. This enables exploration of codon space, which is a large search space given a protein sequence, to optimize for mRNA design.  Applicable to mRNA vaccines, protein therapeutics, and gene expression optimization.