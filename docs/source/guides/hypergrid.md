# Hypergrid Environment

## HyperGrid compositional environments

This guide documents three compositional reward families implemented in `gfn/gym/hypergrid.py` that scale to very large state spaces while rewarding modes that emerge only from learning simple rules and composing them across tiers. Each family ships with five difficulty presets targeting step-distance bands from the origin.

### High-level design constraints
- **Simple base rule near `s0`**: Tier 0 should be discoverable by short random walks and simple policies.
- **Tiered, compositional sparsity**: Higher tiers are the closure of lower-tier modes under a simple composition, becoming dramatically sparser.
- **Deterministic computability**: Given lower tiers, the next tier’s modes are computable by a small, known mathematical function.
- **Massive spaces, few modes**: Choose `height` and `ndim` so the space is huge, while top-tier modes are ultrathin.

### Notation
- The grid is \(X = \{0,\dots,H-1\}^D\) with origin \(s_0 = 0\).
- The step distance is the L1 norm: \(\lVert x \rVert_1 = \sum_{i=1}^D x_i\).
- Rewards are tiered: \( R(x) = R_0 + \sum_{t=0}^{T-1} w_t \; \mathbf{1}[x \in S_t] \), where membership requires satisfying all constraints up to tier \(t\).

## Environment families

### Bitwise/XOR fractal (divide-and-conquer friendly)
**Intuition**: Impose linear parity constraints on bit-planes over GF(2). Each tier adds bit-planes and/or parity checks over a subset of dimensions. Shards can work independently by splitting the space on high-bit prefixes.

GF(2) is the finite field with two elements {0,1} where addition and multiplication are taken modulo 2. In this setting, vector addition corresponds to bitwise XOR, and matrix–vector products (A u) are evaluated entrywise modulo 2.

**Mathematical definition (tier t)**
- Choose constrained dimension indices \(M \subseteq \{1,\dots,D\}\) with \(|M| = m\).
- Choose bit window \([\ell_t, h_t] \subseteq \{0,\dots, B\}\) where typically \(H = 2^B\).
- Optionally choose a parity system over GF(2): \(A_t \in \{0,1\}^{k_t \times m}\), \(c_t \in \{0,1\}^{k_t}\).
- For each bit-plane \(b \in [\ell_t, h_t]\), define \(u^{(b)} \in \{0,1\}^m\) by \(u^{(b)}_j = \big((x_{i_j} \gg b) \; \&\; 1\big)\), where \(i_j\) indexes \(M\).
- Constraint: \(A_t\, u^{(b)} \equiv c_t \pmod{2}\). If \(A_t, c_t\) are omitted, default is even parity: \(\sum_j u^{(b)}_j \equiv 0 \pmod{2}.\)

**Reward**: Add \(w_t\) if all bit-planes in \([\ell_t,h_t]\) satisfy the constraints and all lower tiers are satisfied.

**Distance control**: Typical distance scales like \(m \cdot 2^{\max_b}\), where \(\max_b\) is the highest constrained bit.

**Sharding**: High. Split by high-bit prefixes; constraints are local and uniform.

#### Difficulty presets (indicative)
| Difficulty | Target steps | m (dims) | Bit windows per tier | Notes |
|---|---:|---:|---|---|
| Easy | 50–100 | 3 | [(0,4), (0,5), (0,5)] | Short walks discoverable |
| Medium | 250–500 | 4 | [(0,6), (0,7), (0,7), (0,7)] | Still shardable |
| Hard | 1k–2.5k | 8 | [(0,8)] × 4 | Macro-steps useful |
| Challenging | 2.5k–5k | 6 | [(0,9)] × 4 | Macro-steps recommended |
| Impossible | 5k+ | 12 | [(0,9), (0,10), (0,10), (0,10), (0,10)] | Very sparse |

See in-code: `gym.hypergrid.get_bitwise_xor_presets(D, H)`.

### Multiplicative/Coprime ladder (information sharing required)
**Intuition**: Values must factor over a small shared prime set with bounded exponents, plus cross-dimension coprimality and optional LCM targets. Higher tiers raise caps or add targets. Agents benefit from sharing discovered prime/exponent structure.

**Mathematical definition (tier t)**
- Fix prime set \(P = \{p_1,\dots,p_r\}\) and exponent cap \(c_t\).
- Active dimension set \(A \subseteq \{1,\dots,D\}\), indices relative to \(A\): \(0..m-1\).
- Per active dimension \(i\), require \(x_i = \prod_{j=1}^r p_j^{e_{i,j}}\) with \(0 \le e_{i,j} \le c_t\). Equivalently, repeatedly divide \(x_i\) by primes in \(P\) up to \(c_t\) times each; residue must be 1 (allow \(x_i=1\)).
- Optional pairwise coprimality on pairs \(\mathcal{E} \subseteq \{(i,j)\}\): \(\gcd(x_i, x_j) = 1\) for \((i,j) \in \mathcal{E}\).
- Optional LCM target \(L_t\): write exponent vectors across dims, take per-prime maxima \(\max_i e_{i,j}\); require that equals the exponents of \(L_t\).

**Reward**: Add \(w_t\) if prime-support, coprimality, and optional LCM target hold, and all lower tiers hold.

**Distance control**: Increase primes, exponent caps, and number of active dims to raise \(\sum_i x_i\) into target bands.

**Sharding**: Low–Medium. Without global knowledge of \(P\), caps, and targets, shards waste effort; sharing/broadcast helps.

#### Difficulty presets (indicative)
| Difficulty | Target steps | Primes | Caps | Active dims | Extras |
|---|---:|---|---:|---:|---|
| Easy | 50–100 | {2,3,5} | 2 | 3 | Chain coprime pairs |
| Medium | 250–500 | {2,3,5,7} | 2 | 5 | Light coupling |
| Hard | 1k–2.5k | {2,3,5,7,11} | 3 | 8 | LCM target e.g. \(2^3 3^2 5 7 11\) |
| Challenging | 2.5k–5k | {2,3,5,7,11,13} | 3–4 | 10 | Tighter caps/targets |
| Impossible | 5k+ | up to 29 | 4 | 12 | Multiple global targets |

See in-code: `get_multiplicative_coprime_presets(D, H)`.

### Template Minkowski powers (information sharing recommended)
**Intuition**: Compose a small atom template \(T\) near the origin via repeated Minkowski sums. In practice we gate by thin L1 radius bands per tier, optionally with simple residue constraints, which closely track membership in \(T^{\oplus k}\).

**Mathematical definition (tier t)**
- Choose subset of dimensions \(S \subseteq \{1,\dots,D\}\) used for L1 and residues (default all).
- Choose an L1 band \([r^{\min}_t, r^{\max}_t]\). Define \(\lVert x \rVert_{1,S} = \sum_{i \in S} x_i\).
- Optional sum residue \((m_t, a_t)\): require \( \lVert x \rVert_{1,S} \equiv a_t \pmod{m_t} \).
- Composition perspective: atoms \(T\) with 0 and unit basis imply membership in \(T^{\oplus k}\) when \(\lVert x \rVert_1 \approx k\); the band gates implicitly approximate the tier’s fold \(k\).

**Reward**: Add \(w_t\) if band and residue hold and all lower tiers hold.

**Distance control**: Directly via \(r_t\) bands; combine with residues to thin.

**Sharding**: Medium. You can shard by annuli (bands), but composition and gating choices benefit from shared knowledge.

#### Difficulty presets (indicative)
| Difficulty | Target steps | L1 bands per tier | Residues |
|---|---:|---|---|
| Easy | 50–100 | [(60,61), (80,81), (90,91)] | – |
| Medium | 250–500 | [(300,301), (350,351), (420,421), (480,481)] | mod 4 = 1 (tier 2) |
| Hard | 1k–2.5k | [(1400,1401), (1800,1801), (2200,2201), (2400,2401)] | parity on sum |
| Challenging | 2.5k–5k | [(2800,2801), (3200,3201), (4000,4001), (4800,4801)] | mod 3 = 2 (tier 0) |
| Impossible | 5k–10k | [(6000,6001), …, (10000,10001)] | sparse mod gates |

See in-code: `gym.hypergrid.get_template_minkowski_presets(D, H)`.

## Non-compositional baselines

These reward families do not require composing rules across tiers, and represent the standard hypergrid environment used in the literature.

### Original reward (GFlowNets baseline)
Let \(a_i(x) = \big| x_i/(H-1) - 0.5 \big|\). Define per-dimension bands:
- Outer ring: \(0.25 < a_i(x) \le 0.5\)
- Thin band: \(0.3 < a_i(x) < 0.4\)

Reward:
\[ R(x) = R_0 \; + \; R_1 \; \prod_{i=1}^D \mathbf{1}[0.25 < a_i(x) \le 0.5] \; + \; R_2 \; \prod_{i=1}^D \mathbf{1}[0.3 < a_i(x) < 0.4]. \]

Modes occur at states where both products equal 1 (i.e., the thin band inside the outer ring along all dimensions). See `OriginalReward`.

### Cosine reward (oscillatory, center-weighted)
Define \(a_i(x)\) as above and \(\varphi(z) = (2\pi)^{-1/2} e^{-z^2/2}\). Reward:
\[ R(x) = R_0 \; + \; R_1 \; \prod_{i=1}^D \Big( (\cos(50\, a_i(x)) + 1) \cdot \varphi(5\, a_i(x)) \Big). \]

This produces a peak near the center with oscillatory local maxima along each axis. See `CosineReward`.

### Sparse reward (permutation targets; GAFN paper)
Construct a target set \(T \subseteq \{0,\dots,H-1\}^D\) by taking, for each \(k = 0,\dots,D\), all distinct permutations of the vector with \(k\) ones and \(D-k\) entries equal to \(H-2\). Reward:
\[ R(x) = \sum_{t \in T} \mathbf{1}[x = t] + \varepsilon, \quad \varepsilon > 0. \]

This yields extremely sparse “spikes” at specified corners/edges. See `SparseReward`.

### Deceptive reward (outer cancellation, center emphasis)
With \(a_i(x)\) as above, define
\[ R(x) = (R_0 + R_1) \; - \; R_1 \; \prod_{i=1}^D \mathbf{1}[0.1 < a_i(x)] \; + \; R_2 \; \prod_{i=1}^D \mathbf{1}[0.3 < a_i(x) < 0.4]. \]

Compared to the Original reward, the outer region cancels \(R_1\) while the center keeps \(R_1\); the ring band still adds \(R_2\). Modes lie on the thin band; corners are de-emphasized and the center square is emphasized. See `DeceptiveReward`.

## Sharding vs. information sharing

| Setup | Divide-and-conquer suitability | Info sharing needed | Rationale |
|---|---|---|---|
| Bitwise/XOR fractal | High | Low | Local, uniform GF(2) constraints; shard by bit prefixes |
| Multiplicative/Coprime ladder | Low–Medium | High | Global prime/exponent and LCM/coprime structure |
| Template Minkowski powers | Medium | Medium–High | Bands shard, but composition/gates benefit from shared rules |

## Difficulty bands and macro-steps
- Bands: Easy (50–100), Medium (250–500), Hard (1k–2.5k), Challenging (2.5k–5k), Impossible (5k+).
- If your training struggles with trajectories > 500, you can compose learned sub-trajectories as macro-steps (e.g., 5-step macros multiply effective reach by ~5). Prefer “Hard/Challenging/Impossible” presets for macro-step experiments.

## Mode validation and statistics

The `HyperGrid` environment can validate that modes (as defined by the configured reward’s mode threshold) actually exist, and optionally compute mode statistics for monitoring.

Parameters
- `validate_modes: bool` (default: True)
  - On initialization, raises `ValueError` if no state satisfies the mode threshold.
- `mode_stats: {"none", "approx", "exact"}` (default: "none")
  - `exact`: requires `store_all_states=True`. Enumerates all states to compute exact counts.
  - `approx`: uniform random sampling to estimate counts; controlled by `mode_stats_samples`.
  - `none`: disables statistics collection.
- `mode_stats_samples: int` (default: 20000)
  - Number of random samples used for `approx` statistics.

Properties
- `n_modes`: number of distinct modes, when available
  - `exact`: count of unique `mode_ids` among mode states
  - `approx`: estimated from unique `mode_ids` observed in samples
  - fallback: `2**ndim` (baseline heuristic) if stats are disabled
- `n_mode_states`: number of states inside modes
  - `exact`: integer count
  - `approx`: floating-point estimate (fraction × total states)
  - `None` if stats are disabled

> Warning
> - `mode_stats="exact"` requires `store_all_states=True` and enumerates all `H^D` states; this can be very expensive in time and memory for large `H` or `D`. Prefer `approx` for large grids or when running distributed training.

Example
```python
from gfn.gym.hypergrid import HyperGrid, get_reward_presets

presets = get_reward_presets("bitwise_xor", 16, 2**12)
env = HyperGrid(
    ndim=16,
    height=2**12,
    reward_fn_str="bitwise_xor",
    reward_fn_kwargs=presets["medium"],
    validate_modes=True,
    mode_stats="approx",
    mode_stats_samples=50000,
)
print(env.n_modes, env.n_mode_states)
```

## Usage examples
```python
from gfn.gym.hypergrid import HyperGrid, get_reward_presets

D, H = 32, 2**16
presets = get_reward_presets("bitwise_xor", D, H)
env = HyperGrid(ndim=D, height=H, reward_fn_str="bitwise_xor", reward_fn_kwargs=presets["hard"])
```

## References
- GF(2) and parity checks: [Wikipedia – Finite field of two elements](https://en.wikipedia.org/wiki/Finite_field_arithmetic#Binary_field_(GF(2)))
- Linear codes and parity-check matrices: [Wikipedia – Parity-check matrix](https://en.wikipedia.org/wiki/Parity-check_matrix)
- Minkowski sum: [Wikipedia – Minkowski addition](https://en.wikipedia.org/wiki/Minkowski_addition)
- L1 norm: [Wikipedia – Norm (mathematics)](https://en.wikipedia.org/wiki/Norm_(mathematics))
- Fundamental theorem of arithmetic (unique factorization), gcd/lcm: [Wikipedia – Fundamental theorem of arithmetic](https://en.wikipedia.org/wiki/Fundamental_theorem_of_arithmetic), [Wikipedia – Greatest common divisor](https://en.wikipedia.org/wiki/Greatest_common_divisor), [Wikipedia – Least common multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
- GFlowNets hypergrid baseline: Bengio et al., 2021 – [Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](https://arxiv.org/abs/2106.04399)
- GAFN sparse targets: Pan et al., 2022 – [Exploring Categorical GFlowNets for Discrete Structure Generation](https://arxiv.org/abs/2210.03308)
- Deceptive reward variant (Adaptive Teachers): Kim et al., 2025 – official implementation linked in code docstring


