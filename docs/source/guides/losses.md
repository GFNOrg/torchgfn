# Loss Functions

GFlowNets can be trained with different losses, each of which requires a different parametrization, which we call in this library a `GFlowNet`. A `GFlowNet` includes one or multiple `Estimator`s, at least one of which implements a `to_probability_distribution` function. They also need to implement a `loss` function, that takes as input either [`States`, `Transitions`, or `Trajectories` `Container`](guides/states_actions_containers.md) instances, depending on the loss.

## Available Losses

### Trajectory Balance (TB)

**Class:** `TBGFlowNet`

The most commonly used loss. Enforces flow conservation along entire trajectories by requiring that the product of forward transition probabilities (times Z) equals the product of backward probabilities times the reward.

**Requires:** Forward policy (PF), backward policy (PB), learnable log-partition function (logZ).

**When to use:** Default choice for most problems. Works well across discrete, continuous, and graph environments. Straightforward to implement and debug.

**Tip:** logZ typically benefits from a higher learning rate than the policy parameters (e.g., `lr_Z=0.1` vs `lr=1e-3`). Use separate optimizer parameter groups via `gflownet.pf_pb_parameters()` and `gflownet.logz_parameters()`.

**See:** `train_hypergrid_simple.py` (basic usage), `train_box.py` (continuous), `train_graph_ring.py` (graphs).

---

### Detailed Balance (DB)

**Class:** `DBGFlowNet`

Imposes a stricter, state-level balance constraint. Instead of balancing entire trajectories, enforces that flow is conserved at every individual transition.

**Requires:** Forward policy (PF), backward policy (PB), log state-flow estimator (logF) via `ScalarEstimator`.

**When to use:** When you want fine-grained per-transition learning signal. Can converge faster than TB on some problems but requires an additional estimator.

**Modified variant:** `ModifiedDBGFlowNet` drops the explicit logF estimator. In forward-looking mode, rewards must be defined on edges; the current implementation treats the edge reward as the difference between the successor and current state rewards, so only enable this when that matches your environment.

**See:** `train_hypergrid_simple.py` (with `--loss db`), `train_bit_sequences.py`.

---

### Sub-Trajectory Balance (SubTB)

**Class:** `SubTBGFlowNet`

Generalizes TB by considering all sub-trajectories within a trajectory. Each sub-trajectory is weighted geometrically (within the trajectory) depending on its length. This corresponds to the strategy defined [here](https://www.semanticscholar.org/reader/f2c32fe3f7f3e2e9d36d833e32ec55fc93f900f5). Other strategies exist and are implemented in `src/gfn/losses/sub_trajectory_balance.py`.

**Requires:** Forward policy (PF), backward policy (PB), log state-flow estimator (logF).

**When to use:** When TB is underperforming and you want richer learning signal from each trajectory. Adds computational cost but can improve sample efficiency.

**Note:** When using geometric-based weighting, the `'mean'` reduction is not supported; requests for a mean reduction are coerced to a sum (a warning is emitted when debug is enabled).

**See:** `train_box.py` (with `--loss subtb`), `train_with_compile.py`.

---

### Flow Matching (FM)

**Class:** `FMGFlowNet`

The original GFlowNet loss. Matches incoming and outgoing flows at each state.

**Requires:** Only a log-flow estimator (logF) via `DiscretePolicyEstimator` — no explicit forward/backward policies.

**When to use:** Rarely recommended. Slow to compute and hard to optimize. Included primarily for completeness and for comparison with other losses.

**See:** `train_discreteebm.py`, `train_ising.py`.

---

### Log Partition Variance (ZVar)

**Class:** `LogPartitionVarianceGFlowNet`

Minimizes the variance of the log-partition function estimate across trajectories. Introduced in [this paper](https://arxiv.org/abs/2302.05446).

**Requires:** Forward policy (PF), backward policy (PB).

**When to use:** An alternative to TB that avoids learning an explicit logZ parameter. Can be useful when logZ estimation is unstable.

**See:** `train_hypergrid.py` (with `--loss zvar`).

---

### Relative Trajectory Balance (RTB)

**Class:** `RelativeTrajectoryBalanceGFlowNet`

A variant of TB designed for posterior fine-tuning from a pre-trained prior. Uses a fixed reference policy that does not receive gradients.

**Requires:** Forward policy (PF, trainable), backward policy (PB), fixed prior policy (PF_prior).

**When to use:** When you have a pre-trained model (e.g., from MLE) and want to fine-tune it to match a posterior distribution.

**See:** `train_diffusion_rtb.py` (two-stage prior→posterior pipeline).

---

## Choosing a Loss Function

| Loss | Estimators needed | Learning signal | Computational cost | Recommended for |
|------|------------------|----------------|-------------------|----------------|
| **TB** | PF, PB, logZ | Per-trajectory | Low | Most problems (default choice) |
| **DB** | PF, PB, logF | Per-transition | Medium | Problems where per-state signal helps |
| **SubTB** | PF, PB, logF | Per-sub-trajectory | High | When TB underperforms |
| **FM** | logF only | Per-state flow | High | Completeness / comparison |
| **ZVar** | PF, PB | Per-trajectory | Low | When logZ learning is unstable |
| **RTB** | PF, PB, PF_prior | Per-trajectory | Medium | Posterior fine-tuning |

For a single-script comparison of TB, DB, and FM on the same environment, see `train_hypergrid_simple.py`. For all six losses in a single script, see `train_hypergrid.py`.

## Common Training Patterns

### Separate Learning Rates

Most losses benefit from different learning rates for different parameter groups:

```python
optimizer = torch.optim.Adam([
    {"params": gflownet.pf_pb_parameters(), "lr": 1e-3},
    {"params": gflownet.logz_parameters(), "lr": 1e-1},
])
```

For DB/SubTB, add a third group for the logF estimator.

### On-Policy vs Off-Policy

When training on-policy (no replay buffer, no exploration noise), set `save_logprobs=True` during sampling and `recalculate_all_logprobs=False` during loss computation to avoid redundant forward passes. For off-policy training, log-probs must be recalculated — see the [Off-Policy Training guide](off_policy_training.md).
