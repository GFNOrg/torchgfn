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

### Soft Policy Gradient (VPG)

**Class:** `PolicyGradientGFlowNet`

Trains the forward policy as an entropy-regularized RL agent instead of enforcing a balance condition. With a fixed backward policy, GFlowNet training is equivalent to soft RL at `α = γ = 1` under the per-step reward `log P_B(s_t | s_{t+1})`, plus `log R(x)` on the exit transition; the soft value function is the GFlowNet log-flow, and `V(s_0) = log Z`. Introduced for GFlowNets in [Proximal Policy Optimization for Amortized Discrete Sampling](https://arxiv.org/abs/2606.15793).

Four advantage estimators are available via `advantage=`, in increasing order of sophistication and decreasing variance: `"total"` (the full soft return — this is REINFORCE on the reverse KL, and equals the negated TB score), `"reward_to_go"`, `"baseline"` (reward-to-go minus a learned soft value), and `"gae"` (the default; Generalized Advantage Estimation, `gae_lambda=0.7` in the paper).

**Requires:** Forward policy (PF), backward policy (PB), and — for `"baseline"`/`"gae"` — a soft value estimator (logV) via `ScalarEstimator`.

**When to use:** When you want RL-style variance reduction and multiple gradient steps per rollout. The paper reports that VPG with GAE already beats TB, DB and SubTB on Hypergrid and sequence problems.

**Note:** Unlike the balance objectives, these admit no joint forward/backward loss — changing `P_B` changes the MDP reward itself. Learn `P_B` with `tlm_loss` (Trajectory Likelihood Maximization) as a separate update.

**See:** `train_hypergrid_ppo.py` (with `--method vpg --advantage ...`).

---

### Entropic PPO (Ent-PPO)

**Class:** `EntPPOGFlowNet`

Proximal Policy Optimization adapted to the soft-RL formulation above, from the same paper. Combines the usual clipped importance ratio with an *analytic* KL penalty against the rollout policy, which falls out of soft policy improvement rather than being imposed on top. Standard PPO carries a free entropy coefficient and no cross-entropy term, and its optimum concentrates on `argmax_x R(x)` instead of sampling from `R/Z` — which is why earlier attempts at PPO for GFlowNets reported mode collapse.

**Requires:** Forward policy (PF), backward policy (PB), soft value estimator (logV). For the analytic KL, sample with `save_estimator_outputs=True`; `kl_estimator="importance"` is a fallback for policies with no closed-form KL.

**When to use:** When reward evaluations are the bottleneck. The importance ratio and KL trust region make several update epochs (`K`) per rollout safe, so each batch of trajectories yields more learning.

**Note:** `use_kl=False` and `use_clipping=False` reproduce the paper's ablations. Dropping the KL roughly doubles the final TV distance on a 16x16 HyperGrid at `K=8` — it is the load-bearing term. The clipping ablation does not separate at that scale; the paper reports divergence without it at `K` of 16-32.

**Workflow:** call `to_training_samples(trajectories)` once per rollout to freeze the advantages, value targets and `π_old` log-probs, then call `policy_loss` / `value_loss` as many times as you like on that frozen batch. Slicing the returned `PolicyGradientTrajectories` gives mini-batches that carry the same frozen targets.

**See:** `train_hypergrid_ppo.py`.

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
| **VPG** | PF, PB, logV | Per-step advantage | Medium | RL-style variance reduction |
| **Ent-PPO** | PF, PB, logV | Per-step advantage | Medium | When reward evaluations are the bottleneck |

### Measuring a discrete GFlowNet exactly

`env.validate(...)` estimates the terminating distribution by sampling, which has a noise
floor of its own: over 256 terminating states, a *perfect* model scores a total variation
distance of about 0.06 when measured with 10 000 samples. On small problems that floor can
swamp the differences between objectives.

When the state space is enumerable, use `env.exact_terminating_distribution(gflownet.pf)`
instead. It computes the distribution in closed form by pushing probability mass forward
through the DAG, and returns a tensor aligned with `env.true_dist()`.

Check `env.is_enumerable` first. It answers an *instance*-level question, because most
enumerable environments are only enumerable for some hyperparameters — a `HyperGrid` has
`height ** ndim` states, so a 20x20 grid is trivially enumerable while the same grid in
ten dimensions has 10^13 states and is not. Two conditions are checked:

- `supports_enumeration`, a class attribute, records whether the environment type
  implements the enumeration API at all. It is all-or-nothing: True commits the class to
  `n_states`, `all_states`, `get_states_indices` and the three terminating equivalents,
  with `get_states_indices(all_states) == arange(n_states)`.
- `n_states <= max_enumerable_states`, checked from the *formula* for the state count, so
  it costs nothing and fires before anything tries to materialize the states. Raise
  `env.max_enumerable_states` if you know the space fits in memory.

`env.enumeration_unavailable_reason()` returns a human-readable explanation of which
condition failed, or None. Note that measuring exactly removes sampling noise but not
training noise — comparing objectives still needs multiple seeds.

---

For a single-script comparison of TB, DB, and FM on the same environment, see `train_hypergrid_simple.py`. For all six balance losses in a single script, see `train_hypergrid.py`. For the policy-gradient objectives, see `train_hypergrid_ppo.py`.

## Common Training Patterns

### Separate Learning Rates

Most losses benefit from different learning rates for different parameter groups:

```python
optimizer = torch.optim.Adam([
    {"params": gflownet.pf_pb_parameters(), "lr": 1e-3},
    {"params": gflownet.logz_parameters(), "lr": 1e-1},
])
```

For DB/SubTB, add a third group for the logF estimator. For VPG/Ent-PPO, use
`gflownet.logV_parameters()` — the paper trains the soft value function at one third of
the policy learning rate, and with a separate optimizer, since the two take a different
number of steps per rollout.

### On-Policy vs Off-Policy

When training on-policy (no replay buffer, no exploration noise), set `save_logprobs=True` during sampling and `recalculate_all_logprobs=False` during loss computation to avoid redundant forward passes. For off-policy training, log-probs must be recalculated — see the [Off-Policy Training guide](off_policy_training.md).
