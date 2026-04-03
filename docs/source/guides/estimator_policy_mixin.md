# PolicyMixin: Policies and Rollouts

Estimators become policy-capable by mixing in a small, uniform rollout API. This lets the same `Sampler` and probability utilities drive different estimator families (discrete, graph, conditional, recurrent) without bespoke glue code.

This guide explains:
- The Policy rollout API and `RolloutContext`
- Vectorized vs non‑vectorized probability paths
- How policies integrate with the `Sampler` and probability calculators
- How to implement a new policy mixin or tailor the default behavior

## Concepts and Goals

A policy‑capable estimator exposes:
- `is_vectorized: bool` — whether the estimator can be evaluated in a single vectorized call (no per‑step carry).
- `init_context(batch_size, device, conditions)` — allocate a per‑rollout context.
- `compute_dist(states_active, ctx, step_mask, ...) -> (Distribution, ctx)` — run the model, build a `torch.distributions.Distribution`.
- `log_probs(actions_active, dist, ctx, step_mask, vectorized, ...) -> (Tensor, ctx)` — evaluate log‑probs, optionally padded to batch.
- `get_current_estimator_output(ctx)` — access the last raw model output when requested.

All per‑step artifacts (e.g., log‑probs, raw outputs, recurrent state) are owned by the `RolloutContext` and recorded by the mixin.

## RolloutContext

The `RolloutContext` is a lightweight container created once per rollout:
- `batch_size`, `device`, optional `conditions`
- Optional `carry` (for recurrent policies)
- Per‑step buffers: `trajectory_log_probs`, `trajectory_estimator_outputs`
- `current_estimator_output` for cached reuse or immediate retrieval
- `extras: dict` for arbitrary policy‑specific data

See `src/gfn/estimators.py` for the full definition.

## PolicyMixin (vectorized, default)

`PolicyMixin` enables vectorized evaluation by default (`is_vectorized=True`).

- `init_context(batch_size, device, conditions)` returns a fresh `RolloutContext` with empty buffers.
- `compute_dist(...)`:
  - Slices `conditions` by `step_mask` when provided; uses full `conditions` when `step_mask=None` (vectorized).
  - Optionally reuses `ctx.current_estimator_output` (e.g., PF with cached `trajectories.estimator_outputs`).
  - Calls the estimator module and builds a `Distribution` via `to_probability_distribution`.
  - When `save_estimator_outputs=True`, sets `ctx.current_estimator_output` and records a padded copy to `ctx.trajectory_estimator_outputs` for non‑vectorized calls.
- `log_probs(...)`:
  - `vectorized=True`: returns raw `dist.log_prob(...)` (may include `-inf` for illegal actions) and optionally records to `trajectory_log_probs`.
  - `vectorized=False`: strict inf‑check, pads to shape `(N,)` using `step_mask`, records when requested.

Code reference (log‑probs behavior): `src/gfn/estimators.py`.

## RecurrentPolicyMixin (per‑step)

`RecurrentPolicyMixin` sets `is_vectorized=False` and threads a carry through steps:

- `init_context(...)` requires the estimator to implement `init_carry(batch_size, device)`; stores the result in `ctx.carry`.
- `compute_dist(...)` must call the estimator as `(states_active, ctx.carry) -> (est_out, new_carry)`, update `ctx.carry`, build the `Distribution`, and record outputs when requested (with padding when masked).
- `log_probs(...)` follows the non‑vectorized path (pad and strict checks) and can reuse the same recording semantics as `PolicyMixin`.

Code reference (carry update and padded recording): `src/gfn/estimators.py`.

## Integration with the Sampler

The `Sampler` uses the policy API directly. It creates a single `ctx` per rollout, then repeats `compute_dist` → sample → optional `log_probs` while some trajectories are active. Per‑step artifacts are recorded into `ctx` by the mixin when flags are enabled.

Excerpt (per‑step call pattern): `src/gfn/samplers.py`.

## Integration with probability calculators (PF/PB)

Probability utilities in `utils/prob_calculations.py` branch on `is_vectorized` but call the same two methods in both paths:
- `compute_dist(states_active, ctx, step_mask=None or mask)`
- `log_probs(actions_active, dist, ctx, step_mask=None or mask, vectorized=...)`

Key differences:
- Vectorized (fast path)
  - `step_mask=None`, `vectorized=True`.
  - May reuse cached estimator outputs by pre‑setting `ctx.current_estimator_output`.
  - `log_probs` returns raw `dist.log_prob(...)` and does not raise on `-inf`.
- Non‑vectorized (per‑step path)
  - Uses legacy‑accurate masks and alignments:
    - PF (trajectories): `~states.is_sink_state[t] & ~actions.is_dummy[t]`
    - PB (trajectories): aligns action at `t` with state at `t+1`, using `~states.is_sink_state[t+1] & ~states.is_initial_state[t+1] & ~actions.is_dummy[t] & ~actions.is_exit[t]` (skips `t==0`).
    - Transitions: legacy PB mask on `next_states` with `~actions.is_exit`.
  - `log_probs` pads back to `(N,)` and raises if any `±inf` remains after masking.

See `src/gfn/utils/prob_calculations.py` for full branching.

## Built‑in policy‑capable estimators

- `DiscretePolicyEstimator`: logits → `Categorical` with masking, optional temperature and epsilon‑greedy mixing in log‑space.
- `DiscreteGraphPolicyEstimator`: multi‑head logits (`TensorDict`) → `GraphActionDistribution` with per‑component masks and transforms.
- `RecurrentDiscretePolicyEstimator`: sequence models that maintain a `carry`; requires `init_carry` and returns `(logits, carry)` in `forward`.
- Conditional variants exist for state+condition architectures.

## How to write a new policy (or mixin variant)

Most users only need to implement `to_probability_distribution` (or reuse the provided ones). If you need a new interface or extra tracking, you can either:

1) Use `PolicyMixin` (stateless, vectorized) and override `to_probability_distribution` on your estimator.
2) Use `RecurrentPolicyMixin` (per‑step, carry) and implement `init_carry` plus a `forward(states, carry)` that returns `(estimator_outputs, new_carry)`.
3) Create a custom mixin derived from `PolicyMixin` to tailor `compute_dist`/`log_probs` (e.g., custom caching, diagnostics).

### Minimal stateless policy (discrete)

```python
import torch
from torch import nn
from gfn.estimators import DiscretePolicyEstimator

class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Forward policy over n_actions
policy = DiscretePolicyEstimator(module=SmallMLP(input_dim=32, output_dim=17), n_actions=17)
```

Use with the `Sampler`:

```python
from gfn.samplers import Sampler

sampler = Sampler(policy)
trajectories = sampler.sample_trajectories(env, n=64, save_logprobs=True)
```

### Minimal recurrent policy

```python
import torch
from torch import nn
from gfn.estimators import RecurrentDiscretePolicyEstimator

class TinyRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor, carry: dict[str, torch.Tensor]):
        x = self.embed(tokens)
        h0 = carry.get("h", torch.zeros(1, tokens.size(0), x.size(-1), device=tokens.device))
        y, h = self.rnn(x, h0)
        logits = self.head(y)
        return logits, {"h": h}

    def init_carry(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        return {"h": torch.zeros(1, batch_size, self.embed.embedding_dim, device=device)}

policy = RecurrentDiscretePolicyEstimator(module=TinyRNN(vocab_size=33, hidden=64), n_actions=33)
```

### Custom mixin variant (advanced)

If you need to add diagnostics or custom caching, subclass `PolicyMixin` and override `compute_dist`/`log_probs` to interact with `ctx.extras`.

```python
from typing import Any, Optional
from torch.distributions import Distribution
from gfn.estimators import PolicyMixin

class TracingPolicyMixin(PolicyMixin):
    def compute_dist(self, states_active, ctx, step_mask=None, save_estimator_outputs=False, **kw):
        dist, ctx = super().compute_dist(states_active, ctx, step_mask, save_estimator_outputs, **kw)
        ctx.extras.setdefault("num_compute_calls", 0)
        ctx.extras["num_compute_calls"] += 1
        return dist, ctx

    def log_probs(self, actions_active, dist: Distribution, ctx: Any, step_mask=None, vectorized=False, save_logprobs=False):
        lp, ctx = super().log_probs(actions_active, dist, ctx, step_mask, vectorized, save_logprobs)
        ctx.extras.setdefault("last_lp_mean", lp.mean().detach())
        return lp, ctx
```

Keep `is_vectorized` consistent with your evaluation strategy. If you switch to `False`, ensure your estimator supports per‑step rollouts and masking semantics.
