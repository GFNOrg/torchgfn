# Estimator Adapters

Adapters decouple the generic sampling and probability computation logic from estimator-specific details (conditioning shape, recurrent state/carry, distribution construction, artifact recording). They enable a single sampler and probability utilities to work across different estimator families.

This guide explains:
- The Adapter and RolloutContext
- Vectorized vs non-vectorized probability paths
- How adapters integrate with the Sampler and probability calculators
- How to implement a new Adapter

## Concepts and Goals

An Adapter mediates between three places where estimator logic is needed:
1) The online sampling loop (Sampler) for trajectory rollouts
2) Probability calculators for trajectories (PF/PB) and transitions (PF/PB)
3) Optional artifact capture (per-step log-probs, estimator outputs)

The Sampler remains estimator-agnostic. Adapters own any estimator-specific state (e.g., recurrent carry) and control how to run the estimator and build the policy distribution.

## Adapters

Adapters conform to an abstract class structure:

- Properties
  - `is_backward: bool` — whether the wrapped estimator is a backward policy.
  - `is_vectorized: bool` — whether the adapter supports vectorized probability calculations (no carry). Vectorized adapters always use the faster legacy vectorized paths in probability calculators. Non-vectorized adapters (e.g., recurrent) use per-step paths with masking and alignment identical to the legacy reference.

- Methods
  - `init_context(batch_size: int, device: torch.device, conditioning: Tensor|None) -> Any`
    - Allocates a rollout context once per batch (Sampler). Stores invariants (batch size, device, optional conditioning) and initializes any adapter state (e.g., recurrent carry) along with per-step artifact buffers.

  - `compute_dist(states_active: States, ctx: Any, step_mask: Tensor|None, **policy_kwargs) -> (Distribution, Any)`
    - Runs the estimator forward on the provided rows and returns a torch Distribution over actions.
    - Slices `conditioning` with `step_mask` when provided (non‑vectorized); uses full conditioning when `step_mask=None` (vectorized).
    - Sets `ctx.current_estimator_output` to the raw estimator output. Vectorized callers may prefill `ctx.current_estimator_output` to reuse cached outputs.

  - `log_probs(actions_active: Tensor, dist: Distribution, ctx: Any, step_mask: Tensor|None, vectorized: bool = False) -> (Tensor, Any)`
    - Computes log-probs from `dist` for the given actions.
    - When `vectorized=False`, returns a padded `(N,)` tensor (zeros where `~step_mask`), with a strict inf-check (raises on `±inf`).
    - When `vectorized=True`, returns the raw `dist.log_prob(...)` without padding or inf-check (vectorized paths can legitimately include `-inf` for illegal actions).

  - `record(ctx: Any, step_mask: Tensor, sampled_actions: Tensor, dist: Distribution, log_probs: Optional[Tensor], save_estimator_outputs: bool) -> None`
    - Records per-step artifacts owned by the context. It never recomputes log-probs; pass `log_probs=None` to skip recording them.
    - Pads estimator outputs to `(N, ...)` using `-inf` before appending when `save_estimator_outputs=True`.

  - `finalize(ctx) -> dict[str, Optional[Tensor]]`
    - Stacks per-step buffers into trajectory-level tensors, e.g. `(T, N, ...)`, returning `{"log_probs": Tensor|None, "estimator_outputs": Tensor|None}`.

  - `get_current_estimator_output(ctx: Any) -> Tensor|None`
    - Returns the last estimator output saved during `compute_dist`.

- Context
  - The rollout context (created by `init_context`) owns:
    - `batch_size`, `device`, optional `conditioning`
    - Optional `carry` (recurrent hidden state)
    - Per-step buffers: `trajectory_log_probs`, `trajectory_estimator_outputs`

## Built-in Adapters

- `DefaultEstimatorAdapter`
  - `is_vectorized = True`
  - No carry. Works with both the Sampler and vectorized probability calculators.
  - In the Sampler, it slices conditioning by `step_mask`, runs the estimator, builds the Distribution, and optionally records artifacts.

- `RecurrentEstimatorAdapter`
  - `is_vectorized = False`
  - Maintains a `carry` in the context (initialized via `estimator.init_carry(batch_size, device)`).
  - In the Sampler, it calls the estimator as `(states_active, ctx.carry) -> (est_out, new_carry)`, stores `new_carry`, builds the Distribution, and optionally records artifacts.

## Vectorized vs Non-Vectorized Probability Paths

Probability calculators (PF/PB for trajectories and transitions) branch on `adapter.is_vectorized` but use the same two adapter calls in both paths:

- `compute_dist(states_active, ctx, step_mask=None or mask)`
- `log_probs(actions_active, dist, ctx, step_mask=None or mask, vectorized=...)`

Key differences:

- Vectorized (fast path)
  - `step_mask=None` and `vectorized=True`.
  - May reuse cached estimator outputs by pre-setting `ctx.current_estimator_output` (e.g., PF with stored `trajectories.estimator_outputs`).
  - `log_probs` returns raw `dist.log_prob(...)` and does not raise on `-inf` (illegal actions can produce `-inf`).

- Non‑Vectorized (per-step path)
  - Uses legacy-accurate boolean masks:
    - PF (trajectories): `~states.is_sink_state[t] & ~actions.is_dummy[t]`
    - PB (trajectories): align actions at `t` with states at `t+1`, using `~states.is_sink_state[t+1] & ~states.is_initial_state[t+1] & ~actions.is_dummy[t] & ~actions.is_exit[t]`, skipping `t==0`.
    - Transitions: one per-batch call with legacy masks.
  - `log_probs` pads back to `(N,)` at inactive rows and raises if any `±inf` remains after masking.

## Integration with the Sampler

The Sampler uses the adapter lifecycle:
- `ctx = adapter.init_context(batch_size, device, conditioning)`
- While some trajectories are active:
  - `(dist, ctx) = adapter.compute_dist(states[step_mask], ctx, step_mask, **policy_kwargs)`
  - Sample actions from `dist`; build actions for the full batch
  - `log_probs = adapter.log_probs(valid_actions_tensor, dist, ctx, step_mask, vectorized=False)` (or `None` if skipping)
  - `adapter.record(ctx, step_mask, sampled_actions=valid_actions_tensor, dist=dist, log_probs=log_probs, save_estimator_outputs=...)`
  - Step the environment forward/backward based on `adapter.is_backward`
- After rollout: `artifacts = adapter.finalize(ctx)` and populate `Trajectories`.

## How to Implement a New Adapter

1) Decide on vectorization:
   - If your estimator maintains a recurrent carry, set `is_vectorized = False` and implement carry management in `init_context` and `compute_dist`.
   - Otherwise set `is_vectorized = True` and follow the default adapter pattern.

2) Implement `init_context(batch_size, device, conditioning)`
   - Save invariants and allocate any adapter-specific state. Initialize empty per-step buffers.

3) Implement `compute_dist(states_active, ctx, step_mask, **policy_kwargs)`
   - Slice `conditioning` by `step_mask` for non‑vectorized calls; use full conditioning when `step_mask=None`.
   - Call your estimator, set `ctx.current_estimator_output`, and return a Distribution via `to_probability_distribution`.

4) Implement `log_probs(actions_active, dist, ctx, step_mask, vectorized=False)`
   - Non‑vectorized: strict inf-check, return a padded `(N,)` tensor.
   - Vectorized: return raw `dist.log_prob(...)` (may include `-inf` for illegal actions).

5) Implement `record(ctx, step_mask, sampled_actions, dist, log_probs, save_estimator_outputs)`
   - Never recompute log-probs here; only store what was provided.
   - When saving estimator outputs, pad to `(N, ...)` using `-inf`.

6) Implement `finalize(ctx)`
   - Stack per-step buffers into `(T, N, ...)` tensors and return a dict of artifacts.

7) Set `is_backward` appropriately so the Sampler chooses forward/backward environment steps.

## Reference: Legacy Implementations

The move to adaptors, while allowing for portentially much more complex forms of estimators, introduces significant complexity into the Sampler and probability calculation logic. The legacy, vectorized implementations of these operations exactly re-implemented in the DefaultEstimatorAdapters and the library is designed to use those paths whenever possible (i.e., using vectorized operations), and we have ensured to exactly match the behaviour of this path when using per-step evaluation (the non-vectorized path). These paths are also tested against the legacy code in `test_probability_calculations.py` to ensure correctness. See the reference for details:

- `utils/prob_calculations.py` (master): [link](https://raw.githubusercontent.com/GFNOrg/torchgfn/refs/heads/master/src/gfn/utils/prob_calculations.py)