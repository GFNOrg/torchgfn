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

Adapters conform to an abstract class structure (see `gfn/samplers.py`):

- Properties
  - `is_backward: bool` — whether the wrapped estimator is a backward policy.
  - `is_vectorized: bool` — whether the adapter supports vectorized probability calculations (no carry). Vectorized adapters always use the faster legacy vectorized paths in probability calculators. Non-vectorized adapters (e.g., recurrent) use per-step paths with masking and alignment identical to the legacy reference.

- Methods
  - `init_context(batch_size: int, device: torch.device, conditioning: Tensor|None) -> Any`
    - Allocates a rollout context once per batch (Sampler). Stores invariants (batch size, device, optional conditioning) and initializes any adapter state (e.g., recurrent carry) along with per-step artifact buffers.

  - `compute(states_active: States, ctx: Any, step_mask: Tensor, **policy_kwargs) -> (Distribution, Any)`
    - Runs the estimator forward on the active rows and returns a torch Distribution over actions.
    - Must handle conditioning slicing with `step_mask` when applicable.

  - `record(ctx: Any, step_mask: Tensor, sampled_actions: Tensor, dist: Distribution, save_logprobs: bool, save_estimator_outputs: bool) -> None`
    - Optionally record per-step artifacts into buffers owned by the context (e.g., log-probs, estimator outputs). Padding back to batch size happens here, using zeros for log-probs and `-inf` for estimator outputs to match existing conventions.

  - `log_prob_of_actions(states_active: States, actions_active: Tensor, ctx: Any, step_mask: Tensor, **policy_kwargs) -> (Tensor, Any)`
    - Computes log-probs for a batch of (state, action) pairs corresponding to `True` entries of `step_mask` and returns a padded `(N,)` vector.

  - `finalize(ctx) -> -> dict[str, Optional[torch.Tensor]]`
    - Realizes the buffers of the context object into tensors which can be used by the rest of the library (e.g., Trajectories objects).

  - `get_current_estimator_output(ctx: Any) -> Tensor|None`
    - Convenience to expose the last estimator output after `compute`.

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

Probability calculators (PF/PB for trajectories and transitions) branch on `adapter.is_vectorized`:

- Vectorized (fast path)
  - Used when `adapter is None` or `adapter.is_vectorized is True`.
  - Implements the legacy vectorized logic exactly (see the reference implementation below).
  - No adapter calls are needed; the estimator is called on vectorized masks, and distributions compute `log_prob` over the masked actions. This path is the most efficient and is used during training when possible.

- Non-Vectorized (per-step path)
  - Used when `adapter.is_vectorized is False` (e.g., recurrent adapters).
  - The calculators iterate per-step with legacy-accurate masks and alignment:
    - PF (trajectories): `step_mask = ~states.is_sink_state[t] & ~actions.is_dummy[t]`
    - PB (trajectories): align actions at time `t` with states at time `t+1`, and use `step_mask = ~states.is_sink_state[t+1] & ~states.is_initial_state[t+1] & ~actions.is_dummy[t] & ~actions.is_exit[t]` and skip `t==0`.
    - Transitions: use the same masks as the legacy vectorized functions and make a single adapter call per batch.
  - No mask indexing with action ids is used; masking is solely via the legacy boolean masks and the Distribution handles illegal actions internally.

In both branches, behavior matches the legacy reference exactly, so tests compare outputs between vectorized and non-vectorized paths for parity.

## Integration with the Sampler

The Sampler uses the adapter lifecycle:
- `ctx = adapter.init_context(batch_size, device, conditioning)`
- While some trajectories are active:
  - `(dist, ctx) = adapter.compute(states[step_mask], ctx, step_mask, **policy_kwargs)`
  - Sample actions from `dist`; build actions for the full batch
  - `adapter.record(ctx, step_mask, sampled_actions, dist, save_logprobs, save_estimator_outputs)`
  - Step the environment forward/backward based on `adapter.is_backward`
- After rollout: `artifacts = adapter.finalize(ctx)` and populate `Trajectories`.

## How to Implement a New Adapter

A new Adapter will only likely need changes to `compute`, `record`, and `log_prob_of_actions`. You can rely otherwise on the defaults. However we detail all of the steps below for completeness:

1) Decide if your estimator needs a recurrent carry - some persistent state or cache that is utilized throughout the trajectory.
   - If yes, set `is_vectorized = False` and implement `init_context` to initialize `carry`. Implement `compute` to update `carry` each step.
   - If no, set `is_vectorized = True` and follow the default adapter pattern.

2) Implement `compute`
   - Handle conditioning slicing with `step_mask` when conditioning is provided.
   - Call your estimator and construct a torch Distribution via `to_probability_distribution(states_active, est_out, **policy_kwargs)`.

3) Implement `record`
   - If you want to capture per-step log-probs and/or estimator outputs, compute them for active rows and pad back to `(N,)` (log-probs) or `(N, ...)` (estimator outputs) before appending to the context buffers.

4) Implement `log_prob_of_actions`
   - Given `(states_active, actions_active)` for the active rows, compute the Distribution (reusing the same forward logic) and return a padded `(N,)` vector of `log_prob`.
   - Do not modify masks here; calculators pass in `step_mask` already built from existing masks.

5) Implement `finalize`
   - Given the contents of your context, return the trajectory-level objects required by the Sampler.

5) Mark `is_backward` if your estimator is a backward policy; the sampler will step the environment backward accordingly.

6) Performance Guidance
   - For vectorized adapters, prefer the vectorized probability path (legacy implementation). It’s much faster and avoids per-step overhead.
   - For non-vectorized adapters, keep per-step code minimal and avoid Python-side loops that can be vectorized.

## Reference: Legacy Implementations

The move to adaptors, while allowing for portentially much more complex forms of estimators, introduces significant complexity into the Sampler and probability calculation logic. The legacy, vectorized implementations of these operations exactly re-implemented in the DefaultEstimatorAdapters and the library is designed to use those paths whenever possible (i.e., using vectorized operations), and we have ensured to exactly match the behaviour of this path when using per-step evaluation (the non-vectorized path). These paths are also tested against the legacy code in `test_probability_calculations.py` to ensure correctness. See the reference for details:

- `utils/prob_calculations.py` (master): [link](https://raw.githubusercontent.com/GFNOrg/torchgfn/refs/heads/master/src/gfn/utils/prob_calculations.py)