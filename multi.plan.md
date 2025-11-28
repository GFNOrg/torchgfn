<!-- e404f8a9-7b21-4d7c-ac71-239f6831f927 81843692-b79f-4340-9f6d-3a84f9047966 -->
# Plan: Extend Benchmark to Diffusion Sampling

## 1. Refactor scenario management

- Extract existing HyperGrid logic into a reusable `EnvironmentBenchmark` structure (e.g., dataclass with name, color, scenario list, builder function).
- Keep HyperGrid’s current scenarios (baseline / library fast path / script fast path) but register them under the new structure.
- Update the main loop to iterate over environments sequentially, collecting per-env results (including histories) and tagging each record with both env and scenario identifiers.

## 2. Add diffusion sampling environment support

- Review `tutorials/examples/train_diffusion_sampler.py` to reuse its estimator construction (`DiffusionSampling`, `DiffusionPISGradNetForward`, `DiffusionFixedBackwardModule`, `PinnedBrownianMotionForward/Backward`).
- Implement a new `DiffusionEnvConfig` builder under `build_training_components` (or a dedicated helper) that creates the env, forward/backward estimators, optimizer groups, and default hyperparameters mirroring the standalone script.
- Define diffusion-specific scenarios:
- Baseline: standard sampler, no compilation.
- Library Fast Path: use `CompiledChunkSampler` (env already inherits `EnvFastPathMixin`).
- Script Fast Path: implement a local chunked sampler analogous to `ChunkedHyperGridSampler`, but operating on diffusion states/tensors (handle continuous actions, exit padding, dummy actions). Expose it only for diffusion.

## 3. Integrate new sampler/env wiring

- Update `build_training_components` to dispatch based on the environment key (hypergrid vs diffusion) so each path can select the correct preprocessor, estimator modules, sampler type, and optimizer parameter groups.
- Ensure the diffusion path still returns metrics compatible with the existing training loop (needs `validate`?—if not available for diffusion, skip validation or provide a stub message).

## 4. Expand plotting to multi-row layout

- Adjust `plot_benchmark` to group results by environment and create one row per environment (HyperGrid row retains three scenarios; Diffusion row shows its two/three variants).
- Reuse the existing color mapping for GFlowNet variants; introduce per-environment scenario linestyles (or reuse existing names when overlapping).
- Update subplot titles/labels to mention the environment name so viewers can distinguish rows easily.

## 5. Final polish

- Update CLI help text to mention multi-environment benchmarking and any diffusion-specific knobs (e.g., target selection, num steps) if exposed; otherwise, explain defaults in docstring/comments.
- Verify histories are recorded for both environments so the new loss/timing plots aren’t empty.
- Refresh documentation/comments at the top of the script to describe the new diffusion benchmark capability.