# Diffusion GFlowNets

GFlowNets can be applied to continuous diffusion processes, where trajectories are sequences of noisy states evolving under stochastic differential equations. `torchgfn` provides environments, estimators, and modules for diffusion-based sampling.

## Overview

In a diffusion GFlowNet, the forward policy defines a stochastic process that transforms noise into samples from a target distribution. The backward policy defines the reverse process. Training enforces that the forward and backward processes are consistent with the target density.

This connects GFlowNets to diffusion models and score-based generative modeling, but with the GFlowNet objective (reward-proportional sampling) rather than the standard denoising objective.

## Key Components

### DiffusionSampling Environment

**Class:** `DiffusionSampling`

The environment for continuous diffusion tasks. It defines the target distribution, the time discretization, and the state space.

```python
from gfn.gym import DiffusionSampling

env = DiffusionSampling(target=target_distribution, n_steps=100, device=device)
```

### Pinned Brownian Motion Estimators

The forward and backward processes are parameterized as perturbations of a reference process (pinned Brownian motion):

- **`PinnedBrownianMotionForward`** — wraps a neural network that predicts the score function (gradient of log-density) for the forward process
- **`PinnedBrownianMotionBackward`** — wraps a module for the backward process, which can be fixed (analytical) or learned

```python
from gfn.estimators import PinnedBrownianMotionForward, PinnedBrownianMotionBackward

pf = PinnedBrownianMotionForward(module=forward_net, env=env)
pb = PinnedBrownianMotionBackward(module=backward_net, env=env)
```

### Neural Network Modules

- **`DiffusionPISGradNetForward`** — score network for the forward process, with configurable time embeddings, harmonics, and learned variance
- **`DiffusionPISGradNetBackward`** — learned backward score network (for RTB)
- **`DiffusionFixedBackwardModule`** — analytical backward process (Brownian bridge), no learning required

## Training Approaches

### Standard TB Training

Train a forward policy from scratch using Trajectory Balance:

```python
gflownet = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0)

for iteration in range(n_iterations):
    trajectories = gflownet.sample_trajectories(env, n=batch_size)
    training_samples = gflownet.to_training_samples(trajectories)
    loss = gflownet.loss(env, training_samples)
    loss.backward()
    optimizer.step()
```

**See:** `train_diffusion_sampler.py`.

### Two-Stage Prior→Posterior with RTB

A more advanced approach that first pre-trains a prior via maximum likelihood, then fine-tunes to a posterior using Relative Trajectory Balance:

1. **Stage 1 (Prior):** Train with `MLEDiffusion` to learn a baseline generative model
2. **Stage 2 (Posterior):** Fine-tune with `RelativeTrajectoryBalanceGFlowNet`, using the frozen prior as a reference

```python
from gfn.gflownet import RelativeTrajectoryBalanceGFlowNet

gflownet = RelativeTrajectoryBalanceGFlowNet(
    pf=pf_posterior,
    pb=pb,
    pf_prior=pf_prior,  # Frozen, no gradients
)
```

The prior policy provides a stable baseline, and RTB adjusts the posterior to match the target. This can be more stable than training from scratch.

**See:** `train_diffusion_rtb.py` (complete two-stage pipeline with checkpoint management).

## Exploration in Diffusion GFlowNets

Continuous diffusion benefits from exploration variance scheduling:

```python
trajectories = gflownet.sample_trajectories(
    env, n=batch_size, exploration_std=current_std
)
```

A typical schedule decays `exploration_std` from a high initial value to zero over training, balancing early exploration with later exploitation.

**See:** `train_diffusion_rtb.py` (warm-down schedule for `exploration_std`).

## Evaluation

Evaluate by sampling terminal states and comparing against the known target distribution:

```python
terminating_states = gflownet.sample_terminating_states(env, n=n_eval)
# Compare with env.target.cached_sample() or env.target.visualize()
```

For 2D targets, `viz_2d_slice` provides contour plots with sample overlays.

Bidirectional evaluation (forward→backward and backward→forward trajectory consistency) provides a stronger diagnostic than terminal state comparison alone.

**See:** `train_diffusion_sampler.py` (bidirectional evaluation with `get_trajectory_pfs_and_pbs`).
