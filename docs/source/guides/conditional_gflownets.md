# Conditional GFlowNets

A conditional GFlowNet learns to sample from different distributions depending on an external condition variable. Instead of learning a single distribution `p(x) ∝ R(x)`, it learns a family of distributions `p(x | c) ∝ R(x, c)` parameterized by a condition `c`.

This is useful when:
- The reward function depends on an external parameter (e.g., temperature, task ID)
- You want a single model that interpolates between different target distributions
- The condition represents side information that changes the desirable outputs

## Conditional Estimators

`torchgfn` provides conditional variants of the standard estimators:

### ConditionalDiscretePolicyEstimator

Wraps a neural network that takes both state and condition as input to produce action logits:

```python
from gfn.estimators import ConditionalDiscretePolicyEstimator

pf_estimator = ConditionalDiscretePolicyEstimator(
    module=pf_module,
    n_actions=env.n_actions,
    preprocessor=preprocessor,
    is_backward=False,
)
```

The module must accept concatenated state and condition encodings. A typical architecture uses separate encoders for state and condition, then merges them:

```python
state_encoder = MLP(state_dim, hidden_dim, hidden_dim)
condition_encoder = MLP(condition_dim, hidden_dim, hidden_dim)
trunk = MLP(2 * hidden_dim, hidden_dim, output_dim)
```

### ConditionalLogZEstimator

Estimates the log-partition function as a function of the condition only (not the state):

```python
from gfn.estimators import ConditionalLogZEstimator

logz_estimator = ConditionalLogZEstimator(module=logz_module)
```

This makes sense because Z depends on the reward landscape, which changes with the condition.

### ConditionalScalarEstimator

For DB/SubTB losses, estimates log state-flow conditioned on both state and condition:

```python
from gfn.estimators import ConditionalScalarEstimator

logf_estimator = ConditionalScalarEstimator(
    module=logf_module,
    preprocessor=preprocessor,
)
```

## Sampling with Conditions

Pass conditions when sampling trajectories:

```python
# Sample a batch of conditions
conditions = torch.rand(batch_size, condition_dim, device=device)

trajectories = gflownet.sample_trajectories(
    env, n=batch_size, conditions=conditions
)
```

The conditions are threaded through to all estimators automatically.

## Validation

Validate across a range of condition values to ensure the model generalizes:

```python
for c_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
    conditions = torch.full((n_val, condition_dim), c_value, device=device)
    terminating_states = gflownet.sample_terminating_states(env, n=n_val, conditions=conditions)
    # Compare against env.true_dist(conditions=c_value)
```

Check that the learned distribution matches the target at each condition value, including edge cases (e.g., uniform distribution at one extreme, multimodal at the other).

## Supported Loss Functions

All standard losses (TB, DB, SubTB, FM, ZVar) work with conditional estimators. The GFlowNet classes (`TBGFlowNet`, `DBGFlowNet`, etc.) handle conditions transparently — no changes to the loss computation code are needed.

## Example Environment

`ConditionalHyperGrid` is a built-in environment where the reward function is parameterized by a continuous condition. It provides `true_dist()` and `log_partition()` methods that accept conditions, making it easy to validate the learned conditional distributions.

**See:** `train_conditional.py` for a complete example with five loss variants, condition-dependent validation, and interpolation quality metrics.
