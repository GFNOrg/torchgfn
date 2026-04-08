# Common Operations

This guide covers practical recipes for common tasks when working with `torchgfn`. Each section is self-contained — jump to whatever you need.

## Evaluating a Trained GFlowNet

For discrete environments that implement `true_dist()`, use `env.validate()` to compare the learned distribution against the ground truth:

```python
metrics, terminating_states = env.validate(
    gflownet=gfn,
    n_validation_samples=1000,
)
print(f"L1 distance: {metrics['l1_dist']:.4f}")
print(f"Jensen-Shannon divergence: {metrics['jsd']:.4f}")
if "logZ_diff" in metrics:
    print(f"|logZ_learned - logZ_true|: {metrics['logZ_diff']:.4f}")
```

The returned `terminating_states` are the freshly sampled terminal states used for validation.

**Tip:** For environments with many terminal states, increase `n_validation_samples` to reduce noise. The method will warn you if the sample count is too low relative to the state space.

**See:** `train_hypergrid.py` (periodic validation inside training loop).

## Sampling from a Trained Model

To generate samples (e.g., for inference after training), use the sampler in a no-grad context:

```python
with torch.no_grad():
    trajectories = sampler.sample_trajectories(env=env, n=100)

# Access the terminal states
final_states = trajectories.last_states

# Access the full state sequences
all_states = trajectories.states

# Get the log-rewards of terminal states
log_rewards = trajectories.log_rewards
```

For conditional GFlowNets, pass a condition tensor:

```python
condition = torch.tensor([1.0, 0.0])  # Example condition
conditions = condition.unsqueeze(0).repeat(100, 1)  # Batch it

with torch.no_grad():
    trajectories = sampler.sample_trajectories(
        env=env, n=100, conditioning=conditions
    )
```

## Saving and Loading Checkpoints

Save all learnable components (estimators, optimizer, training state):

```python
torch.save(
    {
        "pf_state_dict": gfn.pf.state_dict(),
        "pb_state_dict": gfn.pb.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": iteration,
    },
    "checkpoint.pt",
)
```

For losses that use `logZ` (like TB):

```python
torch.save(
    {
        "pf_state_dict": gfn.pf.state_dict(),
        "pb_state_dict": gfn.pb.state_dict(),
        "logZ": gfn.logZ,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": iteration,
    },
    "checkpoint.pt",
)
```

Load a checkpoint:

```python
ckpt = torch.load("checkpoint.pt", map_location=device)
gfn.pf.load_state_dict(ckpt["pf_state_dict"])
gfn.pb.load_state_dict(ckpt["pb_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
start_step = ckpt["step"]
```

**See:** `train_diffusion_rtb.py` (checkpoint save/load in a two-stage training pipeline).

## Setting Up Optimizer Parameter Groups

Most GFlowNets benefit from different learning rates for different components. The GFlowNet classes provide helper methods for this:

```python
# TB GFlowNet: separate LR for policies vs logZ
optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})
```

```python
# DB/SubTB GFlowNet: add a third group for logF
optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
optimizer.add_param_group({"params": gfn.logF_parameters(), "lr": 1e-2})
```

You can also pass everything in a single call:

```python
optimizer = torch.optim.Adam([
    {"params": gfn.pf_pb_parameters(), "lr": 1e-3},
    {"params": gfn.logz_parameters(), "lr": 1e-1},
])
```

## Monitoring Gradient and Parameter Norms

Use the utilities in `gfn.utils.training` for training diagnostics:

```python
from gfn.utils.training import grad_norm, param_norm

loss.backward()
print(f"Gradient norm: {grad_norm(gfn.parameters()):.4f}")
print(f"Parameter norm: {param_norm(gfn.parameters()):.4f}")
optimizer.step()
```

For per-group learning-rate-to-gradient ratios:

```python
from gfn.utils.training import lr_grad_ratio

loss.backward()
ratios = lr_grad_ratio(optimizer)
# ratios[i] = (lr * ||grad||) / ||params|| for each param group
```

## Integrating with Weights & Biases

```python
import wandb

wandb.init(project="my-gfn-project", config={"lr": 1e-3, "loss": "tb", ...})

for iteration in range(n_iterations):
    trajectories = sampler.sample_trajectories(env=env, n=batch_size)
    loss = gfn.loss(env, trajectories)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if iteration % log_every == 0:
        metrics, _ = env.validate(gflownet=gfn, n_validation_samples=1000)
        wandb.log({"loss": loss.item(), **metrics}, step=iteration)
```

**See:** `train_hypergrid.py` (full W&B integration with `--wandb_project`).

## Using `torch.compile` for Performance

Compile the loss function for faster training:

```python
compiled_loss = torch.compile(gfn.loss, mode="reduce-overhead")

for iteration in range(n_iterations):
    trajectories = sampler.sample_trajectories(env=env, n=batch_size)
    loss = compiled_loss(env, trajectories)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Note:** The first few iterations will be slower due to compilation overhead. Not all loss functions are equally compatible — TB compiles well, while DB and SubTB may have limitations due to dynamic control flow.

**See:** `train_with_compile.py` (systematic benchmark across losses and compile modes).

## Sharing Parameters Between Forward and Backward Policies

A common pattern is sharing the "trunk" (hidden layers) between PF and PB while keeping separate output heads:

```python
module_PF = MLP(input_dim=input_dim, output_dim=env.n_actions)
module_PB = MLP(
    input_dim=input_dim,
    output_dim=env.n_actions - 1,
    trunk=module_PF.trunk,  # Share all layers except the last
)
```

This reduces the total number of parameters and often improves training stability, since PF and PB learn shared representations of the state space.

## Converting External Data to Trajectories

If you have expert demonstrations or externally generated data as raw tensors, convert them to `Trajectories` objects:

```python
from gfn.utils.training import states_actions_tns_to_traj

# states_tns: shape [traj_len, *state_shape]
# actions_tns: shape [traj_len - 1]  (discrete action indices)
trajectory = states_actions_tns_to_traj(states_tns, actions_tns, env)

# Add to a replay buffer for training
replay_buffer.add(trajectory)
```

This is useful for warm-starting training with known good trajectories.

## Warm-Up Training from a Replay Buffer

Pre-train a GFlowNet on buffered data (e.g., expert demonstrations) before the main training loop:

```python
from gfn.utils.training import warm_up

gfn = warm_up(
    replay_buf=replay_buffer,
    optimizer=optimizer,
    gflownet=gfn,
    env=env,
    n_epochs=100,
    batch_size=32,
)
```

## Using a Uniform Backward Policy

For simpler setups where you don't want to learn PB:

```python
from gfn.utils.modules import UniformPB

module_PB = UniformPB(n_actions=env.n_actions)
pb_estimator = DiscretePolicyEstimator(
    module_PB, env.n_actions, is_backward=True, preprocessor=preprocessor
)
```

## Running Validation Periodically During Training

A typical pattern for monitoring convergence:

```python
for iteration in range(n_iterations):
    trajectories = sampler.sample_trajectories(env=env, n=batch_size, save_logprobs=True)
    optimizer.zero_grad()
    loss = gfn.loss(env, trajectories)
    loss.backward()
    optimizer.step()

    if iteration % 250 == 0:
        metrics, states = env.validate(gflownet=gfn, n_validation_samples=1000)
        print(
            f"Iteration {iteration}: loss={loss.item():.4f}, "
            f"L1={metrics['l1_dist']:.4f}, JSD={metrics['jsd']:.6f}"
        )
```

## Disabling Debug Assertions for Speed

`torchgfn` includes runtime shape and validity checks guarded by Python's `__debug__` flag. These are helpful during development but add overhead in production. Disable them by running Python in optimized mode:

```bash
python -O train.py
```

This skips all `assert` statements and code guarded by `if __debug__:`.

**See:** `tutorials/misc/performance_tuning.py` (benchmarks with and without debug mode).
