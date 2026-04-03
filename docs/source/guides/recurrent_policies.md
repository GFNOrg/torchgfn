# Recurrent and Non-Autoregressive Policies

The default policy estimators in `torchgfn` are feedforward: they process each state independently with no memory of previous steps. For sequential generation tasks where context from prior steps is useful, `torchgfn` provides recurrent policy estimators. Separately, non-autoregressive environments allow the same terminal state to be reached via multiple action orderings.

## Recurrent Policies

### RecurrentDiscretePolicyEstimator

**Class:** `RecurrentDiscretePolicyEstimator`

Wraps a recurrent neural network (LSTM or GRU) that maintains a hidden state ("carry") across trajectory steps. This allows the policy to condition on the full history of actions taken so far.

```python
from gfn.estimators import RecurrentDiscretePolicyEstimator
from gfn.modules import RecurrentDiscreteSequenceModel

model = RecurrentDiscreteSequenceModel(
    input_dim=env.preprocessor.output_dim,
    output_dim=env.n_actions,
    hidden_dim=64,
    rnn_type="lstm",  # or "gru"
)

pf_estimator = RecurrentDiscretePolicyEstimator(module=model, n_actions=env.n_actions)
```

### Carry Management

The recurrent hidden state is managed automatically by the `RecurrentPolicyMixin`:

- **`init_carry(batch_size)`** — called by the sampler at the start of each rollout to allocate the initial hidden state
- The carry is updated at each step and stored in the `RolloutContext`
- When computing log-probabilities for existing trajectories (e.g., during off-policy loss), the carry is re-threaded step-by-step

### Tree DAG Simplification

For environments where the DAG is a tree (each terminal state is reachable by exactly one path), the backward policy is uniform and constant. In this case, you can skip the backward policy entirely:

```python
gflownet = TBGFlowNet(pf=pf_estimator, pb=None, init_logZ=0.0, constant_pb=True)
```

Setting `pb=None` and `constant_pb=True` tells the loss function that the backward probabilities are uniform, avoiding the need to learn or evaluate a backward model.

**See:** `train_bitsequence_recurrent.py` (LSTM policy on bit sequences with tree DAG).

## Non-Autoregressive Generation

### NonAutoregressiveBitSequence

In autoregressive environments, actions are applied in a fixed order (e.g., left-to-right). Non-autoregressive environments allow the same terminal state to be reached by filling in positions in any order, creating a richer DAG structure.

The `NonAutoregressiveBitSequence` environment demonstrates this: each action specifies both a position and a value (`action = position * n_words + word`), and the action mask prevents filling already-occupied positions.

This has implications for training:
- The DAG has more paths to each terminal state, providing more training signal per terminal state
- The backward policy must account for multiple valid predecessors
- Action masking is essential to prevent invalid actions

**See:** `train_bitsequence_non_autoregressive.py`.

## When to Use Recurrent Policies

Recurrent policies are most useful when:
- The optimal action depends on the sequence of previous actions, not just the current state
- The state representation doesn't fully capture the relevant history
- You're generating sequences where long-range dependencies matter

For most environments (grids, graphs, continuous spaces), feedforward policies with a good state representation are sufficient and faster to train. Recurrent policies add overhead from sequential carry threading and are harder to parallelize.

## Comparing Approaches

| Approach | Policy type | DAG structure | Best for |
|----------|------------|---------------|----------|
| Feedforward (default) | `DiscretePolicyEstimator` | Any | Most environments |
| Recurrent | `RecurrentDiscretePolicyEstimator` | Any | Sequential generation with history dependence |
| Non-autoregressive | `DiscretePolicyEstimator` | Richer (multiple paths) | Order-invariant generation |
| Tree DAG + no PB | Any PF, `pb=None` | Tree only | Simplification when backward policy is uniform |
