# 6. Advanced: Defining a New GFlowNet

To define a new `GFlowNet`, the user needs to define a class which subclasses `GFlowNet`
and implements the following methods:

- `sample_trajectories`: Sample a specific number of complete trajectories.
- `loss`: Compute the loss given the training objects.
- `to_training_samples`: Convert trajectories to training samples.

Based on the type of training samples returned by `to_training_samples`, the user should
define the generic type `TrainingSampleType` when subclassing `GFlowNet`. For example,
if the training sample is an instance of `Trajectories`, the `GFlowNet` class should be
subclassed as `GFlowNet[Trajectories]`. Thus, the class definition should look like this:

```python
class MyGFlowNet(GFlowNet[Trajectories]):
    ...
```

**Example: Flow Matching GFlowNet**

Let's consider the example of the `FMGFlowNet` class, which is a subclass of
`GFlowNet` that implements the Flow Matching GFlowNet. The training samples are
pairs of states managed by the `StatePairs` container:

```python
class FMGFlowNet(GFlowNet[StatePairs[DiscreteStates]]):
    ...

    def to_training_samples(
        self, trajectories: Trajectories
    ) -> StatePairs[DiscreteStates]:
        """Converts a batch of trajectories into a batch of training samples."""
        return trajectories.to_state_pairs()
```

This means that the `loss` method of `FMGFlowNet` will receive a
`StatePairs[DiscreteStates]` object as its training samples argument:

```python
def loss(self, env: DiscreteEnv, states: StatePairs[DiscreteStates]) -> torch.Tensor:
    ...
```

**Adding New Training Sample Types**

If your GFlowNet returns a unique type of training samples, you'll need to
expand the `TrainingSampleType` bound. This ensures type-safety and better code
clarity.

**Implementing Class Methods**

As mentioned earlier, your new GFlowNet must implement the following methods:

- `sample_trajectories`: Sample a specific number of complete trajectories.
- `loss`: Compute the loss given the training objects.
- `to_training_samples`: Convert trajectories to training samples.

These methods are defined in
[`src/gfn/gflownet/base.py`](https://github.com/GFNOrg/torchgfn/blob/master/src/gfn/gflownet/base.py)
and are abstract methods, so they must be implemented in your new GFlowNet. If
your GFlowNet has unique functionality which should be represented as additional
class methods, implement them as required. Remember to document new methods to
ensure other developers understand their purposes and use-cases!

**Testing**

Remember to create unit tests for your new GFlowNet to ensure it works as
intended and integrates seamlessly with other parts of the codebase. This
ensures maintainability and reliability of the code!

