# Creating `torchgfn` GFlowNets

To define a new subclass of `GFlowNet`, the user needs to define a class which subclasses `GFlowNet` and implements the following methods:

- `sample_trajectories`: Sample a specific number of complete trajectories.
- `loss`: Compute the loss given the training objects.
- `to_training_samples`: Convert trajectories to training samples.

Based on the type of training samples returned by `to_training_samples`, the user should define the generic type `TrainingSampleType` when subclassing `GFlowNet`. For example, if the training sample is an instance of `Trajectories`, the `GFlowNet` class should be subclassed as `GFlowNet[Trajectories]`. Thus, the class definition should look like this:

```python
class MyGFlowNet(GFlowNet[Trajectories]):
    ...
```

## Example: Flow Matching GFlowNet

Let's consider the example of the `FMGFlowNet` class, which is a subclass of `GFlowNet` that implements the Flow Matching GFlowNet. The training samples are tuples of discrete states, so the class references the type `Tuple[DiscreteStates, DiscreteStates]` when subclassing `GFlowNet`:

```python
class FMGFlowNet(GFlowNet[Tuple[DiscreteStates, DiscreteStates]]):
    ...

    def to_training_samples(
        self, trajectories: Trajectories
    ) -> tuple[DiscreteStates, DiscreteStates]:
        """Converts a batch of trajectories into a batch of training samples."""
        return trajectories.to_non_initial_intermediary_and_terminating_states()

```

## Adding New Training Sample Types

If your GFlowNet returns a unique type of training samples, you'll need to expand the `TrainingSampleType` bound. This ensures type-safety and better code clarity.

In the earlier example, the `FMGFlowNet` used:

```python
GFlowNet[Tuple[DiscreteStates, DiscreteStates]]
```

This means the method `to_training_samples` should return a tuple of `DiscreteStates`.

If the `to_training_sample` method of your new GFlowNet, for example, returns an `int`, you should expand the `TrainingSampleType` in `src/gfn/gflownet/base.py` to include this type in the `bound` of the `TypeVar`:

Before:

```python
TrainingSampleType = TypeVar(
    "TrainingSampleType", bound=Union[Container, tuple[States, ...]]
)
```

After:

```python
TrainingSampleType = TypeVar(
    "TrainingSampleType", bound=Union[Container, tuple[States, ...], int]
)
```

## Implementing Class Methods

As mentioned earlier, your new GFlowNet must implement the following methods:

- `sample_trajectories`: Sample a specific number of complete trajectories.
- `loss`: Compute the loss given the training objects.
- `to_training_samples`: Convert trajectories to training samples.

These methods are defined in `src/gfn/gflownet/base.py` and are abstract methods, so they must be implemented in your new GFlowNet. If your GFlowNet has unique functionality which should be represented as additional class methods, implement them as required. Remember to document new methods to ensure other developers understand their purposes and use-cases!

## Testing

Remember to create unit tests for your new GFlowNet to ensure it works as intended and integrates seamlessly with other parts of the codebase. This ensures maintainability and reliability of the code!
