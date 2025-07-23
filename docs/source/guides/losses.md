# Losses

GFlowNets can be trained with different losses, each of which requires a different parametrization, which we call in this library a `GFlowNet`. A `GFlowNet` includes one or multiple `Estimator`s, at least one of which implements a `to_probability_distribution` function. They also need to implement a `loss` function, that takes as input either [`States`, `Transitions`, or `Trajectories` `Container`](guides/states_actions_containers.md) instances, depending on the loss.

Currently, the implemented losses are:

- Flow Matching: This is the original loss, and is mostly made available for completeness. It is slow to compute this loss, and also hard to optimize, so is generally not recommended for any significantly hard to learn problem.
- Detailed Balance (and it's modified variant).
- Trajectory Balance
- Sub-Trajectory Balance. By default, each sub-trajectory is weighted geometrically (within the trajectory) depending on its length. This corresponds to the strategy defined [here](https://www.semanticscholar.org/reader/f2c32fe3f7f3e2e9d36d833e32ec55fc93f900f5). Other strategies exist and are implemented [here](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/losses/sub_trajectory_balance.py).
- Log Partition Variance loss. Introduced [here](https://arxiv.org/abs/2302.05446)
