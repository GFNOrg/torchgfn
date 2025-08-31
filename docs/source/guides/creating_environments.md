# Creating Environments

To create a basic discrete environment, you need to:

1. Inherit from DiscreteEnv.
2. Implement the following required methods:

   - `__init__()`: Initialize environment parameters.
   - `step()`: Define how actions modify states going forward.
   - `backward_step()`: Define how actions modify states going backward.
   - `update_masks()`: Define which actions are valid in each state.
   - `reward()` or `log_reward()`: Define the reward function.

3. Optionally implement:

   - `make_random_states()`: For random state initialization

For a complete working example, see the
[HyperGrid environment in `torchgfn/gym`](https://github.com/GFNOrg/torchgfn/blob/master/src/gfn/gym/hypergrid.py).
This serves as a reference implementation showing how to properly implement all required methods.

## Advanced Usage

For more complex environments, you can:

1. Customize action representation with `action_shape`, `dummy_action`, `exit_action`
2. Implement state enumeration methods for exact calculations
3. Add environment-specific helper methods

## Creating Environments in General (Discrete or Continuous)

To define an environment, the user needs to define the tensor `s0` representing the
initial state $s_0$, and optionally a tensor representing the sink state $s_f$, which
denotes the end of a trajectory (and can be used for padding). If it is not specified,
`sf` is set to a tensor of the same shape as `s0` filled with $-\infty$.

The user must also define the `action_shape`, which may or may not be of
different dimensionality to the `state_shape`. For example, in many environments
a timestamp exists as part of the state to prevent cycles, and actions cannot
(directly) modify this value.

A `dummy_action` and `exit_action` tensor must also be submitted by the user.
The `exit_action` is a unique action that brings the state to $s_f$. The
`dummy_action` should be different from the `exit_action` (and not be a valid
trajectory) action - it's used to pad batched action tensors (after the
exit action). This is useful when trajectories will be of different lengths
within the batch.

In addition, a number of methods must be defined by the user:

+ `env.step(self, states, actions)` accepts a batch of states and actions, and
  returns a batch of `next_states``. This is used for forward trajectories.
+ `env.backward_step(self, states, actions)` accepts a batch of `next_states`
  and actions and returns a batch of `states`. This is used for backward
  trajectories.
      + These functions do not need to handle masking for discrete
        environments, nor checking whether actions are allowed, nor checking
        whether a state is the sink state, etc... These checks are handled in
        `Env._step` and `Env._backward_step` functions, that are not implemented
        by the user.
+ `env.is_action_valid(self, states, actions, backward)`: This function is used
  to ensure all actions are valid for both forward and backward trajectories
  (these are often different sets of rules) for continuous environments. It
  accepts a batch of states and actions, and returning `True` only if all
  actions can be taken at the given states.
+ `env.make_random_states(self, batch_shape, device)` is an **optional** method
  which is consumed by the States class automatically, which is useful if you
  want random samples you can evaluate under your reward model or policy.
+ `env.reset(self, ...)` can also **optionally** be overwritten by the user
  to support custom logic. For example, for conditional GFlowNets, the
  conditioning tensor can be concatenated to $s_0$ automatically here.
+ `env.log_reward(self, final_states)` must be defined, which calculates the
  log reward of the terminating states (i.e. state with all $s_f$ as a child in
  the DAG). It by default returns the log of `env.reward(self, final_states)`,
  which is not implemented. The user can decide to either implement the `reward`
  method, or if it is simpler / more numerically stable, to override the
  `log_reward` method and leave the `reward` unimplemented.

If the environment is discrete, it is an instance of `DiscreteEnv`, and
therefore total number of actions should be specified as an attribute. The
`action_shape` is assumed to be `(1,)`, as the common use case of a
`DiscreteEnv` would be to sample a single action per step. However, this can be
set to any shape by the user (for example `(1,5)` if the policy is sampling 5
independent actions per step).

In addition to the above methods, in the discrete case, you must also define
the following method:

+ `env.update_masks(self, states)`: in discrete environments, the `States` class
  contains state-dependent forward and backward masks, which define allowable
  forward and backward actions conditioned on the state. Note that in
  calculating these masks, the user can leverage the helper methods
  `DiscreteStates.set_nonexit_action_masks`,
  `DiscreteStates.set_exit_masks`, and
  `DiscreteStates.init_forward_masks`.

The code automatically implements the following two class factories, which the
majority of users will not need to overwrite. However, the user could override
these factories to imbue new functionality into the `States` and `Actions` that
interact with the environment:
- The method `make_states_class` that creates the corresponding subclass of [`States`](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/states.py).
For discrete environments, the resulting class should be a subclass of [`DiscreteStates`](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/states.py),
that implements the `update_masks` method specifying which actions are available at each state.
- The method `make_actions_class` that creates a subclass of [`Actions`](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/actions.py),
simply by specifying the required class variables (the shape of an action tensor, the dummy action, and the exit action).
This method is implemented by default for all `DiscreteEnv`s.

The logic of the environment is handled by the methods `step` and `backward_step`, that need to be implemented,
which specify how an action changes a state (going forward and backward).

For `DiscreteEnv`s, the user can define a `get_states_indices` method that
assigns a unique integer number to each state, and a `n_states` property that
returns an integer representing the number of states (excluding $s_f$) in the environment. The function `get_terminating_states_indices` can also be
implemented and serves the purpose of uniquely identifying terminating states of
the environment, which is useful for
[tabular modules](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/utils/modules.py).
Other properties and functions can be implemented as well, such as the
`log_partition` or the `true_dist` properties.

## Environment Examples

The library includes several example environments showcasing different features:

- `Line`: A continuous environment modeling a mixture of Gaussians. Shows:
   - Continuous state and action spaces
   - Custom reward functions based on probability distributions
   - State tracking with step counters

- `Box`: A continuous environment with complex dynamics. Demonstrates:
   - Custom action validation logic
   - Complex probability distributions for policies
   - Advanced state transitions

- `HyperGrid`: A N-dimensional grid environment. Shows:
   - State enumeration for exact calculations
   - Parameterized reward functions

- `DiscreteEBM`: Energy-based model environment. Features:
   - Complex reward functions
   - Advanced state enumeration

- `GraphBuilding`: Graph-based environment using `GraphEnv`. Illustrates:
   - Graph state representation using `torch_geometric`
   - Complex action spaces for node/edge operations
   - Dynamic state validation

- `RingGraphBuilding`: Specialized graph environment for generating ring structures. Shows:
   - Inheritance and specialization of `GraphBuilding`
   - Discrete action space for graph operations
   - Custom validation for ring topology
   - Support for both directed and undirected graphs

## When to Use Advanced Features

1. **State Enumeration**: Implement when:
   - Your state space is finite and enumerable
   - You need exact calculations of partition functions
   - Example: `DiscreteEBM`'s state indexing

2. **Graph-Based States**: Use `GraphEnv` when:
   - States are naturally represented as graphs
   - You need to handle variable-sized states
   - Example: `GraphBuilding` environment

3. **Custom Action Spaces**: Consider when:
   - Actions have complex structure
   - You need special action validation
   - Example: `Box`'s continuous action space with constraints
