# Creating `gfn` environments

A pointed DAG environment (or GFN environment, or environment for short) is a representation for the pointed DAG. The abstract class [Env](https://github.com/saleml/gfn/blob/master/src/gfn/envs/env.py) specifies the requirements for a valid environment definition. To obtain such a representation, the environment needs to specify the following attributes, properties, or methods:

- The `action_space`. Which should be a `gymnasium.spaces.Discrete` object for discrete environments. The last action should correspond to the exit action.
- The initial state `s_0`, as a `torch.Tensor` of arbitrary dimension.
- (Optional) The sink state `s_f`, as a `torch.Tensor` of the same shape as `s_0`, used to represent complete trajectories only (within a batch of trajectories of different lengths), and never processed by any model. If not specified, it is set to `torch.full_like(s_0, -float('inf'))`.
- The method `make_States_class` that creates a subclass of [States](https://github.com/saleml/gfn/blob/master/src/gfn/containers/states.py). The instances of the resulting class should represent a batch of states of arbitrary shape, which is useful to define a trajectory, or a batch of trajectories. `s_0` and `s_f`, along with a tuple called `state_shape` should be defined as class variables, and the subclass (of `States`) should implement masking methods, that specify which actions are possible, in a discrete environment.
- The methods `maskless_step` and `maskless_backward_step` that specify how an action changes a state (going forward and backward). These functions do not need to handle masking, checking whether actions are allowed, checking whether a state is the sink state, etc... These checks are handled in `Env.step` and `Env.backward_step`
- The `log_reward` function that assigns a nonnegative reward to every terminating state (i.e. state with all $s_f$ as a child in the DAG). If `log_reward` is not implemented, `reward` needs to be.

If the states (as represented in the `States` class) need to be transformed to another format before being processed (by neural networks for example), then the environment should define a `preprocessor` attribute, which should be an instance of the [base preprocessor class](https://github.com/saleml/gfn/blob/master/src/gfn/envs/preprocessors/base.py). If no preprocessor is defined, the states are used as is (actually transformed using  [`IdentityPreprocessor`](https://github.com/saleml/gfn/blob/master/src/gfn/envs/preprocessors/base.py), which transforms the state tensors to `FloatTensor`s). Implementing your own preprocessor requires defining the `preprocess` function, and the `output_shape` attribute, which is a tuple representing the shape of *one* preprocessed state.

Optionally, you can define a static `get_states_indices` method that assigns a unique integer number to each state if the environment allows it, and a `n_states` property that returns an integer representing the number of states (excluding $s_f$) in the environment. `get_terminating_states_indices` can also be implemented and serves the purpose of uniquely identifying terminating states of the environment.

For more details, take a look at [HyperGrid](https://github.com/saleml/gfn/blob/master/src/gfn/envs/hypergrid.py), an environment where all states are terminating states, or at [DiscreteEBM](https://github.com/saleml/gfn/blob/master/src/gfn/envs/discrete_ebm.py), where all trajectories are of the same length but only some states are terminating.