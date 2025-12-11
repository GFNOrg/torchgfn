# States, Actions, and Containers

## States

States are the primitive building blocks for GFlowNet objects such as transitions and trajectories, on which losses operate.

An abstract `States` class is provided. But for each environment, a `States` subclass is needed. A `States` object
is a collection of multiple states (nodes of the DAG). A tensor representation of the states is required for batching. If a state is represented with a tensor of shape `(*state_shape)`, a batch of states is represented with a `States` object, with the attribute `tensor` of shape `(*batch_shape, *state_shape)`. Other
representations are possible (e.g. a state as a string, a `numpy` array, a graph, etc...), but these representations cannot be batched, unless the user specifies a function that transforms these raw states to tensors.

The `batch_shape` attribute is required to keep track of the batch dimension. A trajectory can be represented by a States object with `batch_shape = (n_states,)`. Multiple trajectories can be represented by a States object with `batch_shape = (n_states, n_trajectories)`.

Because multiple trajectories can have different lengths, batching requires appending a dummy tensor to trajectories that are shorter than the longest trajectory. The dummy state is the $s_f$ attribute of the environment (e.g. `[-1, ..., -1]`, or `[-inf, ..., -inf]`, etc...). Which is never processed, and is used to pad the batch of states only.

For discrete environments, the action set is represented with the set $\{0, \dots, n_{actions} - 1\}$, where the $(n_{actions})$-th action always corresponds to the exit or terminate action, i.e. that results in a transition of the type $s \rightarrow s_f$, but not all actions are possible at all states. For discrete environments, each `States` object is endowed with two extra attributes: `forward_masks` and `backward_masks`, representing which actions are allowed at each state and which actions could have led to each state, respectively. Such states are instances of the `DiscreteStates` abstract subclass of `States`. The `forward_masks` tensor is of shape `(*batch_shape, n_{actions})`, and `backward_masks` is of shape `(*batch_shape, n_{actions} - 1)`. Each subclass of `DiscreteStates` needs to implement the `update_masks` function, that uses the environment's logic to define the two tensors.

### Debug guards and factory signatures

To keep compiled hot paths fast, `States`/`DiscreteStates`/`GraphStates` expect a `debug` flag passed at construction time. When `debug=False` (default) no Python-side checks run in hot paths; when `debug=True`, shape/device/type guards run to catch silent bugs. Environments carry an env-level `debug` and pass it when they instantiate `States`.

When defining your own `States` subclass or environment factories, make sure all state factories accept `debug`:

- Constructors: `__init__(..., debug: bool = False, ...)` should store `self.debug` and pass it along when cloning or slicing.
- Factory classmethods: `make_random_states`, `make_initial_states`, `make_sink_states` (and any overrides) **must** accept `debug` (or `**kwargs`) and forward it to `States(...)`. The base class enforces this and will raise a clear `TypeError` otherwise.
- Env helpers: if you override `states_from_tensor` or `reset` in an environment, thread `self.debug` into state construction so all emitted states share the env-level setting.

This pattern avoids graph breaks in `torch.compile` by letting you keep `debug=False` in compiled runs while still enabling strong checks in development and tests.

## Actions

Actions should be though of as internal actions of an agent building a compositional object. They correspond to transitions $s \rightarrow s'$. An abstract `Actions` class is provided. It is automatically subclassed for discrete environments, but needs to be manually subclassed otherwise.

Similar to `States` objects, each action is a tensor of shape `(*batch_shape, *action_shape)`. For discrete environments for instances, `action_shape = (1,)`, representing an integer between $0$ and $n_{actions} - 1$.

Additionally, each subclass needs to define two more class variable tensors:
- `dummy_action`: A tensor that is padded to sequences of actions in the shorter trajectories of a batch of trajectories. It is `[-1]` for discrete environments.
- `exit_action`: A tensor that corresponds to the termination action. It is `[n_{actions} - 1]` fo discrete environments.

## Containers

Containers are collections of `States`, along with other information, such as reward values, or densities $p(s' \mid s)$. Three containers are available:

- [Transitions](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/containers/transitions.py), representing a batch of transitions $s \rightarrow s'$.
- [Trajectories](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/containers/trajectories.py), representing a batch of complete trajectories $\tau = s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_n \rightarrow s_f$.
- [StatesContainer](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/containers/states_container.py), representing a batch of states, particularly useful for flow-matching GFlowNet.

These containers can either be instantiated using a `States` object, or can be initialized as empty containers that can be populated on the fly, allowing the usage of the [ReplayBuffer](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/containers/replay_buffer.py) class.

They inherit from the base `Container` [class](https://github.com/gfnorg/torchgfn/tree/master/src/gfn/containers/base.py), indicating some helpful methods.

In most cases, one needs to sample complete trajectories. From a batch of trajectories, various training samples can be generated:
- Use `Trajectories.to_transitions()` and `Trajectories.to_states()` for edge-decomposable or state-decomposable losses
- Use `Trajectories.to_state_pairs()` for flow matching losses
- Use `GFlowNet.loss_from_trajectories()` as a convenience method that handles the conversion internally

These methods exclude meaningless transitions and dummy states that were added to the batch of trajectories to allow for efficient batching.
