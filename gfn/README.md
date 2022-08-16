## Defining an environment


## States
In this repository, a state is represent as a `torch.Tensor`. The shape of the tensor is arbitrary, but should be consistent across the states of the same environment. For instance, a state can either be a one-dimensional tensor (as in [HyperGrid](envs/hypergrid.py)), or three-dimensional if we consider 2-D RGB images.

The [States](containers/states.py) class represents an abstract base class for a batch of states of any pointed DAG environment. The batch shape is a tuple of arbitrary length. A distinction is made between `state_shape` and `batch_shape`, which are completely independent. The resulting tensor has shape `(*batch_shape, *state_shape)`.

To subclass `States`, 2 class variables are necessary (otherwise a `TypeError` is raised):
- `n_actions`: an integer representing the maximum number of (forward) actions in each state
- `s_0`: The initial state of the pointed DAG.

Additionally, the class variable `s_f` can either be manually instantiated, or inferred from `s_0` (in which case it would a be a tensor of the same shape as `s_0` filled with `-inf`). No computations are ever made on `s_f`, and its sole purpose is to represent complete trajectories.

The class variables `state_shape`, `state_ndim`, and `device` are inferred from `s_0`.

Finally, 2 abstract methods are necessary to finish the definition of a subclass of `States`:
- `update_masks` that overrides a batch of states' `forward_masks` and `backward_masks` attributes, according to the allowed actions in the environment.
- `make_random_states` that takes as input a batch shape and returns a random batch of states of the corresponding environment.

The helper function [make_States_class](containers/states.py) makes it possible to define a States class of any environment by specifying `n_actions`, `s_0`, `make_random_states`, `update_masks`, and optionally `s_f`. In fact, it is this helper function that is used in the [Env](envs/env.py) abstract class representing a DAG environment.