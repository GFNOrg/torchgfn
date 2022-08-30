## Running the code
```bash
git clone https://github.com/saleml/gfn.git
cd gfn
conda create -n gfn python=3.10
conda activate gfn
pip install -r requirements.txt
pip install -e .
python train.py
```

Optionally, for `wandb logging`
```bash
pip install wandb
wandb login
```

For now, only the Trajectory Balance and Detailed Balance losses are implemented. The Trajectory Balance loss has been tested, and the code obtains similar results than those of [The Trajectory Balance paper](https://arxiv.org/pdf/2201.13259.pdf).

To run the code:
```python
python train.py --env HyperGrid --env.ndim 4 --env.height 8 --n_iterations 100000 --parametrization TB --parametrization.tied --validate_with_training_examples --validation_samples 200000 --seed 3 --preprocessor KHot
```


## Contributing
Before the first commit:
```bash
pre-commit install
pre-commit run --all-files
```
Run `pre-commit` after staging, and before committing. Make sure all the tests pass. The codebase uses `black` formatter.


## Defining an environment
A pointed DAG environment (or GFN environment, or environment for short) is a representation for the pointed DAG. The abstract class [Env](envs/env.py) specifies the requirements for a valid environment definition. To obtain such a representation, the environment needs to specify the following attributes or properties:
- The number of actions `n_actions`. The last action (`n_actions - 1`) should correspond to the exit action
- The initial state `s_0`, as a `torch.Tensor` of arbitrary dimension
- (Optional) The sink state `s_f`, as a `torch.Tensor` of the same shape as `s_0`, used to represent complete trajectories only. See [States](#states) for more info about `s_f`.
- The method `update_masks` that specifies which actions are possible at each state (going forward and backward).
- The methods `step_no_worry` and `backward_step_no_worry` that specify how an action changes a state (going forward and backward). These functions do not need to handle masking, checking whether actions are allowed, checking whether a state is the sink state, etc... These checks are handled in `Env.step` and `Env.backward_step`
- The `reward` function that assigns a nonnegative reward to every terminating state (i.e. state with all $s_f$ as a child in the DAG)

The environment also specifies how a batch of states is represented. The attribute `env.States` is a class that inherits from [`States`](containers/states.py). This attribute is created automatically using an additional  method any environment inheriting from `Env` is required to implement: `make_random_states_tensor`, that creates random states ($\neq s_f$) according to the input batch shape.

Optionally, you can define a static `get_states_indices` method that assigns a unique integer number to each state if the environment allows it, and a `n_states` property that returns an integer representing the number of states (excluding $s_f$) in the environment.

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
- `make_random_states_tensor` that takes as input a batch shape and returns a random batch of states of the corresponding environment.

The helper function [make_States_class](containers/states.py) makes it possible to define a States class of any environment by specifying a class name, `n_actions`, `s_0`, `make_random_states_tensor`, `update_masks`, and optionally `s_f`. In fact, it is this helper function that is used in the [Env](envs/env.py) abstract class representing a DAG environment.