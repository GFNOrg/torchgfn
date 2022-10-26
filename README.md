## Installing the library
```bash
git clone https://github.com/saleml/gfn.git
cd gfn
pip install -e .
```

Optionally, to run scripts, and for [wandb](https://wandb.ai) logging
```bash
pip install -r requirements.txt
wandb login
```


## About this repo
This library serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) related algorithms. It decouples the environment definition, the sampling process, and the parametrization used for the GFN loss. 

An example script is provided [here](scripts/train.py). To run the code, use one of the following:
```bash
python train.py --env HyperGrid --env.ndim 4 --env.height 8 --n_iterations 100000 --loss TB 
python train.py --env HyperGrid --env.ndim 2 --env.height 64 --n_iterations 100000 --loss DB --replay_buffer_size 1000 --logit_PB.module_name Uniform --optim sgd --optim.lr 5e-3 
python train.py --env HyperGrid --env.ndim 4 --env.height 8 --env.R0 0.01 --loss FM --optim adam --optim.lr 1e-4
```

### Example, in a few lines
```python
env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

logit_PF = LogitPFEstimator(env=env, module_name='NeuralNet')
logit_PB = LogitPBEstimator(env=env, module_name='NeuralNet', torso=logit_PF.module.torso)  # To share parameters between PF and PB
logZ = LogZEstimator(torch.tensor(0.))

parametrization = TBParametrization(logit_PF, logit_PB, logZ)

actions_sampler = DiscreteActionsSampler(estimator=logit_PF)
trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)

loss_fn = TrajectoryBalance(parametrization=parametrization)

params = [
    {"params": [val for key, val in parametrization.parameters.items() if key != "logZ"],"lr": 0.001},
    {"params": [parametrization.parameters["logZ"]], "lr": 0.1}
]
optimizer = torch.optim.Adam(params)

for i in range(1000):
    trajectories = trajectories_sampler.sample(n_trajectories=16)
    optimizer.zero_grad()
    loss = loss_fn(trajectories)
    loss.backward()
    optimizer.step()
```




# Contributing
Before the first commit:
```bash
pip install pre-commit black pytest
pre-commit install
pre-commit run --all-files
```
Run `pre-commit` after staging, and before committing. Make sure all the tests pass (By running `pytest`).
The codebase uses `black` formatter.


# Details about the codebase

## Defining an environment
A pointed DAG environment (or GFN environment, or environment for short) is a representation for the pointed DAG. The abstract class [Env](gfn/envs/env.py) specifies the requirements for a valid environment definition. To obtain such a representation, the environment needs to specify the following attributes, properties, or methods:
- The `action_space`. Which should be a `gym.spaces.Discrete` object for discrete environments. The last action should correspond to the exit action.
- The initial state `s_0`, as a `torch.Tensor` of arbitrary dimension.
- (Optional) The sink state `s_f`, as a `torch.Tensor` of the same shape as `s_0`, used to represent complete trajectories only (within a batch of trajectories of different lengths), and never processed by any model. If not specified, it is set to `torch.full_like(s_0, -float('inf'))`.
- The method `make_States_class` that creates a subclass of [States](gfn/containers/states.py). The instances of the resulting class should represent a batch of states of arbitrary shape, which is useful to define a trajectory, or a batch of trajectories. `s_0` and `s_f`, along with a tuple called `state_shape` should be defined as class variables, and the subclass (of `States`) should implement masking methods, that specify which actions are possible, in a discrete environment.
- The methods `maskless_step` and `maskless_backward_step` that specify how an action changes a state (going forward and backward). These functions do not need to handle masking, checking whether actions are allowed, checking whether a state is the sink state, etc... These checks are handled in `Env.step` and `Env.backward_step`
- The `reward` function that assigns a nonnegative reward to every terminating state (i.e. state with all $s_f$ as a child in the DAG).

If the states (as represented in the `States` class) need to be transformed to another format before being processed (by neural networks for example), then the environment should define a `preprocessor` attribute, which should be an instance of the [base preprocessor class](gfn/envs/preprocessors/base.py). If no preprocessor is defined, the states are used as is (actually transformed using  [`IdentityPreprocessor`](gfn/envs/preprocessors/base.py), which transforms the state tensors to `FloatTensor`s). Implementing your own preprocessor requires defining the `preprocess` function, and the `output_shape` attribute, which is a tuple representing the shape of *one* preprocessed state.

Optionally, you can define a static `get_states_indices` method that assigns a unique integer number to each state if the environment allows it, and a `n_states` property that returns an integer representing the number of states (excluding $s_f$) in the environment.

For more details, take a look at [HyperGrid](gfn/envs/hypergrid.py).

### Other containers
Besides the `States` class, other containers of states are available:
- [Transitions](gfn/containers/transitions.py), representing a batch of transitions $s \rightarrow s'$.
- [Trajectories](gfn/containers/trajectories.py), representing a batch of complete trajectories $\tau = s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_n \rightarrow s_f$.

These containers can either be instantiated using a `States` object, or can be initialized as empty containers that can be populated on the fly, allowing the usage of [ReplayBuffer](gfn/containers/replay_buffer.py)s.

They inherit from the base `Container` [class](gfn/containers/base.py), indicating some helpful methods.

In most cases, one needs to sample complete trajectories. From a batch of trajectories, a batch of states and batch of transitions can be defined using `Trajectories.to_transitions()` and `Trajectories.to_states()`. These exclude meaningless transitions and states that were added to the batch of trajectories to allow for efficient batching.


## Estimators and Modules
Training GFlowNets requires one or multiple estimators. As of now, only discrete environments are handled. All estimators are subclasses of [FunctionEstimator](gfn/estimators.py), implementing a `__call__` function that takes as input a batch of [States](gfn/containers/states.py). 
- [LogEdgeFlowEstimator](gfn/estimators.py). It outputs a `(*batch_shape, n_actions)` tensor representing $\log F(s \rightarrow s')$, including when $s' = s_f$.
- [LogStateFlowEstimator](gfn/estimators.py). It outputs a `(*batch_shape, 1)` tensor representing $\log F(s)$.
- [LogitPFEstimator](gfn/estimators.py). It outputs a `(*batch_shape, n_actions)` tensor representing $logit(s' \mid s)$, such that $P_F(s' \mid s) = softmax_{s'}\ logit(s' \mid s)$, including when $s' = s_f$.
- [LogitPBEstimator](gfn/estimators.py). It outputs a `(*batch_shape, n_actions - 1)` tensor representing $logit(s' \mid s)$, such that $P_B(s' \mid s) = softmax_{s'}\ logit(s' \mid s)$.

Defining an estimator requires the environment, and a [module](gfn/modules.py) instance. Modules inherit from the [GFNModule](gfn/modules.py) class, which can be seen as an extension of `torch.nn.Module`. Alternatively, a module is created by providing which module type to use (e.g. "NeuralNet" or "Uniform" or "Zero"). A Basic MLP is provided as the [NeuralNet](gfn/modules.py) class, but any function approximator should be possible.


Said differently, a `States` object is first transformed via the environment's preprocessor to a `(*batch_shape, *output_shape)` float tensor. The preprocessor's output shape should match the module input shape (if any). The preprocessed states are then passed as inputs to the module, returning the desired output (either flows or probabilities over children in the DAG).

Each module has a `named_parameters` functions that returns a dictionary of the learnable parameters. This attribute is transferred to the corresponding estimator.


Additionally, a [LogZEstimator](gfn/estimators.py) is provided, which is a container for a scalar tensor representing $\log Z$, the log-partition function, useful for the Trajectory Balance loss for example. This estimator also has a `named_parameters` function.

## Samplers
An [ActionsSampler](gfn/samplers/actions_samplers.py) object defines how actions are sampled at each state of the DAG. As of now, only [DiscreteActionsSampler](gfn/samplers/actions_samplers.py)s are implemented. The require an estimator (of $P_F$, $P_B$, or edge flows) defining the action probabilities. These estimators can contain any type of modules (including random action sampling for example). A [BackwardDiscreteActionsSampler](gfn/samplers/actions_samplers.py) class is provided to sample parents of a state, which is helpful to sample trajectories starting from their last states.

They are at the core of [TrajectoriesSampler](gfn/samplers/trajectories_sampler.py)s, which implements the `sample_trajectories` method, that sample a batch of trajectories starting from a given set of initial states or starting from $s_0$.

## Losses
GFlowNets can be trained with different losses, each of which requires a different parametrization. A parametrization is a dataclass, which can be seen as a container of different estimators. Each parametrization defines a distribution over trajectories, via the `parametrization.Pi` method, and a distribution over terminating states, via the `parametrization.P_T` method. Both distributions should be instances of the classes defined [here](gfn/distributions.py).

The base classes for losses and parametrizations are provided [here](gfn/losses/base.py).

Currently, the implemented losses are:
- Flow Matching
- Detailed Balance
- Trajectory Balance
- Sub-Trajectory Balance. By default, each sub-trajectory is weighted geometrically (within the trajectory) depending on its length. This corresponds to the strategy defined [here](https://www.semanticscholar.org/reader/f2c32fe3f7f3e2e9d36d833e32ec55fc93f900f5). Other strategies exist and are implemented [here](gfn/losses/sub_trajectory_balance.py).

## Solving for the flows using Dynamic Programming
A simple script that propagates trajectories rewards through the DAG to define edge flows in a deterministic way (by visiting each edge once only) is provided [here](scripts/dynamic_programming.py).