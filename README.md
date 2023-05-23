<p align="center">
    <a>
	    <img src='https://img.shields.io/badge/python-3.10%2B-blueviolet' alt='Python' />
	</a>
	<a href='https://gfn.readthedocs.io/en/latest/?badge=latest'>
    	<img src='https://readthedocs.org/projects/gfn/badge/?version=latest' alt='Documentation Status' />
	</a>
    <a>
	    <img src='https://img.shields.io/badge/code%20style-black-black' />
	</a>
</p>

</p>
<p align="center">
  <a href="https://gfn.readthedocs.io/en/latest/">Documentation</a> ~ <a href="https://github.com/saleml/gfn">Code</a>
</p>

# gfn: a Python package for GFlowNets



## Installing the packages

The codebase requires python >= 3.10

```bash
git clone https://github.com/saleml/gfn.git
conda create -n gfn python=3.10
conda activate gfn
cd gfn
pip install .
```

Optionally, to run scripts, and for [wandb](https://wandb.ai) logging

```bash
pip install .[scripts]
wandb login
```

## About this repo

This repo serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) related algorithms. It decouples the environment definition, the sampling process, and the parametrization used for the GFN loss.

An example script is provided [here](https://github.com/saleml/gfn/blob/master/scripts/train.py). To run the code, use one of the following:

```bash
python train.py --env HyperGrid --env.ndim 4 --env.height 8 --n_iterations 100000 --loss TB
python train.py --env DiscreteEBM --env.ndim 4 --env.alpha 0.5 --n_iterations 10000 --batch_size 64 --temperature 2.
python train.py --env HyperGrid --env.ndim 2 --env.height 64 --n_iterations 100000 --loss DB --replay_buffer_size 1000 --logit_PB.module_name Uniform --optim sgd --optim.lr 5e-3
python train.py --env HyperGrid --env.ndim 4 --env.height 8 --env.R0 0.01 --loss FM --optim adam --optim.lr 1e-4
```

### Example, in a few lines

(⬇️ This example require the [`tqdm`](https://github.com/tqdm/tqdm) package to run. `pip install tqdm` or install all extra requirements with `pip install .[scripts]`)

```python
import torch
from tqdm import tqdm

from gfn import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.envs import HyperGrid
from gfn.losses import TBParametrization, TrajectoryBalance
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler

if __name__ == "__main__":

    env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

    logit_PF = LogitPFEstimator(env=env, module_name="NeuralNet")
    logit_PB = LogitPBEstimator(
        env=env,
        module_name="NeuralNet",
        torso=logit_PF.module.torso,  # To share parameters between PF and PB
    )
    logZ = LogZEstimator(torch.tensor(0.0))

    parametrization = TBParametrization(logit_PF, logit_PB, logZ)

    actions_sampler = DiscreteActionsSampler(estimator=logit_PF)
    trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)

    loss_fn = TrajectoryBalance(parametrization=parametrization)

    params = [
        {
            "params": [
                val for key, val in parametrization.parameters.items() if "logZ" not in key
            ],
            "lr": 0.001,
        },
        {"params": [val for key, val in parametrization.parameters.items() if "logZ" in key], "lr": 0.1},
    ]
    optimizer = torch.optim.Adam(params)

    for i in (pbar := tqdm(range(1000))):
        trajectories = trajectories_sampler.sample(n_trajectories=16)
        optimizer.zero_grad()
        loss = loss_fn(trajectories)
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            pbar.set_postfix({"loss": loss.item()})

```

## Contributing

Before the first commit:

```bash
pip install .[dev]
pre-commit install
pre-commit run --all-files
```

Run `pre-commit` after staging, and before committing. Make sure all the tests pass (By running `pytest`).
The codebase uses `black` formatter.

To make the docs locally:
```bash
cd docs
make html
open build/html/index.html
```

## Details about the codebase

### Defining an environment

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

### Other containers

Besides the `States` class, other containers of states are available:

- [Transitions](https://github.com/saleml/gfn/blob/master/src/gfn/containers/transitions.py), representing a batch of transitions $s \rightarrow s'$.
- [Trajectories](https://github.com/saleml/gfn/blob/master/src/gfn/containers/trajectories.py), representing a batch of complete trajectories $\tau = s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_n \rightarrow s_f$.

These containers can either be instantiated using a `States` object, or can be initialized as empty containers that can be populated on the fly, allowing the usage of [ReplayBuffer](https://github.com/saleml/gfn/blob/master/src/gfn/containers/replay_buffer.py)s.

They inherit from the base `Container` [class](https://github.com/saleml/gfn/blob/master/src/gfn/containers/base.py), indicating some helpful methods.

In most cases, one needs to sample complete trajectories. From a batch of trajectories, a batch of states and batch of transitions can be defined using `Trajectories.to_transitions()` and `Trajectories.to_states()`. These exclude meaningless transitions and states that were added to the batch of trajectories to allow for efficient batching.

### Estimators and Modules

Training GFlowNets requires one or multiple estimators. As of now, only discrete environments are handled. All estimators are subclasses of [FunctionEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py), implementing a `__call__` function that takes as input a batch of [States](https://github.com/saleml/gfn/blob/master/src/gfn/containers/states.py).

- [LogEdgeFlowEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py). It outputs a `(*batch_shape, n_actions)` tensor representing $\log F(s \rightarrow s')$, including when $s' = s_f$.
- [LogStateFlowEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py). It outputs a `(*batch_shape, 1)` tensor representing $\log F(s)$. When used with `forward_looking=True`, $\log F(s)$ is parametrized as the sum of a function approximator and $\log R(s)$ - which is only possible for environments where all states are terminating.
- [LogitPFEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py). It outputs a `(*batch_shape, n_actions)` tensor representing $logit(s' \mid s)$, such that $P_F(s' \mid s) = softmax_{s'}\ logit(s' \mid s)$, including when $s' = s_f$.
- [LogitPBEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py). It outputs a `(*batch_shape, n_actions - 1)` tensor representing $logit(s' \mid s)$, such that $P_B(s' \mid s) = softmax_{s'}\ logit(s' \mid s)$.

Defining an estimator requires the environment, and a [module](https://github.com/saleml/gfn/blob/master/src/gfn/modules.py) instance. Modules inherit from the [GFNModule](https://github.com/saleml/gfn/blob/master/src/gfn/modules.py) class, which can be seen as an extension of `torch.nn.Module`. Alternatively, a module is created by providing which module type to use (e.g. "NeuralNet" or "Uniform" or "Zero"). A Basic MLP is provided as the [NeuralNet](https://github.com/saleml/gfn/blob/master/src/gfn/modules.py) class, but any function approximator should be possible.

Said differently, a `States` object is first transformed via the environment's preprocessor to a `(*batch_shape, *output_shape)` float tensor. The preprocessor's output shape should match the module input shape (if any). The preprocessed states are then passed as inputs to the module, returning the desired output (either flows or probabilities over children in the DAG).

Each module has a `named_parameters` functions that returns a dictionary of the learnable parameters. This attribute is transferred to the corresponding estimator.

Additionally, a [LogZEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py) is provided, which is a container for a scalar tensor representing $\log Z$, the log-partition function, useful for the Trajectory Balance loss for example. This estimator also has a `named_parameters` function.

### Samplers

An [ActionsSampler](https://github.com/saleml/gfn/blob/master/src/gfn/samplers/actions_samplers.py) object defines how actions are sampled at each state of the DAG. As of now, only [DiscreteActionsSampler](https://github.com/saleml/gfn/blob/master/src/gfn/samplers/actions_samplers.py)s are implemented. The require an estimator (of $P_F$, $P_B$, or edge flows) defining the action probabilities. These estimators can contain any type of modules (including random action sampling for example). A [BackwardDiscreteActionsSampler](https://github.com/saleml/gfn/blob/master/src/gfn/samplers/actions_samplers.py) class is provided to sample parents of a state, which is helpful to sample trajectories starting from their last states.

They are at the core of [TrajectoriesSampler](https://github.com/saleml/gfn/blob/master/src/gfn/samplers/trajectories_sampler.py)s, which implements the `sample_trajectories` method, that sample a batch of trajectories starting from a given set of initial states or starting from $s_0$.

### Losses

GFlowNets can be trained with different losses, each of which requires a different parametrization. A parametrization is a dataclass, which can be seen as a container of different estimators. Each parametrization defines a distribution over trajectories, via the `parametrization.Pi` method, and a distribution over terminating states, via the `parametrization.P_T` method. Both distributions should be instances of the classes defined [here](https://github.com/saleml/gfn/blob/master/src/gfn/distributions.py).

The base classes for losses and parametrizations are provided [here](https://github.com/saleml/gfn/blob/master/src/gfn/losses/base.py).

Currently, the implemented losses are:

- Flow Matching
- Detailed Balance
- Trajectory Balance
- Sub-Trajectory Balance. By default, each sub-trajectory is weighted geometrically (within the trajectory) depending on its length. This corresponds to the strategy defined [here](https://www.semanticscholar.org/reader/f2c32fe3f7f3e2e9d36d833e32ec55fc93f900f5). Other strategies exist and are implemented [here](https://github.com/saleml/gfn/blob/master/src/gfn/losses/sub_trajectory_balance.py).
- Log Partition Variance loss. Introduced [here](https://arxiv.org/abs/2302.05446)

### Solving for the flows using Dynamic Programming

A simple script that propagates trajectories rewards through the DAG to define edge flows in a deterministic way (by visiting each edge once only) is provided [here](https://github.com/saleml/gfn/blob/master/scripts/dynamic_programming.py). Do not use the script on large environments !
