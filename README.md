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

The codebase requires `python >= 3.10`.

```bash
git clone https://github.com/saleml/gfn.git
conda create -n gfn python=3.10
conda activate gfn
cd gfn
pip install .
```

To install a version of the codebase that supports [wandb](https://wandb.ai) logging,

```bash
pip install .[scripts]
```

## About this repo

This repo serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) related algorithms. It decouples the environment definition, the sampling process, and the parametrization used for the GFN loss.

Example script are provided [here](./examples/scripts/).


### Standalone example

This example, which shows how to use the library for a simple discrete environment, requires [`tqdm`](https://github.com/tqdm/tqdm) package to run. Use `pip install tqdm` or install all extra requirements with `pip install .[scripts]`.

```python
import torch
from tqdm import tqdm

from gfn.estimators import LogZEstimator
from gfn.losses import TBParametrization
from gfn.samplers import ActionsSampler, TrajectoriesSampler
from gfn.utils import NeuralNet, DiscretePFEstimator, DiscretePBEstimator


from gfn.gym import HyperGrid

if __name__ == "__main__":

    env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

    module_PF = NeuralNet(input_dim=env.preprocessor.output_dim, output_dim=env.n_actions)
    module_PB = NeuralNet(input_dim=env.preprocessor.output_dim, output_dim=env.n_actions - 1,     torso=module_PF.torso)

    logit_PF = DiscretePFEstimator(env=env, module=module_PF)
    logit_PB = DiscretePBEstimator(env=env, module=module_PB)
    logZ = LogZEstimator(torch.tensor(0.0))

    parametrization = TBParametrization(logit_PF, logit_PB, logZ)

    actions_sampler = ActionsSampler(estimator=logit_PF)
    trajectories_sampler = TrajectoriesSampler(actions_sampler=actions_sampler)

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
        loss = parametrization.loss(trajectories)
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            pbar.set_postfix({"loss": loss.item()})
```

## Contributing

Before the first commit:

```bash
pip install -e .[dev,scripts]
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

See [examples](./examples/envs/)

### States

**TODO**

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
- [DiscretePFEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py). It outputs a `(*batch_shape, n_actions)` tensor representing $logit(s' \mid s)$, such that $P_F(s' \mid s) = softmax_{s'}\ logit(s' \mid s)$, including when $s' = s_f$.
- [DiscretePBEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py). It outputs a `(*batch_shape, n_actions - 1)` tensor representing $logit(s' \mid s)$, such that $P_B(s' \mid s) = softmax_{s'}\ logit(s' \mid s)$.

Defining an estimator requires the environment, and a [module](https://github.com/saleml/gfn/blob/master/src/gfn/modules.py) instance. Modules inherit from the [GFNModule](https://github.com/saleml/gfn/blob/master/src/gfn/modules.py) class, which can be seen as an extension of `torch.nn.Module`. Alternatively, a module is created by providing which module type to use (e.g. "NeuralNet" or "Uniform" or "Zero"). A Basic MLP is provided as the [NeuralNet](https://github.com/saleml/gfn/blob/master/src/gfn/modules.py) class, but any function approximator should be possible.

Said differently, a `States` object is first transformed via the environment's preprocessor to a `(*batch_shape, *output_shape)` float tensor. The preprocessor's output shape should match the module input shape (if any). The preprocessed states are then passed as inputs to the module, returning the desired output (either flows or probabilities over children in the DAG).

Each module has a `named_parameters` functions that returns a dictionary of the learnable parameters. This attribute is transferred to the corresponding estimator.

Additionally, a [LogZEstimator](https://github.com/saleml/gfn/blob/master/src/gfn/estimators.py) is provided, which is a container for a scalar tensor representing $\log Z$, the log-partition function, useful for the Trajectory Balance loss for example. This estimator also has a `named_parameters` function.

### Samplers

An [ActionsSampler](https://github.com/saleml/gfn/blob/master/src/gfn/samplers/actions_samplers.py) object defines how actions are sampled at each state of the DAG. As of now, only [ActionsSampler](https://github.com/saleml/gfn/blob/master/src/gfn/samplers/actions_samplers.py)s are implemented. The require an estimator (of $P_F$, $P_B$, or edge flows) defining the action probabilities. These estimators can contain any type of modules (including random action sampling for example). A [ActionsSampler](https://github.com/saleml/gfn/blob/master/src/gfn/samplers/actions_samplers.py) class is provided to sample parents of a state, which is helpful to sample trajectories starting from their last states.

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
