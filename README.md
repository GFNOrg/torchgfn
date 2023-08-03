<p align="center">
    <a>
	    <img src='https://img.shields.io/badge/python-3.10%2B-blueviolet' alt='Python' />
	</a>
	<a href='https://torchgfn.readthedocs.io/en/latest/?badge=latest'>
    	<img src='https://readthedocs.org/projects/torchgfn/badge/?version=latest' alt='Documentation Status' />
	</a>
    <a>
	    <img src='https://img.shields.io/badge/code%20style-black-black' />
	</a>
</p>

</p>
<p align="center">
  <a href="https://torchgfn.readthedocs.io/en/latest/">Documentation</a> ~ <a href="https://github.com/saleml/torchgfn">Code</a> ~ <a href="https://arxiv.org/abs/2305.14594">Paper</a>
</p>

# torchgfn: a Python package for GFlowNets

<p align="center"> Please cite <a href="https://arxiv.org/abs/2305.14594">this paper</a> if you are using the library for your research </p>

## Installing the package

The codebase requires python >= 3.10

To install the latest stable version:

```bash
pip install torchgfn
```

Optionally, to run scripts:

```bash
pip install torchgfn[scripts]
```

To install the cutting edge version (from the `main` branch):

```bash
git clone https://github.com/saleml/torchgfn.git
conda create -n gfn python=3.10
conda activate gfn
cd torchgfn
pip install .
```


## About this repo

This repo serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) related algorithms. It decouples the environment definition, the sampling process, and the parametrization of the function approximators used to calculate the GFN loss.

Example scripts and notebooks are provided [here](https://github.com/saleml/torchgfn/tree/master/tutorials/).


### Standalone example

This example, which shows how to use the library for a simple discrete environment, requires [`tqdm`](https://github.com/tqdm/tqdm) package to run. Use `pip install tqdm` or install all extra requirements with `pip install .[scripts]` or `pip install torchgfn[scripts]`.

```python
import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet

if __name__ == "__main__":

    env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions
    )
    module_PB = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        torso=module_PF.torso
    )

    pf_estimator = DiscretePolicyEstimator(env, module_PF, forward=True)
    pb_estimator = DiscretePolicyEstimator(env, module_PB, forward=False)

    gfn = TBGFlowNet(init_logZ=0., pf=pf_estimator, pb=pb_estimator)

    sampler = Sampler(estimator=pf_estimator))

    # Policy parameters have their own LR.
    non_logz_params = [v for k, v in dict(gfn.named_parameters()).items() if k != "logZ"]
    optimizer = torch.optim.Adam(non_logz_params, lr=1e-3)

    # Log Z gets dedicated learning rate (typically higher).
    logz_params = [dict(gfn.named_parameters())["logZ"]]
    optimizer.add_param_group({"params": logz_params, "lr": 1e-2})

    for i in (pbar := tqdm(range(1000))):
        trajectories = sampler.sample_trajectories(n_trajectories=16)
        optimizer.zero_grad()
        loss = gfn.loss(trajectories)
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

See [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md)

### States

States are the primitive building blocks for GFlowNet objects such as transitions and trajectories, on which losses operate.

An abstract `States` class is provided. But for each environment, a `States` subclass is needed. A `States` object
is a collection of multiple states (nodes of the DAG). A tensor representation of the states is required for batching. If a state is represented with a tensor of shape `(*state_shape)`, a batch of states is represented with a `States` object, with the attribute `tensor` of shape `(*batch_shape, *state_shape)`. Other
representations are possible (e.g. a state as a string, a `numpy` array, a graph, etc...), but these representations cannot be batched, unless the user specifies a function that transforms these raw states to tensors.

The `batch_shape` attribute is required to keep track of the batch dimension. A trajectory can be represented by a States object with `batch_shape = (n_states,)`. Multiple trajectories can be represented by a States object with `batch_shape = (n_states, n_trajectories)`.

Because multiple trajectories can have different lengths, batching requires appending a dummy tensor to trajectories that are shorter than the longest trajectory. The dummy state is the $s_f$ attribute of the environment (e.g. `[-1, ..., -1]`, or `[-inf, ..., -inf]`, etc...). Which is never processed, and is used to pad the batch of states only.

For discrete environments, the action set is represented with the set $\{0, \dots, n_{actions} - 1\}$, where the $(n_{actions})$-th action always corresponds to the exit or terminate action, i.e. that results in a transition of the type $s \rightarrow s_f$, but not all actions are possible at all states. Each `States` object is endowed with two extra attributes: `forward_masks` and `backward_masks`, representing which actions are allowed at each state and which actions could have led to each state, respectively. Such states are instances of the `DiscreteStates` abstract subclass of `States`. The `forward_masks` tensor is of shape `(*batch_shape, n_{actions})`, and `backward_masks` is of shape `(*batch_shape, n_{actions} - 1)`. Each subclass of `DiscreteStates` needs to implement the `update_masks` function, that uses the environment's logic to define the two tensors.

### Actions
Actions should be though of as internal actions of an agent building a compositional object. They correspond to transitions $s \rightarrow s'$. An abstract `Actions` class is provided. It is automatically subclassed for discrete environments, but needs to be manually subclassed otherwise.

Similar to `States` objects, each action is a tensor of shape `(*batch_shape, *action_shape)`. For discrete environments for instances, `action_shape = (1,)`, representing an integer between $0$ and $n_{actions} - 1$.

Additionally, each subclass needs to define two more class variable tensors:
- `dummy_action`: A tensor that is padded to sequences of actions in the shorter trajectories of a batch of trajectories. It is `[-1]` for discrete environments.
- `exit_action`: A tensor that corresponds to the termination action. It is `[n_{actions} - 1]` fo discrete environments.

### Containers

Containers are collections of `States`, along with other information, such as reward values, or densities $p(s' \mid s)$. Two containers are available:

- [Transitions](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/transitions.py), representing a batch of transitions $s \rightarrow s'$.
- [Trajectories](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/trajectories.py), representing a batch of complete trajectories $\tau = s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_n \rightarrow s_f$.

These containers can either be instantiated using a `States` object, or can be initialized as empty containers that can be populated on the fly, allowing the usage of the[ReplayBuffer](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/replay_buffer.py) class.

They inherit from the base `Container` [class](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/base.py), indicating some helpful methods.

In most cases, one needs to sample complete trajectories. From a batch of trajectories, a batch of states and batch of transitions can be defined using `Trajectories.to_transitions()` and `Trajectories.to_states()`. These exclude meaningless transitions and dummy states that were added to the batch of trajectories to allow for efficient batching.

### Modules

Training GFlowNets requires one or multiple estimators, called `GFNModule`s, which is an abstract subclass of `torch.nn.Module`. In addition to the usual `forward` function, `GFNModule`s need to implement a `required_output_dim` attribute, to ensure that the outputs have the required dimension for the task at hand; and some (but not all) need to implement a `to_probability_distribution` function. They take the environment `env` as an input at initialization.
- `DiscretePolicyEstimator` is a `GFNModule` that defines the policies $P_F(. \mid s)$ and $P_B(. \mid s)$ for discrete environments. When `backward=False`, the required output dimension is `n = env.n_actions`, and when `backward=True`, it is `n = env.n_actions - 1`. These `n` numbers represent the logits of a Categorical distribution. Additionally, they include exploration parameters, in order to define a tempered version of $P_F$, or a mixture of $P_F$ with a uniform distribution. Naturally, before defining the Categorical distributions, forbidden actions (that are encoded in the `DiscreteStates`' masks attributes), are given 0 probability by setting the corresponding logit to $-\infty$.
- `ScalarModule` is a simple module with required output dimension 1. It is useful to define log-state flows $\log F(s)$.

For non-discrete environments, the user needs to specify their own policies $P_F$ and $P_B$. The module, taking as input a batch of states (as a `States`) object, should return the batched parameters of a `torch.Distribution`. The distribution depends on the environment. The `to_probability_distribution` function handles the conversion of the parameter outputs to an actual batched `Distribution` object, that implements at least the `sample` and `log_prob` functions. An example is provided [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/gym/helpers/box_utils.py), for a square environment in which the forward policy has support either on a quarter disk, or on an arc-circle, such that the angle, and the radius (for the quarter disk part) are scaled samples from a mixture of Beta distributions. The provided example shows an intricate scenario, and it is not expected that user defined environment need this much level of details.

In all `GFNModule`s, note that the input of the `forward` function is a `States` object. Meaning that they first need to be transformed to tensors. However, `states.tensor` does not necessarily include the structure that a neural network can used to generalize. It is common in these scenarios to have a function that transforms these raw tensor states to ones where the structure is clearer, via a `Preprocessor` object, that is part of the environment. More on this [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md). The default preprocessor of an environment is the identity preprocessor. The `forward` pass thus first calls the `preprocessor` attribute of the environment on `States`, before performing any transformation.

For discrete environments, tabular modules are provided, where a lookup table is used instead of a neural network. Additionally, a `UniformPB` module is provided, implementing a uniform backward policy.

### Samplers

A [Sampler](https://github.com/saleml/torchgfn/tree/master/src/gfn/samplers.py) object defines how actions are sampled (`sample_actions()`) at each state, and trajectories  (`sample_trajectories()`), which can sample a batch of trajectories starting from a given set of initial states or starting from $s_0$. It requires a `GFNModule` that implements the `to_probability_distribution` function.


### Losses

GFlowNets can be trained with different losses, each of which requires a different parametrization, which we call in this library a `GFlowNet`. A `GFlowNet` is a `GFNModule` that includes one or multiple `GFNModules`, at least one of which implements a `to_probability_distribution` function. They also need to implement a `loss` function, that takes as input either states, transitions, or trajectories, depending on the loss.

Currently, the implemented losses are:

- Flow Matching
- Detailed Balance (and it's modified variant).
- Trajectory Balance
- Sub-Trajectory Balance. By default, each sub-trajectory is weighted geometrically (within the trajectory) depending on its length. This corresponds to the strategy defined [here](https://www.semanticscholar.org/reader/f2c32fe3f7f3e2e9d36d833e32ec55fc93f900f5). Other strategies exist and are implemented [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/losses/sub_trajectory_balance.py).
- Log Partition Variance loss. Introduced [here](https://arxiv.org/abs/2302.05446)

# Scripts
Example scripts are provided [here](https://github.com/saleml/torchgfn/tree/master/tutorials/examples/). They can be used to reproduce published results in the HyperGrid environment, and the Box environment.