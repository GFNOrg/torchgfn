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

This repo serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) (GFN) related algorithms. It decouples the environment definition, the sampling process, and the parametrization of the function approximators used to calculate the GFN loss. It aims to accompany researchers and engineers in learning about GFlowNets, and in developing new algorithms.

Currently, the library is shipped with three environments: two discrete environments (Discrete Energy Based Model and Hyper Grid) and a continuous box environment. The library is designed to allow users to define their own environments. See [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md) for more details.

### Scripts and notebooks

Example scripts and notebooks for the three environments are provided [here](https://github.com/saleml/torchgfn/tree/master/tutorials/examples). For the hyper grid and the box environments, the provided scripts are supposed to reproduce published results.


### Standalone example

This example, which shows how to use the library for a simple discrete environment, requires [`tqdm`](https://github.com/tqdm/tqdm) package to run. Use `pip install tqdm` or install all extra requirements with `pip install .[scripts]` or `pip install torchgfn[scripts]`.

```python
import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet  # We use a GFlowNet with the Trajectory Balance (TB) loss
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet  # NeuralNet is a simple multi-layer perceptron (MLP)

if __name__ == "__main__":

    # 1 - We define the environment

    env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

    # 2 - We define the needed modules (neural networks)

    # The environment has a preprocessor attribute, which is used to preprocess the state before feeding it to the policy estimator
    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions
    )  # Neural network for the forward policy, with as many outputs as there are actions
    module_PB = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        torso=module_PF.torso  # We share all the parameters of P_F and P_B, except for the last layer
    )

    # 3 - We define the estimators

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)

    # 4 - We define the GFlowNet

    gfn = TBGFlowNet(init_logZ=0., pf=pf_estimator, pb=pb_estimator)  # We initialize logZ to 0

    # 5 - We define the sampler and the optimizer

    sampler = Sampler(estimator=pf_estimator)  # We use an on-policy sampler, based on the forward policy

    # Policy parameters have their own LR.
    non_logz_params = [v for k, v in dict(gfn.named_parameters()).items() if k != "logZ"]
    optimizer = torch.optim.Adam(non_logz_params, lr=1e-3)

    # Log Z gets dedicated learning rate (typically higher).
    logz_params = [dict(gfn.named_parameters())["logZ"]]
    optimizer.add_param_group({"params": logz_params, "lr": 1e-1})

    # 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration

    for i in (pbar := tqdm(range(1000))):
        trajectories = sampler.sample_trajectories(env=env, n_trajectories=16)
        optimizer.zero_grad()
        loss = gfn.loss(env, trajectories)
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

Run `pre-commit` after staging, and before committing. Make sure all the tests pass (By running `pytest`). Note that the `pytest` hook of `pre-commit` only runs the tests in the `testing/` folder. To run all the tests, which take longer, run `pytest` manually.
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

For discrete environments, the action set is represented with the set $\{0, \dots, n_{actions} - 1\}$, where the $(n_{actions})$-th action always corresponds to the exit or terminate action, i.e. that results in a transition of the type $s \rightarrow s_f$, but not all actions are possible at all states. For discrete environments, each `States` object is endowed with two extra attributes: `forward_masks` and `backward_masks`, representing which actions are allowed at each state and which actions could have led to each state, respectively. Such states are instances of the `DiscreteStates` abstract subclass of `States`. The `forward_masks` tensor is of shape `(*batch_shape, n_{actions})`, and `backward_masks` is of shape `(*batch_shape, n_{actions} - 1)`. Each subclass of `DiscreteStates` needs to implement the `update_masks` function, that uses the environment's logic to define the two tensors.

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

These containers can either be instantiated using a `States` object, or can be initialized as empty containers that can be populated on the fly, allowing the usage of the [ReplayBuffer](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/replay_buffer.py) class.

They inherit from the base `Container` [class](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/base.py), indicating some helpful methods.

In most cases, one needs to sample complete trajectories. From a batch of trajectories, a batch of states and batch of transitions can be defined using `Trajectories.to_transitions()` and `Trajectories.to_states()`, in order to train GFlowNets with losses that are edge-decomposable or state-decomposable.  These exclude meaningless transitions and dummy states that were added to the batch of trajectories to allow for efficient batching.

### Modules

Training GFlowNets requires one or multiple estimators, called `GFNModule`s, which is an abstract subclass of `torch.nn.Module`. In addition to the usual `forward` function, `GFNModule`s need to implement a `required_output_dim` attribute, to ensure that the outputs have the required dimension for the task at hand; and some (but not all) need to implement a `to_probability_distribution` function.

- `DiscretePolicyEstimator` is a `GFNModule` that defines the policies $P_F(. \mid s)$ and $P_B(. \mid s)$ for discrete environments. When `is_backward=False`, the required output dimension is `n = env.n_actions`, and when `is_backward=True`, it is `n = env.n_actions - 1`. These `n` numbers represent the logits of a Categorical distribution. The corresponding `to_probability_distribution` function transforms the logits by masking illegal actions (according to the forward or backward masks), then return a Categorical distribution. The masking is done by setting the corresponding logit to $-\infty$. The function also includes exploration parameters, in order to define a tempered version of $P_F$, or a mixture of $P_F$ with a uniform distribution. `DiscretePolicyEstimator`` with `is_backward=False`` can be used to represent log-edge-flow estimators $\log F(s \rightarrow s')$.
- `ScalarModule` is a simple module with required output dimension 1. It is useful to define log-state flows $\log F(s)$.

For non-discrete environments, the user needs to specify their own policies $P_F$ and $P_B$. The module, taking as input a batch of states (as a `States`) object, should return the batched parameters of a `torch.Distribution`. The distribution depends on the environment. The `to_probability_distribution` function handles the conversion of the parameter outputs to an actual batched `Distribution` object, that implements at least the `sample` and `log_prob` functions. An example is provided [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/gym/helpers/box_utils.py), for a square environment in which the forward policy has support either on a quarter disk, or on an arc-circle, such that the angle, and the radius (for the quarter disk part) are scaled samples from a mixture of Beta distributions. The provided example shows an intricate scenario, and it is not expected that user defined environment need this much level of details.

In all `GFNModule`s, note that the input of the `forward` function is a `States` object. Meaning that they first need to be transformed to tensors. However, `states.tensor` does not necessarily include the structure that a neural network can used to generalize. It is common in these scenarios to have a function that transforms these raw tensor states to ones where the structure is clearer, via a `Preprocessor` object, that is part of the environment. More on this [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md). The default preprocessor of an environment is the identity preprocessor. The `forward` pass thus first calls the `preprocessor` attribute of the environment on `States`, before performing any transformation. The `preprocessor` is thus an attribute of the module. If it is not explicitly defined, it is set to the identity preprocessor.

For discrete environments, a `Tabular` module is provided, where a lookup table is used instead of a neural network. Additionally, a `UniformPB` module is provided, implementing a uniform backward policy. These modules are provided [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/utils/modules.py).

### Samplers

A [Sampler](https://github.com/saleml/torchgfn/tree/master/src/gfn/samplers.py) object defines how actions are sampled (`sample_actions()`) at each state, and trajectories  (`sample_trajectories()`), which can sample a batch of trajectories starting from a given set of initial states or starting from $s_0$. It requires a `GFNModule` that implements the `to_probability_distribution` function. For off-policy sampling, the parameters of `to_probability_distribution` can be directly passed when initializing the `Sampler`.


### Losses

GFlowNets can be trained with different losses, each of which requires a different parametrization, which we call in this library a `GFlowNet`. A `GFlowNet` is a `GFNModule` that includes one or multiple `GFNModule`s, at least one of which implements a `to_probability_distribution` function. They also need to implement a `loss` function, that takes as input either states, transitions, or trajectories, depending on the loss.

Currently, the implemented losses are:

- Flow Matching
- Detailed Balance (and it's modified variant).
- Trajectory Balance
- Sub-Trajectory Balance. By default, each sub-trajectory is weighted geometrically (within the trajectory) depending on its length. This corresponds to the strategy defined [here](https://www.semanticscholar.org/reader/f2c32fe3f7f3e2e9d36d833e32ec55fc93f900f5). Other strategies exist and are implemented [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/losses/sub_trajectory_balance.py).
- Log Partition Variance loss. Introduced [here](https://arxiv.org/abs/2302.05446)
