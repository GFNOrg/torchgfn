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

The codebase requires python >= 3.10. To install the latest stable version:

```bash
pip install torchgfn
```

Optionally, to run scripts:

```bash
pip install torchgfn[scripts]
```

To install the cutting edge version (from the `main` branch):

```bash
git clone https://github.com/GFNOrg/torchgfn.git
conda create -n gfn python=3.10
conda activate gfn
cd torchgfn
pip install -e ".[all]"
```

## Installing `oneccl` bindings for multinode training.

You can determine the version of `pytorch` installed using the command

```
echo $(python -c $"import torch; print(torch.__version__)")
```

after which you can install the closest matching version from [this table](https://github.com/intel/torch-ccl?tab=readme-ov-file#install-prebuilt-wheel) (otherwise, you must build from source). You can see the specific wheels [here](https://pytorch-extension.intel.com/release-whl/stable/cpu/us/oneccl-bind-pt/).

```
pip install oneccl_bind_pt=={pytorch_version} -f https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

for example, if your pytorch version is `2.0.1+cu117`, you would run `python -m pip install oneccl_bind_pt==2.0.0+cpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/`.


***TODO: Rough instructions - to integrate into docs (just moving them here from email) -
```
# Create & activate conda env.
conda create -n gfn python=3.10
conda activate gfn

# Install the package.
pip install .[scripts]  # Includes `tqdm`.

# We will use torch-ccl library for multinode implementation. The latest torch-ccl is compatible with PyTorch 2.2.0. The above command installs the latest torch. So, we need to uninstall it and install latest torch. If you agree that we can make it the default version, I can update it in pyproject.toml.
pip uninstall torch -y
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch

# Install torch-ccl
git clone https://github.com/intel/torch-ccl.git torch-ccl && cd torch-ccl
git checkout tags/v2.2.0+cpu
git submodule sync
git submodule update --init --recursive

# TODO: this didn't work for me -- I had to use a prebuilt wheel.
# ONECCL_BINDINGS_FOR_PYTORCH_BACKEND=cpu python setup.py install
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

# Installation is complete now.

You can submit a job by modifying one of the slurm scripts and submitting. For example, ddp_gfn.small.8.slurm. Please note that you need to modify the conda env name in the slurm script to the name of your env. Also, change the paths and dimensions if needed. I submit the script using the following command:

sbatch ddp_gfn.small.4.mila.slurm
```


## About this repo

This repo serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) (GFN) related algorithms. It decouples the environment definition, the sampling process, and the parametrization of the function approximators used to calculate the GFN loss. It aims to accompany researchers and engineers in learning about GFlowNets, and in developing new algorithms.

Currently, the library is shipped with three environments: two discrete environments (Discrete Energy Based Model and Hyper Grid) and a continuous box environment. The library is designed to allow users to define their own environments. See [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md) for more details.

### Scripts and notebooks

Example scripts and notebooks for the three environments are provided [here](https://github.com/saleml/torchgfn/tree/master/tutorials/examples). For the hyper grid and the box environments, the provided scripts are supposed to reproduce published results.


### Standalone example

This example, which shows how to use the library for a simple discrete environment, requires [`tqdm`](https://github.com/tqdm/tqdm) package to run. Use `pip install tqdm` or install all extra requirements with `pip install .[scripts]` or `pip install torchgfn[scripts]`. In the first example, we will train a Tarjectory Balance GFlowNet:

```python
import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.preprocessors import KHotPreprocessor
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP  # is a simple multi-layer perceptron (MLP)

# 1 - We define the environment.
env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8
preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)

# 2 - We define the needed modules (neural networks).
module_PF = MLP(
    input_dim=preprocessor.output_dim,
    output_dim=env.n_actions
)  # Neural network for the forward policy, with as many outputs as there are actions

module_PB = MLP(
    input_dim=preprocessor.output_dim,
    output_dim=env.n_actions - 1,
    trunk=module_PF.trunk  # We share all the parameters of P_F and P_B, except for the last layer
)

# 3 - We define the estimators.
pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=preprocessor)
pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=preprocessor)

# 4 - We define the GFlowNet.
gfn = TBGFlowNet(logZ=0., pf=pf_estimator, pb=pb_estimator)  # We initialize logZ to 0

# 5 - We define the sampler and the optimizer.
sampler = Sampler(estimator=pf_estimator)  # We use an on-policy sampler, based on the forward policy

# Different policy parameters can have their own LR.
# Log Z gets dedicated learning rate (typically higher).
optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})

# 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration
for i in (pbar := tqdm(range(1000))):
    trajectories = sampler.sample_trajectories(env=env, n=16, save_logprobs=True)  # The save_logprobs=True makes on-policy training faster
    optimizer.zero_grad()
    loss = gfn.loss(env, trajectories)
    loss.backward()
    optimizer.step()
    if i % 25 == 0:
        pbar.set_postfix({"loss": loss.item()})
```

and in this example, we instead train using Sub Trajectory Balance. You can see we simply assemble our GFlowNet from slightly different building blocks:

```python
import torch
from tqdm import tqdm

from gfn.gflownet import SubTBGFlowNet
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.preprocessors import KHotPreprocessor
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP  # MLP is a simple multi-layer perceptron (MLP)

# 1 - We define the environment.
env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8
preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)

# 2 - We define the needed modules (neural networks).
# The environment has a preprocessor attribute, which is used to preprocess the state before feeding it to the policy estimator
module_PF = MLP(
    input_dim=preprocessor.output_dim,
    output_dim=env.n_actions
)  # Neural network for the forward policy, with as many outputs as there are actions

module_PB = MLP(
    input_dim=preprocessor.output_dim,
    output_dim=env.n_actions - 1,
    trunk=module_PF.trunk  # We share all the parameters of P_F and P_B, except for the last layer
)
module_logF = MLP(
    input_dim=preprocessor.output_dim,
    output_dim=1,  # Important for ScalarEstimators!
)

# 3 - We define the estimators.
pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=preprocessor)
pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=preprocessor)
logF_estimator = ScalarEstimator(module=module_logF, preprocessor=env.preprocessor)

# 4 - We define the GFlowNet.
gfn = SubTBGFlowNet(pf=pf_estimator, pb=pb_estimator, logF=logF_estimator, lamda=0.9)

# 5 - We define the sampler and the optimizer.
sampler = Sampler(estimator=pf_estimator)

# Different policy parameters can have their own LR.
# Log F gets dedicated learning rate (typically higher).
optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
optimizer.add_param_group({"params": gfn.logF_parameters(), "lr": 1e-2})

# 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration
for i in (pbar := tqdm(range(1000))):
    # We are going to sample trajectories off policy, by tempering the distribution.
    # We should not save the sampling logprobs, as we are not using them for training.
    # We should save the estimator outputs to make training faster.
    trajectories = sampler.sample_trajectories(env=env, n=16, save_logprobs=False, save_estimator_outputs=True, temperature=1.5)
    optimizer.zero_grad()
    loss = gfn.loss(env, trajectories)
    loss.backward()
    optimizer.step()
    if i % 25 == 0:
        pbar.set_postfix({"loss": loss.item()})

```

## Contributing

Please see the [Contributing Guidelines](.github/CONTRIBUTING.md).

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

Containers are collections of `States`, along with other information, such as reward values, or densities $p(s' \mid s)$. Three containers are available:

- [Transitions](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/transitions.py), representing a batch of transitions $s \rightarrow s'$.
- [Trajectories](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/trajectories.py), representing a batch of complete trajectories $\tau = s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_n \rightarrow s_f$.
- [StatePairs](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/state_pairs.py), representing pairs of states with optional conditioning, particularly useful for flow matching algorithms.

These containers can either be instantiated using a `States` object, or can be initialized as empty containers that can be populated on the fly, allowing the usage of the [ReplayBuffer](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/replay_buffer.py) class.

They inherit from the base `Container` [class](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/base.py), indicating some helpful methods.

In most cases, one needs to sample complete trajectories. From a batch of trajectories, various training samples can be generated:
- Use `Trajectories.to_transitions()` and `Trajectories.to_states()` for edge-decomposable or state-decomposable losses
- Use `Trajectories.to_state_pairs()` for flow matching losses
- Use `GFlowNet.loss_from_trajectories()` as a convenience method that handles the conversion internally

These methods exclude meaningless transitions and dummy states that were added to the batch of trajectories to allow for efficient batching.

### Modules

Training GFlowNets requires one or multiple estimators, called `GFNModule`s, which is an abstract subclass of `torch.nn.Module`. In addition to the usual `forward` function, `GFNModule`s need to implement a `required_output_dim` attribute, to ensure that the outputs have the required dimension for the task at hand; and some (but not all) need to implement a `to_probability_distribution` function.

- `DiscretePolicyEstimator` is a `GFNModule` that defines the policies $P_F(. \mid s)$ and $P_B(. \mid s)$ for discrete environments. When `is_backward=False`, the required output dimension is `n = env.n_actions`, and when `is_backward=True`, it is `n = env.n_actions - 1`. These `n` numbers represent the logits of a Categorical distribution. The corresponding `to_probability_distribution` function transforms the logits by masking illegal actions (according to the forward or backward masks), then return a Categorical distribution. The masking is done by setting the corresponding logit to $-\infty$. The function also includes exploration parameters, in order to define a tempered version of $P_F$, or a mixture of $P_F$ with a uniform distribution. `DiscretePolicyEstimator` with `is_backward=False` can be used to represent log-edge-flow estimators $\log F(s \rightarrow s')$.
- `ScalarModule` is a simple module with required output dimension 1. It is useful to define log-state flows $\log F(s)$.

For non-discrete environments, the user needs to specify their own policies $P_F$ and $P_B$. The module, taking as input a batch of states (as a `States`) object, should return the batched parameters of a `torch.Distribution`. The distribution depends on the environment. The `to_probability_distribution` function handles the conversion of the parameter outputs to an actual batched `Distribution` object, that implements at least the `sample` and `log_prob` functions. An example is provided [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/gym/helpers/box_utils.py), for a square environment in which the forward policy has support either on a quarter disk, or on an arc-circle, such that the angle, and the radius (for the quarter disk part) are scaled samples from a mixture of Beta distributions. The provided example shows an intricate scenario, and it is not expected that user defined environment need this much level of details.

In general, (and perhaps obviously) the `to_probability_distribution` method is used to calculate a probability distribution from a policy. Therefore, in order to go off-policy, one needs to modify the computations in this method during sampling. One accomplishes this using `policy_kwargs`, a `dict` of kwarg-value pairs which are used by the `Estimator` when calculating the new policy. In the discrete case, where common settings apply, one can see their use in `DiscretePolicyEstimator`'s `to_probability_distribution` method by passing a softmax `temperature`, `sf_bias` (a scalar to subtract from the exit action logit) or `epsilon` which allows for e-greedy style exploration. In the continuous case, it is not possible to foresee the methods used for off-policy exploration (as it depends on the details of the `to_probability_distribution` method, which is not generic for continuous GFNs), so this must be handled by the user, using custom `policy_kwargs`.

In all `GFNModule`s, note that the input of the `forward` function is a `States` object. Meaning that they first need to be transformed to tensors. However, `states.tensor` does not necessarily include the structure that a neural network can used to generalize. It is common in these scenarios to have a function that transforms these raw tensor states to ones where the structure is clearer, via a `Preprocessor` object, that is part of the environment. More on this [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md). The `forward` pass thus first calls the `preprocessor` attribute of the environment on `States`, before performing any transformation. The `preprocessor` is thus an attribute of the module. If it is not explicitly defined, it is set to the identity preprocessor.

For discrete environments, a `Tabular` module is provided, where a lookup table is used instead of a neural network. Additionally, a `UniformPB` module is provided, implementing a uniform backward policy. These modules are provided [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/utils/modules.py).

### Samplers

A [Sampler](https://github.com/saleml/torchgfn/tree/master/src/gfn/samplers.py) object defines how actions are sampled (`sample_actions()`) at each state, and trajectories  (`sample_trajectories()`), which can sample a batch of trajectories starting from a given set of initial states or starting from $s_0$. It requires a `GFNModule` that implements the `to_probability_distribution` function. For simple off-policy sampling (e.g., epsilon-noisy or tempering), you can pass appropriate `policy_kwargs` to the `Sampler` object, which will be used by the `GFNModule`. If you need more complex off-policy sampling, you can subclass the `Sampler` object, and override the `sample_actions` and `sample_trajectories` methods.

Currently, the library provides two samplers:

- Sampler
- LocalSearchSampler (references: [EB-GFN](https://arxiv.org/abs/2202.01361), [LS-GFN](https://arxiv.org/abs/2310.02710))


### Losses

GFlowNets can be trained with different losses, each of which requires a different parametrization, which we call in this library a `GFlowNet`. A `GFlowNet` includes one or multiple `GFNModule`s, at least one of which implements a `to_probability_distribution` function. They also need to implement a `loss` function, that takes as input either states, transitions, or trajectories, depending on the loss.

Currently, the implemented losses are:

- Flow Matching
- Detailed Balance (and it's modified variant).
- Trajectory Balance
- Sub-Trajectory Balance. By default, each sub-trajectory is weighted geometrically (within the trajectory) depending on its length. This corresponds to the strategy defined [here](https://www.semanticscholar.org/reader/f2c32fe3f7f3e2e9d36d833e32ec55fc93f900f5). Other strategies exist and are implemented [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/losses/sub_trajectory_balance.py).
- Log Partition Variance loss. Introduced [here](https://arxiv.org/abs/2302.05446)

### Extending GFlowNets

To define a new `GFlowNet`, the user needs to define a class which subclasses `GFlowNet` and implements the following methods:

- `sample_trajectories`: Sample a specific number of complete trajectories.
- `loss`: Compute the loss given the training objects.
- `to_training_samples`: Convert trajectories to training samples.

Based on the type of training samples returned by `to_training_samples`, the user should define the generic type `TrainingSampleType` when subclassing `GFlowNet`. For example, if the training sample is an instance of `Trajectories`, the `GFlowNet` class should be subclassed as `GFlowNet[Trajectories]`. Thus, the class definition should look like this:

```python
class MyGFlowNet(GFlowNet[Trajectories]):
    ...
```

**Example: Flow Matching GFlowNet**

Let's consider the example of the `FMGFlowNet` class, which is a subclass of `GFlowNet` that implements the Flow Matching GFlowNet. The training samples are pairs of states managed by the `StatePairs` container:

```python
class FMGFlowNet(GFlowNet[StatePairs[DiscreteStates]]):
    ...

    def to_training_samples(
        self, trajectories: Trajectories
    ) -> StatePairs[DiscreteStates]:
        """Converts a batch of trajectories into a batch of training samples."""
        return trajectories.to_state_pairs()
```

This means that the `loss` method of `FMGFlowNet` will receive a `StatePairs[DiscreteStates]` object as its training samples argument:

```python
def loss(self, env: DiscreteEnv, states: StatePairs[DiscreteStates]) -> torch.Tensor:
    ...
```

**Adding New Training Sample Types**

If your GFlowNet returns a unique type of training samples, you'll need to expand the `TrainingSampleType` bound. This ensures type-safety and better code clarity.

**Implementing Class Methods**

As mentioned earlier, your new GFlowNet must implement the following methods:

- `sample_trajectories`: Sample a specific number of complete trajectories.
- `loss`: Compute the loss given the training objects.
- `to_training_samples`: Convert trajectories to training samples.

These methods are defined in `src/gfn/gflownet/base.py` and are abstract methods, so they must be implemented in your new GFlowNet. If your GFlowNet has unique functionality which should be represented as additional class methods, implement them as required. Remember to document new methods to ensure other developers understand their purposes and use-cases!

**Testing**

Remember to create unit tests for your new GFlowNet to ensure it works as intended and integrates seamlessly with other parts of the codebase. This ensures maintainability and reliability of the code!


## Training Examples

The repository includes several example environments and training scripts. Below are three different implementations of training on the HyperGrid environment, which serve as good starting points for understanding GFlowNets:

1. `tutorials/examples/train_hypergrid.py`: The main training script with full features:
   - Multiple loss functions (FM, TB, DB, SubTB, ZVar, ModifiedDB)
   - Weights & Biases integration for experiment tracking
   - Support for replay buffers (including prioritized)
   - Visualization capabilities for 2D environments:
     * True probability distribution
     * Learned probability distribution
     * L1 distance evolution over training
   - Various hyperparameter options
   - Reproduces results from multiple papers (see script docstring)

2. `tutorials/examples/train_hypergrid_simple.py`: A simplified version focused on core concepts:
   - Uses only Trajectory Balance (TB) loss
   - Minimal architecture with shared trunks
   - No extra features (no replay buffer, no wandb)
   - Great starting point for understanding GFlowNets

3. `tutorials/examples/train_hypergrid_simple_ls.py`: Demonstrates advanced sampling strategies:
   - Implements local search sampling
   - Configurable local search parameters
   - Optional Metropolis-Hastings acceptance criterion
   - Shows how to extend basic GFlowNet training with sophisticated sampling

Other environments available in the package include:
- Discrete Energy Based Model (`tutorials/examples/train_discreteebm.py`): A simple environment for learning energy-based distributions
- Box Environment (`tutorials/examples/train_box.py`): A continuous environment for sampling from distributions in bounded spaces
- Ring Environment (`tutorials/examples/train_graph_ring.py`): A simple graph building environment for learning to generate ring graphs
- Bayesian Structure Learning: A graph building environment for learning Bayesian structures ([Deleu et al., 2022](https://arxiv.org/abs/2202.13903))
- Custom environments can be added by following the environment creation guide in `tutorials/ENV.md`

## Usage Examples

To train with Weights & Biases tracking:
```bash
python tutorials/examples/train_hypergrid.py --ndim 4 --height 8 --wandb_project your_project_name
```

To train with visualization (2D environments only):
```bash
python tutorials/examples/train_hypergrid.py --ndim 2 --height 8 --plot
```

To try the simple version with epsilon-greedy exploration:
```bash
python tutorials/examples/train_hypergrid_simple.py --ndim 2 --height 8 --epsilon 0.1
```

To experiment with local search:
```bash
python tutorials/examples/train_hypergrid_simple_ls.py --ndim 2 --height 8 --n_local_search_loops 2 --back_ratio 0.5 --use_metropolis_hastings
```

For more options and configurations, check the help of each script:
```bash
python tutorials/examples/train_hypergrid.py --help
python tutorials/examples/train_hypergrid_simple.py --help
python tutorials/examples/train_hypergrid_simple_ls.py --help
```
