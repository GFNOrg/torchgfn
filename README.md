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

## About this repo

This repo serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) (GFN) related algorithms. It decouples the environment definition, the sampling process, and the parametrization of the function approximators used to calculate the GFN loss. It aims to accompany researchers and engineers in learning about GFlowNets, and in developing new algorithms.

Currently, the library is shipped with three environments: two discrete environments (Discrete Energy Based Model and Hyper Grid) and a continuous box environment. The library is designed to allow users to define their own environments. See [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md) for more details.

### Getting Started with example scripts & notebooks

See [this example](docs/example.md) for a concise example of how to train a simple gflownet.

[Example scripts and notebooks](https://github.com/saleml/torchgfn/tree/master/tutorials/examples) for the included environments are provided. Where indicated, these scripts are intended to reproduce published results.

## Contributing

Please see the [Contributing Guidelines](.github/CONTRIBUTING.md).

## Components of the Library

+ [Defining Environments](docs/markdown/defining_environments.md): For most applications of `torchgfn`, the main challenge will be to define a stateless environment which will produce a valid sampler.
+ [States, Actions, & Containers](docs/markdown/states_actions_containers.md): The two core elements of `torchgfn` are the concepts of states, which are emitted by stateless environments, and actions, which transform states through the the environment logic. These are encapsulated with metadata in containers.
+ [Modules, Estimators, & Samplers](docs/markdown/modules_estimators_samplers.md): The components of a GFlowNet policy (those which select actions given a state), are encapsulated in Estimators. If the policy is a neural network, that logic is captures in a Module. This estimator can then be used with a Sampler to produce states (in the form of trajectories or transitions).
+ [Losses](docs/markdown/losses.md): Each type of `GFlowNet` has a specific parameterization related to a specific loss.

## Advanced Usage: Extending `torchgfn` with Custom GFlowNets

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
- Discrete Energy Based Model: A simple environment for learning energy-based distributions
- Box Environment: A continuous environment for sampling from distributions in bounded spaces
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
