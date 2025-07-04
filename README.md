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
+ [Advanced Usage: Extending `torchgfn` with Custom GFlowNets](docs/markdown/advanced.md): While `torchgfn` aims to support major usages of GFlowNets, we hope this will also serve as a platform for the community to extend the possible use cases of the technology. This guide details how one can extend
