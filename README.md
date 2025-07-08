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
  <a href="https://torchgfn.readthedocs.io/en/latest/">Documentation</a> ~ <a href="https://github.com/gfnorg/torchgfn">Code</a> ~ <a href="https://arxiv.org/abs/2305.14594">Paper</a>
</p>

# torchgfn

**A Python Package for GFLowNets.** Please cite [this paper](https://arxiv.org/abs/2305.14594) if you are using the library for your research!

## Installing the package

The codebase requires python >= 3.10. To install the latest stable version with the core dependencies:

```bash
pip install torchgfn
```

`torchgfn` supports installation with multiple sets of dependencies, under the following tags:

- `dev`: dependencies required for development of the core library.
- `scripts`: dependencies needed to run examples in `tutorials/examples/'.
- `all`: everything.

and can be called by running

```bash
pip install torchgfn[scripts]
```

Or to install the latest release (from the `main` branch) with all dependencies in a Conda environment:

```bash
git clone https://github.com/GFNOrg/torchgfn.git
conda create -n gfn python=3.10
conda activate gfn
cd torchgfn
pip install -e ".[all]"
```

## About this repo

This repo serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) (GFN) related algorithms. It decouples the environment definition, the sampling process, and the parametrization of the function approximators used to calculate the GFN loss. It aims to accompany researchers and engineers in learning about GFlowNets, and in developing new algorithms.

The library is shipped with many environments under the `gym`, including discrete environments (e.g., Discrete Energy Based Model, Hyper Grid, Graph Generation), and continuous environments (e.g., Box). The library is designed to allow users to define their own environments relatively easily. See [here](guides/creating_environments.rst) for more details.

### Getting Started with Example Scripts & Notebooks

+ [Simple example](guides/example.md): a concise description of how to train a gflownet using the library.
+ [Tutorials README](tutorials/README.md) for an overview of the included [example scripts and notebooks](https://github.com/gfnorg/torchgfn/tree/master/tutorials/examples) drawn from the included gym environments. Where indicated, these scripts are intended to reproduce published results.

### Contributing

Please see the [Contributing Guidelines](https://github.com/GFNOrg/torchgfn/blob/master/.github/CONTRIBUTING.md).

## Components of the Library

+ [States, Actions, & Containers](https://torchgfn.readthedocs.io/en/latest/guides/states_actions_containers.html): The two core elements of `torchgfn` are the concepts of states, which are emitted by stateless environments, and actions, which transform states through the the environment logic. These are encapsulated with metadata in containers.
+ [Modules, Estimators, & Samplers](https://torchgfn.readthedocs.io/en/latest/guides/modules_estimators_samplers.html): The components of a GFlowNet policy (those which select actions given a state), are encapsulated in Estimators. If the policy is a neural network, that logic is captures in a Module. This estimator can then be used with a Sampler to produce states (in the form of trajectories or transitions).
+ [Losses](https://torchgfn.readthedocs.io/en/latest/guides/losses.html): Each type of `GFlowNet` has a specific parameterization related to a specific loss.
+ [Defining Environments](https://torchgfn.readthedocs.io/en/latest/guides/creating_environments.html): For most applications of `torchgfn`, the main challenge will be to define a stateless environment which will produce a valid sampler.
+ [Advanced Usage: Extending `torchgfn` with Custom GFlowNets](https://torchgfn.readthedocs.io/en/latest/guides/advanced.html): While `torchgfn` aims to support major usages of GFlowNets, we hope this will also serve as a platform for the community to extend the possible use cases of the technology. This guide details how one can extend
+ [Full API Reference](https://torchgfn.readthedocs.io/en/latest/autoapi/index.html): A breakdown of all major components of both the core GFN library, the associated gym, and the included tutorials.
