# Tutorials

1. Learn the building blocks of your GflowNet with [notebooks](https://github.com/gfnorg/torchgfn/tree/master/tutorials/notebooks/)
2. See `torchgfn` in action with example [training scripts](https://github.com/gfnorg/torchgfn/tree/master/tutorials/examples/)
3. Creating your own [Environment](guides/creating_environments.md).

## Training Examples

The repository includes several example environments and training scripts. There are three different implementations of training on the HyperGrid environment, which serve as good starting points for understanding GFlowNets at multiple levels of complexity:

1. [`train_hypergrid_simple.py`](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_hypergrid_simple.py) - Focused on core concepts:
   - Uses only Trajectory Balance (TB) loss.
   - Minimal architecture with shared trunks.
   - No extra features (no replay buffer, no wandb).
   - A starting point for understanding GFlowNets.

2. [`train_hypergrid.py`](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_hypergrid.py) - The main training script with many features:
   - Multiple loss functions (FM, TB, DB, SubTB, ZVar, ModifiedDB).
   - Weights & Biases integration for experiment tracking.
   - Support for replay buffers (including prioritized).
   - Visualization capabilities for 2D environments:
     * True probability distribution.
     * Learned probability distribution.
     * L1 distance evolution over training.
   - Various hyperparameter options.
   - Reproduces results from multiple papers (see script docstring).
   - Scalable, multi node training.

3. [`train_hypergrid_simple_ls.py`](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_hypergrid_simple_ls.py) - Demonstrates advanced sampling strategies:
   - Implements local search sampling with configurable parameters.
   - Optional Metropolis-Hastings acceptance criterion.
   - Shows how to extend basic GFlowNet training with sophisticated sampling.

Other environments available in the package include:

- [Discrete Energy Based Model](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_discreteebm.py): A simple environment for learning energy-based distributions.
- [Box Environment](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_box.py): A continuous environment for sampling from distributions in bounded spaces.
- [Ring Environment](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_graph_ring.py): A simple graph building environment for learning to generate ring graphs.
- Bayesian Structure Learning: A graph building environment for learning Bayesian structures ([Deleu et al., 2022](https://arxiv.org/abs/2202.13903)).


**Custom environments can be added by following the [environment creation guide](guides/creating_environments.md)**.

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
