Creating Environments
===================

Quick Start Guide
----------------
To create a basic discrete environment, you need to:

1. Inherit from DiscreteEnv
2. Implement the following required methods:

   - ``__init__()``: Initialize environment parameters
   - ``step()``: Define how actions modify states going forward
   - ``backward_step()``: Define how actions modify states going backward
   - ``update_masks()``: Define which actions are valid in each state
   - ``reward()`` or ``log_reward()``: Define the reward function

3. Optionally implement:

   - ``make_random_states()``: For random state initialization

For a complete working example, see the HyperGrid environment in ``src/gfn/gym/hypergrid.py``.
This serves as a reference implementation showing how to properly implement all required methods.

Advanced Usage
-------------
For more complex environments, you can:

1. Customize action representation with ``action_shape``, ``dummy_action``, ``exit_action``
2. Implement state enumeration methods for exact calculations
3. Add environment-specific helper methods

Environment Examples
------------------
The library includes several example environments showcasing different features:

- ``Line``: A continuous environment modeling a mixture of Gaussians. Shows:
   - Continuous state and action spaces
   - Custom reward functions based on probability distributions
   - State tracking with step counters

- ``Box``: A continuous environment with complex dynamics. Demonstrates:
   - Custom action validation logic
   - Complex probability distributions for policies
   - Advanced state transitions

- ``HyperGrid``: A N-dimensional grid environment. Shows:
   - State enumeration for exact calculations
   - Parameterized reward functions

- ``DiscreteEBM``: Energy-based model environment. Features:
   - Complex reward functions
   - Advanced state enumeration

- ``GraphBuilding``: Graph-based environment using ``GraphEnv``. Illustrates:
   - Graph state representation using ``torch_geometric``
   - Complex action spaces for node/edge operations
   - Dynamic state validation

- ``RingGraphBuilding``: Specialized graph environment for generating ring structures. Shows:
   - Inheritance and specialization of ``GraphBuilding``
   - Discrete action space for graph operations
   - Custom validation for ring topology
   - Support for both directed and undirected graphs

When to Use Advanced Features
---------------------------

1. **State Enumeration**: Implement when:
   - Your state space is finite and enumerable
   - You need exact calculations of partition functions
   - Example: ``DiscreteEBM``'s state indexing

2. **Graph-Based States**: Use ``GraphEnv`` when:
   - States are naturally represented as graphs
   - You need to handle variable-sized states
   - Example: ``GraphBuilding`` environment

3. **Custom Action Spaces**: Consider when:
   - Actions have complex structure
   - You need special action validation
   - Example: ``Box``'s continuous action space with constraints 