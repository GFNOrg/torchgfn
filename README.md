## Running the code
```bash
git clone https://github.com/saleml/gfn.git
cd gfn
conda create -n gfn python=3.10
conda activate gfn
pip install -r requirements.txt
pip install -e .
python train.py
```

Optionally, for `wandb logging`
```bash
pip install wandb
wandb login
```

For now, only the Trajectory Balance and Detailed Balance losses are implemented. The Trajectory Balance loss has been tested, and the code obtains similar results than those of [The Trajectory Balance paper](https://arxiv.org/pdf/2201.13259.pdf).

To run the code:
```python
python train.py --env HyperGrid --env.ndim 4 --env.height 8 --n_iterations 100000 --parametrization TB 
```

## Example, in a few lines
```python
env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

logZ_tensor = torch.tensor(0.)

logit_PF = LogitPFEstimator(env=env, module_name='NeuralNet')
logit_PB = LogitPBEstimator(env=env, module_name='NeuralNet', torso=logit_PF.module.torso)  # To share parameters between PF and PB
logZ = LogZEstimator(logZ_tensor)

parametrization = TBParametrization(logit_PF, logit_PB, logZ)

actions_sampler = LogitPFActionsSampler(estimator=logit_PF)
trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)

loss_fn = TrajectoryBalance(parametrization=parametrization)

params = [
    {"params": [val for key, val in parametrization.parameters.items() if key != "logZ"],"lr": 0.001},
    {"params": [parametrization.parameters["logZ"]], "lr": 0.1}
]
optimizer = torch.optim.Adam(params)

for i in range(1000):
    trajectories = trajectories_sampler.sample(n_objects=16)
    optimizer.zero_grad()
    loss = loss_fn(trajectories)
    loss.backward()
    optimizer.step()
```


## Contributing
Before the first commit:
```bash
pre-commit install
pre-commit run --all-files
```
Run `pre-commit` after staging, and before committing. Make sure all the tests pass. The codebase uses `black` formatter.


## Defining an environment
A pointed DAG environment (or GFN environment, or environment for short) is a representation for the pointed DAG. The abstract class [Env](gfn/envs/env.py) specifies the requirements for a valid environment definition. To obtain such a representation, the environment needs to specify the following attributes, properties, or methods:
- The `action_space`. Which should be a `gym.spaces.Discrete` object for discrete environments. The last action should correspond to the exit action.
- The initial state `s_0`, as a `torch.Tensor` of arbitrary dimension.
- (Optional) The sink state `s_f`, as a `torch.Tensor` of the same shape as `s_0`, used to represent complete trajectories only (within a batch of trajectories of different lengths), and never processed by any model. If not specified, it is set to `torch.fill_like(s_0, -float('inf'))`.
- The method `make_States_class` that creates a subclass of [States](gfn/containers/states.py). The instances of the resulting class should represent a batch of states of arbitrary shape, which is useful to define a trajectory, or a batch of trajectories. `s_0` and `s_f`, along with a tuple called `state_shape` should be defined as class variables, and the subclass (of `States`) should implement masking methods, that specify which actions are possible, in a discrete environment.
- The methods `maskless_step` and `maskless_backward_step` that specify how an action changes a state (going forward and backward). These functions do not need to handle masking, checking whether actions are allowed, checking whether a state is the sink state, etc... These checks are handled in `Env.step` and `Env.backward_step`
- The `reward` function that assigns a nonnegative reward to every terminating state (i.e. state with all $s_f$ as a child in the DAG).

If the states (as represented in the `States` class) need to be transformed to another format before being processed (by neural networks for example), then the environment should define a `preprocessor` attribute, which should be an instance of the [base preprocessor class](gfn/envs/preprocessors/base.py). If no preprocessor is defined, the states are used as is (actually transformed using  [`IdentityPreprocessor`](gfn/envs/preprocessors/base.py), which transforms the state tensors to `FloatTensor`s). Implementing your own preprocessor requires defining the `preprocess` function, and the `output_shape` attribute, which is a tuple representing the shape of *one* preprocessed state.

Optionally, you can define a static `get_states_indices` method that assigns a unique integer number to each state if the environment allows it, and a `n_states` property that returns an integer representing the number of states (excluding $s_f$) in the environment.

For more details, take a look at [HyperGrid](gfn/envs/hypergrid.py).


## Estimators and Modules
Training GFlowNets requires one or multiple estimators. As of now, only discrete environments are handled. All estimators are subclasses of [DiscreteFunctionEstimator](gfn/estimators.py), implementing a `__call__` function that takes as input a batch of [States](gfn/containers/states.py). 
- [LogEdgeFlowEstimator](gfn/estimators.py). It outputs a `(*batch_shape, n_actions)` tensor representing $\log F(s \rightarrow s')$, including when $s' = s_f$.
- [LogStateFlowEstimator](gfn/estimators.py). It outputs a `(*batch_shape, 1)` tensor representing $\log F(s)$.
- [LogitPFEstimator](gfn/estimators.py). It outputs a `(*batch_shape, n_actions)` tensor representing $logit(s' \mid s)$, such that $P_F(s' \mid s) = softmax_{s'}\ logit(s' \mid s)$, including when $s' = s_f$.
- [LogitPBEstimator](gfn/estimators.py). It outputs a `(*batch_shape, n_actions - 1)` tensor representing $logit(s' \mid s)$, such that $P_B(s' \mid s) = softmax_{s'}\ logit(s' \mid s)$.

These estimators require a [module](gfn/modules.py). Modules inherit from the [GFNModule](gfn/modules.py) class, which can be seen as an extension of `torch.nn.Module`.

Said differently, a `States` object is first transformed via the environment's preprocessor to a `(*batch_shape, *output_shape)` float tensor


Additionally, a [LogZEstimaor](gfn/estimators.py) is provided, which is a container for a scalar tensor representing $\log Z$, the log-partition function, useful for the Trajectory Balance loss for example. Each module takes a preprocessed tensor as input, and outputs a tensor of logits, or log flows

## Action Samplers