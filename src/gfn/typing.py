from torchtyping import TensorType
import torch

# Types specific to the handling of actions masking.
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]

# Types specific to handling a batch of states.
StatesFloatTensor = TensorType["batch_shape", "state_shape", torch.float]
StatesBoolTensor = TensorType["batch_shape", "state_shape", torch.bool]
StatesShortTensor = TensorType["batch_shape", "state_shape", torch.short]
StatesLongTensor = TensorType["batch_shape", "state_shape", torch.long]

# Types specific to handling a bactch of scalars (e.g., rewards, flows).
BatchFloatTensor = TensorType["batch_shape", torch.float]
BatchBoolTensor = TensorType["batch_shape", torch.bool]
BatchShortTensor = TensorType["batch_shape", torch.short]
BatchLongTensor = TensorType["batch_shape", torch.long]
BatchGenericTensor = TensorType["batch_shape"]

# Types specific to transitions, n_transitions is either int or Tuple[int].
TransitionLongTensor = TensorType["n_transitions", torch.long]
TransitionBoolTensor = TensorType["n_transitions", torch.bool]
TransitionFloatTensor = TensorType["n_transitions", torch.float]
TransitionPairFloatTensor = TensorType["n_transitions", 2, torch.float]

# Types specific to handling a batch of actions or steps.
BatchActionsTensor = TensorType["batch_shape", "action_shape"]
BatchStepsTensor = TensorType["batch_size", "n_steps"]  # TODO: remove? unusued?

# Types specific to environment preprocessing.
BatchInputTensor = TensorType["batch_shape", "input_dim"]  # Note: Should we specify precision?
BatchInputFloatTensor = TensorType["batch_shape", "input_dim", float]  
BatchOutputFloatTensor = TensorType["batch_shape", "output_dim", float]

# Types specific to handling single states/actions.
OneStateTensor = TensorType["state_shape", torch.float]
OneActionTensor = TensorType["action_shape"]

# Types specific to handling distributions, losses, etc.
PmfTensor = TensorType["n_states", torch.float]
LossTensor = TensorType[0, float]  # TODO: Rename to ScalarTensor.

# Types specific to transitions. n_transitions is an int.
TrajectoriesBoolTensor1D = TensorType["n_trajectories", torch.bool]
TrajectoriesFloatTensor1D = TensorType["n_trajectories", torch.float]
TrajectoriesLongTensor1D = TensorType["n_trajectories", torch.long]
TrajectoriesStatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
TrajectoriesFloatTensor2D = TensorType["max_length", "n_trajectories", torch.float]
TrajectoriesLongTensor2D = TensorType["max_length", "n_trajectories", torch.long]

# To Remove:
#TrajectoriesTensor2D2 = TensorType["n_trajectories", "shape"]
#TrajectoriesTensor1D = TensorType["n_trajectories", torch.long]
#TrajectoriesFloatTensor1D = TensorType["n_trajectories", torch.float]

#Tensor2D = TensorType["max_length", "n_trajectories", torch.long]
#Tensor2D2 = TensorType["n_trajectories", "shape"]
