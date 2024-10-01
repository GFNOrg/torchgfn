#!/usr/bin/env python
import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import ConditionalDiscretePolicyEstimator, ScalarEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet

torch.manual_seed(0)
exploration_rate = 0.5
learning_rate = 0.0005

# Setup the Environment.
env = HyperGrid(
    ndim=5,
    height=2,
    device_str="cuda" if torch.cuda.is_available() else "cpu",
)

# Build the GFlowNet -- Modules pre-concatenation.
CONCAT_SIZE = 16
module_PF = NeuralNet(
    input_dim=env.preprocessor.output_dim,
    output_dim=CONCAT_SIZE,
    hidden_dim=256,
)
module_PB = NeuralNet(
    input_dim=env.preprocessor.output_dim,
    output_dim=CONCAT_SIZE,
    hidden_dim=256,
    torso=module_PF.torso,
)

# Encoder for the Conditioning information.
module_cond = NeuralNet(
    input_dim=1,
    output_dim=CONCAT_SIZE,
    hidden_dim=256,
)

# Modules post-concatenation.
module_final_PF = NeuralNet(
    input_dim=CONCAT_SIZE * 2,
    output_dim=env.n_actions,
)
module_final_PB = NeuralNet(
    input_dim=CONCAT_SIZE * 2,
    output_dim=env.n_actions - 1,
    torso=module_final_PF.torso,
)

module_logZ = NeuralNet(
    input_dim=1,
    output_dim=1,
    hidden_dim=16,
    n_hidden_layers=2,
)

pf_estimator = ConditionalDiscretePolicyEstimator(
    module_PF,
    module_cond,
    module_final_PF,
    env.n_actions,
    is_backward=False,
    preprocessor=env.preprocessor,
)
pb_estimator = ConditionalDiscretePolicyEstimator(
    module_PB,
    module_cond,
    module_final_PB,
    env.n_actions,
    is_backward=True,
    preprocessor=env.preprocessor,
)

logZ_estimator = ScalarEstimator(module_logZ)
gflownet = TBGFlowNet(logZ=logZ_estimator, pf=pf_estimator, pb=pb_estimator)

# Feed pf to the sampler.
sampler = Sampler(estimator=pf_estimator)

# Move the gflownet to the GPU.
if torch.cuda.is_available():
    gflownet = gflownet.to("cuda")

# Policy parameters have their own LR. Log Z gets dedicated learning rate
# (typically higher).
optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=1e-3)
optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": 1e-1})

n_iterations = int(1e4)
batch_size = int(1e5)

for i in (pbar := tqdm(range(n_iterations))):
    conditioning = torch.rand((batch_size, 1))
    conditioning = (conditioning > 0.5).to(torch.float)  # Randomly 1 and zero.

    trajectories = gflownet.sample_trajectories(
        env,
        n=batch_size,
        conditioning=conditioning,
        save_logprobs=False,
        save_estimator_outputs=True,
        epsilon=exploration_rate,
    )
    optimizer.zero_grad()
    loss = gflownet.loss(env, trajectories)
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"loss": loss.item()})
