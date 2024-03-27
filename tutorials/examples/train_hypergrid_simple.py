#!/usr/bin/env python
import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
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

# Build the GFlowNet.
module_PF = NeuralNet(
    input_dim=env.preprocessor.output_dim,
    output_dim=env.n_actions,
)
module_PB = NeuralNet(
    input_dim=env.preprocessor.output_dim,
    output_dim=env.n_actions - 1,
    torso=module_PF.torso,
)
pf_estimator = DiscretePolicyEstimator(
    module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor
)
pb_estimator = DiscretePolicyEstimator(
    module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor
)
gflownet = TBGFlowNet(init_logZ=0.0, pf=pf_estimator, pb=pb_estimator)

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
    trajectories = sampler.sample_trajectories(
        env,
        n_trajectories=batch_size,
        save_logprobs=False,
        save_estimator_outputs=True,
        epsilon=exploration_rate,
    )
    optimizer.zero_grad()
    loss = gflownet.loss(env, trajectories)
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"loss": loss.item()})
