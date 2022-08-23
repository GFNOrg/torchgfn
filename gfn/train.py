import torch

import wandb
from gfn.envs import HyperGrid
from gfn.envs.utils import get_flat_grid, get_true_dist_pmf
from gfn.estimators import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.losses import TrajectoryBalance
from gfn.models import NeuralNet, Tabular, Uniform
from gfn.parametrizations import O_PFBZ, O_PFZ
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor
from gfn.samplers import LogitPFActionSampler, TrajectoriesSampler

wandb.init(project="gfn_tests")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 16
LR = 1e-2
LR_Z = 0.1

wandb.config = {"lr": LR, "lr_Z": LR_Z, "batch_size": batch_size}

env = HyperGrid(ndim=2, height=4)
preprocessor = KHotPreprocessor(env)
preprocessor = IdentityPreprocessor(env)
logit_PF_module = NeuralNet(
    input_dim=preprocessor.output_dim, output_dim=env.n_actions, hidden_dim=16
).to(device=device)


tie_PB_parameters = True
logit_PB_module = NeuralNet(
    input_dim=preprocessor.output_dim,
    output_dim=env.n_actions - 1,
    hidden_dim=16,
    torso=logit_PF_module.torso if tie_PB_parameters else None,
).to(device=device)

logit_PF = LogitPFEstimator(preprocessor=preprocessor, module=logit_PF_module)
logit_PB = LogitPBEstimator(preprocessor=preprocessor, module=logit_PB_module)

logit_PF_module = Tabular(env=env, output_dim=env.n_actions)
logit_PF = LogitPFEstimator(preprocessor=preprocessor, module=logit_PF_module)

logZ = torch.tensor(0.0, device=device)
logZ = LogZEstimator(logZ=logZ)

o = O_PFBZ(logit_PF=logit_PF, logit_PB=logit_PB, logZ=logZ, tied=tie_PB_parameters)
print(o.parameters.keys())

logit_PB = LogitPBEstimator(
    preprocessor=preprocessor, module=Uniform(output_dim=env.n_actions - 1)
)
o = O_PFZ(logit_PF=logit_PF, logit_PB=logit_PB, logZ=logZ)

optimizer = torch.optim.SGD(
    [
        {
            "params": [val for key, val in o.parameters.items() if key != "logZ"],
            "lr": LR,
        },
        {"params": [o.parameters["logZ"]], "lr": LR_Z},
    ]
)

loss_fn = TrajectoryBalance(o)

action_sampler = LogitPFActionSampler(logit_PF=logit_PF)
trajectories_sampler = TrajectoriesSampler(env, action_sampler)

true_dist_pmf = get_true_dist_pmf(env)
true_logZ = env.reward(get_flat_grid(env)).sum().log().item()

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[200, 500, 1000, 2000, 5000], gamma=0.5
)

last_states_indices = []

for i in range(10000):
    trajectories = trajectories_sampler.sample_trajectories(
        n_trajectories=batch_size,
        temperature=1.0,
    )
    # print(env.get_states_indices(trajectories.last_states).numpy().tolist())
    last_states_indices.extend(
        env.get_states_indices(trajectories.last_states).numpy().tolist()
    )
    # print(trajectories)

    optimizer.zero_grad()
    loss = loss_fn(trajectories)
    loss.backward()
    # print(logZ.logZ.grad)
    # for i in range(env.height):
    #     print(
    #         i,
    #         logit_PF_module.tensors[i].detach().numpy(),
    #         logit_PF_module.tensors[i].grad,
    #     )
    optimizer.step()
    scheduler.step()
    wandb.log(
        {
            "trajectories": env.get_states_indices(trajectories.states).transpose(0, 1),
            "loss": loss,
            "last_states_indices": wandb.Histogram(last_states_indices),
        },
        step=i,
    )
    if i % 100 == 0:
        # print("xx", sum(list(map(lambda x: x.sum(), logit_PF_module.parameters()))))
        # print("yy", sum(list(map(lambda x: x.sum(), logit_PB_module.parameters()))))

        print(true_logZ, o.logZ)
        n_samples = 1000
        final_states_dist = o.P_T(env, n_samples)

        l1_dist = (final_states_dist.pmf() - true_dist_pmf).abs().mean()
        print(loss)
        wandb.log({"l1_dist": l1_dist}, step=i)


assert False
