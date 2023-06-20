"""
The goal of this script is to reproduce some of the published results on the HyperGrid
environment. Run one of the following commands to reproduce some of the results in
[Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

python train_hypergrid.py --ndim 4 --height 8 --R0 {0.1, 0.01, 0.001} --tied {--uniform} --loss {TB, DB}
python train_hypergrid.py --ndim 2 --height 64 --R0 {0.1, 0.01, 0.001} --tied {--uniform} --loss {TB, DB}

And run one of the following to reproduce some of the results in
[Learning GFlowNets from partial episodes for improved convergence and stability](https://arxiv.org/abs/2209.12782)
python train_hypergrid.py --ndim {2, 4} --height 12 --R0 {1e-3, 1e-4} --tied --loss {TB, DB, SubTB}
"""

from argparse import ArgumentParser

import torch
import wandb
from tqdm import tqdm, trange

from gfn.envs import HyperGrid
from gfn.estimators import LogEdgeFlowEstimator, LogStateFlowEstimator, LogZEstimator
from gfn.losses import (
    DBParametrization,
    FMParametrization,
    LogPartitionVarianceParametrization,
    SubTBParametrization,
    TBParametrization,
)
from gfn.utils.common import trajectories_to_training_samples, validate
from gfn.utils.estimators import DiscretePBEstimator, DiscretePFEstimator
from gfn.utils.modules import DiscreteUniform, NeuralNet, Tabular

parser = ArgumentParser()

parser.add_argument("--no_cuda", action="store_true")

parser.add_argument("--ndim", type=int, default=2)
parser.add_argument("--height", type=int, default=64)
parser.add_argument("--R0", type=float, default=0.1)
parser.add_argument("--R1", type=float, default=0.5)
parser.add_argument("--R2", type=float, default=2.0)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument(
    "--loss", type=str, choices=["FM", "TB", "DB", "SubTB", "ZVar"], default="TB"
)
parser.add_argument(
    "--subTB_weighing",
    type=str,
    default="geometric_within",
)
parser.add_argument("--subTB_lambda", type=float, default=0.9)

parser.add_argument(
    "--tabular", action="store_true", help="Use a lookup table for F, PF, PB"
)
parser.add_argument("--uniform", action="store_true", help="Use a uniform PB")
parser.add_argument(
    "--tied", action="store_true", help="Tie the parameters of PF, PB, and F"
)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--n_hidden", type=int, default=2)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_Z", type=float, default=0.1)

parser.add_argument("--n_trajectories", type=int, default=int(1e6))

parser.add_argument("--validation_interval", type=int, default=100)
parser.add_argument(
    "--validation_samples",
    type=int,
    default=200000,
    help="Number of validation samples to use to evaluate the pmf.",
)

parser.add_argument("--wandb", type=str, default="")


args = parser.parse_args()

seed = args.seed if args.seed != 0 else torch.randint(int(10e10), (1,))[0].item()
torch.manual_seed(args.seed)

device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

use_wandb = len(args.wandb) > 0
if use_wandb:
    wandb.init(project=args.wandb)
    wandb.config.update(args)

# 1. Create the environment
env = HyperGrid(
    args.ndim, args.height, args.R0, args.R1, args.R2, device_str=device_str
)

# 2. Create the necessary modules, estimators, and parametrizations
if args.loss == "FM":
    # We need a LogEdgeFlowEstimator
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
    else:
        module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )
    estimator = LogEdgeFlowEstimator(env=env, module=module)
    parametrization = FMParametrization(estimator)
else:
    # We need a DiscretePFEstimator and a DiscretePBEstimator
    if args.tabular:
        pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        if not args.uniform:
            pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
    else:
        pf_module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )
        if not args.uniform:
            pb_module = NeuralNet(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                torso=pf_module.torso if args.tied else None,
            )
    if args.uniform:
        pb_module = DiscreteUniform(env.n_actions - 1)
    pf_estimator = DiscretePFEstimator(env=env, module=pf_module)
    pb_estimator = DiscretePBEstimator(env=env, module=pb_module)

if args.loss in ("DB", "SubTB"):
    # We need a LogStateFlowEstimator
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=1)
    else:
        module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            torso=pf_module.torso if args.tied else None,
        )
    logF_estimator = LogStateFlowEstimator(env=env, module=module)

    if args.loss == "DB":
        parametrization = DBParametrization(
            pf=pf_estimator, pb=pb_estimator, logF=logF_estimator, on_policy=True
        )
    else:
        parametrization = SubTBParametrization(
            pf=pf_estimator,
            pb=pb_estimator,
            logF=logF_estimator,
            on_policy=True,
            weighing=args.subTB_weighing,
            lamda=args.subTB_lambda,
        )
elif args.loss == "TB":
    # We need a LogZEstimator
    logZ = LogZEstimator(tensor=torch.tensor(0.0, device=env.device))
    parametrization = TBParametrization(
        pf=pf_estimator, pb=pb_estimator, logZ=logZ, on_policy=True
    )
elif args.loss == "ZVar":
    parametrization = LogPartitionVarianceParametrization(
        pf=pf_estimator, pb=pb_estimator, on_policy=True
    )

# 3. Create the optimizer
params = [
    {
        "params": [
            val for key, val in parametrization.parameters.items() if "logZ" not in key
        ],
        "lr": args.lr,
    }
]
if "logZ_logZ" in parametrization.parameters:
    params.append(
        {
            "params": [parametrization.parameters["logZ_logZ"]],
            "lr": args.lr_Z,
        }
    )

optimizer = torch.optim.Adam(params)


visited_terminating_states = env.States.from_batch_shape((0,))

states_visited = 0
n_iterations = args.n_trajectories // args.batch_size
for iteration in trange(n_iterations):
    trajectories = parametrization.sample_trajectories(n_samples=args.batch_size)
    training_samples = trajectories_to_training_samples(trajectories, parametrization)
    training_objects = training_samples

    optimizer.zero_grad()
    loss = parametrization.loss(training_objects)
    loss.backward()
    optimizer.step()

    visited_terminating_states.extend(trajectories.last_states)

    states_visited += len(trajectories)

    to_log = {"loss": loss.item(), "states_visited": states_visited}
    if use_wandb:
        wandb.log(to_log, step=iteration)
    if iteration % args.validation_interval == 0:
        validation_info = validate(
            env, parametrization, args.validation_samples, visited_terminating_states
        )
        if use_wandb:
            wandb.log(validation_info, step=iteration)
        to_log.update(validation_info)
        tqdm.write(f"{iteration}: {to_log}")
