import torch
from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import encode

import wandb
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.envs import HyperGrid
from gfn.estimators import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.losses import TrajectoryBalance
from gfn.modules import NeuralNet, Uniform
from gfn.parametrizations import TBParametrization
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor, OneHotPreprocessor
from gfn.samplers import LogitPFActionsSampler, TrajectoriesSampler
from gfn.utils import validate_TB_for_HyperGrid

parser = ArgumentParser()
parser.add_argument("--ndim", type=int, default=2)
parser.add_argument("--height", type=int, default=8)

parser.add_argument(
    "--preprocessor", type=str, choices=["identity", "KHot", "OneHot"], default="KHot"
)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_iterations", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_Z", type=float, default=0.1)
parser.add_argument("--replay_buffer_size", type=int, default=0)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--use_tb", action="store_true", default=False)
parser.add_argument("--use_baseline", action="store_true", default=False)
parser.add_argument("--wandb", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--validation_samples",
    type=int,
    default=100000,
    help="Number of validation samples to use to evaluate the pmf.",
)
parser.add_argument("--validation_interval", type=int, default=100)
parser.add_argument(
    "--validate_with_training_examples",
    action="store_true",
    default=False,
    help="If true, the pmf is obtained from the latest visited terminating states",
)

args = parser.parse_args()

torch.manual_seed(args.seed)
if args.no_cuda:
    device_str = "cpu"
else:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"


env = HyperGrid(args.ndim, args.height)
if args.preprocessor == "Identity":
    preprocessor = IdentityPreprocessor(env)
elif args.preprocessor == "OneHot":
    preprocessor = OneHotPreprocessor(env)
else:
    preprocessor = KHotPreprocessor(env)

logit_PF = NeuralNet(
    input_dim=preprocessor.output_dim,
    output_dim=env.n_actions,
    hidden_dim=256,
    n_hidden_layers=2,
)
logit_PB = Uniform(env=env, output_dim=env.n_actions - 1)

logZ_tensor = torch.tensor(0.0)
logit_PF = LogitPFEstimator(preprocessor=preprocessor, module=logit_PF)
logit_PB = LogitPBEstimator(preprocessor=preprocessor, module=logit_PB)
logZ = LogZEstimator(logZ_tensor)
parametrization = TBParametrization(logit_PF, logit_PB, logZ)


actions_sampler = LogitPFActionsSampler(estimator=logit_PF)
trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)

loss_fn = TrajectoryBalance(parametrization=parametrization)


params = [
    {
        "params": [
            val for key, val in parametrization.parameters.items() if key != "logZ"
        ],
        "lr": args.lr,
    }
]
if args.use_tb and "logZ" in parametrization.parameters:
    params.append({"params": [parametrization.parameters["logZ"]], "lr": args.lr_Z})
optimizer = torch.optim.Adam(params)

use_replay_buffer = False
if args.replay_buffer_size > 0:
    use_replay_buffer = True
    replay_buffer = ReplayBuffer(
        env, capacity=args.replay_buffer_size, objects="trajectories"
    )


use_wandb = len(args.wandb) > 0


if use_wandb:
    wandb.init(project=args.wandb)
    wandb.config.update(encode(args))

visited_terminating_states = (
    env.States() if args.validate_with_training_examples else None
)


for i in range(args.n_iterations):
    trajectories = trajectories_sampler.sample(n_objects=args.batch_size)
    if use_replay_buffer:
        replay_buffer.add(trajectories)  # type: ignore
        training_objects = replay_buffer.sample(n_objects=args.batch_size)  # type: ignore
    else:
        training_objects = trajectories

    optimizer.zero_grad()
    if args.use_tb:
        loss = loss_fn(training_objects)
    else:
        logPF_trajectories, scores = loss_fn.get_scores(trajectories)
        if args.use_baseline:
            scores = scores - torch.mean(scores)
        loss = torch.mean(logPF_trajectories * scores.detach())
    loss.backward()

    optimizer.step()
    if args.validate_with_training_examples:
        visited_terminating_states.extend(training_objects.last_states)  # type: ignore
    if use_wandb:
        wandb.log({"loss": loss.item()}, step=i)
        wandb.log({"states_visited": (i + 1) * args.batch_size}, step=i)
    if i % args.validation_interval == 0:
        true_logZ, validation_info = validate_TB_for_HyperGrid(
            env, parametrization, args.validation_samples, visited_terminating_states
        )
        if use_wandb:
            wandb.log(validation_info, step=i)
            if i == 0:
                wandb.log({"true_logZ": true_logZ})
        print(f"{i}: {validation_info} - Loss: {loss} - True logZ: {true_logZ}")