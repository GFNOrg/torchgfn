import torch
from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import encode
from tqdm import tqdm, trange

import wandb
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.envs import HyperGrid
from gfn.estimators import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.losses import TrajectoryBalance
from gfn.modules import NeuralNet, Tabular, Uniform
from gfn.parametrizations import TBParametrization
from gfn.preprocessors import (
    EnumPreprocessor,
    IdentityPreprocessor,
    KHotPreprocessor,
    OneHotPreprocessor,
)
from gfn.samplers import LogitPFActionsSampler, TrajectoriesSampler
from gfn.validate import validate

parser = ArgumentParser()
parser.add_argument("--ndim", type=int, default=2)
parser.add_argument("--height", type=int, default=8)
parser.add_argument("--R0", type=float, default=0.1)
parser.add_argument(
    "--preprocessor",
    type=str,
    choices=["Identity", "KHot", "OneHot", "Enum"],
    default="KHot",
)
parser.add_argument("--tabular", action="store_true")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_iterations", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_Z", type=float, default=0.1)
parser.add_argument("--schedule", type=float, default=1.0)
parser.add_argument("--learn_PB", action="store_true")
parser.add_argument("--tie_PB", action="store_true")
parser.add_argument("--replay_buffer_size", type=int, default=0)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--use_tb", action="store_true", default=False)
parser.add_argument("--use_chi2", action="store_true", default=False)
parser.add_argument("--v2", action="store_true", default=False)
parser.add_argument("--use_baseline", action="store_true", default=False)
parser.add_argument("--wandb", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--validation_samples",
    type=int,
    default=200000,
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
print(encode(args))

torch.manual_seed(args.seed)
if args.no_cuda:
    device_str = "cpu"
else:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"


env = HyperGrid(args.ndim, args.height, R0=args.R0)
if args.preprocessor == "Identity":
    preprocessor = IdentityPreprocessor(env)
elif args.preprocessor == "OneHot":
    preprocessor = OneHotPreprocessor(env)
elif args.preprocessor == "KHot":
    preprocessor = KHotPreprocessor(env)
else:
    preprocessor = EnumPreprocessor(env)

if args.tabular:
    logit_PF = Tabular(env, output_dim=env.n_actions)
else:
    logit_PF = NeuralNet(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=256,
        n_hidden_layers=2,
    )
if args.learn_PB:
    if args.tabular:
        logit_PB = Tabular(env, output_dim=env.n_actions - 1)
    else:
        logit_PB = NeuralNet(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=256,
            n_hidden_layers=2,
            torso=logit_PF.torso if args.tie_PB else None,
        )
else:
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
    optimizer_Z = torch.optim.Adam([parametrization.parameters["logZ"]], lr=args.lr_Z)
    scheduler_Z = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_Z, milestones=list(range(2000, 100000, 2000)), gamma=args.schedule
    )
else:
    optimizer_Z = None
    scheduler_Z = None
optimizer = torch.optim.Adam(params)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=list(range(2000, 100000, 2000)), gamma=args.schedule
)


use_replay_buffer = args.replay_buffer_size > 0
if args.replay_buffer_size > 0:
    use_replay_buffer = True
    replay_buffer = ReplayBuffer(
        env, capacity=args.replay_buffer_size, objects="trajectories"
    )


use_wandb = len(args.wandb) > 0


if use_wandb:
    wandb.init(project=args.wandb)
    wandb.config.update(encode(args))
    run_name = (
        ("TB_" if not args.use_chi2 else "TB2_")
        if args.use_tb
        else ("VI_" if not args.use_chi2 else "CHI2_")
    )
    if args.use_tb:
        run_name += f"sch{args.schedule}_"
    run_name += "v2_" if args.v2 else ""
    run_name += "baseline_" if args.use_baseline else ""
    run_name += f"_{args.ndim}_{args.height}_{args.R0}_{args.seed}_"
    wandb.run.name = run_name + wandb.run.name.split("-")[-1]  # type: ignore


visited_terminating_states = (
    env.States() if args.validate_with_training_examples else None
)


for i in trange(args.n_iterations):
    trajectories = trajectories_sampler.sample(n_objects=args.batch_size)
    if use_replay_buffer:
        replay_buffer.add(trajectories)  # type: ignore
        training_objects = replay_buffer.sample(n_objects=args.batch_size)  # type: ignore
    else:
        training_objects = trajectories

    optimizer.zero_grad()
    if optimizer_Z is not None:
        optimizer_Z.zero_grad()
    logPF_trajectories, logPB_trajectories, scores = loss_fn.get_scores(trajectories)
    if args.use_tb:
        loss = (scores + parametrization.logZ.tensor).pow(2)
        if args.v2 and not args.use_chi2:
            loss = loss.pow(2)
        elif args.use_chi2:
            loss = (torch.exp(-scores) - torch.exp(parametrization.logZ.tensor)).pow(2)
        loss = loss.mean()

    else:
        if args.use_chi2:
            if args.v2:
                exp_scores = torch.exp(-2 * scores.detach())
                loss = torch.mean(
                    (2 * logPB_trajectories - logPF_trajectories) * exp_scores
                    + (
                        logPF_trajectories * torch.mean(exp_scores)
                        if args.use_baseline
                        else 0.0
                    )
                )
            else:
                loss = 0.5 * torch.mean(torch.exp(-2 * scores))
        else:
            if args.use_baseline:
                scores = scores - torch.mean(scores).detach()
            loss = torch.mean(scores**2)
    loss.backward()

    optimizer.step()
    scheduler.step()
    if optimizer_Z is not None:
        optimizer_Z.step()
    if scheduler_Z is not None:
        scheduler_Z.step()
    if args.validate_with_training_examples:
        visited_terminating_states.extend(training_objects.last_states)  # type: ignore
    to_log = {"loss": loss.item(), "states_visited": (i + 1) * args.batch_size}
    if use_wandb:
        wandb.log(to_log, step=i)
    if i % args.validation_interval == 0:
        validation_info = validate(
            env, parametrization, args.validation_samples, visited_terminating_states
        )
        if use_wandb:
            wandb.log(validation_info, step=i)
        to_log.update(validation_info)
        tqdm.write(f"{i}: {to_log}")
