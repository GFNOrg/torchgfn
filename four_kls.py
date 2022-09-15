import os
import torch
from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import encode
from tqdm import tqdm, trange

import wandb
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.containers.trajectories import Trajectories
from gfn.envs import HyperGrid
from gfn.estimators import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.losses import TrajectoryBalance
from gfn.parametrizations import TBParametrization

from gfn.samplers import LogitPFActionsSampler, TrajectoriesSampler
from gfn.samplers.actions_samplers import LogitPBActionsSampler
from gfn.validate import validate

parser = ArgumentParser()
parser.add_argument("--ndim", type=int, default=4)
parser.add_argument("--height", type=int, default=8)
parser.add_argument("--R0", type=float, default=0.1)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_iterations", type=int, default=40000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_Z", type=float, default=0.1)
parser.add_argument("--schedule", type=float, default=1.0)
parser.add_argument("--replay_buffer_size", type=int, default=0)
parser.add_argument(
    "--baseline", type=str, choices=["global", "local", "None"], default="None"
)
parser.add_argument("--wandb", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--uniform_pb", action="store_true", default=False)
parser.add_argument("--tied", action="store_true", default=False)
parser.add_argument("--off_policy", action="store_true", default=False)
parser.add_argument(
    "--mode",
    type=str,
    choices=["tb", "forward_kl", "reverse_kl", "rws", "reverse_rws"],
    default="tb",
)
parser.add_argument(
    "--validation_samples",
    type=int,
    default=200000,
    help="Number of validation samples to use to evaluate the pmf.",
)
parser.add_argument("--validation_interval", type=int, default=100)
parser.add_argument("--sample_from_reward", action="store_true", default=False)
parser.add_argument("--reweight", action="store_true", default=False)
parser.add_argument("--no_cuda", action="store_true", default=False)

parser.add_argument("--models_directory", type=str, default=None)
parser.add_argument("--wandb_dir", type=str, default=None)


parser.add_argument("--config_id", type=int, default=None)

# Slurm specific arguments
parser.add_argument("--task_id", type=int)
parser.add_argument("--total", type=int)
parser.add_argument("--offset", type=int, default=0)
args = parser.parse_args()

# If launched via slurm scheduler or args.config_id is specified, change the args as follows:
if os.environ.get("SLURM_PROCID") is not None or args.config_id is not None:
    if args.config_id is None:
        slurm_process_id = int(os.environ["SLURM_PROCID"])
        args.config_id = args.offset + slurm_process_id * args.total + args.task_id
    config_id = args.config_id
    print(config_id)  # config_id starts with 1
    args.wandb = "four_kls_final"
    args.no_cuda = True
    changing_vars = [
        "seed",
        "ndim",
        "height",
        "R0",
        "mode",
        "baseline",
        "sample_from_reward",
        "reweight",
    ]
    seeds = range(10, 15)
    env_configs = [(4, 8, 0.1), (2, 64, 0.01)]
    modes = ["tb", "forward_kl", "reverse_kl", "rws", "reverse_rws"]
    baselines = ["None", "local", "global"]
    sample_from_rewards = [False, True]
    reweights = [False, True]
    config = []
    for seed in seeds:
        for ndim, height, R0 in env_configs:
            for mode in modes:
                for baseline in baselines if mode not in ["tb", "rws"] else ["None"]:
                    for sample_from_reward in (
                        sample_from_rewards if mode != "reverse_kl" else [False]
                    ):
                        for reweight in (
                            reweights if mode not in ["tb", "reverse_kl"] else [False]
                        ):
                            config.append(
                                dict(
                                    seed=seed,
                                    ndim=ndim,
                                    height=height,
                                    R0=R0,
                                    mode=mode,
                                    baseline=baseline,
                                    sample_from_reward=sample_from_reward,
                                    reweight=reweight,
                                )
                            )
    print(f"Total number of configs: {len(config)}. Config id: {config_id}")
    config = config[config_id - 1]
    for var in changing_vars:
        setattr(args, var, config[var])
print(encode(args))
if args.config_id is not None:
    model_hash = str(args.wandb + str(args.config_id))
else:
    model_hash = "temporary_model"
if args.models_directory is not None:
    models_directory = args.models_directory
else:
    models_directory = "models"
save_path = os.path.join(models_directory, model_hash)
print(save_path)
loading_model = (
    save_path is not None
    and os.path.exists(save_path)
    and model_hash != "temporary_model"
)

torch.manual_seed(args.seed)
device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
print(device_str)

env = HyperGrid(args.ndim, args.height, R0=args.R0)


logZ_tensor = torch.tensor(0.0)
logit_PF = LogitPFEstimator(env=env, module_name="NeuralNet")
logit_PB = LogitPBEstimator(
    env=env,
    module_name="Uniform" if args.uniform_pb else "NeuralNet",
    torso=logit_PF.module.torso if args.tied else None,
)
logZ = LogZEstimator(logZ_tensor)
parametrization = TBParametrization(logit_PF, logit_PB, logZ)

if not args.sample_from_reward:
    if not args.off_policy:
        actions_sampler = LogitPFActionsSampler(estimator=logit_PF)
    else:
        actions_sampler = LogitPFActionsSampler(estimator=logit_PF, temperature=0.1)
else:
    actions_sampler = LogitPBActionsSampler(estimator=logit_PB)
    validation_actions_sampler = LogitPFActionsSampler(estimator=logit_PF)
    validation_trajectories_sampler = TrajectoriesSampler(
        env=env, actions_sampler=validation_actions_sampler
    )

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

optimizer = torch.optim.Adam(params, lr=args.lr)
if args.mode == "tb":
    optimizer.add_param_group(
        {"params": [parametrization.parameters["logZ"]], "lr": args.lr_Z}
    )
else:
    optimizer_Z = torch.optim.Adam([parametrization.parameters["logZ"]], lr=args.lr_Z)
    scheduler_Z = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_Z, milestones=list(range(2000, 100000, 2000)), gamma=args.schedule
    )

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=list(range(2000, 100000, 2000)), gamma=args.schedule
)


use_replay_buffer = args.replay_buffer_size > 0
if args.replay_buffer_size > 0:
    use_replay_buffer = True
    replay_buffer = ReplayBuffer(
        env, capacity=args.replay_buffer_size, objects="trajectories"
    )


visited_terminating_states = env.States()
if loading_model:
    parametrization.load_state_dict(save_path)
    print("Loaded model from", save_path)
    with open(os.path.join(save_path, "metadata.txt"), "r") as f:
        lines = f.readlines()
        iteration = int(lines[0].split(":")[-1].strip())
        wandb_id = lines[1].strip()
    optimizer.load_state_dict(torch.load(os.path.join(save_path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(save_path, "scheduler.pt")))
    if args.mode != "tb":
        optimizer_Z.load_state_dict(
            torch.load(os.path.join(save_path, "optimizer_Z.pt"))
        )
        scheduler_Z.load_state_dict(
            torch.load(os.path.join(save_path, "scheduler_Z.pt"))
        )
    print("Loaded optimizer from", save_path)
    visited_terminating_states.load(save_path)
else:
    iteration = 0
    wandb_id = None

use_wandb = len(args.wandb) > 0


if use_wandb:
    if args.wandb_dir is not None:
        os.environ["WANDB_DIR"] = args.wandb_dir
    wandb.init(project=args.wandb, id=wandb_id, resume="allow")
    wandb.config.update(encode(args))
    if args.config_id is not None:
        wandb.run.name = f"{args.wandb}_{args.config_id}"
    else:
        run_name = args.mode
        run_name += f"_bs_{args.batch_size}_"
        run_name += f"_{args.ndim}_{args.height}_{args.R0}_{args.seed}_"
        wandb.run.name = run_name + wandb.run.name.split("-")[-1]  # type: ignore

if (args.mode, args.sample_from_reward, args.reweight) in [
    ("tb", True, True),
    ("tb", False, True),
    ("forward_kl", True, True),
    ("reverse_kl", True, False),
    ("reverse_kl", True, True),
    ("reverse_kl", False, True),
]:
    raise ValueError("Invalid combination of parameters.")
if save_path is not None and not os.path.exists(save_path):
    os.makedirs(save_path)
for i in trange(iteration, args.n_iterations):
    if args.sample_from_reward:
        samples_idx = torch.distributions.Categorical(probs=env.true_dist_pmf).sample(
            (args.batch_size,)
        )
        states = env.all_states[samples_idx]
        trajectories = trajectories_sampler.sample_trajectories(states=states)
        trajectories = Trajectories.revert_backward_trajectories(trajectories)
        validation_trajectories = validation_trajectories_sampler.sample(
            n_objects=args.batch_size
        )
    else:
        trajectories = trajectories_sampler.sample(n_objects=args.batch_size)

    if use_replay_buffer:
        replay_buffer.add(trajectories)  # type: ignore
        training_objects = replay_buffer.sample(n_objects=args.batch_size)  # type: ignore
    else:
        training_objects = trajectories

    logPF_trajectories, logPB_trajectories, scores = loss_fn.get_scores(trajectories)

    optimizer.zero_grad()
    if args.mode != "tb":
        optimizer_Z.zero_grad()
        loss_Z = (scores.detach() + logZ_tensor).pow(2).mean()
        loss_Z.backward()
        optimizer_Z.step()
        scheduler_Z.step()

    if args.baseline == "local":
        baseline = scores.mean().detach()
    elif args.baseline == "global":
        baseline = -logZ_tensor.detach()
    else:
        baseline = 0.0

    if args.reweight:
        if args.sample_from_reward:
            weights = torch.exp(scores) / torch.exp(scores).sum()
        else:
            weights = torch.exp(-scores) / torch.exp(-scores).sum()
        weights = weights.detach()

    if args.mode == "tb":
        loss = (scores + parametrization.logZ.tensor).pow(2)
    elif args.mode == "forward_kl":
        loss = -logPB_trajectories * (scores.detach() - baseline) - logPF_trajectories
        if args.reweight:
            loss = loss * weights
    elif args.mode == "reverse_kl":
        loss = logPF_trajectories * (scores.detach() - baseline) - logPB_trajectories
    elif args.mode == "rws":
        loss_pf = -logPF_trajectories
        if not args.sample_from_reward and args.reweight:
            loss_pf = loss_pf * weights
        loss_pb = -logPB_trajectories
        if args.sample_from_reward and args.reweight:
            loss_pb = loss_pb * weights
        loss = loss_pf + loss_pb
    elif args.mode == "reverse_rws":
        loss_pf = logPF_trajectories * (scores.detach() - baseline)
        if args.sample_from_reward and args.reweight:
            loss_pf = loss_pf * weights
        loss_pb = -logPB_trajectories * (scores.detach() - baseline)
        if not args.sample_from_reward and args.reweight:
            loss_pb = loss_pb * weights
        loss = loss_pf + loss_pb

    loss = loss.mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    visited_terminating_states.extend(training_objects.last_states if not args.sample_from_reward else validation_trajectories.last_states)  # type: ignore
    to_log = {"states_visited": (i + 1) * args.batch_size, "loss": loss.item()}
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

        if save_path is not None and os.path.exists(save_path):
            parametrization.save_state_dict(save_path)
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            visited_terminating_states.save(save_path)
            if args.mode != "tb":
                torch.save(
                    optimizer_Z.state_dict(), os.path.join(save_path, "optimizer_Z.pt")
                )
                torch.save(
                    scheduler_Z.state_dict(), os.path.join(save_path, "scheduler_Z.pt")
                )
            with open(os.path.join(save_path, "metadata.txt"), "w") as f:
                f.write("Iteration: " + str(i) + "\n")
                f.write(wandb.run.id)
