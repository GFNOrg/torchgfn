import torch
import wandb
from configs import EnvConfig, LossConfig, OptimConfig, SamplerConfig
from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import encode
from tqdm import tqdm, trange

from gfn.containers.replay_buffer import ReplayBuffer
from gfn.utils import trajectories_to_training_samples, validate

parser = ArgumentParser()

parser.add_arguments(EnvConfig, dest="env_config")
parser.add_arguments(LossConfig, dest="loss_config")
parser.add_arguments(OptimConfig, dest="optim_config")
parser.add_arguments(SamplerConfig, dest="sampler_config")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_iterations", type=int, default=1000)
parser.add_argument("--replay_buffer_size", type=int, default=0)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--wandb", type=str, default="")
parser.add_argument("--validation_interval", type=int, default=100)
parser.add_argument(
    "--validation_samples",
    type=int,
    default=200000,
    help="Number of validation samples to use to evaluate the pmf.",
)
parser.add_argument(
    "--resample_for_validation",
    action="store_true",
    default=False,
    help="If False (default), the pmf is obtained from the latest visited terminating states",
)


args = parser.parse_args()

torch.manual_seed(args.seed)
if args.no_cuda:
    device_str = "cpu"
else:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

env_config: EnvConfig = args.env_config
loss_config: LossConfig = args.loss_config
optim_config: OptimConfig = args.optim_config
sampler_config: SamplerConfig = args.sampler_config

env = env_config.parse(device_str)
parametrization, loss_fn = loss_config.parse(env)
optimizer, scheduler = optim_config.parse(parametrization)
trajectories_sampler, on_policy = sampler_config.parse(env, parametrization)
loss_fn.on_policy = on_policy

use_replay_buffer = False
replay_buffer = None
if args.replay_buffer_size > 0:
    use_replay_buffer = True
    replay_buffer = ReplayBuffer(env, loss_fn, capacity=args.replay_buffer_size)

print(env_config, loss_config, optim_config, sampler_config)
print(args)
print(device_str)

use_wandb = len(args.wandb) > 0


if use_wandb:
    wandb.init(project=args.wandb)
    wandb.config.update(encode(args))

visited_terminating_states = (
    env.States.from_batch_shape((0,)) if not args.resample_for_validation else None
)

states_visited = 0
for i in trange(args.n_iterations):
    trajectories = trajectories_sampler.sample(n_trajectories=args.batch_size)
    training_samples = trajectories_to_training_samples(trajectories, loss_fn)
    if replay_buffer is not None:
        replay_buffer.add(training_samples)
        training_objects = replay_buffer.sample(n_trajectories=args.batch_size)
    else:
        training_objects = training_samples

    optimizer.zero_grad()
    loss = loss_fn(training_objects)
    loss.backward()

    optimizer.step()
    scheduler.step()
    if visited_terminating_states is not None:
        visited_terminating_states.extend(trajectories.last_states)

    states_visited += len(trajectories)
    to_log = {"loss": loss.item(), "states_visited": states_visited}
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
