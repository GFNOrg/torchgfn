import torch
from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import encode
from tqdm import tqdm, trange

import wandb
from gfn.configs import EnvConfig, OptimConfig, ParametrizationConfig, SamplerConfig
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.parametrizations.forward_probs import TBParametrization
from gfn.utils import hypergrid_validate

parser = ArgumentParser()

parser.add_arguments(EnvConfig, dest="env_config")
parser.add_arguments(ParametrizationConfig, dest="parametrization_config")
parser.add_arguments(OptimConfig, dest="optim_config")
parser.add_arguments(SamplerConfig, dest="sampler_config")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_iterations", type=int, default=1000)
parser.add_argument("--replay_buffer_size", type=int, default=0)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--wandb", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--validation_samples",
    type=int,
    default=1000,
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

env_config: EnvConfig = args.env_config
parametrization_config: ParametrizationConfig = args.parametrization_config
optim_config: OptimConfig = args.optim_config
sampler_config: SamplerConfig = args.sampler_config

env = env_config.parse(device_str)
parametrization, loss_fn = parametrization_config.parse(env)
optimizer, scheduler = optim_config.parse(parametrization)
training_sampler = sampler_config.parse(env, parametrization)

use_replay_buffer = False
if args.replay_buffer_size > 0:
    use_replay_buffer = True
    if isinstance(parametrization, TBParametrization):
        objects = "trajectories"
    else:
        objects = "transitions"
    print(objects)
    replay_buffer = ReplayBuffer(env, capacity=args.replay_buffer_size, objects=objects)

print(env_config, parametrization_config, optim_config, sampler_config)
print(args)
print(device_str)

use_wandb = len(args.wandb) > 0


if use_wandb:
    wandb.init(project=args.wandb)
    wandb.config.update(encode(args))

visited_terminating_states = (
    env.States() if args.validate_with_training_examples else None
)

for i in trange(args.n_iterations):
    training_samples = training_sampler.sample(n_objects=args.batch_size)
    if use_replay_buffer:
        replay_buffer.add(training_samples)  # type: ignore
        training_objects = replay_buffer.sample(n_objects=args.batch_size)  # type: ignore
    else:
        training_objects = training_samples

    optimizer.zero_grad()
    loss = loss_fn(training_objects)
    loss.backward()

    optimizer.step()
    scheduler.step()
    if args.validate_with_training_examples:
        visited_terminating_states.extend(training_objects.last_states)  # type: ignore
    to_log = {"loss": loss.item(), "states_visited": (i + 1) * args.batch_size}
    if use_wandb:
        wandb.log(to_log, step=i)
    if i % args.validation_interval == 0:
        validation_info = hypergrid_validate(
            env, parametrization, args.validation_samples, visited_terminating_states
        )
        if use_wandb:
            wandb.log(validation_info, step=i)
        to_log.update(validation_info)
        tqdm.write(f"{i}: {to_log}")
