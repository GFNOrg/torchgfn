from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import encode

import wandb
from gfn.configs import EnvConfig, OptimConfig, ParametrizationConfig, SamplerConfig
from gfn.utils import validate_TB_parametrization_for_HyperGrid

parser = ArgumentParser()

parser.add_arguments(EnvConfig, dest="env_config")
parser.add_arguments(ParametrizationConfig, dest="parametrization_config")
parser.add_arguments(OptimConfig, dest="optim_config")
parser.add_arguments(SamplerConfig, dest="sampler_config")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_iterations", type=int, default=1000)
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--validation_samples", type=int, default=1000)
parser.add_argument("--validation_interval", type=int, default=100)

args = parser.parse_args()

# config = args.config
env_config: EnvConfig = args.env_config
parametrization_config: ParametrizationConfig = args.parametrization_config
optim_config: OptimConfig = args.optim_config
sampler_config: SamplerConfig = args.sampler_config

env = env_config.parse(args.device)
parametrization, loss_fn = parametrization_config.parse(env)
optimizer, scheduler = optim_config.parse(parametrization)
training_sampler, validation_trajectories_sampler = sampler_config.parse(
    env, parametrization
)

print(env_config, parametrization_config, optim_config, sampler_config)

use_wandb = False

if use_wandb:
    wandb.init(project="gfn_tests_2")
    wandb.config.update(encode(args))
    # wandb.config.update(encode(env_config))
    # wandb.config.update(encode(parametrization_config))
    # wandb.config.update(encode(optim_config))
    # wandb.config.update(encode(sampler_config))
    # wandb.config.update({"batch_size": args.batch_size})
    # wandb.config.update({"n_iterations": args.n_iterations})
    # wandb.config.update({"device": args.device})


for i in range(args.n_iterations):
    training_objects = training_sampler.sample(n_objects=args.batch_size)

    optimizer.zero_grad()
    loss = loss_fn(training_objects)
    loss.backward()

    optimizer.step()
    scheduler.step()
    if use_wandb:
        wandb.log(
            {
                # "trajectories": env.get_states_indices(trajectories.states).transpose(0, 1),
                "loss": loss,
                # "last_states_indices": wandb.Histogram(last_states_indices),
            },
            step=i,
        )
    if i % args.validation_interval == 0:
        validation_info = validate_TB_parametrization_for_HyperGrid(
            env, parametrization, args.validation_samples
        )
        if use_wandb:
            wandb.log(validation_info, step=i)
        print(f"{i}: {validation_info} - Loss: {loss}")
