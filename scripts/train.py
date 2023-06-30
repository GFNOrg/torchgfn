import torch
import wandb
from configs import load_config, make_env, make_loss, make_optim, make_sampler
import json
from tqdm import tqdm, trange
from argparse import ArgumentParser

from gfn.containers.replay_buffer import ReplayBuffer
from gfn.utils.common import trajectories_to_training_samples, validate

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default="hypergrid",
        help="Name of the environment config to load. Must exist as ./env/{env}.yaml."
        + " Note that ./env/base.yaml is always loaded first, "
        + "even if this argument is not specified.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="trajectory-balance",
        help="Name of the loss config to load. Must exist as ./loss/{loss}.yaml."
        + " Note that ./loss/base.yaml is always loaded first, "
        + "even if this argument is not specified.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="Name of the optim config to load. Must exist as ./optim/{optim}.yaml."
        + " Note that ./optim/base.yaml is always loaded first, "
        + "even if this argument is not specified.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        help="Name of the sampler config to load. Must exist as ./sampler/{sampler}.yaml."
        + " Note that ./sampler/base.yaml is always loaded first, "
        + "even if this argument is not specified.",
    )
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

    config = load_config(parser)

    torch.manual_seed(config["seed"])
    if config["no_cuda"]:
        device_str = "cpu"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    config["device"] = device_str

    env = make_env(config)
    parametrization, loss_fn = make_loss(config, env)
    optimizer = make_optim(config, parametrization)
    trajectories_sampler, on_policy = make_sampler(config, env, parametrization)
    loss_fn.on_policy = on_policy

    use_replay_buffer = False
    replay_buffer = None
    if config["replay_buffer_size"] > 0:
        use_replay_buffer = True
        replay_buffer = ReplayBuffer(
            env, loss_fn, capacity=config["replay_buffer_size"]
        )

    print("Config:")
    print(json.dumps(config, indent=2))

    use_wandb = len(config["wandb"]) > 0

    if use_wandb:
        wandb.init(project=config["wandb"])
        wandb.config.update(config)

    visited_terminating_states = (
        env.States.from_batch_shape((0,))
        if not config["resample_for_validation"]
        else None
    )

    states_visited = 0
    for i in trange(config["n_iterations"]):
        trajectories = trajectories_sampler.sample(n_trajectories=config["batch_size"])
        training_samples = trajectories_to_training_samples(trajectories, loss_fn)
        if replay_buffer is not None:
            replay_buffer.add(training_samples)
            training_objects = replay_buffer.sample(n_trajectories=config["batch_size"])
        else:
            training_objects = training_samples

        optimizer.zero_grad()
        loss = loss_fn(training_objects)
        loss.backward()

        optimizer.step()
        if visited_terminating_states is not None:
            visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)
        to_log = {"loss": loss.item(), "states_visited": states_visited}
        if use_wandb:
            wandb.log(to_log, step=i)
        if i % config["validation_interval"] == 0:
            validation_info = validate(
                env,
                parametrization,
                config["validation_samples"],
                visited_terminating_states,
            )
            if use_wandb:
                wandb.log(validation_info, step=i)
            to_log.update(validation_info)
            tqdm.write(f"{i}: {to_log}")
