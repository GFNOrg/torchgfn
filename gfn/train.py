from simple_parsing import ArgumentParser

import wandb
from gfn.envs.utils import get_true_dist_pmf
from gfn.working_config import GFlowNetConfig

parser = ArgumentParser()
parser.add_arguments(GFlowNetConfig, dest="config")
args = parser.parse_args()
config: GFlowNetConfig = args.config


(
    env,
    parametrization,
    loss_fn,
    optimizer,
    scheduler,
    training_sampler,
    validation_trajectories_sampler,
    batch_size,
    n_iterations,
    device,
) = config.parse()

print(config)

wandb.init(project="gfn_tests")
wandb.config.update(config)

true_dist_pmf = get_true_dist_pmf(env)

for i in range(n_iterations):
    training_objects = training_sampler.sample(n_objects=batch_size)

    optimizer.zero_grad()
    loss = loss_fn(training_objects)
    loss.backward()

    optimizer.step()
    scheduler.step()
    wandb.log(
        {
            # "trajectories": env.get_states_indices(trajectories.states).transpose(0, 1),
            "loss": loss,
            # "last_states_indices": wandb.Histogram(last_states_indices),
        },
        step=i,
    )
    if i % 100 == 0:
        n_samples = 1000
        final_states_dist = parametrization.P_T(env, n_samples)

        l1_dist = (final_states_dist.pmf() - true_dist_pmf).abs().mean()
        wandb.log({"l1_dist": l1_dist}, step=i)
        print(f"Step {i} - l1_dist: {l1_dist} - loss: {loss}")
