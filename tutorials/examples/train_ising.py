from argparse import ArgumentParser

import torch
import wandb
from tqdm import tqdm

from gfn.gflownet import FMGFlowNet
from gfn.gym import DiscreteEBM
from gfn.gym.discrete_ebm import IsingModel
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.modules import MLP


def main(args):
    # Configs

    use_wandb = len(args.wandb_project) > 0
    if use_wandb:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    device = torch.device(args.device)
    torch.set_num_threads(args.n_threads)
    hidden_dim = 512

    n_hidden = 2
    acc_fn = "relu"
    lr = 0.001

    def make_J(L, coupling_constant):
        """Ising model parameters."""

        def ising_n_to_ij(L, n):
            i = n // L
            j = n - i * L
            return (i, j)

        N = L**2
        J = torch.zeros((N, N), device=device)
        for k in range(N):
            for m in range(k):
                x1, y1 = ising_n_to_ij(L, k)
                x2, y2 = ising_n_to_ij(L, m)
                if x1 == x2 and abs(y2 - y1) == 1:
                    J[k][m] = 1
                    J[m][k] = 1
                elif y1 == y2 and abs(x2 - x1) == 1:
                    J[k][m] = 1
                    J[m][k] = 1

        for k in range(L):
            J[k * L][(k + 1) * L - 1] = 1
            J[(k + 1) * L - 1][k * L] = 1
            J[k][k + N - L] = 1
            J[k + N - L][k] = 1

        return coupling_constant * J

    # Ising model env
    N = args.L**2
    J = make_J(args.L, args.J)
    ising_energy = IsingModel(J)
    env = DiscreteEBM(N, alpha=1, energy=ising_energy, device=device)

    # Parametrization and losses
    pf_module = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
        activation_fn=acc_fn,
    )

    pf_estimator = DiscretePolicyEstimator(
        pf_module, env.n_actions, env.preprocessor, is_backward=False
    )
    gflownet = FMGFlowNet(pf_estimator).to(device)
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=lr)

    # Learning
    states_visited = 0

    for i in tqdm(range(args.n_iterations)):
        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_estimator_outputs=False,
            save_logprobs=False,
        )
        training_samples = gflownet.to_training_samples(trajectories)
        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=True)
        loss.backward()
        optimizer.step()

        states_visited += len(trajectories)
        to_log = {"loss": loss.item(), "states_visited": states_visited}

        if i % 25 == 0:
            tqdm.write(f"{i}: {to_log}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on",
    )

    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="Number of threads used by PyTorch",
    )

    parser.add_argument(
        "-L",
        type=int,
        default=6,
        help="Length of the grid",
    )

    parser.add_argument(
        "-J",
        type=float,
        default=0.44,
        help="J (Magnetic coupling constant)",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10000,
        help="Number of iterations",
    )

    args = parser.parse_args()
    main(args)
