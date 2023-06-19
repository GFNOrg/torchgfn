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

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument(
        "--ndim", type=int, default=2, help="Dimensionality of the hyper grid"
    )
    parser.add_argument("--height", type=int, default=64, help="Size of each dimension")
    parser.add_argument("--R0", type=float, default=0.1, help="R0 reward factor")
    parser.add_argument("--R1", type=float, default=0.5, help="R1 reward factor")
    parser.add_argument("--R2", type=float, default=2.0, help="R2 reward factor")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["FM", "TB", "DB", "SubTB", "ZVar"],
        default="TB",
        help="Loss parameterization to train with",
    )
    parser.add_argument(
        "--subTB_weighing",
        type=str,
        choices=[
            "DB",
            "ModifiedDB",
            "TB",
            "geometric",
            "equal",
            "geometric_within",
            "equal_within",
        ],
        default="geometric_within",
        help="Weighing scheme for the SubTB loss",
    )
    parser.add_argument(
        "--subTB_lambda", type=float, default=0.9, help="SubTB lambda scale factor"
    )

    parser.add_argument(
        "--tabular",
        action="store_true",
        help="Use a lookup table for F, PF, PB instead of an estimator",
    )
    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform PB")
    parser.add_argument(
        "--tied", action="store_true", help="Tie the parameters of PF, PB, and F"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the estimators' neural network modules.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=2,
        help="Number of hidden layers (of size `hidden_dim`) in the estimators'"
        + " neural network modules",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_Z",
        type=float,
        default=0.1,
        help="Specific learning rate for Z (only used for TB loss)",
    )

    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(1e6),
        help="Total budget of trajectories to train on. "
        + "Training iterations = n_trajectories // batch_size",
    )

    parser.add_argument(
        "--validation_interval",
        type=int,
        default=100,
        help="How often (in training steps) to validate the parameterization",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Number of validation samples to use to evaluate the pmf.",
    )

    parser.add_argument(
        "--wandb",
        type=str,
        default="",
        help="Wandb project name. Disabled by default. Set to a string to enable "
        + "wandb logging",
    )

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
    parametrization = pb_module = pf_module = pf_estimator = pb_estimator = None

    # 2. Create the parameterization.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
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
            if not args.uniform_pb:
                pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
        else:
            pf_module = NeuralNet(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
            if not args.uniform_pb:
                pb_module = NeuralNet(
                    input_dim=env.preprocessor.output_dim,
                    output_dim=env.n_actions - 1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    torso=pf_module.torso if args.tied else None,
                )
        if args.uniform_pb:
            pb_module = DiscreteUniform(env.n_actions - 1)

        assert (
            pf_module is not None
        ), f"pf_module is None. Command-line arguments: {args}"
        assert (
            pb_module is not None
        ), f"pb_module is None. Command-line arguments: {args}"

        pf_estimator = DiscretePFEstimator(env=env, module=pf_module)
        pb_estimator = DiscretePBEstimator(env=env, module=pb_module)

        if args.loss in ("DB", "SubTB"):
            # We need a LogStateFlowEstimator

            assert (
                pf_estimator is not None
            ), f"pf_estimator is None. Command-line arguments: {args}"
            assert (
                pb_estimator is not None
            ), f"pb_estimator is None. Command-line arguments: {args}"

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
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    on_policy=True,
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
                pf=pf_estimator,
                pb=pb_estimator,
                logZ=logZ,
                on_policy=True,
            )
        elif args.loss == "ZVar":
            parametrization = LogPartitionVarianceParametrization(
                pf=pf_estimator,
                pb=pb_estimator,
                on_policy=True,
            )

    assert parametrization is not None, f"No parametrization for loss {args.loss}"

    # 3. Create the optimizer
    params = [
        {
            "params": [
                val
                for key, val in parametrization.parameters.items()
                if "logZ" not in key
            ],
            "lr": args.lr,
        }
    ]
    if "logZ.logZ" in parametrization.parameters:
        params.append(
            {
                "params": [parametrization.parameters["logZ.logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size
    for iteration in trange(n_iterations):
        trajectories = parametrization.sample_trajectories(n_samples=args.batch_size)
        training_samples = trajectories_to_training_samples(
            trajectories, parametrization
        )
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
                env,
                parametrization,
                args.validation_samples,
                visited_terminating_states,
            )
            if use_wandb:
                wandb.log(validation_info, step=iteration)
            to_log.update(validation_info)
            tqdm.write(f"{iteration}: {to_log}")
