r"""
The goal of this script is to reproduce some of the published results on the HyperGrid
environment. Run one of the following commands to reproduce some of the results in
[Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

python train_hypergrid.py --ndim 4 --height 8 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}
python train_hypergrid.py --ndim 2 --height 64 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}

And run one of the following to reproduce some of the results in
[Learning GFlowNets from partial episodes for improved convergence and stability](https://arxiv.org/abs/2209.12782)
python train_hypergrid.py --ndim {2, 4} --height 12 --R0 {1e-3, 1e-4} --tied --loss {TB, DB, SubTB}
"""

from argparse import ArgumentParser

# didnt help.
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import time
from tqdm import tqdm, trange
from math import ceil
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gfn.containers import ReplayBuffer
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.utils.common import set_seed
from gfn.utils.modules import DiscreteUniform, NeuralNet, Tabular
from gfn.utils.training import validate

DEFAULT_SEED = 4444


def dist_init(dist_backend : str = "ccl"):
    import os
    global my_rank
    global my_size
    print("PMI_SIZE={}".format(int(os.environ.get("PMI_SIZE", "0"))))

    dist_backend = "ccl"
    if int(os.environ.get("PMI_SIZE", "0")) > 1:
        if dist_backend == "ccl":
            print("+ CCL backend requested...")
            try:
                import oneccl_bindings_for_pytorch
            except ImportError as e:
                raise Exception(
                    "import oneccl_bindings_for_pytorch failed, {}".format(e)
                )

        elif dist_backend == "mpi":
            print("+ MPI backend requested...")
            assert torch.distributed.is_mpi_available()
            try:
                import torch_mpi
            except ImportError as e:
                raise Exception ("import torch_mpi failed, {}".format(e))
        else:
            raise Exception(f"invalid backend requested: {dist_backend}")


        os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")
        print("+ OMP_NUM_THREADS = ", os.getenv('OMP_NUM_THREADS'))
        dist.init_process_group(
            backend=dist_backend,
            init_method="env://",
            world_size=int(os.environ.get("WORLD_SIZE")),
            rank=int(os.environ.get("RANK")),
        )

        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        print(f"+ My rank: {my_rank} size: {my_size}")


def main(args):  # noqa: C901
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    off_policy_sampling = False if args.replay_buffer_size == 0 else True
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    use_wandb = len(args.wandb_project) > 0
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    dist_init()
    rank = dist.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"Running with DDP on rank {rank}/{world_size}.")

    # 1. Create the environment
    env = HyperGrid(
        args.ndim, args.height, args.R0, args.R1, args.R2, device_str=device_str
    )

    # 2. Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    gflownet = None
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
        module = DDP(module)
        estimator = DiscretePolicyEstimator(
            module=module,
            n_actions=env.n_actions,
            preprocessor=env.preprocessor,
        )
        gflownet = FMGFlowNet(estimator)
    else:
        pb_module = None
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

        pf_module = DDP(pf_module)
        pb_module = DDP(pb_module)
        pf_estimator = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            preprocessor=env.preprocessor,
        )
        pb_estimator = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=env.preprocessor,
        )

        if args.loss == "ModifiedDB":
            gflownet = ModifiedDBGFlowNet(
                pf_estimator,
                pb_estimator,
                off_policy_sampling,
            )

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
            # TODO: make module also DDP?
            logF_estimator = ScalarEstimator(
                module=module, preprocessor=env.preprocessor
            )
            if args.loss == "DB":
                gflownet = DBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    off_policy=off_policy_sampling,
                )
            else:
                gflownet = SubTBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    off_policy=off_policy_sampling,
                    weighting=args.subTB_weighting,
                    lamda=args.subTB_lambda,
                )
        elif args.loss == "TB":
            gflownet = TBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                off_policy=off_policy_sampling,
            )
        elif args.loss == "ZVar":
            gflownet = LogPartitionVarianceGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                off_policy=off_policy_sampling,
            )

    assert gflownet is not None, f"No gflownet for loss {args.loss}"

    # Initialize the replay buffer ?
    replay_buffer = None
    if args.replay_buffer_size > 0:
        if args.loss in ("TB", "SubTB", "ZVar"):
            objects_type = "trajectories"
        elif args.loss in ("DB", "ModifiedDB"):
            objects_type = "transitions"
        elif args.loss == "FM":
            objects_type = "states"
        else:
            raise NotImplementedError(f"Unknown loss: {args.loss}")
        replay_buffer = ReplayBuffer(
            env, objects_type=objects_type, capacity=args.replay_buffer_size
        )

    # 3. Create the optimizer

    # Policy parameters have their own LR.
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]

    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.states_from_batch_shape((0,))

    states_visited = 0
    n_iterations = ceil(args.n_trajectories / args.batch_size)
    my_batch_size = args.batch_size // world_size
    validation_info = {"l1_dist": float("inf")}
    sample_time = 0
    to_train_samples_time = 0
    loss_time = 0
    loss_backward_time = 0
    opt_time = 0
    rest_time = 0
    print ("n_iterations = ", n_iterations)
    print ("my_batch_size = ", my_batch_size)
    time_start = time.time()
    discovered_modes = set()

    for iteration in trange(n_iterations):
        sample_start = time.time()
        trajectories = gflownet.sample_trajectories(
            env, n_samples=my_batch_size, sample_off_policy=off_policy_sampling
        )
        sample_end = time.time()
        sample_time += (sample_end - sample_start)
        to_train_samples_start = time.time()
        training_samples = gflownet.to_training_samples(trajectories)
        to_train_samples_end = time.time()
        to_train_samples_time += (to_train_samples_end - to_train_samples_start)
        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                training_objects = replay_buffer.sample(n_trajectories=my_batch_size)
        else:
            training_objects = training_samples

        optimizer.zero_grad()
        loss_start = time.time()
        loss = gflownet.loss(env, training_objects)
        loss_end = time.time()
        loss_time += (loss_end - loss_start)
        loss_backward_start = time.time()
        loss.backward()
        loss_backward_end = time.time()
        loss_backward_time += (loss_backward_end - loss_backward_start)
        opt_start = time.time()
        optimizer.step()
        opt_end = time.time()
        opt_time += (opt_end - opt_start)

        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        if my_rank == 0:
            to_log = {"loss": loss.item(), "states_visited": states_visited}
            if use_wandb:
                wandb.log(to_log, step=iteration)
            if (iteration % args.validation_interval == 0) or (iteration == n_iterations - 1):
                validation_info, discovered_modes = validate_hypergrid(
                    env,
                    gflownet,
                    args.validation_samples,
                    visited_terminating_states,
                    discovered_modes,
                )
                if use_wandb:
                    wandb.log(validation_info, step=iteration)
                to_log.update(validation_info)
                tqdm.write(f"{iteration}: {to_log}")

    time_end = time.time()
    total_time = time_end - time_start
    rest_time = total_time - (sample_time + to_train_samples_time + loss_time + loss_backward_time + opt_time)
    dist.barrier()
    if (my_rank == 0):
        print ("total_time, sample_time, to_train_samples_time, loss_time, loss_backward_time, opt_time, rest_time")
        print (total_time, sample_time, to_train_samples_time, loss_time, loss_backward_time, opt_time, rest_time)

    return validation_info["l1_dist"]


def validate_hypergrid(
    env,
    gflownet,
    n_validation_samples,
    visited_terminating_states,
    discovered_modes,
):
    validation_info = validate(  # Standard validation shared across envs.
        env,
        gflownet,
        n_validation_samples,
        visited_terminating_states,
    )

    # Add the mode counting metric.
    states, scale = visited_terminating_states.tensor, env.scale_factor

    normalized_states = ((states * scale) - (scale / 2) * (env.height - 1)).abs()

    modes = torch.all(
        (normalized_states > (0.3 * scale) * (env.height - 1))
        & (normalized_states <= (0.4 * scale) * (env.height - 1)),
        dim=-1,
    )
    modes_found = set([tuple(s.tolist()) for s in states[modes.bool()]])
    discovered_modes.update(modes_found)
    validation_info["n_modes_found"] = len(discovered_modes)

    return validation_info, discovered_modes


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=8, help="Height of the environment"
    )
    parser.add_argument("--R0", type=float, default=0.1, help="Environment's R0")
    parser.add_argument("--R1", type=float, default=0.5, help="Environment's R1")
    parser.add_argument("--R2", type=float, default=2.0, help="Environment's R2")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed, if 0 then a random seed is used",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=0,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["FM", "TB", "DB", "SubTB", "ZVar", "ModifiedDB"],
        default="TB",
        help="Loss function to use",
    )
    parser.add_argument(
        "--subTB_weighting",
        type=str,
        default="geometric_within",
        help="weighting scheme for SubTB",
    )
    parser.add_argument(
        "--subTB_lambda", type=float, default=0.9, help="Lambda parameter for SubTB"
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
        help="How often (in training steps) to validate the gflownet",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )

    args = parser.parse_args()

    print(main(args))
