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

import datetime
import os
import signal
import sys
import threading
import time
from argparse import ArgumentParser
from math import ceil
from typing import Callable, Optional, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm, trange

from gfn.containers import NormBasedDiversePrioritizedReplayBuffer, ReplayBuffer
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
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform, Tabular
from gfn.utils.training import validate

DEFAULT_SEED = 4444


def average_gradients(model):
    """All-Reduce gradients across all models."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_models(model):
    """Averages model weights across all ranks."""
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        param_tensor = param.data.clone()  # clone to avoid inplace operations
        dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        param.data = param_tensor / world_size


def initialize_distributed_compute(dist_backend: str = "ccl"):
    """Initalizes distributed compute using either ccl or mpi backends."""
    # global my_rank  # TODO: remove globals?
    # global my_size  # TODO: remove globals?

    pmi_size = int(os.environ.get("PMI_SIZE", "0"))  # 0 or 1 default value?
    print("+ Initalizing distributed compute, PMI_SIZE={}".format(pmi_size))

    if pmi_size > 1:
        if dist_backend == "ccl":
            print("+ CCL backend requested...")
            try:
                # Note - intel must be imported before oneccl!
                import oneccl_bindings_for_pytorch  # noqa: F401
            except ImportError as e:
                raise Exception(
                    "import oneccl_bindings_for_pytorch failed, {}".format(e)
                )

        elif dist_backend == "mpi":
            print("+ MPI backend requested...")
            assert torch.distributed.is_mpi_available()
            try:
                import torch_mpi  # noqa: F401
            except ImportError as e:
                raise Exception("import torch_mpi failed, {}".format(e))
        else:
            raise Exception(f"Invalid backend requested: {dist_backend}")

        os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")

        print("+ OMP_NUM_THREADS = ", os.getenv("OMP_NUM_THREADS"))
        dist.init_process_group(
            backend=dist_backend,
            init_method="env://",
            world_size=int(os.environ.get("WORLD_SIZE")),
            rank=int(os.environ.get("RANK")),
            timeout=datetime.timedelta(minutes=5),
        )

        my_rank = dist.get_rank()  # Global!
        my_size = dist.get_world_size()  # Global!

        # for now, let us enforce that each agent gets equal number of ranks.
        # TODO: later, we can relax this condition.
        assert my_size % args.num_agent_groups == 0
        agent_group_size = my_size // args.num_agent_groups
        agent_group_rank_list = [
            list(range(i * agent_group_size, (i + 1) * agent_group_size))
            for i in range(args.num_agent_groups)
        ]
        print(agent_group_rank_list)
        agent_group_list = [
            dist.new_group(
                agent_group_rank_list[i],
                backend=dist_backend,
                timeout=datetime.timedelta(minutes=5),
            )
            for i in range(args.num_agent_groups)
        ]

        print(f"+ My rank: {my_rank} size: {my_size}")

        return (my_rank, my_size, agent_group_size, agent_group_list)


class DistributedErrorHandler:
    def __init__(
        self,
        device_str: str,
        rank: int,
        world_size: int,
        error_check_interval: float = 1.0,
        cleanup_callback: Optional[Callable] = None,
    ):
        """
        Initialize error handler for distributed training.

        Args:
            device_str: String representing the current device.
            rank: Current process rank
            world_size: Total number of processes
            error_check_interval: How often to check for errors (in seconds)
            cleanup_callback: Optional function to call before shutdown
        """
        self.device_str = device_str
        self.rank = rank
        self.world_size = world_size
        self.error_check_interval = error_check_interval
        self.cleanup_callback = cleanup_callback
        self.shutdown_flag = threading.Event()
        self.error_tensor = torch.zeros(1, dtype=torch.uint8, device=self.device_str)

        # Set up error checking thread
        self.checker_thread = threading.Thread(target=self._error_checker, daemon=True)

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def start(self):
        """Start error checking thread"""
        self.checker_thread.start()

    def _signal_handler(self, signum, frame):
        """Handle external signals"""
        print(f"Process {self.rank} received signal {signum}")
        self.shutdown_flag.set()
        self._cleanup()
        sys.exit(1)

    def _error_checker(self):
        """Periodically check for errors across all processes"""
        while not self.shutdown_flag.is_set():
            try:
                # Use all_reduce to check if any process has errored
                error_count = torch.zeros_like(self.error_tensor)
                dist.all_reduce(error_count, op=dist.ReduceOp.SUM)

                if error_count.item() > 0:
                    print(f"Process {self.rank}: Detected error in another process")
                    self.shutdown_flag.set()
                    self._cleanup()
                    sys.exit(1)

            except Exception as e:
                print("Process {}: Error in error checker: {}".format(self.rank, e))
                self.signal_error()
                break

            time.sleep(self.error_check_interval)

    def signal_error(self):
        """Signal that this process has encountered an error"""
        try:
            self.error_tensor.fill_(1)
            dist.all_reduce(self.error_tensor, op=dist.ReduceOp.SUM)
        except:
            pass  # If this fails, processes will eventually timeout

        self.shutdown_flag.set()
        self._cleanup()
        sys.exit(1)

    def _cleanup(self):
        """Perform cleanup before shutdown"""
        if self.cleanup_callback:
            try:
                self.cleanup_callback()
            except Exception as e:
                print(f"Process {self.rank}: Error in cleanup: {str(e)}")

        try:
            dist.destroy_process_group()
        except:
            pass


def gather_distributed_data(
    local_tensor: torch.Tensor,
    world_size: int = None,
    rank: int = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Gather data from all processes in a distributed setting.

    Args:
        local_data: Data from the current process (List or Tensor)
        world_size: Number of processes (optional, will get from env if None)
        rank: Current process rank (optional, will get from env if None)

    Returns:
        On rank 0: Concatenated tensor from all processes
        On other ranks: None
    """
    if verbose:
        print("syncing distributed data")

    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()

    # First gather batch_sizes to allocate correct buffer sizes.
    local_batch_size = torch.tensor(
        [local_tensor.shape[0]], device=local_tensor.device, dtype=local_tensor.dtype
    )
    if rank == 0:
        # Assumes same dimensionality on all ranks!
        batch_size_list = [
            torch.zeros((1,), device=local_tensor.device, dtype=local_tensor.dtype)
            for _ in range(world_size)
        ]
    else:
        batch_size_list = None

    if verbose:
        print("rank={}, batch_size_list={}".format(rank, batch_size_list))
        print(
            "+ gather of local_batch_size={} to batch_size_list".format(local_batch_size)
        )
    dist.gather(local_batch_size, gather_list=batch_size_list, dst=0)
    dist.barrier()  # Add synchronization

    # Pad local tensor to maximum size.
    if verbose:
        print("+ padding local tensor")

    if rank == 0:
        max_batch_size = max(bs for bs in batch_size_list)
    else:
        max_batch_size = 0

    state_size = local_tensor.shape[1]  # assume states are 1-d, is true for this env.

    # Broadcast max_size to all processes for padding
    max_batch_size_tensor = torch.tensor(max_batch_size, device=local_tensor.device)
    dist.broadcast(max_batch_size_tensor, src=0)

    # Pad local tensor to maximum size.
    if local_tensor.shape[0] < max_batch_size:
        padding = torch.zeros(
            (max_batch_size - local_tensor.shape[0], state_size),
            dtype=local_tensor.dtype,
            device=local_tensor.device,
        )
        local_tensor = torch.cat((local_tensor, padding), dim=0)

    # Gather padded tensors.
    if rank == 0:
        tensor_list = [
            torch.zeros(
                (max_batch_size, state_size),
                dtype=local_tensor.dtype,
                device=local_tensor.device,
            )
            for _ in range(world_size)
        ]
    else:
        tensor_list = None

    if verbose:
        print("+ gathering all tensors from world_size={}".format(world_size))
        print("rank={}, tensor_list={}".format(rank, tensor_list))
    dist.gather(local_tensor, gather_list=tensor_list, dst=0)
    dist.barrier()  # Add synchronization

    # Only rank 0 processes the results
    if rank == 0:
        results = []
        for tensor, batch_size in zip(tensor_list, batch_size_list):
            trimmed_tensor = tensor[: batch_size.item(), ...]
            results.append(trimmed_tensor)

        if verbose:
            print("distributed n_results={}".format(len(results)))

        for r in results:
            print("    {}".format(r.shape))

        return torch.cat(results, dim=0)  # Concatenates along the batch dimension.

    return None  # For all non-zero ranks.


def main(args):  # noqa: C901
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    use_wandb = args.wandb_project != ""
    if use_wandb:

        if args.wandb_local:
            os.environ["WANDB_MODE"] = "offline"

        import wandb

        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    if args.distributed:
        my_rank, my_size, agent_group_size, agent_group_list = (
            initialize_distributed_compute()
        )
        my_rank = dist.get_rank()
        world_size = torch.distributed.get_world_size()
        my_agent_group_id = my_rank // agent_group_size
        print(f"Running with DDP on rank {my_rank}/{world_size}.")
        print(
            f"agent_group_size, my_agent_group_id = {agent_group_size, my_agent_group_id}"
        )
    else:
        world_size = 1  # Single machine.
        my_rank = 0  # Single machine.

    set_seed(seed + my_rank)

    # 1. Create the environment
    env = HyperGrid(
        args.ndim,
        args.height,
        args.R0,
        args.R1,
        args.R2,
        device_str=device_str,
        calculate_partition=args.calculate_partition,
        calculate_all_states=args.calculate_all_states,
    )

    # 2. Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    gflownet, pf_module, pb_module = None, None, None
    pf_estimator, pb_estimator = None, None

    if args.loss == "FM":
        # We need a LogEdgeFlowEstimator.
        if args.tabular:
            module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        else:
            module = MLP(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )

        if args.distributed:
            module = DDP(module, process_group=agent_group_list[my_agent_group_id])

        estimator = DiscretePolicyEstimator(
            module=module,
            n_actions=env.n_actions,
            preprocessor=env.preprocessor,
        )
        gflownet = FMGFlowNet(estimator)
    else:
        # We need a DiscretePFEstimator and a DiscretePBEstimator.
        if args.tabular:
            pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
            if not args.uniform_pb:
                pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
        else:
            pf_module = MLP(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
            if not args.uniform_pb:
                pb_module = MLP(
                    input_dim=env.preprocessor.output_dim,
                    output_dim=env.n_actions - 1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    trunk=pf_module.trunk if args.tied else None,
                )
        if args.uniform_pb:
            pb_module = DiscreteUniform(env.n_actions - 1)

        for v in ["pf_module", "pb_module"]:
            assert locals()[v] is not None, f"{v} is None, Args: {args}"

        if args.distributed:
            pf_module = DDP(pf_module, process_group=agent_group_list[my_agent_group_id])
            pb_module = DDP(pb_module, process_group=agent_group_list[my_agent_group_id])

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
            for v in ["pf_estimator", "pb_estimator"]:
                assert locals()[v] is not None, f"{v} is None, Args: {args}"
            gflownet = ModifiedDBGFlowNet(pf_estimator, pb_estimator)

        elif args.loss in ("DB", "SubTB"):
            # We need a LogStateFlowEstimator.
            for v in ["pf_estimator", "pb_estimator"]:
                assert locals()[v] is not None, f"{v} is None, Args: {args}"

            # We also need a LogStateFlowEstimator.
            if args.tabular:
                module = Tabular(n_states=env.n_states, output_dim=1)
            else:
                module = MLP(
                    input_dim=env.preprocessor.output_dim,
                    output_dim=1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    trunk=pf_module.trunk if args.tied else None,
                )

            if args.distributed:
                module = DDP(module, process_group=agent_group_list[my_agent_group_id])

            logF_estimator = ScalarEstimator(
                module=module, preprocessor=env.preprocessor
            )
            if args.loss == "DB":
                gflownet = DBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                )
            else:
                gflownet = SubTBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    weighting=args.subTB_weighting,
                    lamda=args.subTB_lambda,
                )
        elif args.loss == "TB":
            for v in ["pf_estimator", "pb_estimator"]:
                assert locals()[v] is not None, f"{v} is None, Args: {args}"
            gflownet = TBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
            )
        elif args.loss == "ZVar":
            for v in ["pf_estimator", "pb_estimator"]:
                assert locals()[v] is not None, f"{v} is None, Args: {args}"
            gflownet = LogPartitionVarianceGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
            )

        assert gflownet is not None, f"gflownet is None, Args: {args}"

    # Create replay buffer if needed
    replay_buffer = None
    object_type_mapping = {
        "TB": "trajectories",
        "SubTB": "trajectories",
        "ZVar": "trajectories",
        "DB": "transitions",
        "ModifiedDB": "transitions",
        "FM": "states",
    }
    if args.replay_buffer_size > 0:
        if args.diverse_replay_buffer:
            replay_buffer = NormBasedDiversePrioritizedReplayBuffer(
                env,
                capacity=args.replay_buffer_size,
                cutoff_distance=args.cutoff_distance,
                p_norm_distance=args.p_norm_distance,
            )
        else:
            replay_buffer = ReplayBuffer(
                env,
                capacity=args.replay_buffer_size,
                prioritized=False,
            )

    # Move the gflownet to the GPU.
    gflownet = gflownet.to(device_str)

    # 3. Create the optimizer
    non_logz_params = [
        v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
    ]
    if "logZ" in dict(gflownet.named_parameters()):
        logz_params = [dict(gflownet.named_parameters())["logZ"]]
    else:
        logz_params = []

    params = [
        {"params": non_logz_params, "lr": args.lr},
        # Log Z gets dedicated learning rate (typically higher).
        {"params": logz_params, "lr": args.lr_Z},
    ]
    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.states_from_batch_shape((0,))

    states_visited = 0
    n_iterations = ceil(args.n_trajectories / args.batch_size)
    per_node_batch_size = args.batch_size // world_size
    validation_info = {"l1_dist": float("inf")}
    discovered_modes = set()
    is_on_policy = args.replay_buffer_size == 0
    print("+ n_iterations = ", n_iterations)
    print("+ per_node_batch_size = ", per_node_batch_size)

    # Timing.
    total_sample_time, total_to_train_samples_time = 0, 0
    total_loss_time, total_loss_backward_time = 0, 0
    total_opt_time, total_rest_time = 0, 0
    total_average_time = 0
    total_bar0_time, total_bar1_time = 0, 0
    time_start = time.time()

    if args.profile:
        keep_active = args.trajectories_to_profile // args.batch_size
        prof = profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=keep_active, repeat=1
            ),
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    if args.distributed:
        # Create and start error handler.
        def cleanup():
            print(f"Process {rank}: Cleaning up...")

        rank = os.environ["RANK"]
        world_size = os.environ["WORLD_SIZE"]
        handler = DistributedErrorHandler(
            device_str,
            rank,
            world_size,
            cleanup_callback=cleanup,
        )
        # handler.start()

    if args.distributed:
        world_size = torch.distributed.get_world_size()
    for iteration in trange(n_iterations):

        iteration_start = time.time()

        # Time sample_trajectories method.
        sample_start = time.time()

        # Use the optional profiler.
        if args.profile:
            prof.step()
            if iteration >= 1 + 1 + keep_active:
                break

        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=is_on_policy,
            save_estimator_outputs=False,
        )
        sample_end = time.time()
        sample_time = sample_end - sample_start
        total_sample_time += sample_time

        # Time to_training_samples method.
        to_train_samples_start = time.time()
        training_samples = gflownet.to_training_samples(trajectories)

        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                training_objects = replay_buffer.sample(
                    n_trajectories=per_node_batch_size
                )
        else:
            training_objects = training_samples
        to_train_samples_end = time.time()
        to_train_samples_time = to_train_samples_end - to_train_samples_start
        total_to_train_samples_time += to_train_samples_time

        optimizer.zero_grad()

        # Time the loss computation
        loss_start = time.time()
        loss = gflownet.loss(
            env,
            training_objects,
            reduction="sum" if args.distributed or args.loss == "SubTB" else "mean",
        )

        # Normalize the loss by the local batch size if distributed
        if args.distributed:
            loss = loss / (per_node_batch_size)
        loss_end = time.time()
        loss_time = loss_end - loss_start
        total_loss_time += loss_time

        bar0_start = time.time()
        if args.distributed:
            dist.barrier()
        bar0_end = time.time()
        total_bar0_time += bar0_end - bar0_start

        # Time backpropagation computation.
        loss_backward_start = time.time()
        loss.backward()
        loss_backward_end = time.time()
        loss_backward_time = loss_backward_end - loss_backward_start
        total_loss_backward_time += loss_backward_time

        # Time optimizer step.
        opt_start = time.time()
        optimizer.step()
        opt_end = time.time()
        opt_time = opt_end - opt_start
        total_opt_time += opt_time

        bar1_start = time.time()
        if args.distributed:
            dist.barrier()
        bar1_end = time.time()
        total_bar1_time += bar1_end - bar1_start

        average_start = time.time()
        if args.distributed and (iteration % args.average_every == 0):
            print("before averaging model, iteration = ", iteration)
            average_models(gflownet)
            print("after averaging model, iteration = ", iteration)
        average_end = time.time()
        average_time = average_end - average_start
        total_average_time += average_time

        # Keep track of trajectories / states.
        last_states = trajectories.last_states
        last_states = cast(DiscreteStates, last_states)
        visited_terminating_states.extend(last_states)
        states_visited += len(trajectories)

        # Calculate how long this iteration took.
        iteration_time = time.time() - iteration_start
        rest_time = iteration_time - sum(
            [
                sample_time,
                to_train_samples_time,
                loss_time,
                loss_backward_time,
                opt_time,
                average_time,
            ]
        )

        log_this_iter = (
            iteration % args.validation_interval == 0
        ) or iteration == n_iterations - 1

        print(
            "before distributed -- orig_shape={}".format(
                visited_terminating_states.tensor.shape
            )
        )
        if args.distributed and log_this_iter:
            try:
                all_visited_terminating_states = gather_distributed_data(
                    visited_terminating_states.tensor
                )
            except Exception as e:
                print("Process {}: Caught error: {}".format(my_rank, e))
                # handler.signal_error()
                sys.exit(1)
        else:
            all_visited_terminating_states = visited_terminating_states.tensor

        if my_rank == 0:
            print(
                "after distributed -- gathered_shape={}, orig_shape={}".format(
                    all_visited_terminating_states.shape,
                    visited_terminating_states.tensor.shape,
                )
            )

        # If we are on the master node, calculate the validation metrics.
        if my_rank == 0:
            to_log = {
                "loss": loss.item(),
                "states_visited": states_visited,
                "sample_time": sample_time,
                "to_train_samples_time": to_train_samples_time,
                "loss_time": loss_time,
                "loss_backward_time": loss_backward_time,
                "opt_time": opt_time,
                "average_time": average_time,
                "rest_time": rest_time,
            }

            if use_wandb:
                wandb.log(to_log, step=iteration)

            if log_this_iter:
                print("logging thjs iteration!")
                validation_info, discovered_modes = validate_hypergrid(
                    env,
                    gflownet,
                    args.validation_samples,
                    all_visited_terminating_states,
                    discovered_modes,
                )

                if use_wandb:
                    wandb.log(validation_info, step=iteration)

                to_log.update(validation_info)
                tqdm.write(f"{iteration}: {to_log}")

    time_end = time.time()
    total_time = time_end - time_start
    total_rest_time = total_time - sum(
        [
            total_sample_time,
            total_to_train_samples_time,
            total_loss_time,
            total_loss_backward_time,
            total_opt_time,
            total_average_time,
        ]
    )

    if args.distributed:
        dist.barrier()

    if my_rank == 0:
        to_log = {
            "total_sample_time": total_sample_time,
            "total_to_train_samples_time": total_to_train_samples_time,
            "total_loss_time": total_loss_time,
            "total_actor_barrier_time": total_bar0_time,
            "total_loss_backward_time": total_loss_backward_time,
            "total_opt_time": total_opt_time,
            "total_average_time": total_average_time,
            "total_learner_barrier_time": total_bar1_time,
            "total_rest_time": total_rest_time,
            "total_time": total_time,
        }

        print("+ Final timing.")
        for k, v in to_log.items():
            print("  {}: {:.6f}".format(k, v))

    if args.profile:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        prof.export_chrome_trace("trace.json")
    try:
        return validation_info["l1_dist"]
    except KeyError:
        print(validation_info.keys())
        return validation_info["n_modes_found"]


def validate_hypergrid(
    env,
    gflownet,
    n_validation_samples,
    visited_terminating_states: torch.Tensor | None,
    discovered_modes,
):
    # Standard validation shared across envs.
    validation_info, visited_terminating_states = validate(
        env,
        gflownet,
        n_validation_samples,
        visited_terminating_states,
    )
    # validation_info = {}

    # Add the mode counting metric.
    states, scale = visited_terminating_states, env.scale_factor
    mode_reward_threshold = 1.0  # Assumes height >= 5. TODO - verify.

    # Modes will have a reward greater than 1.
    modes = states[env.reward(states) >= mode_reward_threshold]
    modes_found = set([tuple(s.tolist()) for s in modes])
    discovered_modes.update(modes_found)
    validation_info["n_modes_found"] = len(discovered_modes)
    print(len(discovered_modes))

    # Old way of counting modes -- potentially buggy - to be removed.
    # # Add the mode counting metric.
    # states, scale = visited_terminating_states.tensor, env.scale_factor
    # normalized_states = ((states * scale) - (scale / 2) * (env.height - 1)).abs()

    # modes = torch.all(
    #     (normalized_states > (0.3 * scale) * (env.height - 1))
    #     & (normalized_states <= (0.4 * scale) * (env.height - 1)),
    #     dim=-1,
    # )
    # modes_found = set([tuple(s.tolist()) for s in states[modes.bool()]])
    # discovered_modes.update(modes_found)
    # validation_info["n_modes_found"] = len(discovered_modes)

    return validation_info, discovered_modes


if __name__ == "__main__":
    parser = ArgumentParser()

    # Machine setting.
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed, if 0 then a random seed is used",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Prevent CUDA usage",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Initalizes distributed computation (torch.distributed)",
    )

    # Environment settings.
    parser.add_argument(
        "--num_agent_groups",
        type=int,
        default=1,
        help="Number of agents learning together",
    )

    # Environment settings.
    parser.add_argument(
        "--average_every",
        type=int,
        default=20,
        help="Number of epochs after which we average model across all agents",
    )

    # Environment settings.
    parser.add_argument(
        "--ndim",
        type=int,
        default=2,
        help="Number of dimensions in the environment",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=8,
        help="Height of the environment",
    )
    parser.add_argument(
        "--R0",
        type=float,
        default=0.1,
        help="Environment's R0",
    )
    parser.add_argument(
        "--R1",
        type=float,
        default=0.5,
        help="Environment's R1",
    )
    parser.add_argument(
        "--R2",
        type=float,
        default=2.0,
        help="Environment's R2",
    )

    # Misc settings.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=10000,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )
    parser.add_argument(
        "--diverse_replay_buffer",
        action="store_true",
        help="Use a diverse replay buffer",
    )

    # Loss settings.
    parser.add_argument(
        "--loss",
        type=str,
        choices=["FM", "TB", "DB", "SubTB", "ZVar", "ModifiedDB"],
        default="FM",
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

    # Policy architecture.
    parser.add_argument(
        "--tabular",
        action="store_true",
        help="Use a lookup table for F, PF, PB instead of an estimator",
    )
    parser.add_argument(
        "--uniform_pb",
        action="store_true",
        help="Use a uniform PB",
    )
    parser.add_argument(
        "--tied",
        action="store_true",
        help="Tie the parameters of PF, PB, and F",
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

    # Training settings.
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

    # Validation settings.
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

    # WandB settings.
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="torchgfn_multinode_hypergrid",
        help="Name of the wandb project. If empty, don't use wandb",
    )
    parser.add_argument(
        "--wandb_local",
        action="store_true",
        help="Stores wandb results locally, to be uploaded later.",
    )

    # Settings relevant to the problem size -- toggle off for larger problems.
    parser.add_argument(
        "--calculate_all_states",
        action="store_true",
        help="Enumerates all states.",
    )
    parser.add_argument(
        "--calculate_partition",
        action="store_true",
        help="Calculates the true partition function.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profiles the execution using PyTorch Profiler.",
    )
    parser.add_argument(
        "--trajectories_to_profile",
        type=int,
        default=2048,
        help="Number of trajectories to profile using the Pytorch Profiler."
        + " Preferably, a multiple of batch size.",
    )

    args = parser.parse_args()
    result = main(args)
    print("+ Training complete - final_score={:.6f}".format(result))
