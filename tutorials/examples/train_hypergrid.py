r"""
The goal of this script is to reproduce some of the published results on the HyperGrid
environment. Run one of the following commands to reproduce some of the results in
[Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

python train_hypergrid.py --ndim 4 --height 8 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}
python train_hypergrid.py --ndim 2 --height 64 --R0 {0.1, 0.01, 0.001} --tied {--uniform_pb} --loss {TB, DB}

And run one of the following to reproduce some of the results in
[Learning GFlowNets from partial episodes for improved convergence and stability](https://arxiv.org/abs/2209.12782)
python train_hypergrid.py --ndim {2, 4} --height 12 --R0 {1e-3, 1e-4} --tied --loss {TB, DB, SubTB}

SELECTIVE AVERAGING:
This script also supports selective model averaging for distributed training, where instead of
averaging all models, the worst performing models are replaced with averaged weights from the
better performing ones. Use the following flags:

--use_selective_averaging: Enable selective averaging instead of standard averaging
--replacement_ratio 0.2: Replace the worst 20% of models (adjustable 0.0-1.0)
--averaging_strategy mean: How to combine good models ("mean", "weighted_mean", "best_only")
--momentum 0.0: Momentum factor for combining with previous weights (0.0-1.0, default 0.0)

Example with selective averaging:
python train_hypergrid.py --distributed --use_selective_averaging --replacement_ratio 0.3 --averaging_strategy mean --momentum 0.1

This script also provides a function `get_exact_P_T` that computes the exact terminating state
distribution for the HyperGrid environment, which is useful for evaluation and visualization.
"""

import os
import time
from argparse import ArgumentParser
from math import ceil
from typing import Tuple, cast

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from matplotlib.gridspec import GridSpec
from torch.profiler import ProfilerActivity, profile
from tqdm import trange

from gfn.containers import NormBasedDiversePrioritizedReplayBuffer, ReplayBuffer
from gfn.containers.replay_buffer_manager import ReplayBufferManager
from gfn.estimators import DiscretePolicyEstimator, Estimator, ScalarEstimator
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    GFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.states import DiscreteStates
from gfn.utils.common import Timer, set_seed
from gfn.utils.distributed import DistributedContext, initialize_distributed_compute
from gfn.utils.modules import MLP, DiscreteUniform, Tabular
from tutorials.examples.multinode.spawn_policy import (
    AsyncSelectiveAveragingPolicy,
    AverageAllPolicy,
)


class ModesReplayBufferManager(ReplayBufferManager):
    def __init__(
        self,
        env: HyperGrid,
        rank: int,
        num_training_ranks: int,
        diverse_replay_buffer: bool = False,
        capacity: int = 10000,
        remote_manager_rank: int | None = None,
    ):
        super().__init__(
            env,
            rank,
            num_training_ranks,
            scoring_function=self.scoring_function,
            diverse_replay_buffer=diverse_replay_buffer,
            capacity=capacity,
            remote_manager_rank=remote_manager_rank,
        )
        self.discovered_modes = set()
        self.env = env

    def scoring_function(self, obj) -> float:
        modes_found = self.env.modes_found(obj.terminating_states)
        score = len(modes_found - self.discovered_modes)
        self.discovered_modes.update(modes_found)
        return float(score)

    def _compute_metadata(self) -> dict:
        return {"n_modes_found": len(self.discovered_modes)}


def get_exact_P_T(env: HyperGrid, gflownet: GFlowNet) -> torch.Tensor:
    r"""Evaluates the exact terminating state distribution P_T for HyperGrid.

    For each state s', the terminating state probability is computed as:

    .. math::
        P_T(s') = u(s') P_F(s_f | s')

    where u(s') satisfies the recursion:

    .. math::
        u(s') = \sum_{s \in \text{Par}(s')} u(s) P_F(s' | s)

    with the base case u(s_0) = 1.

    Args:
        env: The HyperGrid environment
        gflownet: The GFlowNet model

    Returns:
        The exact terminating state distribution as a tensor
    """
    if env.ndim != 2:
        raise ValueError("plotting is only supported for 2D environments")

    grid = env.all_states
    assert grid is not None, "all_states is not implemented in the environment"

    # Get the forward policy distribution for all states
    with torch.no_grad():
        # Handle both FM and other GFlowNet types
        policy: Estimator = cast(
            Estimator, gflownet.logF if isinstance(gflownet, FMGFlowNet) else gflownet.pf
        )

        estimator_outputs = policy(grid)
        dist = policy.to_probability_distribution(grid, estimator_outputs)
        probabilities = torch.exp(dist.logits)  # Get raw probabilities

    u = torch.ones(grid.batch_shape)

    indices = env.all_indices()
    for index in indices[1:]:
        parents = [
            tuple(list(index[:i]) + [index[i] - 1] + list(index[i + 1 :]) + [i])
            for i in range(len(index))
            if index[i] > 0
        ]
        parents_tensor = torch.tensor(parents)
        parents_indices = parents_tensor[:, :-1].long()  # All but last column for u
        action_indices = parents_tensor[:, -1].long()  # Last column for probabilities

        # Compute u values for parent states.
        parent_u_values = []
        for p in parents_indices:
            grid_idx = torch.all(grid.tensor == p, 1)  # index along flattened grid.
            parent_u_values.append(u[grid_idx])
            # parent_u_values.append(u[tuple(p.tolist())])
            # # torch.all(grid.tensor == p, 1)
        parent_u_values = torch.stack(parent_u_values)
        # parent_u_values = torch.stack([u[tuple(p.tolist())] for p in parents_indices])

        # Compute probabilities for parent transitions.
        parent_probs = []
        for p, a in zip(parents_indices, action_indices):
            grid_idx = torch.all(grid.tensor == p, 1)  # index along flattened grid.
            parent_probs.append(probabilities[grid_idx, a])
        parent_probs = torch.stack(parent_probs)

        u[indices.index(index)] = torch.sum(parent_u_values * parent_probs)

    return (u * probabilities[..., -1]).detach().cpu()


def _sample_new_strategy(
    args,
    agent_group_id: int,
    iteration: int,
    prev_eps: float,
    prev_temp: float,
    prev_noisy: int,
) -> dict:
    """Select a new exploration strategy, including noisy layers.

    The strategy only defines exploration-time parameters and the count of
    noisy layers to use when building/rebuilding the networks.

    We pick deterministically from a small candidate pool, excluding the
    previous configuration when possible, to ensure diversity across
    restarts without requiring synchronization.

    Returns:
        A dict with keys: name, epsilon, temperature, n_noisy_layers,
        and noisy_std_init (if present in args, default 0.5 otherwise).
    """
    # TODO: Generate a new exploration strategy instead of selecting from a pre-defined
    # list.
    candidates = [
        {"name": "on_policy", "epsilon": 0.0, "temperature": 1.0, "n_noisy_layers": 0},
        {"name": "epsilon_0.1", "epsilon": 0.1, "temperature": 1.0, "n_noisy_layers": 0},
        {"name": "temp_1.5", "epsilon": 0.0, "temperature": 1.5, "n_noisy_layers": 0},
        {"name": "noisy_1", "epsilon": 0.0, "temperature": 1.0, "n_noisy_layers": 1},
        {
            "name": "noisy_2_temp_1.5",
            "epsilon": 0.0,
            "temperature": 1.5,
            "n_noisy_layers": 2,
        },
    ]
    choices = [
        c
        for c in candidates
        if (
            c["epsilon"] != prev_eps
            or c["temperature"] != prev_temp
            or c["n_noisy_layers"] != prev_noisy
        )
    ]
    if not choices:
        choices = candidates
    idx_seed = int(args.seed) + int(agent_group_id) * 7919 + int(iteration) * 104729
    idx = idx_seed % len(choices)
    strat = choices[idx]
    strat["noisy_std_init"] = float(getattr(args, "agent_noisy_std_init", 0.5))
    return strat


def _make_optimizer_for(gflownet, args) -> torch.optim.Optimizer:
    """Build a fresh Adam optimizer for a (re)built GFlowNet with logZ group."""
    named = dict(gflownet.named_parameters())
    non_logz = [v for k, v in named.items() if k != "logZ"]
    logz = [named["logZ"]] if "logZ" in named else []

    return torch.optim.Adam(
        [{"params": non_logz, "lr": args.lr}, {"params": logz, "lr": args.lr_Z}]
    )


def set_up_fm_gflownet(args, env, preprocessor, agent_group_list, my_agent_group_id):
    """Returns a FM GFlowNet."""
    # We need a LogEdgeFlowEstimator.
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
    else:
        module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )

    estimator = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    return FMGFlowNet(estimator)


def set_up_pb_pf_estimators(
    args, env, preprocessor, agent_group_list, my_agent_group_id
):
    """Returns a pair of estimators for the forward and backward policies."""
    if args.tabular:
        pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        if not args.uniform_pb:
            pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
    else:
        # Forward module: honor per-agent noisy layers for exploration diversity.
        pf_module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            n_noisy_layers=getattr(args, "agent_n_noisy_layers", 0),
            std_init=getattr(args, "agent_noisy_std_init", 0.5),
        )
        if not args.uniform_pb:
            # Backward module: if sharing trunk (tied), PB may only add at most one
            # noisy layer (its output) to remain compatible with the shared trunk.
            pb_module = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk=(
                    pf_module.trunk
                    if args.tied and isinstance(pf_module.trunk, torch.nn.Module)
                    else None
                ),
                n_noisy_layers=(
                    1 if getattr(args, "agent_n_noisy_layers", 0) > 0 else 0
                ),
                std_init=getattr(args, "agent_noisy_std_init", 0.5),
            )
    if args.uniform_pb:
        pb_module = DiscreteUniform(env.n_actions - 1)

    for v in ["pf_module", "pb_module"]:
        assert locals()[v] is not None, f"{v} is None, Args: {args}"

    assert pf_module is not None
    assert pb_module is not None
    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )

    return (pf_estimator, pb_estimator)


def set_up_logF_estimator(
    args, env, preprocessor, agent_group_list, my_agent_group_id, pf_module
):
    """Returns a LogStateFlowEstimator."""
    if args.tabular:
        module = Tabular(n_states=env.n_states, output_dim=1)
    else:
        module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=(
                pf_module.trunk
                if args.tied and isinstance(pf_module.trunk, torch.nn.Module)
                else None
            ),
        )

    return ScalarEstimator(module=module, preprocessor=preprocessor)


def set_up_gflownet(args, env, preprocessor, agent_group_list, my_agent_group_id):
    """Returns a GFlowNet complete with the required estimators."""
    # Initialize per-agent exploration strategy.
    # Default (tests stable): on-policy, no noisy layers.
    # When --use_random_strategies is provided, sample a random initial strategy.
    if getattr(args, "use_random_strategies", False):
        init_cfg = _sample_new_strategy(
            args,
            agent_group_id=my_agent_group_id,
            iteration=0,
            prev_eps=9999.0,
            prev_temp=9999.0,
            prev_noisy=9999,
        )
        args.agent_epsilon = float(init_cfg.get("epsilon", 0.0))
        args.agent_temperature = float(init_cfg.get("temperature", 1.0))
        args.agent_n_noisy_layers = int(init_cfg.get("n_noisy_layers", 0))
        args.agent_noisy_std_init = float(init_cfg.get("noisy_std_init", 0.5))
    else:
        # Disable off-policy training.
        args.agent_epsilon = 0.0
        args.agent_temperature = 1.0
        args.agent_n_noisy_layers = 0
        args.agent_noisy_std_init = 0.5

    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (forward, backward, logZ/logF) estimators for DB, TB.

    if args.loss == "FM":
        return set_up_fm_gflownet(
            args,
            env,
            preprocessor,
            agent_group_list,
            my_agent_group_id,
        )
    else:
        # We need a DiscretePFEstimator and a DiscretePBEstimator.
        pf_estimator, pb_estimator = set_up_pb_pf_estimators(
            args,
            env,
            preprocessor,
            agent_group_list,
            my_agent_group_id,
        )
        assert pf_estimator is not None
        assert pb_estimator is not None

        if args.loss == "ModifiedDB":
            return ModifiedDBGFlowNet(pf_estimator, pb_estimator)

        elif args.loss == "TB":
            return TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)

        elif args.loss == "ZVar":
            return LogPartitionVarianceGFlowNet(pf=pf_estimator, pb=pb_estimator)

        elif args.loss in ("DB", "SubTB"):
            # We also need a LogStateFlowEstimator.
            logF_estimator = set_up_logF_estimator(
                args,
                env,
                preprocessor,
                agent_group_list,
                my_agent_group_id,
                pf_estimator,
            )

            if args.loss == "DB":
                return DBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                )
            elif args.loss == "SubTB":
                return SubTBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    weighting=args.subTB_weighting,
                    lamda=args.subTB_lambda,
                )


def plot_results(env, gflownet, l1_distances, validation_steps):
    # Create figure with 3 subplots with proper spacing
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 0.1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])  # Colorbar axis
    ax3 = fig.add_subplot(gs[3])

    # Get distributions and find global min/max for consistent color scaling
    true_dist = env.true_dist.reshape(args.height, args.height).cpu().numpy()
    learned_dist = get_exact_P_T(env, gflownet).reshape(args.height, args.height).numpy()

    # Ensure consistent orientation by transposing
    true_dist = true_dist.T
    learned_dist = learned_dist.T

    vmin = min(true_dist.min(), learned_dist.min())
    vmax = max(true_dist.max(), learned_dist.max())

    # True reward distribution
    im1 = ax1.imshow(
        true_dist,
        cmap="viridis",
        interpolation="none",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("True Distribution")

    # Learned reward distribution
    _ = ax2.imshow(
        learned_dist,
        cmap="viridis",
        interpolation="none",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("Learned Distribution")

    # Add colorbar in its own axis
    plt.colorbar(im1, cax=cax)

    # L1 distances over time
    states_per_validation = args.batch_size * args.validation_interval
    validation_states = [i * states_per_validation for i in range(len(l1_distances))]
    ax3.plot(validation_states, l1_distances)
    ax3.set_xlabel("States Visited")
    ax3.set_ylabel("L1 Distance")
    ax3.set_title("L1 Distance Evolution")
    ax3.set_yscale("log")  # Set log scale for y-axis

    plt.tight_layout()
    plt.show()
    plt.close()


def main(args):  # noqa: C901
    """Trains a GFlowNet on the Hypergrid Environment, potentially distributed."""

    if args.half_precision:
        torch.set_default_dtype(torch.bfloat16)

    print("+ Using default dtype: ", torch.get_default_dtype())

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Check if plotting is allowed.
    if args.plot:
        if args.wandb_project:
            raise ValueError("plot argument is incompatible with wandb_project")
        if args.ndim != 2:
            raise ValueError("plotting is only supported for 2D environments")

    # Initialize distributed compute.
    if args.distributed:
        distributed_context = initialize_distributed_compute(
            dist_backend=args.dist_backend,
            num_remote_buffers=args.num_remote_buffers,
            num_agent_groups=args.num_agent_groups,
        )

        print(f"Running distributed with following settings: {distributed_context}")
    else:
        distributed_context = DistributedContext(
            my_rank=0, world_size=1, num_training_ranks=1, agent_group_size=1
        )

    set_seed(args.seed + distributed_context.my_rank)

    # Initialize the environment.
    env = HyperGrid(
        args.ndim,
        args.height,
        device=device,
        reward_fn_str="original",
        reward_fn_kwargs={
            "R0": args.R0,
            "R1": args.R1,
            "R2": args.R2,
        },
        calculate_partition=args.calculate_partition,
        store_all_states=args.store_all_states,
    )

    if args.distributed and distributed_context.is_buffer_rank():
        if distributed_context.assigned_training_ranks is None:
            num_training_ranks = 0
        else:
            num_training_ranks = len(distributed_context.assigned_training_ranks)

        replay_buffer_manager = ModesReplayBufferManager(
            env=env,
            rank=distributed_context.my_rank,
            num_training_ranks=num_training_ranks,
            diverse_replay_buffer=args.diverse_replay_buffer,
            capacity=args.global_replay_buffer_size,
        )
        replay_buffer_manager.run()
        return 0.0

    # Initialize WandB.
    use_wandb = args.wandb_project != ""
    if use_wandb:
        if args.wandb_local:
            os.environ["WANDB_MODE"] = "offline"

        import wandb

        # Generate shared group name for wandb across all processes
        group_name = None
        if args.distributed:
            # Use the training group and perform in-place broadcasts
            pg = distributed_context.train_global_group
            is_root = distributed_context.my_rank == 0

            if is_root:
                group_name = wandb.util.generate_id()
                group_name_bytes = group_name.encode("utf-8")
                group_name_len_tensor = torch.tensor(
                    [len(group_name_bytes)], dtype=torch.long
                )
            else:
                group_name_bytes = None
                group_name_len_tensor = torch.zeros(1, dtype=torch.long)

            # Broadcast the length
            dist.broadcast(group_name_len_tensor, src=0, group=pg)
            group_name_len = int(group_name_len_tensor.item())

            # Broadcast the payload
            if is_root:
                assert group_name_bytes is not None
                payload = torch.tensor(list(group_name_bytes), dtype=torch.uint8)
            else:
                payload = torch.empty(group_name_len, dtype=torch.uint8)

            dist.broadcast(payload, src=0, group=pg)
            group_name = bytes(payload.tolist()).decode("utf-8")
        else:
            group_name = wandb.util.generate_id()

        wandb.init(project=args.wandb_project, group=group_name)
        wandb.config.update(args)

    # Initialize the preprocessor.
    preprocessor = KHotPreprocessor(height=args.height, ndim=args.ndim)

    # Builder closure to create a fresh model + optimizer (used by spawn policy as well)
    def _model_builder() -> Tuple[GFlowNet, torch.optim.Optimizer]:
        model = set_up_gflownet(
            args,
            env,
            preprocessor,
            distributed_context.agent_groups,
            distributed_context.agent_group_id,
        )
        assert model is not None
        model = model.to(device)
        optim = _make_optimizer_for(model, args)
        return model, optim

    # Build the initial model and optimizer
    gflownet, optimizer = _model_builder()

    # Create replay buffer if needed
    replay_buffer = None

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
                prioritized_capacity=False,
                remote_manager_rank=distributed_context.assigned_buffer,
                remote_buffer_freq=1,
            )

    gflownet = gflownet.to(device)

    n_iterations = ceil(args.n_trajectories / args.batch_size)
    per_node_batch_size = args.batch_size // distributed_context.world_size
    validation_info = {"l1_dist": float("inf")}
    modes_found = set()
    # n_pixels_per_mode = round(env.height / 10) ** env.ndim
    # Note: on/off-policy depends on the current strategy; recomputed inside the loop.

    print("+ n_iterations = ", n_iterations)
    print("+ per_node_batch_size = ", per_node_batch_size)

    # Initialize the profiler.
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

        rank = torch.distributed.get_rank()
        torch.distributed.get_world_size()

    # Initialize some variables before the training loop.
    timing = {}
    time_start = time.time()
    l1_distances, validation_steps = [], []

    # Used for calculating the L1 distance across all nodes.
    all_visited_terminating_states = env.states_from_batch_shape((0,))

    # Barrier for pre-processing. Wait for all processes to reach this point before starting training.
    with Timer(
        timing, "Pre-processing_barrier", enabled=(args.timing and args.distributed)
    ):
        if args.distributed and args.timing:
            dist.barrier(group=distributed_context.train_global_group)

    # Set up averaging policy (called every iteration; internal guard checks cadence/distributed)
    averaging_policy = None
    if args.distributed:
        if args.use_selective_averaging:
            averaging_policy = AsyncSelectiveAveragingPolicy(  # type: ignore[abstract]
                model_builder=_model_builder,
                average_every=args.average_every,
                replacement_ratio=args.replacement_ratio,
                averaging_strategy=args.averaging_strategy,
                momentum=args.momentum,
            )
        else:
            averaging_policy = AverageAllPolicy(average_every=args.average_every)

    # Training loop.
    pbar = trange(n_iterations)
    for iteration in pbar:
        iteration_start = time.time()

        # Keep track of visited terminating states on this node.
        with Timer(
            timing, "track_visited_states", enabled=args.timing
        ) as visited_states_timer:
            visited_terminating_states = env.states_from_batch_shape((0,))

            # Profiler.
            if args.profile:
                prof.step()
                if iteration >= 1 + 1 + keep_active:
                    break

        # Restarts are handled by selective averaging policy via spawn; no-op here.

        # Sample trajectories.
        with Timer(timing, "generate_samples", enabled=args.timing) as sample_timer:
            # Determine on-policy for this iteration based on current strategy.
            is_on_policy_iter = (
                (args.replay_buffer_size == 0)
                and (float(getattr(args, "agent_epsilon", 0.0)) == 0.0)
                and (float(getattr(args, "agent_temperature", 1.0)) == 1.0)
                and (int(getattr(args, "agent_n_noisy_layers", 0)) == 0)
            )
            trajectories = gflownet.sample_trajectories(
                env,
                n=args.batch_size,
                save_logprobs=is_on_policy_iter,  # Reuse on-policy log-probs.
                save_estimator_outputs=not is_on_policy_iter,  # Off-policy caches estimator outputs.
                epsilon=float(getattr(args, "agent_epsilon", 0.0)),
                temperature=float(getattr(args, "agent_temperature", 1.0)),
            )

        # Training objects (incl. possible replay buffer sampling).
        with Timer(
            timing, "to_training_samples", enabled=args.timing
        ) as to_train_samples_timer:
            training_samples = gflownet.to_training_samples(trajectories)

            score = None
            if replay_buffer is not None:
                with torch.no_grad():
                    score = replay_buffer.add(training_samples)
                    training_objects = replay_buffer.sample(
                        n_samples=per_node_batch_size
                    )
            else:
                training_objects = training_samples

        # Loss.
        with Timer(timing, "calculate_loss", enabled=args.timing) as loss_timer:

            optimizer.zero_grad()
            # Recompute whether we are off-policy for loss logprob recalculation.
            is_on_policy_iter = (
                (args.replay_buffer_size == 0)
                and (float(getattr(args, "agent_epsilon", 0.0)) == 0.0)
                and (float(getattr(args, "agent_temperature", 1.0)) == 1.0)
                and (int(getattr(args, "agent_n_noisy_layers", 0)) == 0)
            )
            loss = gflownet.loss(
                env,
                training_objects,  # type: ignore
                recalculate_all_logprobs=(not is_on_policy_iter),
                reduction="sum" if args.distributed or args.loss == "SubTB" else "mean",  # type: ignore
            )

            # Normalize the loss by the local batch size if distributed.
            if args.distributed:
                loss = loss / (per_node_batch_size)

        # Barrier.
        with Timer(
            timing, "barrier 0", enabled=(args.timing and args.distributed)
        ) as bar0_timer:
            if args.distributed and args.timing:
                dist.barrier(group=distributed_context.train_global_group)

        # Backpropagation.
        with Timer(timing, "loss_backward", enabled=args.timing) as loss_backward_timer:
            loss.backward()

        # Optimization.
        with Timer(timing, "optimizer", enabled=args.timing) as opt_timer:
            optimizer.step()

        # Barrier.
        with Timer(
            timing, "barrier 1", enabled=(args.timing and args.distributed)
        ) as bar1_timer:
            if args.distributed and args.timing:
                dist.barrier(group=distributed_context.train_global_group)

        # Model averaging.
        averaging_info = {}
        with Timer(
            timing, "averaging_model", enabled=args.timing
        ) as model_averaging_timer:
            if averaging_policy is not None:
                gflownet, optimizer, averaging_info = averaging_policy(
                    iteration=iteration,
                    model=gflownet,
                    optimizer=optimizer,
                    local_metric=score if score is not None else -loss.item(),
                    group=distributed_context.train_global_group,
                )

        # Calculate how long this iteration took.
        iteration_time = time.time() - iteration_start
        rest_time = iteration_time - sum(
            [
                t
                for t in [
                    visited_states_timer.elapsed,
                    sample_timer.elapsed,
                    to_train_samples_timer.elapsed,
                    loss_timer.elapsed,
                    bar0_timer.elapsed,
                    loss_backward_timer.elapsed,
                    opt_timer.elapsed,
                    bar1_timer.elapsed,
                    model_averaging_timer.elapsed,
                ]
                if t is not None
            ]
        )

        log_this_iter = (
            iteration % args.validation_interval == 0
        ) or iteration == n_iterations - 1

        # Keep track of trajectories / states.
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        # If we are on the master node, calculate the validation metrics.
        with Timer(timing, "validation", enabled=args.timing):
            assert visited_terminating_states is not None
            all_visited_terminating_states.extend(visited_terminating_states)
            my_rank = distributed_context.my_rank
            to_log = {
                f"loss_{my_rank}": loss.item(),
                f"sample_time_{my_rank}": sample_timer.elapsed,
                f"to_train_samples_time_{my_rank}": to_train_samples_timer.elapsed,
                f"loss_time_{my_rank}": loss_timer.elapsed,
                f"loss_backward_time_{my_rank}": loss_backward_timer.elapsed,
                f"opt_time_{my_rank}": opt_timer.elapsed,
                f"model_averaging_time_{my_rank}": model_averaging_timer.elapsed,
                f"rest_time_{my_rank}": rest_time,
                f"score_{my_rank}": score,
                f"l1_dist_{my_rank}": None,  # only logged if calculate_partition.
            }
            to_log.update({f"{k}_{my_rank}": v for k, v in averaging_info.items()})

            if log_this_iter:
                validation_info, all_visited_terminating_states = env.validate(
                    gflownet,
                    args.validation_samples,
                    all_visited_terminating_states,
                )
                assert all_visited_terminating_states is not None
                to_log.update({f"{k}_{my_rank}": v for k, v in validation_info.items()})

                if my_rank == 0:
                    if args.distributed:
                        manager_rank = distributed_context.assigned_buffer
                        assert manager_rank is not None
                        metadata = ReplayBufferManager.get_metadata(manager_rank)
                        to_log.update(metadata)
                    else:
                        modes_found.update(env.modes_found(visited_terminating_states))
                        n_modes_found = len(modes_found)
                        to_log["n_modes_found"] = n_modes_found

                    pbar.set_postfix(
                        loss_0=to_log["loss_0"],
                        l1_dist_0=to_log[
                            "l1_dist_0"
                        ],  # only logged if calculate_partition.
                        n_modes_found=to_log["n_modes_found"],
                    )

                if use_wandb:
                    wandb.log(to_log, step=iteration)

        with Timer(timing, "barrier 2", enabled=(args.timing and args.distributed)):
            if args.distributed and args.timing:
                dist.barrier(group=distributed_context.train_global_group)

    print("+ Finished all iterations")
    total_time = time.time() - time_start
    if args.timing:
        timing["total_rest_time"] = [total_time - sum(sum(v) for k, v in timing.items())]

    timing["total_time"] = [total_time]

    if args.distributed:
        dist.barrier(group=distributed_context.train_global_group)
        assert averaging_policy is not None
        try:
            averaging_policy.shutdown()
        except Exception:
            pass

    # Log the final timing results.
    if args.timing:
        if distributed_context.my_rank == 0:
            print("\n" + "=" * 80)
            print("\n Timing information:")
            if args.distributed:
                print("-" * 80)
                print("Distributed run: showing local timings for rank 0 only.")
            print("=" * 80)

        # Print local timings only (avoid collective communication)
        if (not args.distributed) or (distributed_context.my_rank == 0):
            print(f"{'Step Name':<25} {'Time (s)':>12}")
            print("-" * 80)
            for k, v in timing.items():
                print(f"{k:<25} {sum(v):>10.4f}s")

    # Stop the profiler if it's active.
    if args.profile:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        prof.export_chrome_trace("trace.json")

    # Plot the results if requested & possible.
    if args.plot:
        # Create figure with 3 subplots with proper spacing.
        plot_results(env, gflownet, l1_distances, validation_steps)

    try:
        result = validation_info["l1_dist"]
    except KeyError:
        result = validation_info["n_modes_found"]

    if distributed_context.my_rank == 0:
        print("+ Training complete - final_score={:.6f}".format(result))

    if (
        args.distributed
        and distributed_context.is_training_rank()
        and (distributed_context.assigned_buffer is not None)
    ):
        # Send a termination signal to the replay buffer manager.
        ReplayBufferManager.send_termination_signal(distributed_context.assigned_buffer)

    return result


if __name__ == "__main__":
    parser = ArgumentParser()

    # Machine setting.
    parser.add_argument("--seed", type=int, default=4444, help="Random seed.")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Prevent CUDA usage",
    )

    # Distributed settings.
    parser.add_argument(
        "--average_every",
        type=int,
        default=100,
        help="Number of epochs after which we average model across all agents",
    )
    parser.add_argument(
        "--num_agent_groups",
        type=int,
        default=1,
        help="Number of agents learning together",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Initializes distributed computation (torch.distributed)",
    )
    parser.add_argument(
        "--num_remote_buffers",
        type=int,
        default=1,
        help="Number of remote replay buffer managers (only if using distributed computation)",
    )
    parser.add_argument(
        "--global_replay_buffer_size",
        type=int,
        default=10000,
        help="Global replay buffer size (only if using distributed computation)",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="gloo",
        help="Distributed backend to use: gloo, ccl or mpi",
    )

    # Selective averaging settings.
    parser.add_argument(
        "--use_selective_averaging",
        action="store_true",
        help="Use selective averaging instead of standard averaging",
    )

    parser.add_argument(
        "--replacement_ratio",
        type=float,
        default=0.2,
        help="Fraction of worst performing models to replace (0.0 to 1.0)",
    )
    parser.add_argument(
        "--averaging_strategy",
        type=str,
        choices=["mean", "weighted_mean", "best_only", "reset_weights"],
        default="mean",
        help="Strategy for combining good models",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.01,
        help="Momentum factor for combining with previous weights (0.0 = no momentum, 1.0 = keep old weights)",
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

    # Training settings.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=1000,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )
    parser.add_argument(
        "--restart_init_mode",
        type=str,
        default="random",
        choices=["random", "mean_others"],
        help=(
            "How to reinitialize an agent when restarted: "
            "'random' resets with module defaults; 'mean_others' averages only canonical "
            "(shape-compatible) parameters across other training ranks; non-canonical parameters "
            "such as NoisyLinear sigmas remain at default initialization."
        ),
    )
    parser.add_argument(
        "--diverse_replay_buffer",
        action="store_true",
        help="Use a diverse replay buffer",
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
        help=(
            "Total budget of trajectories to train on. "
            "Training iterations = n_trajectories // batch_size"
        ),
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
        help=(
            "Number of hidden layers (of size `hidden_dim`) in the estimators'"
            " neural network modules"
        ),
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
        default="torchgfn",
        help="Name of the wandb project. If empty, don't use wandb",
    )
    parser.add_argument(
        "--wandb_local",
        action="store_true",
        help="Stores wandb results locally, to be uploaded later.",
    )

    # Settings relevant to the problem size -- toggle off for larger problems.
    parser.add_argument(
        "--store_all_states",
        action="store_true",
        default=False,
        help="Whether to store all states.",
    )
    parser.add_argument(
        "--calculate_partition",
        action="store_true",
        default=False,
        help="Whether to calculate the true partition function.",
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
        help=(
            "Number of trajectories to profile using the Pytorch Profiler. "
            "Preferably, a multiple of batch size."
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of true and learned distributions (only works for 2D, incompatible with wandb)",
    )

    parser.add_argument(
        "--timing",
        action="store_true",
        default=True,
        help="Report timing information at the end of training",
    )

    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Use half precision for the model",
    )

    parser.add_argument(
        "--use_random_strategies",
        action="store_true",
        help="Use a random strategy for the initial gflownet and restarts.",
    )
    parser.add_argument(
        "--use_restarts",
        action="store_true",
        help="Use restarts.",
    )

    args = parser.parse_args()
    result = main(args)
