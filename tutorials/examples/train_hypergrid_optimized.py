#!/usr/bin/env python
r"""
Optimized HyperGrid training script with optional torch.compile, vmap, and benchmarking.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

import torch
from torch.func import vmap
from tqdm import tqdm

from gfn.containers import Trajectories
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet.detailed_balance import DBGFlowNet
from gfn.gflownet.flow_matching import FMGFlowNet
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.compile import try_compile_gflownet
from gfn.utils.modules import MLP, DiscreteUniform
from gfn.utils.training import validate


# Local subclasses for benchmarking-only optimizations (no core library changes)
class HyperGridWithTensorStep(HyperGrid):
    def step_tensor(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert states.dtype == torch.long
        device = states.device
        batch = states.shape[0]
        ndim = self.ndim
        exit_idx = self.n_actions - 1

        if actions.ndim == 1:
            actions_idx = actions.view(-1, 1)
        else:
            assert actions.shape[-1] == 1
            actions_idx = actions

        is_exit = actions_idx.squeeze(-1) == exit_idx

        next_states = states.clone()
        non_exit_mask = ~is_exit
        if torch.any(non_exit_mask):
            sel_states = next_states[non_exit_mask]
            sel_actions = actions_idx[non_exit_mask]
            sel_states = sel_states.scatter(-1, sel_actions, 1, reduce="add")
            next_states[non_exit_mask] = sel_states
        if torch.any(is_exit):
            # Ensure exit actions land exactly on the sink state so downstream
            # `is_sink_state` masks match the action padding semantics assumed
            # by `Trajectories` and probability calculations.
            next_states[is_exit] = self.sf.to(device=device)

        next_forward_masks = torch.ones(
            (batch, self.n_actions), dtype=torch.bool, device=device
        )
        next_forward_masks[:, :ndim] = next_states != (self.height - 1)
        next_forward_masks[:, ndim] = True
        return next_states, next_forward_masks, is_exit

    def forward_action_masks(self, states_tensor: torch.Tensor) -> torch.Tensor:
        """Returns forward-action masks for a batch of state tensors."""
        base = states_tensor != (self.height - 1)
        return torch.cat(
            [
                base,
                torch.ones(
                    (states_tensor.shape[0], 1),
                    dtype=torch.bool,
                    device=states_tensor.device,
                ),
            ],
            dim=-1,
        )


class ChunkedHyperGridSampler(Sampler):
    def __init__(self, estimator, chunk_size: int):
        super().__init__(estimator)
        self.chunk_size = int(chunk_size)

    def sample_trajectories(  # noqa: C901
        self,
        env: HyperGridWithTensorStep,
        n: int | None = None,
        states: DiscreteStates | None = None,
        conditions: torch.Tensor | None = None,
        save_estimator_outputs: bool = False,  # unused in chunked fast path
        save_logprobs: bool = False,  # unused in chunked fast path
        **policy_kwargs: Any,
    ):
        assert self.chunk_size > 0
        assert hasattr(env, "step_tensor")
        epsilon = float(policy_kwargs.get("epsilon", 0.0))

        if states is None:
            assert n is not None
            states_obj = env.reset(batch_shape=(n,))
        else:
            states_obj = states

        estimator = self.estimator
        module = getattr(estimator, "module", None)
        assert module is not None
        height = int(env.height)
        exit_idx = env.n_actions - 1

        curr_states = states_obj.tensor
        batch = curr_states.shape[0]
        device = curr_states.device

        forward_masks = env.forward_action_masks(curr_states)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        actions_seq: List[torch.Tensor] = []
        dones_seq: List[torch.Tensor] = []

        def sample_actions_from_logits(
            logits: torch.Tensor, masks: torch.Tensor, eps: float
        ) -> torch.Tensor:
            masked_logits = logits.masked_fill(~masks, float("-inf"))
            probs = torch.softmax(masked_logits, dim=-1)

            if eps > 0.0:
                valid_counts = masks.sum(dim=-1, keepdim=True).clamp_min(1)
                uniform = masks.to(probs.dtype) / valid_counts.to(probs.dtype)
                probs = (1.0 - eps) * probs + eps * uniform

            # Ensure exit actions have probability 1.0 so that they land exactly on
            # the sink state and downstream `is_sink_state` masks match the action
            # padding semantics assumed by `Trajectories` and probability calculations.
            nan_rows = torch.isnan(probs).any(dim=-1)
            if nan_rows.any():
                probs[nan_rows] = 0.0
                probs[nan_rows, exit_idx] = 1.0

            return torch.multinomial(probs, 1)

        def _chunk_loop(
            current_states: torch.Tensor,
            current_masks: torch.Tensor,
            done_mask: torch.Tensor,
        ):
            actions_list: List[torch.Tensor] = []
            dones_list: List[torch.Tensor] = []
            for _ in range(self.chunk_size):
                if done_mask.any():
                    current_masks = current_masks.clone()
                    current_masks[done_mask] = False
                    current_masks[done_mask, exit_idx] = True
                khot = torch.nn.functional.one_hot(
                    current_states, num_classes=height
                ).to(dtype=torch.get_default_dtype())
                khot = khot.view(current_states.shape[0], -1)
                logits = module(khot)
                actions = sample_actions_from_logits(logits, current_masks, epsilon)
                next_states, next_masks, is_exit = env.step_tensor(
                    current_states, actions
                )
                record_actions = actions.clone()

                # Replace actions for already-finished trajectories with the dummy
                # action so that their timeline matches the padded semantics expected
                # by Trajectories (actions.is_dummy aligns with states.is_sink_state[:-1]).
                if done_mask.any():
                    dummy_val = env.dummy_action.to(device=device)
                    record_actions[done_mask] = dummy_val
                actions_list.append(record_actions)
                dones_list.append(is_exit)

                current_states = next_states
                current_masks = next_masks
                done_mask = done_mask | is_exit

                if bool(done_mask.all().item()):
                    break

            return current_states, current_masks, done_mask, actions_list, dones_list

        chunk_fn = _chunk_loop
        if hasattr(torch, "compile"):
            try:
                chunk_fn = torch.compile(_chunk_loop, mode="reduce-overhead")  # type: ignore
            except Exception:
                pass

        while not bool(done.all().item()):
            curr_states, forward_masks, done, actions_chunk, dones_chunk = chunk_fn(
                curr_states, forward_masks, done
            )
            if actions_chunk:
                actions_seq.extend(actions_chunk)
                dones_seq.extend(dones_chunk)

        if actions_seq:
            actions_tsr = torch.stack([a for a in actions_seq], dim=0)
            T = actions_tsr.shape[0]
            s = states_obj.tensor
            states_stack = [s]
            for t in range(T):
                s, fm, is_exit = env.step_tensor(s, actions_tsr[t])
                states_stack.append(s)
            states_tsr = torch.stack(states_stack, dim=0)
            is_exit_seq = torch.stack(dones_seq, dim=0)
            first_exit = torch.argmax(is_exit_seq.to(torch.long), dim=0)
            never_exited = ~is_exit_seq.any(dim=0)
            first_exit = torch.where(
                never_exited, torch.tensor(T - 1, device=device), first_exit
            )
            terminating_idx = first_exit + 1
        else:
            states_tsr = states_obj.tensor.unsqueeze(0)
            actions_tsr = env.actions_from_batch_shape((0, states_tsr.shape[1])).tensor
            terminating_idx = torch.zeros(
                states_tsr.shape[1], dtype=torch.long, device=device
            )

        trajectories = Trajectories(
            env=env,
            states=env.states_from_tensor(states_tsr),
            conditions=None,
            actions=env.actions_from_tensor(actions_tsr),
            terminating_idx=terminating_idx,
            is_backward=False,
            log_rewards=None,
            log_probs=None,
            estimator_outputs=None,
        )
        return trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["FM", "TB", "DB"], default="TB")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--R0", type=float, default=0.1)
    parser.add_argument("--R1", type=float, default=0.5)
    parser.add_argument("--R2", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_logz", type=float, default=1e-1)
    parser.add_argument("--uniform_pb", action="store_true")
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--validation_samples", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device to run on; auto prefers CUDA>MPS>CPU.",
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="Mode passed to torch.compile.",
    )
    parser.add_argument("--use-vmap", action="store_true", help="Use vmap TB loss.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Enable chunked sampler fast path when > 0.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=str(Path.home() / "hypergrid_benchmark.png"),
        help="Output path for benchmark plot.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=50,
        help="Warmup iterations before timing (benchmark mode).",
    )
    return parser.parse_args()


def init_metrics() -> Dict[str, Any]:
    return {
        "validation_info": {"l1_dist": float("inf")},
        "discovered_modes": set(),
        "total_steps": 0,
        "measured_steps": 0,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.benchmark:
        base_scenarios: list[tuple[str, bool, bool, bool]] = [
            ("Baseline", False, False, False),
            (f"Compile ({args.compile_mode})", True, False, False),
            ("Vmap", False, True, False),
            (f"Compile+Vmap ({args.compile_mode})", True, True, False),
        ]
        if args.chunk_size > 0:
            base_scenarios += [
                (f"Chunk ({args.chunk_size})", False, False, True),
                (
                    f"Compile+Chunk ({args.compile_mode},{args.chunk_size})",
                    True,
                    False,
                    True,
                ),
                (f"Chunk+Vmap ({args.chunk_size})", False, True, True),
                (
                    f"Compile+Chunk+Vmap ({args.compile_mode},{args.chunk_size})",
                    True,
                    True,
                    True,
                ),
            ]
        scenarios = base_scenarios
        results: list[dict[str, Any]] = []
        for label, enable_compile, use_vmap, use_chunk in scenarios:
            result = train_with_options(
                args,
                device,
                enable_compile=enable_compile,
                use_vmap=use_vmap,
                warmup_iters=args.warmup_iters,
                quiet=True,
                timing=True,
                record_history=True,
                use_chunk=use_chunk,
            )
            result["label"] = label
            results.append(result)

        baseline_elapsed = results[0]["elapsed"]
        print("Benchmark summary (speedups vs baseline):")
        for result in results:
            speedup = (
                baseline_elapsed / result["elapsed"]
                if result["elapsed"]
                else float("inf")
            )
            print(
                f"- {result['label']}: {result['elapsed']:.2f}s "
                f"({speedup:.2f}x) | compile_mode={result['compile_mode']} "
                f"| vmap={'on' if result['effective_vmap'] else 'off'} "
                f"| chunk={'on' if result.get('chunk_size_effective', 0) > 0 else 'off'}"
            )

        plot_benchmark(results, args.benchmark_output)
        return

    train_with_options(
        args,
        device,
        enable_compile=args.compile,
        use_vmap=args.use_vmap,
        warmup_iters=0,
        quiet=False,
        timing=False,
        record_history=False,
        use_chunk=args.chunk_size > 0,
    )


def train_with_options(
    args: argparse.Namespace,
    device: torch.device,
    *,
    enable_compile: bool,
    use_vmap: bool,
    warmup_iters: int,
    quiet: bool,
    timing: bool,
    record_history: bool,
    use_chunk: bool = False,
) -> dict[str, Any]:
    set_seed(args.seed)
    (
        env,
        gflownet,
        sampler,
        optimizer,
        visited_states,
    ) = build_training_components(args, device, use_chunk=use_chunk)
    metrics = init_metrics()

    compile_mode = args.compile_mode if enable_compile else "none"
    if enable_compile:
        compile_results = try_compile_gflownet(
            gflownet,
            mode=args.compile_mode,
        )
        if not quiet:
            formatted = ", ".join(
                f"{name}:{'✓' if success else 'x'}"
                for name, success in compile_results.items()
            )
            print(f"[compile] {formatted}")

    requested_vmap = use_vmap
    if use_vmap and not isinstance(gflownet, TBGFlowNet):
        if not quiet:
            print("vmap is currently only supported for TBGFlowNet; ignoring flag.")
        use_vmap = False
    effective_vmap = use_vmap

    if warmup_iters > 0:
        run_iterations(
            env,
            gflownet,
            sampler,
            optimizer,
            visited_states,
            metrics,
            args,
            n_iters=warmup_iters,
            use_vmap=use_vmap,
            quiet=True,
            collect_metrics=False,
            track_time=False,
            record_history=False,
        )

    elapsed, history = run_iterations(
        env,
        gflownet,
        sampler,
        optimizer,
        visited_states,
        metrics,
        args,
        n_iters=args.n_iterations,
        use_vmap=use_vmap,
        quiet=quiet,
        collect_metrics=True,
        track_time=timing,
        record_history=record_history,
    )

    if not quiet:
        validation_info = metrics["validation_info"]
        l1 = validation_info.get("l1_dist", float("nan"))
        print(
            f"Finished training | iterations={metrics['measured_steps']} | "
            f"modes={len(metrics['discovered_modes'])} / {env.n_modes} | "
            f"L1 distance={l1:.6f}"
        )

    return {
        "elapsed": elapsed or 0.0,
        "losses": history["losses"] if history else None,
        "iter_times": history["iter_times"] if history else None,
        "compile_mode": compile_mode,
        "use_compile": enable_compile,
        "requested_vmap": requested_vmap,
        "effective_vmap": effective_vmap,
        "chunk_size_effective": (args.chunk_size if use_chunk else 0),
    }


def run_iterations(
    env: HyperGrid,
    gflownet: TBGFlowNet | DBGFlowNet | FMGFlowNet,
    sampler: Sampler,
    optimizer: torch.optim.Optimizer,
    visited_states: DiscreteStates,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    *,
    n_iters: int,
    use_vmap: bool,
    quiet: bool,
    collect_metrics: bool,
    track_time: bool,
    record_history: bool,
) -> tuple[float | None, Dict[str, list[float]] | None]:
    if n_iters <= 0:
        empty_history = {"losses": [], "iter_times": []} if record_history else None
        return (0.0 if track_time else None), empty_history

    iterator: Iterable[int]
    if quiet:
        iterator = range(n_iters)
    else:
        iterator = tqdm(range(n_iters), dynamic_ncols=True)

    start_time = time.perf_counter() if track_time else None
    last_loss = 0.0
    losses_history: list[float] | None = [] if record_history else None
    iter_time_history: list[float] | None = [] if record_history else None

    for _ in iterator:
        iter_start = time.perf_counter() if (track_time or record_history) else None
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
        )

        terminating_states = cast(DiscreteStates, trajectories.terminating_states)
        visited_states.extend(terminating_states)

        optimizer.zero_grad()
        loss = compute_loss(gflownet, env, trajectories, use_vmap=use_vmap)
        loss.backward()
        gflownet.assert_finite_gradients()
        torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 1.0)
        optimizer.step()
        gflownet.assert_finite_parameters()

        metrics["total_steps"] += 1
        if collect_metrics:
            metrics["measured_steps"] += 1

        last_loss = loss.item()
        if (
            record_history
            and (losses_history is not None)
            and (iter_time_history is not None)
        ):
            losses_history.append(last_loss)
            iter_duration = (
                (time.perf_counter() - iter_start) if iter_start is not None else 0.0
            )
            iter_time_history.append(iter_duration)

        if collect_metrics:
            run_validation_if_needed(
                env,
                gflownet,
                visited_states,
                metrics,
                args,
                quiet=quiet,
            )

        if not quiet and isinstance(iterator, tqdm):
            iterator.set_postfix(
                {
                    "loss": last_loss,
                    "trajectories_sampled": (
                        metrics["measured_steps"] * args.batch_size
                    ),
                }
            )

    if track_time:
        synchronize_if_needed(env.device)
        assert start_time is not None
        elapsed_time = time.perf_counter() - start_time
    else:
        elapsed_time = None

    history = None
    if record_history and losses_history is not None and iter_time_history is not None:
        history = {
            "losses": losses_history,
            "iter_times": iter_time_history,
        }

    return elapsed_time, history


def compute_loss(
    gflownet: TBGFlowNet | DBGFlowNet | FMGFlowNet,
    env: HyperGrid,
    trajectories,
    *,
    use_vmap: bool,
) -> torch.Tensor:
    if use_vmap and isinstance(gflownet, TBGFlowNet):
        return trajectory_balance_loss_vmap(gflownet, trajectories)

    return gflownet.loss_from_trajectories(
        env, trajectories, recalculate_all_logprobs=False
    )


def trajectory_balance_loss_vmap(
    gflownet: TBGFlowNet,
    trajectories,
) -> torch.Tensor:
    log_pf, log_pb = gflownet.get_pfs_and_pbs(
        trajectories, recalculate_all_logprobs=False
    )
    log_rewards = trajectories.log_rewards
    if log_rewards is None:
        raise ValueError("Log rewards required for TB loss.")

    def tb_residual(
        log_pf_seq: torch.Tensor, log_pb_seq: torch.Tensor, log_reward: torch.Tensor
    ) -> torch.Tensor:
        return log_pf_seq.sum() - log_pb_seq.sum() - log_reward

    residuals = vmap(tb_residual)(
        log_pf.transpose(0, 1),
        log_pb.transpose(0, 1),
        log_rewards,
    )

    log_z = gflownet.logZ
    if isinstance(log_z, ScalarEstimator):
        if trajectories.conditions is None:
            raise ValueError("Conditional logZ requires conditions tensor.")
        log_z_value = log_z(trajectories.conditions)
    else:
        log_z_value = log_z

    if isinstance(log_z_value, torch.Tensor):
        log_z_tensor = log_z_value
    else:
        log_z_tensor = torch.as_tensor(log_z_value, device=residuals.device)
    log_z_tensor = log_z_tensor.squeeze()
    scores = (residuals + log_z_tensor).pow(2)

    return scores.mean()


def run_validation_if_needed(
    env: HyperGrid,
    gflownet: TBGFlowNet | DBGFlowNet | FMGFlowNet,
    visited_states: DiscreteStates,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    *,
    quiet: bool,
) -> None:
    if args.validation_interval <= 0:
        return
    measured_steps = metrics["measured_steps"]
    if measured_steps == 0:
        return
    if measured_steps % args.validation_interval != 0:
        return

    validation_info, _ = validate(
        env,
        gflownet,
        args.validation_samples,
        visited_states,
    )
    metrics["validation_info"] = validation_info
    modes_found = env.modes_found(visited_states)
    metrics["discovered_modes"].update(modes_found)

    if not quiet:
        str_info = (
            f"Iter {measured_steps}: "
            f"L1 distance={validation_info.get('l1_dist', float('nan')):.8f} "
            f"modes discovered={len(metrics['discovered_modes'])} / {env.n_modes} "
            f"n terminating states {len(visited_states)}"
        )
        print(str_info)


def build_training_components(
    args: argparse.Namespace, device: torch.device, *, use_chunk: bool = False
) -> tuple[
    HyperGrid,
    TBGFlowNet | DBGFlowNet | FMGFlowNet,
    Sampler,
    torch.optim.Optimizer,
    DiscreteStates,
]:
    EnvClass = (
        HyperGridWithTensorStep if (use_chunk and args.chunk_size > 0) else HyperGrid
    )
    env = EnvClass(
        ndim=args.ndim,
        height=args.height,
        reward_fn_str="original",
        reward_fn_kwargs={
            "R0": args.R0,
            "R1": args.R1,
            "R2": args.R2,
        },
        device=device,
        calculate_partition=True,
        store_all_states=True,
        check_action_validity=__debug__,
    )

    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    if not args.uniform_pb:
        module_PB = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            trunk=module_PF.trunk,
        )
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)

    if args.loss == "FM":
        logF_estimator = DiscretePolicyEstimator(
            module=module_PF,
            n_actions=env.n_actions,
            preprocessor=preprocessor,
        )
        gflownet: TBGFlowNet | DBGFlowNet | FMGFlowNet = FMGFlowNet(logF_estimator).to(
            device
        )
        optimizer = torch.optim.Adam(gflownet.logF.parameters(), lr=args.lr)
        sampler = (
            ChunkedHyperGridSampler(estimator=logF_estimator, chunk_size=args.chunk_size)
            if use_chunk and args.chunk_size > 0
            else Sampler(estimator=logF_estimator)
        )
    else:
        pf_estimator = DiscretePolicyEstimator(
            module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
        )
        pb_estimator = DiscretePolicyEstimator(
            module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
        )

        if args.loss == "DB":
            logF_module = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=1,
            )
            logF_estimator = ScalarEstimator(
                module=logF_module,
                preprocessor=preprocessor,
            )
            gflownet = DBGFlowNet(pf=pf_estimator, pb=pb_estimator, logF=logF_estimator)
        else:
            gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)

        gflownet = gflownet.to(device)
        optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
        if isinstance(gflownet, DBGFlowNet):
            optimizer.add_param_group(
                {"params": gflownet.logF.parameters(), "lr": args.lr}
            )
        else:
            optimizer.add_param_group(
                {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
            )
        sampler = (
            ChunkedHyperGridSampler(estimator=pf_estimator, chunk_size=args.chunk_size)
            if use_chunk and args.chunk_size > 0
            else Sampler(estimator=pf_estimator)
        )

    visited_states = env.states_from_batch_shape((0,))
    return env, gflownet, sampler, optimizer, visited_states


def _mps_backend_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_backend_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "mps" and not _mps_backend_available():
        raise RuntimeError("MPS requested but not available.")
    return device


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and _mps_backend_available() and hasattr(torch, "mps"):
        torch.mps.synchronize()


def plot_benchmark(results: list[Dict[str, Any]], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting; install it or omit --benchmark."
        ) from exc

    def summarize_iteration_times(times: list[float]) -> tuple[float, float]:
        if not times:
            return 0.0, 0.0
        mean_time = statistics.fmean(times)
        std_time = statistics.pstdev(times) if len(times) > 1 else 0.0
        return mean_time, std_time

    labels = [res.get("label", f"Run {idx+1}") for idx, res in enumerate(results)]
    times = [res["elapsed"] for res in results]
    losses_list = [res.get("losses") or [] for res in results]
    iter_times_list = [res.get("iter_times") or [] for res in results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Subplot 1: total time comparison
    colors = ["#6c757d", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bar_colors = [colors[i % len(colors)] for i in range(len(results))]
    bars = axes[0].bar(labels, times, color=bar_colors)
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Total Training Time")
    baseline_time = times[0] if times else 1.0
    for i, (bar, value) in enumerate(zip(bars, times)):
        speedup = baseline_time / value if value else float("inf")
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}s\n{speedup:.2f}x",
            ha="center",
            va="bottom",
        )

    # Subplot 2: training curves
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))]

    for idx, losses in enumerate(losses_list):
        if not losses:
            continue
        axes[1].plot(
            range(1, len(losses) + 1),
            losses,
            label=labels[idx],
            color=bar_colors[idx],
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=2.0,
            alpha=0.5,
        )
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # Subplot 3: per-iteration timing with error bars
    summary_stats = [summarize_iteration_times(times) for times in iter_times_list]
    means_ms = [mean * 1000.0 for mean, _ in summary_stats]
    stds_ms = [std * 1000.0 for _, std in summary_stats]
    axes[2].bar(
        labels,
        means_ms,
        yerr=stds_ms,
        capsize=6,
        color=bar_colors,
    )
    axes[2].set_ylabel("Per-iteration time (ms)")
    axes[2].set_title("Iteration Timing (mean ± std)")

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved benchmark plot to {output}")


if __name__ == "__main__":
    main()
