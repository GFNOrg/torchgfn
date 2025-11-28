#!/usr/bin/env python
r"""
Optimized multi-environment (HyperGrid + Diffusion) training/benchmark script with
optional torch.compile, vmap, and chunked sampling across several GFlowNet variants.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, cast

import torch
from torch.func import vmap
from tqdm import tqdm

from gfn.containers import Trajectories
from gfn.env import Env, EnvFastPathMixin
from gfn.estimators import (
    DiscretePolicyEstimator,
    FastPolicyMixin,
    PinnedBrownianMotionBackward,
    PinnedBrownianMotionForward,
    ScalarEstimator,
)
from gfn.gflownet import PFBasedGFlowNet, SubTBGFlowNet
from gfn.gflownet.detailed_balance import DBGFlowNet
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor
from gfn.samplers import CompiledChunkSampler, Sampler
from gfn.states import DiscreteStates, States
from gfn.utils.common import set_seed
from gfn.utils.compile import try_compile_gflownet
from gfn.utils.modules import (
    MLP,
    DiffusionFixedBackwardModule,
    DiffusionPISGradNetForward,
)
from gfn.utils.training import validate


def _mark_cudagraph_step() -> None:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(marker):
        marker()


def _fill_like_reference(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Broadcasts `value` to the shape/dtype/device of `reference`."""
    fill = value.to(device=reference.device, dtype=reference.dtype)
    while fill.ndim < reference.ndim:
        fill = fill.unsqueeze(0)
    return fill.expand_as(reference).clone()


# Default HyperGrid configuration (easy to extend to multiple envs later on).
HYPERGRID_KWARGS: Dict[str, Any] = {
    "ndim": 2,
    "height": 32,
    "reward_fn_str": "original",
    "reward_fn_kwargs": {"R0": 0.1, "R1": 0.5, "R2": 2.0},
    "calculate_partition": False,
    "store_all_states": False,
    "check_action_validity": __debug__,
}

DEFAULT_CHUNK_SIZE = 32
DEFAULT_COMPILE_MODE = "reduce-overhead"


@dataclass
class ScenarioConfig:
    name: str
    description: str
    sampler: Literal["standard", "compiled_chunk", "script_chunk"]
    use_script_env: bool
    use_compile: bool
    use_vmap: bool


@dataclass(frozen=True)
class FlowVariant:
    key: Literal["tb", "dbg", "subtb"]
    label: str
    description: str
    requires_logf: bool
    supports_vmap: bool


HYPERGRID_SCENARIOS: list[ScenarioConfig] = [
    ScenarioConfig(
        name="Baseline (core)",
        description="Stock library path: standard env + sampler, no compilation.",
        sampler="standard",
        use_script_env=False,
        use_compile=False,
        use_vmap=False,
    ),
    ScenarioConfig(
        name="VMap Only",
        description="VMAP Accelerated Loss.",
        sampler="standard",
        use_script_env=False,
        use_compile=False,
        use_vmap=True,
    ),
    ScenarioConfig(
        name="Compile Only (core)",
        description="Standard env + sampler with torch.compile but no chunking.",
        sampler="standard",
        use_script_env=False,
        use_compile=True,
        use_vmap=True,
    ),
    ScenarioConfig(
        name="Library Fast Path",
        description="Core EnvFastPath + CompiledChunkSampler + compile + vmap TB.",
        sampler="compiled_chunk",
        use_script_env=False,
        use_compile=True,
        use_vmap=True,
    ),
    ScenarioConfig(
        name="Script Fast Path",
        description="Script-local tensor env/sampler, compile, and vmap TB.",
        sampler="script_chunk",
        use_script_env=True,
        use_compile=True,
        use_vmap=True,
    ),
]

DIFFUSION_SCENARIOS: list[ScenarioConfig] = [
    ScenarioConfig(
        name="Diffusion Baseline",
        description="Pinned Brownian sampler without compilation or chunking.",
        sampler="standard",
        use_script_env=False,
        use_compile=False,
        use_vmap=False,
    ),
    ScenarioConfig(
        name="Diffusion VMap Only",
        description="Pinned Brownian sampler without compilation or chunking.",
        sampler="standard",
        use_script_env=False,
        use_compile=False,
        use_vmap=True,
    ),
    ScenarioConfig(
        name="Diffusion Compile Only",
        description="Standard diffusion sampler with torch.compile but no chunking.",
        sampler="standard",
        use_script_env=False,
        use_compile=True,
        use_vmap=True,
    ),
    ScenarioConfig(
        name="Diffusion Library Fast Path",
        description="EnvFastPath + CompiledChunkSampler (library implementation).",
        sampler="compiled_chunk",
        use_script_env=False,
        use_compile=True,
        use_vmap=False,
    ),
    ScenarioConfig(
        name="Diffusion Script Fast Path",
        description="Script-local tensor sampler tailored to diffusion states.",
        sampler="script_chunk",
        use_script_env=False,
        use_compile=True,
        use_vmap=False,
    ),
]


FLOW_VARIANTS: dict[str, FlowVariant] = {
    "tb": FlowVariant(
        key="tb",
        label="TBGFlowNet",
        description="Trajectory Balance baseline with optional torch.compile/vmap.",
        requires_logf=False,
        supports_vmap=True,
    ),
    "dbg": FlowVariant(
        key="dbg",
        label="DBGFlowNet",
        description="Detailed Balance loss with learned log-state flows.",
        requires_logf=True,
        supports_vmap=False,
    ),
    "subtb": FlowVariant(
        key="subtb",
        label="SubTBGFlowNet",
        description="Sub-trajectory balance variant with configurable weighting.",
        requires_logf=True,
        supports_vmap=False,
    ),
}

DEFAULT_FLOW_ORDER = ["tb", "dbg", "subtb"]

# Plot styling: consistent colors for GFlowNet variants, linestyles for scenarios.
VARIANT_COLORS: dict[str, str] = {
    "tb": "#000000",  # Trajectory Balance -> black
    "subtb": "#d62728",  # SubTB -> red
    "dbg": "#1f77b4",  # Detailed Balance -> blue
}
SCENARIO_LINESTYLES: dict[str, Any] = {
    "Baseline (core)": "-",
    "Compile Only (core)": "-.",
    "Library Fast Path": "--",  # fast-path compiled
    "Script Fast Path": ":",
    "Diffusion Baseline": "-",
    "Diffusion Compile Only": "-.",
    "Diffusion Library Fast Path": "--",
    "Diffusion Script Fast Path": ":",
}
LOSS_LINE_ALPHA = 0.5


@dataclass
class EnvironmentBenchmark:
    key: Literal["hypergrid", "diffusion"]
    label: str
    description: str
    color: str
    scenarios: list[ScenarioConfig]
    supported_flows: list[str]
    supports_validation: bool


ENVIRONMENT_BENCHMARKS: dict[str, EnvironmentBenchmark] = {
    "hypergrid": EnvironmentBenchmark(
        key="hypergrid",
        label="HyperGrid",
        description="High-dimensional discrete lattice with known reward landscape.",
        color="#4a90e2",
        scenarios=HYPERGRID_SCENARIOS,
        supported_flows=list(DEFAULT_FLOW_ORDER),
        supports_validation=True,
    ),
    "diffusion": EnvironmentBenchmark(
        key="diffusion",
        label="Diffusion Sampling",
        description="Continuous-state diffusion sampling benchmark (Pinned Brownian).",
        color="#a17be7",
        scenarios=DIFFUSION_SCENARIOS,
        supported_flows=list(DEFAULT_FLOW_ORDER),
        supports_validation=False,
    ),
}
DEFAULT_ENV_ORDER = ["hypergrid", "diffusion"]


def _normalize_flow_keys(requested: list[str]) -> list[str]:
    normalized: list[str] = []
    for key in requested:
        alias = key.lower()
        if alias not in FLOW_VARIANTS:
            supported = ", ".join(sorted(FLOW_VARIANTS))
            raise ValueError(
                f"Unsupported GFlowNet variant '{key}'. Choose from {supported}."
            )
        if alias not in normalized:
            normalized.append(alias)
    return normalized


def _normalize_env_keys(requested: list[str]) -> list[str]:
    normalized: list[str] = []
    available = ENVIRONMENT_BENCHMARKS
    for key in requested:
        alias = key.lower()
        if alias not in available:
            supported = ", ".join(sorted(available))
            raise ValueError(
                f"Unsupported environment '{key}'. Choose from {supported}."
            )
        if alias not in normalized:
            normalized.append(alias)
    return normalized or list(DEFAULT_ENV_ORDER)


# Local subclasses for benchmarking-only optimizations (no core library changes)
class HyperGridWithTensorStep(HyperGrid):
    def step_tensor(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Env.TensorStepResult:
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
        non_exit_mask_exp = non_exit_mask.unsqueeze(-1)
        safe_actions = torch.where(
            non_exit_mask_exp, actions_idx, torch.zeros_like(actions_idx)
        )
        delta = torch.zeros_like(next_states)
        delta = delta.scatter(-1, safe_actions, 1, reduce="add")
        delta = delta * non_exit_mask_exp.to(next_states.dtype)
        next_states = next_states + delta

        # Ensure exit actions land exactly on the sink state so downstream
        # `is_sink_state` masks match the action padding semantics assumed
        # by `Trajectories` and probability calculations.
        sink_state = self.sf.to(device=device).unsqueeze(0).expand_as(next_states)
        next_states = torch.where(is_exit.unsqueeze(-1), sink_state, next_states)

        next_forward_masks = torch.ones(
            (batch, self.n_actions), dtype=torch.bool, device=device
        )
        next_forward_masks[:, :ndim] = next_states != (self.height - 1)
        next_forward_masks[:, ndim] = True
        backward_masks = next_states != 0

        return self.TensorStepResult(
            next_states=next_states,
            is_sink_state=is_exit,
            forward_masks=next_forward_masks,
            backward_masks=backward_masks,
        )

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
        self._compiled_chunk_cache: dict[tuple[int, str], Callable] = {}

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
        policy_kwargs = dict(policy_kwargs)
        epsilon = float(policy_kwargs.pop("epsilon", 0.0))

        if states is None:
            assert n is not None
            states_obj = env.reset(batch_shape=(n,))
        else:
            states_obj = states

        if not isinstance(self.estimator, FastPolicyMixin):
            raise TypeError(
                "ChunkedHyperGridSampler requires a FastPolicy-compatible estimator."
            )
        policy = cast(FastPolicyMixin, self.estimator)
        chunk_size = max(1, self.chunk_size)
        exit_idx = env.n_actions - 1

        curr_states = states_obj.tensor
        batch = curr_states.shape[0]
        device = curr_states.device

        def compute_forward_masks(states_tensor: torch.Tensor) -> torch.Tensor:
            if hasattr(env, "forward_action_masks"):
                return env.forward_action_masks(states_tensor)
            if hasattr(env, "forward_action_masks_tensor"):
                return env.forward_action_masks_tensor(states_tensor)
            raise TypeError(
                "HyperGrid environment must expose forward action masks for fast path."
            )

        def step_tensor(
            states_tensor: torch.Tensor, actions_tensor: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            step_result = env.step_tensor(states_tensor, actions_tensor)
            if isinstance(step_result, Env.TensorStepResult):
                next_states = step_result.next_states
                next_masks = step_result.forward_masks
                if next_masks is None:
                    next_masks = compute_forward_masks(next_states)
                is_exit_states = step_result.is_sink_state
                if is_exit_states is None:
                    is_exit_states = env.states_from_tensor(next_states).is_sink_state
                return next_states, next_masks, is_exit_states
            assert isinstance(step_result, tuple) and len(step_result) == 3
            next_states, next_masks, is_exit_states = step_result
            return next_states, next_masks, is_exit_states

        exit_action_value = env.exit_action.to(device=device)
        dummy_action_value = env.dummy_action.to(device=device)

        forward_masks = compute_forward_masks(curr_states)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        actions_seq: List[torch.Tensor] = []
        dones_seq: List[torch.Tensor] = []
        states_stack: List[torch.Tensor] = [curr_states.clone()]

        def _expand_front(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
            expand_dims = target_ndim - tensor.ndim
            if expand_dims <= 0:
                return tensor
            view_shape = (1,) * expand_dims + tuple(tensor.shape)
            return tensor.view(view_shape)

        def _expand_back(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
            expand_dims = target_ndim - tensor.ndim
            if expand_dims <= 0:
                return tensor
            view_shape = tuple(tensor.shape) + (1,) * expand_dims
            return tensor.view(view_shape)

        device_type = curr_states.device.type
        compile_allowed = (
            hasattr(torch, "compile") and device_type in ("cuda", "cpu") and conditions is None and not policy_kwargs
        )
        compile_key = (id(env), device_type)
        chunk_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple] | None = None
        chunk_fn_compiled = False
        if compile_allowed:
            chunk_fn = self._compiled_chunk_cache.get(compile_key)
            if chunk_fn is not None:
                chunk_fn_compiled = True

        if chunk_fn is None:

            def _chunk_loop(
                current_states: torch.Tensor,
                current_masks: torch.Tensor,
                done_mask: torch.Tensor,
            ):
                actions_list: List[torch.Tensor] = []
                dones_list: List[torch.Tensor] = []
                states_list: List[torch.Tensor] = []
                action_template: torch.Tensor | None = None
                steps_taken = 0
                for _ in range(chunk_size):
                    if bool(done_mask.all().item()):
                        assert action_template is not None
                        pad_actions = _fill_like_reference(action_template, dummy_action_value)
                        actions_list.append(pad_actions)
                        dones_list.append(done_mask.clone())
                        states_list.append(current_states.clone())
                        continue

                    masks = current_masks
                    if done_mask.any():
                        masks = masks.clone()
                        masks[done_mask] = False
                        masks[done_mask, exit_idx] = True

                    features = policy.fast_features(
                        current_states,
                        forward_masks=masks,
                        backward_masks=None,
                        conditions=conditions,
                    )
                    dist = policy.fast_distribution(
                        features,
                        forward_masks=masks,
                        backward_masks=None,
                        states_tensor=current_states,
                        epsilon=epsilon,
                        **policy_kwargs,
                    )
                    sampled_actions = dist.sample()
                    step_actions = sampled_actions
                    record_actions = sampled_actions

                    if done_mask.any():
                        mask = _expand_back(done_mask, sampled_actions.ndim)
                        exit_fill = _expand_front(
                            exit_action_value.to(
                                device=sampled_actions.device, dtype=sampled_actions.dtype
                            ),
                            sampled_actions.ndim,
                        )
                        dummy_fill = _expand_front(
                            dummy_action_value.to(
                                device=sampled_actions.device, dtype=sampled_actions.dtype
                            ),
                            sampled_actions.ndim,
                        )
                        step_actions = torch.where(mask, exit_fill, sampled_actions)
                        record_actions = torch.where(mask, dummy_fill, sampled_actions)

                    next_states, next_masks, is_exit = step_tensor(
                        current_states, step_actions
                    )

                    actions_list.append(record_actions)
                    action_template = record_actions.detach()
                    dones_list.append(is_exit)
                    states_list.append(next_states.clone())

                    current_states = next_states
                    current_masks = next_masks
                    done_mask = done_mask | is_exit
                    steps_taken += 1

                return (
                    current_states,
                    current_masks,
                    done_mask,
                    actions_list,
                    dones_list,
                    states_list,
                    torch.tensor(steps_taken, device=current_states.device),
                )

            chunk_fn = _chunk_loop
            if compile_allowed:
                try:
                    chunk_fn = torch.compile(_chunk_loop, mode="reduce-overhead")  # type: ignore[arg-type]
                    self._compiled_chunk_cache[compile_key] = chunk_fn
                    chunk_fn_compiled = True
                except Exception:
                    chunk_fn = _chunk_loop

        while not bool(done.all().item()):
            if chunk_fn_compiled:
                _mark_cudagraph_step()
            (
                curr_states,
                forward_masks,
                done,
                actions_chunk,
                dones_chunk,
                states_chunk,
                steps_taken_tensor,
            ) = chunk_fn(curr_states, forward_masks, done)
            steps_taken = int(steps_taken_tensor.item())
            if steps_taken:
                actions_seq.extend(actions_chunk[:steps_taken])
                dones_seq.extend(dones_chunk[:steps_taken])
                states_stack.extend(states_chunk[:steps_taken])

        if actions_seq:
            actions_tsr = torch.stack(actions_seq, dim=0)
            states_tsr = torch.stack(states_stack, dim=0)
            action_shape = getattr(env, "action_shape", None)
            if action_shape:
                tail_shape = tuple(actions_tsr.shape[-len(action_shape):])
                if tail_shape != tuple(action_shape):
                    if tuple(action_shape) == (1,):
                        actions_tsr = actions_tsr.unsqueeze(-1)
                    else:
                        raise ValueError(
                            "ChunkedHyperGridSampler produced actions with shape "
                            f"{actions_tsr.shape}, expected trailing dims {action_shape}."
                        )
            is_exit_seq = torch.stack(dones_seq, dim=0)
            T = actions_tsr.shape[0]
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


class ChunkedDiffusionSampler(Sampler):
    """Chunked fast-path sampler specialized for DiffusionSampling states."""

    def __init__(self, estimator: PinnedBrownianMotionForward, chunk_size: int):
        super().__init__(estimator)
        self.chunk_size = int(chunk_size)
        self._compiled_chunk_cache: dict[tuple[int, str], Callable] = {}

    def sample_trajectories(  # noqa: C901
        self,
        env: DiffusionSampling,
        n: int | None = None,
        states: States | None = None,
        conditions: torch.Tensor | None = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        if save_estimator_outputs or save_logprobs:
            raise NotImplementedError(
                "ChunkedDiffusionSampler does not record estimator outputs/log-probs yet."
            )
        if not isinstance(env, EnvFastPathMixin):
            raise TypeError(
                "ChunkedDiffusionSampler requires environments with tensor fast paths."
            )
        if not isinstance(self.estimator, FastPolicyMixin):
            raise TypeError(
                "ChunkedDiffusionSampler requires a FastPolicy-compatible estimator."
            )

        policy = cast(FastPolicyMixin, self.estimator)
        chunk_size = max(1, self.chunk_size)

        if states is None:
            assert n is not None
            states_obj = env.reset(batch_shape=(n,))
        else:
            states_obj = states

        curr_states = states_obj.tensor
        done = states_obj.is_sink_state.clone()
        exit_action_value = env.exit_action.to(device=curr_states.device)
        dummy_action_value = env.dummy_action.to(device=curr_states.device)

        recorded_actions_seq: List[torch.Tensor] = []
        sink_seq: List[torch.Tensor] = []
        states_stack: List[torch.Tensor] = [curr_states.clone()]

        exit_template_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        dummy_template_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}

        def _expand_front(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
            expand_dims = target_ndim - tensor.ndim
            if expand_dims <= 0:
                return tensor
            view_shape = (1,) * expand_dims + tuple(tensor.shape)
            return tensor.view(view_shape)

        def _expand_back(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
            expand_dims = target_ndim - tensor.ndim
            if expand_dims <= 0:
                return tensor
            view_shape = tuple(tensor.shape) + (1,) * expand_dims
            return tensor.view(view_shape)

        def _get_template(
            cache: dict[tuple[int, torch.dtype], torch.Tensor],
            base_value: torch.Tensor,
            target_ndim: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> torch.Tensor:
            key = (target_ndim, dtype)
            tensor = cache.get(key)
            if tensor is None:
                tensor = _expand_front(base_value.to(device=device, dtype=dtype), target_ndim)
                cache[key] = tensor
            return tensor

        def _chunk_loop(current_states: torch.Tensor, done_mask: torch.Tensor) -> tuple[
            torch.Tensor,
            torch.Tensor,
            List[torch.Tensor],
            List[torch.Tensor],
            List[torch.Tensor],
            torch.Tensor,
        ]:
            local_recorded_actions: List[torch.Tensor] = []
            local_sinks: List[torch.Tensor] = []
            local_states: List[torch.Tensor] = []
            action_template: torch.Tensor | None = None
            steps_taken = 0

            for _ in range(chunk_size):
                if bool(done_mask.all().item()):
                    assert action_template is not None
                    pad_actions = _fill_like_reference(action_template, dummy_action_value)
                    local_recorded_actions.append(pad_actions)
                    local_sinks.append(done_mask.clone())
                    local_states.append(current_states.clone())
                    continue

                features = policy.fast_features(
                    current_states,
                    forward_masks=None,
                    backward_masks=None,
                    conditions=conditions,
                )
                dist = policy.fast_distribution(
                    features,
                    forward_masks=None,
                    backward_masks=None,
                    states_tensor=current_states,
                    **policy_kwargs,
                )
                sampled_actions = dist.sample()
                step_actions = sampled_actions
                record_actions = sampled_actions

                if torch.any(done_mask):
                    mask = _expand_back(done_mask, sampled_actions.ndim)
                    exit_fill = _get_template(
                        exit_template_cache,
                        exit_action_value,
                        sampled_actions.ndim,
                        sampled_actions.dtype,
                        sampled_actions.device,
                    )
                    dummy_fill = _get_template(
                        dummy_template_cache,
                        dummy_action_value,
                        sampled_actions.ndim,
                        sampled_actions.dtype,
                        sampled_actions.device,
                    )
                    step_actions = torch.where(mask, exit_fill, sampled_actions)
                    record_actions = torch.where(mask, dummy_fill, sampled_actions)

                step_res = env.step_tensor(current_states, step_actions)
                current_states = step_res.next_states
                sinks = step_res.is_sink_state
                if sinks is None:
                    sinks = env.states_from_tensor(current_states).is_sink_state

                done_mask = done_mask | sinks
                local_recorded_actions.append(record_actions)
                action_template = record_actions.detach()
                local_sinks.append(sinks)
                local_states.append(current_states.clone())
                steps_taken += 1

            return (
                current_states,
                done_mask,
                local_recorded_actions,
                local_sinks,
                local_states,
                torch.tensor(steps_taken, device=current_states.device),
            )

        chunk_fn: Callable = _chunk_loop
        chunk_fn_compiled = False
        device_type = curr_states.device.type
        compile_allowed = (
            hasattr(torch, "compile") and device_type in ("cuda", "cpu") and conditions is None and not policy_kwargs
        )
        cache_key = (id(env), device_type)
        if compile_allowed:
            cached = self._compiled_chunk_cache.get(cache_key)
            if cached is not None:
                chunk_fn = cached
                chunk_fn_compiled = True
            else:
                try:
                    compiled = torch.compile(_chunk_loop, mode="reduce-overhead")  # type: ignore[arg-type]
                    self._compiled_chunk_cache[cache_key] = compiled
                    chunk_fn = compiled
                    chunk_fn_compiled = True
                except Exception as exc:  # pragma: no cover - compile fallback
                    warnings.warn(
                        f"Compilation of diffusion chunk loop failed ({exc}); using eager version.",
                        stacklevel=2,
                    )
                    chunk_fn = _chunk_loop

        while not bool(done.all().item()):
            if chunk_fn_compiled:
                _mark_cudagraph_step()
            (
                curr_states,
                done,
                recorded_actions_chunk,
                sinks_chunk,
                states_chunk,
                steps_taken_tensor,
            ) = chunk_fn(curr_states, done)
            steps_taken = int(steps_taken_tensor.item())
            if steps_taken:
                recorded_actions_seq.extend(recorded_actions_chunk[:steps_taken])
                sink_seq.extend(sinks_chunk[:steps_taken])
                states_stack.extend(states_chunk[:steps_taken])

        if recorded_actions_seq:
            actions_tsr = torch.stack(recorded_actions_seq, dim=0)
            states_tsr = torch.stack(states_stack, dim=0)
            action_shape = getattr(env, "action_shape", None)
            if action_shape:
                tail_shape = tuple(actions_tsr.shape[-len(action_shape):])
                if tail_shape != tuple(action_shape):
                    if tuple(action_shape) == (1,):
                        actions_tsr = actions_tsr.unsqueeze(-1)
                    else:
                        raise ValueError(
                            "ChunkedDiffusionSampler produced actions with shape "
                            f"{actions_tsr.shape}, expected trailing dims {action_shape}."
                        )
            T = actions_tsr.shape[0]
            sinks_tsr = torch.stack(sink_seq, dim=0)
            first_sink = torch.argmax(sinks_tsr.to(torch.long), dim=0)
            never_sink = ~sinks_tsr.any(dim=0)
            first_sink = torch.where(
                never_sink,
                torch.tensor(T - 1, device=curr_states.device),
                first_sink,
            )
            terminating_idx = first_sink + 1
        else:
            states_tsr = states_obj.tensor.unsqueeze(0)
            actions_tsr = env.actions_from_batch_shape((0, states_tsr.shape[1])).tensor
            terminating_idx = torch.zeros(
                states_tsr.shape[1], dtype=torch.long, device=curr_states.device
            )
            return Trajectories(
                env=env,
                states=env.states_from_tensor(states_tsr),
                conditions=conditions,
                actions=env.actions_from_tensor(actions_tsr),
                terminating_idx=terminating_idx,
                is_backward=False,
                log_rewards=None,
                log_probs=None,
                estimator_outputs=None,
            )

        trajectories = Trajectories(
            env=env,
            states=env.states_from_tensor(states_tsr),
            conditions=conditions,
            actions=env.actions_from_tensor(actions_tsr),
            terminating_idx=terminating_idx,
            is_backward=False,
            log_rewards=None,
            log_probs=None,
            estimator_outputs=None,
        )
        return trajectories


class FastKHotDiscretePolicyEstimator(FastPolicyMixin, DiscretePolicyEstimator):
    """Discrete forward policy with tensor-only helpers for HyperGrid."""

    def __init__(
        self,
        env: HyperGrid,
        module: torch.nn.Module,
        preprocessor: KHotPreprocessor,
    ) -> None:
        super().__init__(
            module=module,
            n_actions=env.n_actions,
            preprocessor=preprocessor,
            is_backward=False,
        )
        self.height = int(env.height)
        self.ndim = int(env.ndim)
        self.exit_idx = env.n_actions - 1

    def fast_features(
        self,
        states_tensor: torch.Tensor,
        *,
        forward_masks: torch.Tensor | None = None,
        backward_masks: torch.Tensor | None = None,
        conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert states_tensor.dtype == torch.long
        sink_mask = states_tensor < 0  # HyperGrid sink state stores -1 in every dim.
        safe_states = torch.where(
            sink_mask, torch.zeros_like(states_tensor), states_tensor
        )
        khot = torch.nn.functional.one_hot(
            safe_states, num_classes=self.height
        ).to(dtype=torch.get_default_dtype())
        if sink_mask.any():
            khot = khot * (~sink_mask).unsqueeze(-1).to(khot.dtype)
        return khot.view(states_tensor.shape[0], -1)

    def fast_distribution(
        self,
        features: torch.Tensor,
        *,
        states_tensor: torch.Tensor | None = None,
        forward_masks: torch.Tensor | None = None,
        backward_masks: torch.Tensor | None = None,
        epsilon: float = 0.0,
        **policy_kwargs: Any,
    ) -> torch.distributions.Categorical:
        if states_tensor is None:
            raise ValueError(
                "states_tensor is required for FastKHotDiscretePolicyEstimator."
            )

        logits = self.module(features)
        batch = states_tensor.shape[0]
        masks = torch.zeros(
            batch,
            self.ndim + 1,
            dtype=torch.bool,
            device=states_tensor.device,
        )
        masks[:, : self.ndim] = states_tensor < (self.height - 1)
        masks[:, self.exit_idx] = True

        masked_logits = logits.masked_fill(~masks, float("-inf"))
        probs = torch.softmax(masked_logits, dim=-1)

        if epsilon > 0.0:
            valid_counts = masks.sum(dim=-1, keepdim=True).clamp_min(1)
            uniform = masks.to(probs.dtype) / valid_counts.to(probs.dtype)
            probs = (1.0 - epsilon) * probs + epsilon * uniform

        return torch.distributions.Categorical(probs=probs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs. fast-path HyperGrid training pipelines."
    )
    parser.add_argument("--n-iterations", type=int, default=50, dest="n_iterations")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("--warmup-iters", type=int, default=25, dest="warmup_iters")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-logz", type=float, default=1e-1, dest="lr_logz")
    parser.add_argument("--lr-logf", type=float, default=1e-3, dest="lr_logf")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--environments",
        nargs="+",
        choices=sorted(ENVIRONMENT_BENCHMARKS),
        default=list(DEFAULT_ENV_ORDER),
        help="Benchmark environments to include (e.g., hypergrid diffusion).",
    )
    parser.add_argument(
        "--validation-interval", type=int, default=100, dest="validation_interval"
    )
    parser.add_argument(
        "--validation-samples", type=int, default=200_000, dest="validation_samples"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device to run on; auto prefers CUDA>MPS>CPU.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=str(Path.home() / "hypergrid_benchmark.png"),
        help="Output path for optional benchmark plot.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip writing the benchmark plot (still prints the summary).",
    )
    parser.add_argument(
        "--gflownets",
        nargs="+",
        default=DEFAULT_FLOW_ORDER,
        help="GFlowNet variants to benchmark (any of: tb, dbg, subtb).",
    )
    parser.add_argument(
        "--subtb-weighting",
        choices=[
            "DB",
            "ModifiedDB",
            "TB",
            "geometric",
            "equal",
            "equal_within",
        ],
        default="ModifiedDB",
        dest="subtb_weighting",
        help="Weighting strategy for SubTBGFlowNet runs.",
    )
    parser.add_argument(
        "--subtb-lamda",
        type=float,
        default=0.9,
        dest="subtb_lamda",
        help="Lambda discount factor for SubTBGFlowNet geometric weighting.",
    )
    # Diffusion-specific knobs (ignored unless `diffusion` is selected).
    parser.add_argument(
        "--diffusion-target",
        type=str,
        default="gmm2",
        help="Diffusion target alias (see DiffusionSampling.DIFFUSION_TARGETS).",
    )
    parser.add_argument(
        "--diffusion-dim",
        type=int,
        default=None,
        help="Override target dimensionality when supported.",
    )
    parser.add_argument(
        "--diffusion-num-components",
        type=int,
        default=None,
        help="Override mixture component count for Gaussian targets.",
    )
    parser.add_argument(
        "--diffusion-target-seed",
        type=int,
        default=2,
        help="Seed controlling random targets (centers, covariances, etc.).",
    )
    parser.add_argument(
        "--diffusion-num-steps",
        type=int,
        default=32,
        help="Number of discretization steps for the diffusion process.",
    )
    parser.add_argument(
        "--diffusion-sigma",
        type=float,
        default=5.0,
        help="Pinned Brownian motion diffusion coefficient.",
    )
    parser.add_argument(
        "--diffusion-harmonics-dim",
        type=int,
        default=64,
        help="Harmonics embedding dimension for DiffusionPISGradNetForward.",
    )
    parser.add_argument(
        "--diffusion-t-emb-dim",
        type=int,
        default=64,
        help="Temporal embedding dimension for diffusion forward model.",
    )
    parser.add_argument(
        "--diffusion-s-emb-dim",
        type=int,
        default=64,
        help="State embedding dimension for diffusion forward model.",
    )
    parser.add_argument(
        "--diffusion-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for diffusion forward model.",
    )
    parser.add_argument(
        "--diffusion-joint-layers",
        type=int,
        default=2,
        help="Joint layers count for diffusion forward model.",
    )
    parser.add_argument(
        "--diffusion-zero-init",
        action="store_true",
        help="Initialize diffusion forward model heads to zero.",
    )
    return parser.parse_args()


def init_metrics() -> Dict[str, Any]:
    return {
        "validation_info": {"l1_dist": float("nan")},
        "discovered_modes": set(),
        "total_steps": 0,
        "measured_steps": 0,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    flow_keys = _normalize_flow_keys(args.gflownets)
    env_keys = _normalize_env_keys(args.environments)
    if not flow_keys:
        raise ValueError("At least one GFlowNet variant must be specified.")

    results: list[dict[str, Any]] = []
    grouped_results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for env_key in env_keys:
        env_cfg = ENVIRONMENT_BENCHMARKS[env_key]
        env_flow_keys = [
            flow_key for flow_key in flow_keys if flow_key in env_cfg.supported_flows
        ]
        if not env_flow_keys:
            print(
                f"\nSkipping environment '{env_cfg.label}' "
                f"(no compatible flows among {', '.join(flow_keys)})."
            )
            continue

        grouped_results.setdefault(env_key, {})
        print(f"\n### Environment: {env_cfg.label} ###\n" f"{env_cfg.description}\n")

        for flow_key in env_flow_keys:
            flow_variant = FLOW_VARIANTS[flow_key]
            grouped_results[env_key].setdefault(flow_key, [])
            print(
                f"\n=== GFlowNet Variant: {flow_variant.label} "
                f"@ {env_cfg.label} ===\n{flow_variant.description}\n"
            )
            for scenario in env_cfg.scenarios:
                print(
                    f"\n--- Scenario: {scenario.name} | "
                    f"{flow_variant.label} ({env_cfg.label}) ---\n"
                    f"{scenario.description}\n"
                )
                result = run_scenario(args, device, scenario, flow_variant, env_cfg)
                result["label"] = scenario.name
                result["description"] = scenario.description
                result["env_key"] = env_cfg.key
                result["env_label"] = env_cfg.label
                results.append(result)
                grouped_results[env_key][flow_key].append(result)

    print("\nBenchmark summary (speedups vs. per-environment baselines):")
    for env_key in env_keys:
        env_cfg = ENVIRONMENT_BENCHMARKS.get(env_key)
        if env_cfg is None:
            continue
        env_flow_results = grouped_results.get(env_key, {})
        if not env_flow_results:
            continue

        baseline_name = env_cfg.scenarios[0].name if env_cfg.scenarios else "baseline"
        print(f"\n[{env_cfg.label}] scenario baseline = {baseline_name}")

        for flow_key, flow_results in env_flow_results.items():
            if not flow_results:
                continue
            flow_variant = FLOW_VARIANTS[flow_key]
            baseline_candidate = next(
                (res for res in flow_results if res.get("label") == baseline_name),
                flow_results[0],
            )
            baseline_time = baseline_candidate.get("elapsed", 0.0) or 1.0
            print(
                f"\n  - {flow_variant.label}: "
                f"{baseline_time:.2f}s baseline ({baseline_candidate['label']})"
            )
            for result in flow_results:
                elapsed = result["elapsed"]
                speedup = baseline_time / elapsed if elapsed else float("inf")
                print(
                    f"    • {result['label']}: {elapsed:.2f}s ({speedup:.2f}x) | "
                    f"compile={'yes' if result['use_compile'] else 'no'} | "
                    f"vmap={'yes' if result['use_vmap'] else 'no'} | "
                    f"sampler={result['sampler']}"
                )

    if not args.skip_plot:
        plot_benchmark(results, args.benchmark_output)


def run_scenario(
    args: argparse.Namespace,
    device: torch.device,
    scenario: ScenarioConfig,
    flow_variant: FlowVariant,
    env_cfg: EnvironmentBenchmark,
) -> dict[str, Any]:
    set_seed(args.seed)
    (
        env,
        gflownet,
        sampler,
        optimizer,
        visited_states,
    ) = build_training_components(args, device, scenario, flow_variant, env_cfg)
    metrics = init_metrics()
    use_vmap = scenario.use_vmap and flow_variant.supports_vmap

    if scenario.use_compile:
        compile_results = try_compile_gflownet(
            gflownet,
            mode=DEFAULT_COMPILE_MODE,
        )
        formatted = ", ".join(
            f"{name}:{'✓' if success else 'x'}"
            for name, success in compile_results.items()
        )
        print(f"[compile] {formatted}")

    if args.warmup_iters > 0:
        run_iterations(
            env,
            gflownet,
            sampler,
            optimizer,
            visited_states,
            metrics,
            args,
            n_iters=args.warmup_iters,
            use_vmap=use_vmap,
            quiet=True,
            collect_metrics=False,
            track_time=False,
            record_history=False,
            supports_validation=env_cfg.supports_validation,
            mark_compiled_step=scenario.use_compile,
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
        quiet=False,
        collect_metrics=True,
        track_time=True,
        record_history=True,
        supports_validation=env_cfg.supports_validation,
        mark_compiled_step=scenario.use_compile,
    )

    validation_info = metrics["validation_info"]
    l1 = validation_info.get("l1_dist", float("nan"))
    modes_total = getattr(env, "n_modes", None)
    if modes_total is None:
        modes_str = "modes=n/a"
    else:
        modes_str = f"modes={len(metrics['discovered_modes'])} / {modes_total}"
    if env_cfg.supports_validation:
        validation_str = f"L1 distance={l1:.6f}"
    else:
        validation_str = "validation=skipped"
    print(
        f"Finished training ({env_cfg.label}) | "
        f"iterations={metrics['measured_steps']} | "
        f"{modes_str} | {validation_str}"
    )

    return {
        "elapsed": elapsed or 0.0,
        "losses": history["losses"] if history else None,
        "iter_times": history["iter_times"] if history else None,
        "use_compile": scenario.use_compile,
        "use_vmap": use_vmap,
        "sampler": scenario.sampler,
        "gflownet_key": flow_variant.key,
        "gflownet_label": flow_variant.label,
    }


def run_iterations(
    env: Env,
    gflownet: PFBasedGFlowNet,
    sampler: Sampler,
    optimizer: torch.optim.Optimizer,
    visited_states,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    *,
    n_iters: int,
    use_vmap: bool,
    quiet: bool,
    collect_metrics: bool,
    track_time: bool,
    record_history: bool,
    supports_validation: bool,
    mark_compiled_step: bool = False,
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
        if mark_compiled_step:
            _mark_cudagraph_step()
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
        )

        terminating_states = cast(States, trajectories.terminating_states)
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
        if record_history and losses_history is not None and iter_time_history is not None:
            losses_history.append(last_loss)
            iter_duration = (
                (time.perf_counter() - iter_start) if iter_start is not None else 0.0
            )
            iter_time_history.append(iter_duration)

        if collect_metrics and supports_validation:
            run_validation_if_needed(
                cast(HyperGrid, env),
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
        env_device = getattr(env, "device", torch.device("cpu"))
        synchronize_if_needed(env_device)
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
    gflownet: PFBasedGFlowNet,
    env: Env,
    trajectories,
    *,
    use_vmap: bool,
) -> torch.Tensor:
    if use_vmap:
        if not isinstance(gflownet, TBGFlowNet):
            raise ValueError("vmap trajectory balance loss requires a TBGFlowNet.")
        return trajectory_balance_loss_vmap(cast(TBGFlowNet, gflownet), trajectories)

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

    log_z_value = gflownet.logZ
    if not isinstance(log_z_value, torch.Tensor):
        log_z_tensor = torch.as_tensor(log_z_value, device=residuals.device)
    else:
        log_z_tensor = log_z_value
    log_z_tensor = log_z_tensor.squeeze()
    scores = (residuals + log_z_tensor).pow(2)

    return scores.mean()


def run_validation_if_needed(
    env: HyperGrid,
    gflownet: PFBasedGFlowNet,
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
    args: argparse.Namespace,
    device: torch.device,
    scenario: ScenarioConfig,
    flow_variant: FlowVariant,
    env_cfg: EnvironmentBenchmark,
) -> tuple[Env, PFBasedGFlowNet, Sampler, torch.optim.Optimizer, States]:
    if env_cfg.key == "hypergrid":
        return _build_hypergrid_components(args, device, scenario, flow_variant)
    if env_cfg.key == "diffusion":
        return _build_diffusion_components(args, device, scenario, flow_variant)
    raise ValueError(f"Unsupported environment key: {env_cfg.key}")


def _build_hypergrid_components(
    args: argparse.Namespace,
    device: torch.device,
    scenario: ScenarioConfig,
    flow_variant: FlowVariant,
) -> tuple[HyperGrid, PFBasedGFlowNet, Sampler, torch.optim.Optimizer, DiscreteStates]:
    env_kwargs = dict(HYPERGRID_KWARGS)
    env_kwargs["device"] = device
    EnvClass = HyperGridWithTensorStep if scenario.use_script_env else HyperGrid
    env = EnvClass(**env_kwargs)

    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    module_pf = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    module_pb = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        trunk=module_pf.trunk,
    )

    if scenario.sampler in {"compiled_chunk", "script_chunk"}:
        pf_estimator = FastKHotDiscretePolicyEstimator(env, module_pf, preprocessor)
    else:
        pf_estimator = DiscretePolicyEstimator(
            module_pf, env.n_actions, preprocessor=preprocessor, is_backward=False
        )
    pb_estimator = DiscretePolicyEstimator(
        module_pb, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    logF_estimator: ScalarEstimator | None = None
    if flow_variant.requires_logf:
        logF_module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=1,
        )
        logF_estimator = ScalarEstimator(module=logF_module, preprocessor=preprocessor)

    if flow_variant.key == "tb":
        gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
    elif flow_variant.key == "dbg":
        assert logF_estimator is not None
        gflownet = DBGFlowNet(pf=pf_estimator, pb=pb_estimator, logF=logF_estimator)
    elif flow_variant.key == "subtb":
        assert logF_estimator is not None
        gflownet = SubTBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            logF=logF_estimator,
            weighting=args.subtb_weighting,
            lamda=args.subtb_lamda,
        )
    else:
        raise ValueError(f"Unsupported GFlowNet variant: {flow_variant.key}")

    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)

    logz_params = getattr(gflownet, "logz_parameters", None)
    if callable(logz_params):
        params = logz_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logz})

    logf_params = getattr(gflownet, "logF_parameters", None)
    if callable(logf_params):
        params = logf_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logf})

    if scenario.sampler == "compiled_chunk":
        sampler: Sampler = CompiledChunkSampler(
            estimator=pf_estimator, chunk_size=DEFAULT_CHUNK_SIZE
        )
    elif scenario.sampler == "script_chunk":
        sampler = ChunkedHyperGridSampler(
            estimator=pf_estimator, chunk_size=DEFAULT_CHUNK_SIZE
        )
    else:
        sampler = Sampler(estimator=pf_estimator)

    visited_states = env.states_from_batch_shape((0,))
    return env, gflownet, sampler, optimizer, visited_states


def _build_diffusion_components(
    args: argparse.Namespace,
    device: torch.device,
    scenario: ScenarioConfig,
    flow_variant: FlowVariant,
) -> tuple[DiffusionSampling, PFBasedGFlowNet, Sampler, torch.optim.Optimizer, States]:
    target_kwargs: dict[str, Any] = {"seed": args.diffusion_target_seed}
    if args.diffusion_dim is not None:
        target_kwargs["dim"] = args.diffusion_dim
    if args.diffusion_num_components is not None:
        target_kwargs["num_components"] = args.diffusion_num_components

    env = DiffusionSampling(
        target_str=args.diffusion_target,
        target_kwargs=target_kwargs,
        num_discretization_steps=args.diffusion_num_steps,
        device=device,
        check_action_validity=False,
    )

    s_dim = env.dim
    pf_module = DiffusionPISGradNetForward(
        s_dim=s_dim,
        harmonics_dim=args.diffusion_harmonics_dim,
        t_emb_dim=args.diffusion_t_emb_dim,
        s_emb_dim=args.diffusion_s_emb_dim,
        hidden_dim=args.diffusion_hidden_dim,
        joint_layers=args.diffusion_joint_layers,
        zero_init=args.diffusion_zero_init,
    )
    pb_module = DiffusionFixedBackwardModule(s_dim=s_dim)

    pf_estimator = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=args.diffusion_sigma,
        num_discretization_steps=args.diffusion_num_steps,
    )
    pb_estimator = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=args.diffusion_sigma,
        num_discretization_steps=args.diffusion_num_steps,
    )

    logF_estimator: ScalarEstimator | None = None
    if flow_variant.requires_logf:
        logF_module = MLP(
            input_dim=env.state_shape[-1],
            output_dim=1,
        )
        logF_preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])
        logF_estimator = ScalarEstimator(
            module=logF_module, preprocessor=logF_preprocessor
        )

    if flow_variant.key == "tb":
        gflownet: PFBasedGFlowNet = TBGFlowNet(
            pf=pf_estimator, pb=pb_estimator, init_logZ=0.0
        )
    elif flow_variant.key == "dbg":
        assert logF_estimator is not None
        gflownet = DBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            logF=logF_estimator,
        )
    elif flow_variant.key == "subtb":
        assert logF_estimator is not None
        gflownet = SubTBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            logF=logF_estimator,
            weighting=args.subtb_weighting,
            lamda=args.subtb_lamda,
        )
    else:
        raise ValueError(
            f"Unsupported GFlowNet variant for diffusion: {flow_variant.key}"
        )

    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)

    logz_params = getattr(gflownet, "logz_parameters", None)
    if callable(logz_params):
        params = logz_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logz})

    logf_params = getattr(gflownet, "logF_parameters", None)
    if callable(logf_params):
        params = logf_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logf})

    if scenario.sampler == "compiled_chunk":
        sampler: Sampler = CompiledChunkSampler(
            estimator=pf_estimator, chunk_size=DEFAULT_CHUNK_SIZE
        )
    elif scenario.sampler == "script_chunk":
        sampler = ChunkedDiffusionSampler(
            estimator=pf_estimator, chunk_size=DEFAULT_CHUNK_SIZE
        )
    else:
        sampler = Sampler(estimator=pf_estimator)

    visited_states = env.states_from_batch_shape((0,))
    return env, gflownet, sampler, optimizer, visited_states


def _mps_backend_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(requested: str) -> torch.device:
    """MPS backend is not supported for the Diffusion Sampling environment."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # if _mps_backend_available():
        #     return torch.device("mps")
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


def _summarize_iteration_times(times: list[float]) -> tuple[float, float]:
    if not times:
        return 0.0, 0.0
    mean_time = statistics.fmean(times)
    std_time = statistics.pstdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time


def _render_env_row(
    row_axes,
    env_results: list[Dict[str, Any]],
    env_cfg: EnvironmentBenchmark | None,
    palette: list[str],
) -> None:
    env_label = env_cfg.label if env_cfg else env_results[0].get("env_label", "Env")
    labels = [
        f"{res.get('label', f'Run {idx+1}')} [{res.get('gflownet_label', '?')}]"
        for idx, res in enumerate(env_results)
    ]
    times = [res.get("elapsed", 0.0) for res in env_results]
    bar_colors = [palette[i % len(palette)] for i in range(len(env_results))]

    baseline_name = env_cfg.scenarios[0].name if env_cfg and env_cfg.scenarios else None

    # Determine per-flow baselines (default to the baseline scenario if present, else first run).
    flow_baselines: dict[str, float] = {}
    for res in env_results:
        flow_key = res.get("gflownet_key")
        if flow_key is None or flow_key in flow_baselines:
            continue
        if baseline_name is not None and res.get("label") == baseline_name:
            flow_baselines[flow_key] = res.get("elapsed", 0.0) or 0.0
    for res in env_results:
        flow_key = res.get("gflownet_key")
        if flow_key is None:
            continue
        flow_baselines.setdefault(flow_key, res.get("elapsed", 0.0) or 0.0)

    bars = row_axes[0].bar(labels, times, color=bar_colors)
    row_axes[0].set_ylabel("Wall-clock time (s)")
    row_axes[0].set_title(f"{env_label} | Total Training Time")

    for bar, value, res in zip(bars, times, env_results):
        if value == 0.0:
            continue
        flow_key = res.get("gflownet_key", "")
        flow_baseline = flow_baselines.get(flow_key, value) or value
        pct_speedup = (
            (flow_baseline / value - 1.0) * 100.0 if value > 0.0 else float("inf")
        )
        row_axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}s\n{pct_speedup:+.1f}%",
            ha="center",
            va="bottom",
            color="black",
            fontsize=9,
        )

    # Subplot 2: training curves
    loss_ax = row_axes[1]
    for idx, res in enumerate(env_results):
        losses = res.get("losses") or []
        if not losses:
            continue
        variant_key = res.get("gflownet_key", "")
        scenario_label = res.get("label", "")
        color = VARIANT_COLORS.get(variant_key, palette[idx % len(palette)])
        linestyle = SCENARIO_LINESTYLES.get(scenario_label, "-")
        loss_ax.plot(
            range(1, len(losses) + 1),
            losses,
            label=labels[idx],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            alpha=LOSS_LINE_ALPHA,
        )
    loss_ax.set_title(f"{env_label} | Training Loss")
    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel("Loss")
    if loss_ax.lines:
        loss_ax.legend(fontsize="small")

    # Subplot 3: per-iteration timing with error bars

    iter_ax = row_axes[2]
    iter_stats = [
        _summarize_iteration_times(res.get("iter_times") or []) for res in env_results
    ]
    means_ms = [mean * 1000.0 for mean, _ in iter_stats]
    stds_ms = [std * 1000.0 for _, std in iter_stats]
    iter_ax.bar(
        labels,
        means_ms,
        yerr=stds_ms,
        capsize=6,
        color=bar_colors,
    )
    iter_ax.set_ylabel("Per-iteration time (ms)")
    iter_ax.set_title(f"{env_label} | Iteration Timing")

    for ax in row_axes:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")


def plot_benchmark(results: list[Dict[str, Any]], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting; install it or omit --benchmark."
        ) from exc

    if not results:
        print("No benchmark results to plot.")
        return

    env_order: list[str] = []
    for res in results:
        env_key = res.get("env_key", "unknown")
        if env_key not in env_order:
            env_order.append(env_key)

    n_rows = max(1, len(env_order))
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]  # type: ignore[list-item]

    palette = ["#6c757d", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for row_idx, env_key in enumerate(env_order):
        env_results = [res for res in results if res.get("env_key") == env_key]
        if not env_results:
            continue

        env_cfg = ENVIRONMENT_BENCHMARKS.get(env_key)
        row_axes = axes[row_idx]
        _render_env_row(row_axes, env_results, env_cfg, palette)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved benchmark plot to {output}")


if __name__ == "__main__":
    main()
