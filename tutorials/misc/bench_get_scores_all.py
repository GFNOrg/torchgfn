"""
Unified micro-benchmark runner for GFlowNet losses/get_scores.

Runs, in order:
- Trajectory Balance (TB) loss
- Log Partition Variance (LPV) loss
- Sub-trajectory Balance (SubTB) get_scores
- Detailed Balance (DB) get_scores
- Modified Detailed Balance (ModDB) get_scores

For each loss, the script reports four timings:
    original (baseline, frozen copy)
    original+compile (torch.compile applied to the baseline function)
    current (eager)
    current+compile (torch.compile applied to the current function)
Two speedups are printed: current/original and current+compile/original.

All benchmarks use embedded base sizes, scaled by a single --size-scale
multiplier. Correctness is checked once per size before timing and is
excluded from the timing loops.
"""

from __future__ import annotations

import argparse
import math
from types import MethodType
from typing import Any, Callable, Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils import benchmark

from gfn.gflownet.detailed_balance import DBGFlowNet, ModifiedDBGFlowNet
from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet
from gfn.gflownet.trajectory_balance import (
    LogPartitionVarianceGFlowNet,
    TBGFlowNet,
)
from gfn.utils.handlers import (
    has_conditions_exception_handler,
    no_conditions_exception_handler,
)

# -------------------------
# TB / LPV (loss benchmark)
# -------------------------


class _TBTrajectoriesStub:
    def __init__(
        self, log_rewards: torch.Tensor, conditions: torch.Tensor | None = None
    ):
        self._log_rewards = log_rewards
        self.n_trajectories = log_rewards.shape[0]
        self.conditions = conditions

    @property
    def log_rewards(self) -> torch.Tensor:
        return self._log_rewards


def _build_tb(
    T: int, N: int, device: torch.device, dtype: torch.dtype
) -> Tuple[TBGFlowNet, _TBTrajectoriesStub, torch.Tensor, torch.Tensor]:
    log_pf = torch.randn(T, N, device=device, dtype=dtype)
    log_pb = torch.randn(T, N, device=device, dtype=dtype)
    log_rewards = torch.randn(N, device=device, dtype=dtype)
    trajectories = _TBTrajectoriesStub(log_rewards)

    model = TBGFlowNet.__new__(TBGFlowNet)
    nn.Module.__init__(model)
    model.debug = False
    model.log_reward_clip_min = -float("inf")
    model.logZ = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

    def _get_pfs_and_pbs(self, _trajectories, recalculate_all_logprobs: bool = True):
        return log_pf, log_pb

    model.get_pfs_and_pbs = MethodType(_get_pfs_and_pbs, model)
    return model, trajectories, log_pf, log_pb


def _build_lpv(
    T: int, N: int, device: torch.device, dtype: torch.dtype
) -> Tuple[
    LogPartitionVarianceGFlowNet, _TBTrajectoriesStub, torch.Tensor, torch.Tensor
]:
    log_pf = torch.randn(T, N, device=device, dtype=dtype)
    log_pb = torch.randn(T, N, device=device, dtype=dtype)
    log_rewards = torch.randn(N, device=device, dtype=dtype)
    trajectories = _TBTrajectoriesStub(log_rewards)

    model = LogPartitionVarianceGFlowNet.__new__(LogPartitionVarianceGFlowNet)
    nn.Module.__init__(model)
    model.debug = False
    model.log_reward_clip_min = -float("inf")

    def _get_pfs_and_pbs(self, _trajectories, recalculate_all_logprobs: bool = True):
        return log_pf, log_pb

    model.get_pfs_and_pbs = MethodType(_get_pfs_and_pbs, model)
    return model, trajectories, log_pf, log_pb


def _tb_original_get_scores(
    model, trajectories, log_pf: torch.Tensor, log_pb: torch.Tensor
):
    total_log_pf_trajectories = log_pf.sum(dim=0)
    total_log_pb_trajectories = log_pb.sum(dim=0)

    log_rewards = trajectories.log_rewards
    if math.isfinite(model.log_reward_clip_min):
        log_rewards = log_rewards.clamp_min(model.log_reward_clip_min)

    return total_log_pf_trajectories - total_log_pb_trajectories - log_rewards


def _tb_original_loss(model, trajectories, log_pf: torch.Tensor, log_pb: torch.Tensor):
    scores = _tb_original_get_scores(model, trajectories, log_pf, log_pb)
    logZ = torch.as_tensor(model.logZ).squeeze()
    scores = (scores + logZ).pow(2)
    return scores.mean()


def _lpv_original_loss(model, trajectories, log_pf: torch.Tensor, log_pb: torch.Tensor):
    scores = _tb_original_get_scores(model, trajectories, log_pf, log_pb)
    centered = scores - scores.mean()
    return centered.pow(2).mean()


# -------------------------
# SubTB (get_scores)
# -------------------------


class _SubTBDummyTrajectories:
    def __init__(self, terminating_idx: torch.Tensor, max_length: int):
        self.terminating_idx = terminating_idx
        self.max_length = max_length
        self.n_trajectories = terminating_idx.shape[0]

    def __len__(self) -> int:
        return self.n_trajectories


def _subtb_build_model_and_data(
    max_len: int, n_traj: int, seed: int = 0, device: str | torch.device = "cpu"
) -> Tuple[
    SubTBGFlowNet, _SubTBDummyTrajectories, list[torch.Tensor], list[torch.Tensor]
]:
    torch.manual_seed(seed)
    device = torch.device(device)
    terminating_idx = torch.randint(1, max_len + 1, (n_traj,), device=device)
    log_rewards = torch.randn(n_traj, device=device)
    log_pf_trajectories = torch.randn(max_len, n_traj, device=device)
    log_pb_trajectories = torch.randn(max_len, n_traj, device=device)
    log_state_flows = torch.randn(max_len, n_traj, device=device)
    sink_states_mask = torch.zeros(max_len, n_traj, dtype=torch.bool, device=device)
    is_terminal_mask = torch.zeros(max_len, n_traj, dtype=torch.bool, device=device)

    preds_list = [
        torch.randn(max_len + 1 - i, n_traj, device=device)
        for i in range(1, max_len + 1)
    ]
    targets_list = [
        torch.randn(max_len + 1 - i, n_traj, device=device)
        for i in range(1, max_len + 1)
    ]

    trajectories = _SubTBDummyTrajectories(
        terminating_idx=terminating_idx, max_length=max_len
    )

    model = SubTBGFlowNet.__new__(SubTBGFlowNet)
    torch.nn.Module.__init__(model)
    model.debug = False
    model.log_reward_clip_min = float("-inf")

    model.get_pfs_and_pbs = MethodType(
        lambda self, traj, recalculate_all_logprobs=True: (
            log_pf_trajectories,
            log_pb_trajectories,
        ),
        model,
    )
    model.calculate_log_state_flows = MethodType(
        lambda self, _env, _traj, _log_pf: log_state_flows, model
    )
    model.calculate_masks = MethodType(
        lambda self, _log_state_flows, _traj: (sink_states_mask, is_terminal_mask),
        model,
    )
    trajectories.log_rewards = log_rewards
    model.calculate_preds = MethodType(
        lambda self, _log_pf_cum, _log_state_flows, i: preds_list[i - 1], model
    )
    model.calculate_targets = MethodType(
        lambda self, _traj, _preds, _log_pb_cum, _log_state_flows, _term_mask, _sink_mask, i: targets_list[
            i - 1
        ],
        model,
    )

    return model, trajectories, preds_list, targets_list


def _subtb_original_get_scores(
    model: SubTBGFlowNet, env, trajectories
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    log_pf_trajectories_, log_pb_trajectories_ = model.get_pfs_and_pbs(
        trajectories, recalculate_all_logprobs=True
    )

    log_pf_trajectories_cum = model.cumulative_logprobs(
        trajectories, log_pf_trajectories_
    )
    log_pb_trajectories_cum = model.cumulative_logprobs(
        trajectories, log_pb_trajectories_
    )

    log_state_flows_ = model.calculate_log_state_flows(
        env, trajectories, log_pf_trajectories_
    )
    sink_states_mask_, is_terminal_mask_ = model.calculate_masks(
        log_state_flows_, trajectories
    )

    flattening_masks_orig = []
    scores_orig = []
    for i in range(1, 1 + trajectories.max_length):
        preds = model.calculate_preds(log_pf_trajectories_cum, log_state_flows_, i)
        targets = model.calculate_targets(
            trajectories,
            preds,
            log_pb_trajectories_cum,
            log_state_flows_,
            is_terminal_mask_,
            sink_states_mask_,
            i,
        )

        flattening_mask = trajectories.terminating_idx.lt(
            torch.arange(
                i,
                trajectories.max_length + 1,
                device=trajectories.terminating_idx.device,
            ).unsqueeze(-1)
        )

        flat_preds = preds[~flattening_mask]
        if model.debug and torch.any(torch.isnan(flat_preds)):
            raise ValueError("NaN in preds")

        flat_targets = targets[~flattening_mask]
        if model.debug and torch.any(torch.isnan(flat_targets)):
            raise ValueError("NaN in targets")

        flattening_masks_orig.append(flattening_mask)
        scores_orig.append(preds - targets)

    return scores_orig, flattening_masks_orig


# -------------------------
# DB / ModDB (get_scores)
# -------------------------


class _DBDummyStates:
    def __init__(self, tensor: torch.Tensor, is_sink_state: torch.Tensor | None = None):
        self.tensor = tensor
        self.is_sink_state = (
            is_sink_state
            if is_sink_state is not None
            else torch.zeros(tensor.shape[0], dtype=torch.bool, device=tensor.device)
        )

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx) -> "_DBDummyStates":
        return _DBDummyStates(self.tensor[idx], self.is_sink_state[idx])

    @property
    def batch_shape(self) -> torch.Size:
        return self.tensor.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.tensor.device


class _DBDummyActions:
    exit_action = torch.tensor(0, dtype=torch.long)

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.is_exit = torch.zeros_like(tensor, dtype=torch.bool)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx) -> "_DBDummyActions":
        return _DBDummyActions(self.tensor[idx])

    @property
    def batch_shape(self) -> torch.Size:
        return self.tensor.shape


class _DBDummyTransitions:
    def __init__(
        self,
        states: _DBDummyStates,
        next_states: _DBDummyStates,
        actions: _DBDummyActions,
        is_terminating: torch.Tensor,
        log_rewards: torch.Tensor,
        conditions: torch.Tensor | None = None,
    ):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.is_terminating = is_terminating
        self.is_backward = False
        self.conditions = conditions
        self.log_rewards = log_rewards
        self.device = states.device
        self.n_transitions = len(states)

    def __len__(self) -> int:
        return self.n_transitions


class _DBDummyEnv:
    def __init__(self, log_reward_fn: Callable[[Any, Any | None], torch.Tensor]):
        self._log_reward_fn = log_reward_fn

    def log_reward(self, states: Any, conditions: Any | None = None) -> torch.Tensor:
        return self._log_reward_fn(states, conditions)


def _db_build_model_and_data(
    n_transitions: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
    forward_looking: bool = False,
) -> Tuple[DBGFlowNet, _DBDummyEnv, _DBDummyTransitions]:
    torch.manual_seed(seed)
    device = torch.device(device)

    states_tensor = torch.randn(n_transitions, 4, device=device)
    next_states_tensor = torch.randn(n_transitions, 4, device=device)
    is_sink_state = torch.zeros(n_transitions, dtype=torch.bool, device=device)
    states = _DBDummyStates(states_tensor, is_sink_state=is_sink_state)
    next_states = _DBDummyStates(next_states_tensor, is_sink_state=is_sink_state.clone())

    is_terminating = torch.zeros(n_transitions, dtype=torch.bool, device=device)
    is_terminating[::3] = True
    actions = _DBDummyActions(
        torch.zeros(n_transitions, dtype=torch.long, device=device)
    )
    log_rewards = torch.randn(n_transitions, device=device)

    transitions = _DBDummyTransitions(
        states=states,
        next_states=next_states,
        actions=actions,
        is_terminating=is_terminating,
        log_rewards=log_rewards,
        conditions=None,
    )

    log_pf = torch.randn(n_transitions, device=device)
    log_pb = torch.randn(n_transitions, device=device)
    logF_states = torch.randn(n_transitions, 1, device=device)
    logF_next = torch.randn(n_transitions, 1, device=device)
    log_reward_states = torch.randn(n_transitions, device=device)
    log_reward_next = torch.randn(n_transitions, device=device)

    def get_pfs_and_pbs_stub(_self, _transitions, recalculate_all_logprobs: bool = True):
        return log_pf, log_pb

    def logF_stub(_self, s, _conditions=None):
        length = len(s)
        if length == n_transitions:
            return logF_states
        return logF_next[:length]

    def log_reward_stub(_states, _conditions=None):
        length = len(_states)
        if length == n_transitions:
            return log_reward_states
        return log_reward_next[:length]

    env = _DBDummyEnv(log_reward_stub)

    model = DBGFlowNet.__new__(DBGFlowNet)
    torch.nn.Module.__init__(model)
    model.debug = False
    model.forward_looking = forward_looking
    model.log_reward_clip_min = -float("inf")
    model.get_pfs_and_pbs = MethodType(get_pfs_and_pbs_stub, model)
    model.logF = MethodType(logF_stub, model)

    return model, env, transitions


def _db_original_get_scores(
    model: DBGFlowNet,
    env: _DBDummyEnv,
    transitions: _DBDummyTransitions,
    recalculate_all_logprobs: bool = True,
) -> torch.Tensor:
    if model.debug and transitions.is_backward:
        raise ValueError("Backward transitions are not supported")

    states = transitions.states
    transitions.actions

    if len(states) == 0:
        return torch.tensor(0.0, device=transitions.device)

    log_pf, log_pb = model.get_pfs_and_pbs(
        transitions, recalculate_all_logprobs=recalculate_all_logprobs
    )

    if transitions.conditions is not None:
        with has_conditions_exception_handler("logF", model.logF):
            log_F_s = model.logF(states, transitions.conditions).squeeze(-1)
    else:
        with no_conditions_exception_handler("logF", model.logF):
            log_F_s = model.logF(states).squeeze(-1)

    log_F_s_next = torch.zeros_like(log_F_s)
    is_terminating = transitions.is_terminating
    is_intermediate = ~is_terminating

    interm_next_states = transitions.next_states[is_intermediate]
    if transitions.conditions is not None:
        with has_conditions_exception_handler("logF", model.logF):
            log_F_s_next[is_intermediate] = model.logF(
                interm_next_states,
                transitions.conditions[is_intermediate],
            ).squeeze(-1)
    else:
        with no_conditions_exception_handler("logF", model.logF):
            log_F_s_next[is_intermediate] = model.logF(interm_next_states).squeeze(-1)

    if model.forward_looking:
        if transitions.conditions is not None:
            log_rewards_state = env.log_reward(states, transitions.conditions)
            log_rewards_next = env.log_reward(
                interm_next_states, transitions.conditions[is_intermediate]
            )
        else:
            log_rewards_state = env.log_reward(states)
            log_rewards_next = env.log_reward(interm_next_states)

        log_rewards_state = log_rewards_state.clamp_min(model.log_reward_clip_min)
        log_rewards_next = log_rewards_next.clamp_min(model.log_reward_clip_min)

        log_F_s = log_F_s + log_rewards_state
        log_F_s_next[is_intermediate] = log_F_s_next[is_intermediate] + log_rewards_next

    log_rewards = transitions.log_rewards
    log_rewards = log_rewards.clamp_min(model.log_reward_clip_min)
    log_F_s_next[is_terminating] = log_rewards[is_terminating]

    preds = log_pf + log_F_s
    targets = log_pb + log_F_s_next
    scores = preds - targets
    return scores


# ----- Modified DB -----


class _ModDBDummyStates:
    def __init__(self, tensor: torch.Tensor, is_sink_state: torch.Tensor | None = None):
        self.tensor = tensor
        self.is_sink_state = (
            is_sink_state
            if is_sink_state is not None
            else torch.zeros(tensor.shape[0], dtype=torch.bool, device=tensor.device)
        )

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx) -> "_ModDBDummyStates":
        return _ModDBDummyStates(self.tensor[idx], self.is_sink_state[idx])

    @property
    def device(self) -> torch.device:
        return self.tensor.device


class _ModDBDummyActions:
    exit_action = torch.tensor(0, dtype=torch.long)

    def __init__(self, tensor: torch.Tensor, is_exit: torch.Tensor | None = None):
        self.tensor = tensor
        self.is_exit = (
            is_exit
            if is_exit is not None
            else torch.zeros_like(tensor, dtype=torch.bool)
        )

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx) -> "_ModDBDummyActions":
        return _ModDBDummyActions(self.tensor[idx], self.is_exit[idx])


class _ModDBDummyTransitions:
    def __init__(
        self,
        states: _ModDBDummyStates,
        next_states: _ModDBDummyStates,
        actions: _ModDBDummyActions,
        all_log_rewards: torch.Tensor,
        is_backward: bool = False,
        log_probs: torch.Tensor | None = None,
        has_log_probs: bool = False,
        conditions: torch.Tensor | None = None,
    ):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.all_log_rewards = all_log_rewards
        self.is_backward = is_backward
        self.log_probs = log_probs
        self.has_log_probs = has_log_probs
        self.conditions = conditions
        self.device = states.device
        self.n_transitions = len(states)

    def __len__(self) -> int:
        return self.n_transitions

    def __getitem__(self, idx) -> "_ModDBDummyTransitions":
        return _ModDBDummyTransitions(
            self.states[idx],
            self.next_states[idx],
            self.actions[idx],
            self.all_log_rewards[idx],
            self.is_backward,
            self.log_probs[idx] if self.log_probs is not None else None,
            self.has_log_probs,
            self.conditions[idx] if self.conditions is not None else None,
        )


class _ModDBFakeDist:
    def __init__(self, log_action: torch.Tensor, log_exit: torch.Tensor):
        self._log_action = log_action
        self._log_exit = log_exit

    def log_prob(self, action_tensor: torch.Tensor) -> torch.Tensor:
        n = action_tensor.shape[0]
        if action_tensor.shape == self._log_exit.shape:
            return self._log_exit
        return self._log_action[:n]


class _ModDBDummyEstimator:
    def __init__(
        self,
        log_action: torch.Tensor,
        log_exit: torch.Tensor,
    ):
        self._log_action = log_action
        self._log_exit = log_exit

    def __call__(self, states: _ModDBDummyStates, conditions=None):
        return None

    def to_probability_distribution(self, states: _ModDBDummyStates, module_output=None):
        return _ModDBFakeDist(self._log_action, self._log_exit)


def _moddb_build_model_and_data(
    n_transitions: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Tuple[ModifiedDBGFlowNet, _ModDBDummyTransitions]:
    torch.manual_seed(seed)
    device = torch.device(device)

    states_tensor = torch.randn(n_transitions, 4, device=device)
    next_states_tensor = torch.randn(n_transitions, 4, device=device)

    is_sink_state = torch.zeros(n_transitions, dtype=torch.bool, device=device)
    is_sink_state[::4] = True
    states = _ModDBDummyStates(
        states_tensor, is_sink_state=torch.zeros_like(is_sink_state)
    )
    next_states = _ModDBDummyStates(next_states_tensor, is_sink_state=is_sink_state)

    actions_tensor = torch.randint(0, 5, (n_transitions,), device=device)
    is_exit = torch.zeros_like(actions_tensor, dtype=torch.bool)
    actions = _ModDBDummyActions(actions_tensor, is_exit=is_exit)

    all_log_rewards = torch.randn(n_transitions, 2, device=device)

    transitions = _ModDBDummyTransitions(
        states=states,
        next_states=next_states,
        actions=actions,
        all_log_rewards=all_log_rewards,
        has_log_probs=False,
        log_probs=None,
        conditions=None,
    )

    non_sink_count = int((~is_sink_state).sum().item())
    log_pf_action = torch.randn(non_sink_count, device=device)
    log_pf_exit = torch.randn(non_sink_count, device=device)
    log_pf_exit_next = torch.randn(non_sink_count, device=device)
    log_pb_action = torch.randn(non_sink_count, device=device)

    pf_estimator = _ModDBDummyEstimator(log_pf_action, log_pf_exit)
    pb_estimator = _ModDBDummyEstimator(log_pb_action, log_pf_exit_next)

    model = ModifiedDBGFlowNet.__new__(ModifiedDBGFlowNet)
    torch.nn.Module.__init__(model)
    model.debug = False
    model.constant_pb = False
    model.pf = pf_estimator
    model.pb = pb_estimator
    model.log_reward_clip_min = -float("inf")

    return model, transitions


def _moddb_original_get_scores(
    model: ModifiedDBGFlowNet,
    transitions: _ModDBDummyTransitions,
    recalculate_all_logprobs: bool = True,
) -> torch.Tensor:
    if model.debug and transitions.is_backward:
        raise ValueError("Backward transitions are not supported")

    if len(transitions) == 0:
        return torch.tensor(0.0, device=transitions.device)

    mask = ~transitions.next_states.is_sink_state
    states = transitions.states[mask]
    valid_next_states = transitions.next_states[mask]
    actions = transitions.actions[mask]
    all_log_rewards = transitions.all_log_rewards[mask]

    if transitions.conditions is not None:
        with has_conditions_exception_handler("pf", model.pf):
            module_output = model.pf(states, transitions.conditions[mask])
    else:
        with no_conditions_exception_handler("pf", model.pf):
            module_output = model.pf(states)

    if len(states) == 0:
        return torch.tensor(0.0, device=transitions.device)

    pf_dist = model.pf.to_probability_distribution(states, module_output)

    if transitions.has_log_probs and not recalculate_all_logprobs:
        valid_log_pf_actions = transitions[mask].log_probs
        assert valid_log_pf_actions is not None
    else:
        valid_log_pf_actions = pf_dist.log_prob(actions.tensor)
    exit_action_tensor = actions.__class__.exit_action.to(
        actions.tensor.device, dtype=actions.tensor.dtype
    ).expand_as(actions.tensor)
    valid_log_pf_s_exit = pf_dist.log_prob(exit_action_tensor)

    if transitions.conditions is not None:
        with has_conditions_exception_handler("pf", model.pf):
            module_output = model.pf(valid_next_states, transitions.conditions[mask])
    else:
        with no_conditions_exception_handler("pf", model.pf):
            module_output = model.pf(valid_next_states)

    valid_log_pf_s_prime_exit = model.pf.to_probability_distribution(
        valid_next_states, module_output
    ).log_prob(exit_action_tensor[: len(valid_next_states)])

    non_exit_actions = actions[~actions.is_exit]

    if model.pb is not None:
        if transitions.conditions is not None:
            with has_conditions_exception_handler("pb", model.pb):
                module_output = model.pb(valid_next_states, transitions.conditions[mask])
        else:
            with no_conditions_exception_handler("pb", model.pb):
                module_output = model.pb(valid_next_states)

        valid_log_pb_actions = model.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)
    else:
        valid_log_pb_actions = torch.zeros_like(valid_log_pf_s_exit)

    preds = all_log_rewards[:, 0] + valid_log_pf_actions + valid_log_pf_s_prime_exit
    targets = all_log_rewards[:, 1] + valid_log_pb_actions + valid_log_pf_s_exit

    scores = preds - targets
    return scores


# -------------------------
# Helpers
# -------------------------


def _select_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 is CUDA-only for this benchmark")
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}")


def _scale_int(value: int, scale: float) -> int:
    return max(1, int(round(value * scale)))


def _scale_pair(pair: Tuple[int, int], scale: float) -> Tuple[int, int]:
    return _scale_int(pair[0], scale), _scale_int(pair[1], scale)


def _time_fn(fn: Callable[[], Any]) -> float:
    t = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        num_threads=torch.get_num_threads(),
    ).blocked_autorange(min_run_time=0.5)
    return t.median


def _maybe_compile(
    fn: Callable[..., Any] | None, enabled: bool
) -> Callable[..., Any] | None:
    if fn is None or not enabled:
        return None
    return torch.compile(fn, fullgraph=False, dynamic=False, mode="reduce-overhead")


def _format_ms(value: float | None) -> str:
    if value is None:
        return "     -    "
    return f"{value*1e3:10.3f}"


def _run_with_compile_variants(
    eager_fn: Callable[[], Any],
    compile_enabled: bool,
) -> tuple[float, float | None]:
    t_eager = _time_fn(eager_fn)
    compiled_fn = _maybe_compile(eager_fn, compile_enabled)
    t_compiled = _time_fn(compiled_fn) if compiled_fn is not None else None
    return t_eager, t_compiled


def _run_tb_or_lpv(
    variant: str,
    sizes: Iterable[int],
    T: int,
    device: torch.device,
    dtype: torch.dtype,
    repeat: int,
    compile_enabled: bool,
):
    print(f"\n=== {variant.upper()} loss ===")
    print(
        f"{'N':>10}  {'chk':>4}  {'orig(ms)':>10}  {'orig+c(ms)':>10}  {'curr(ms)':>10}  {'curr+c(ms)':>10}  {'spd':>6}  {'spd_c':>6}"
    )
    for size in sizes:
        N = int(size)
        # correctness (once, not timed)
        if variant == "tb":
            model, trajectories, log_pf, log_pb = _build_tb(T, N, device, dtype)
            orig_val = _tb_original_loss(model, trajectories, log_pf, log_pb)
            curr_val = model.loss(None, trajectories, recalculate_all_logprobs=False)  # type: ignore[arg-type]
        else:
            model, trajectories, log_pf, log_pb = _build_lpv(T, N, device, dtype)
            orig_val = _lpv_original_loss(model, trajectories, log_pf, log_pb)
            curr_val = model.loss(None, trajectories, recalculate_all_logprobs=False)  # type: ignore[arg-type]
        diff = (orig_val - curr_val).abs().max().item()
        tol = 1e-6 if dtype == torch.float32 else 5e-4
        status = "PASS" if diff <= tol else "FAIL"

        orig_times = []
        origc_times = []
        curr_times = []
        currc_times = []
        for _ in range(repeat):
            t_orig, t_origc = _run_with_compile_variants(
                eager_fn=lambda: (
                    _tb_original_loss(model, trajectories, log_pf, log_pb)
                    if variant == "tb"
                    else _lpv_original_loss(model, trajectories, log_pf, log_pb)
                ),
                compile_enabled=compile_enabled,
            )
            t_curr, t_currc = _run_with_compile_variants(
                eager_fn=lambda: model.loss(
                    None, trajectories, recalculate_all_logprobs=False  # type: ignore[arg-type]
                ),
                compile_enabled=compile_enabled,
            )
            orig_times.append(t_orig)
            curr_times.append(t_curr)
            if t_origc is not None:
                origc_times.append(t_origc)
            if t_currc is not None:
                currc_times.append(t_currc)

        t_orig_ms = torch.tensor(orig_times).median().item() * 1e3
        t_origc_ms = (
            torch.tensor(origc_times).median().item() * 1e3 if origc_times else None
        )
        t_curr_ms = torch.tensor(curr_times).median().item() * 1e3
        t_currc_ms = (
            torch.tensor(currc_times).median().item() * 1e3 if currc_times else None
        )
        speedup = t_orig_ms / t_curr_ms if t_curr_ms > 0 else float("inf")
        speedup_c = (
            t_orig_ms / t_currc_ms if t_currc_ms and t_currc_ms > 0 else float("inf")
        )
        print(
            f"{N:10d}  {status:>4}  {_format_ms(t_orig_ms)}  {_format_ms(t_origc_ms)}  {_format_ms(t_curr_ms)}  {_format_ms(t_currc_ms)}  {speedup:6.2f}  {speedup_c:6.2f}"
        )


def _run_subtb(
    sizes: Iterable[Tuple[int, int]],
    device: torch.device,
    repeat: int,
    compile_enabled: bool,
):
    print("\n=== SubTB get_scores ===")
    print(
        f"{'size':>12}  {'chk':>4}  {'orig(ms)':>10}  {'orig+c(ms)':>10}  {'curr(ms)':>10}  {'curr+c(ms)':>10}  {'spd':>6}  {'spd_c':>6}"
    )
    for max_len, n_traj in sizes:
        model, trajectories, _, _ = _subtb_build_model_and_data(
            max_len, n_traj, device=device
        )
        env_obj: Any = object()

        # correctness (once, not timed)
        orig_scores, _ = _subtb_original_get_scores(model, env_obj, trajectories)
        curr_scores, _ = model.get_scores(env_obj, trajectories)  # type: ignore[arg-type]
        max_abs = max(
            (orig - curr).abs().max().item()
            for orig, curr in zip(orig_scores, curr_scores)
        )
        tol = 1e-6
        status = "PASS" if max_abs <= tol else "FAIL"

        orig_times = []
        origc_times = []
        curr_times = []
        currc_times = []
        for _ in range(repeat):
            t_orig, t_origc = _run_with_compile_variants(
                eager_fn=lambda: _subtb_original_get_scores(
                    model, env_obj, trajectories
                ),
                compile_enabled=compile_enabled,
            )
            t_curr, t_currc = _run_with_compile_variants(
                eager_fn=lambda: model.get_scores(env_obj, trajectories),  # type: ignore[arg-type]
                compile_enabled=compile_enabled,
            )
            orig_times.append(t_orig)
            curr_times.append(t_curr)
            if t_origc is not None:
                origc_times.append(t_origc)
            if t_currc is not None:
                currc_times.append(t_currc)

        t_orig_ms = torch.tensor(orig_times).median().item() * 1e3
        t_origc_ms = (
            torch.tensor(origc_times).median().item() * 1e3 if origc_times else None
        )
        t_curr_ms = torch.tensor(curr_times).median().item() * 1e3
        t_currc_ms = (
            torch.tensor(currc_times).median().item() * 1e3 if currc_times else None
        )
        speedup = t_orig_ms / t_curr_ms if t_curr_ms > 0 else float("inf")
        speedup_c = (
            t_orig_ms / t_currc_ms if t_currc_ms and t_currc_ms > 0 else float("inf")
        )
        print(
            f"{max_len}x{n_traj:5d}  {status:>4}  {_format_ms(t_orig_ms)}  {_format_ms(t_origc_ms)}  {_format_ms(t_curr_ms)}  {_format_ms(t_currc_ms)}  {speedup:6.2f}  {speedup_c:6.2f}"
        )


def _run_db(
    sizes: Iterable[int],
    device: torch.device,
    repeat: int,
    compile_enabled: bool,
    forward_looking: bool,
):
    print("\n=== DB get_scores ===")
    print(
        f"{'n_trans':>10}  {'chk':>4}  {'orig(ms)':>10}  {'orig+c(ms)':>10}  {'curr(ms)':>10}  {'curr+c(ms)':>10}  {'spd':>6}  {'spd_c':>6}"
    )
    for n_transitions in sizes:
        model, env, transitions = _db_build_model_and_data(
            n_transitions=n_transitions,
            device=device,
            forward_looking=forward_looking,
        )

        # correctness (once, not timed)
        orig = _db_original_get_scores(
            model, env, transitions, recalculate_all_logprobs=True
        )
        curr = model.get_scores(env, transitions)  # type: ignore[arg-type]
        max_abs = (orig - curr).abs().max().item()
        tol = 1e-6
        status = "PASS" if max_abs <= tol else "FAIL"

        orig_times = []
        origc_times = []
        curr_times = []
        currc_times = []
        for _ in range(repeat):
            t_orig, t_origc = _run_with_compile_variants(
                eager_fn=lambda: _db_original_get_scores(
                    model, env, transitions, recalculate_all_logprobs=True
                ),
                compile_enabled=compile_enabled,
            )
            t_curr, t_currc = _run_with_compile_variants(
                eager_fn=lambda: model.get_scores(env, transitions),  # type: ignore[arg-type]
                compile_enabled=compile_enabled,
            )
            orig_times.append(t_orig)
            curr_times.append(t_curr)
            if t_origc is not None:
                origc_times.append(t_origc)
            if t_currc is not None:
                currc_times.append(t_currc)

        t_orig_ms = torch.tensor(orig_times).median().item() * 1e3
        t_origc_ms = (
            torch.tensor(origc_times).median().item() * 1e3 if origc_times else None
        )
        t_curr_ms = torch.tensor(curr_times).median().item() * 1e3
        t_currc_ms = (
            torch.tensor(currc_times).median().item() * 1e3 if currc_times else None
        )
        speedup = t_orig_ms / t_curr_ms if t_curr_ms > 0 else float("inf")
        speedup_c = (
            t_orig_ms / t_currc_ms if t_currc_ms and t_currc_ms > 0 else float("inf")
        )
        print(
            f"{n_transitions:10d}  {status:>4}  {_format_ms(t_orig_ms)}  {_format_ms(t_origc_ms)}  {_format_ms(t_curr_ms)}  {_format_ms(t_currc_ms)}  {speedup:6.2f}  {speedup_c:6.2f}"
        )


def _run_moddb(
    sizes: Iterable[int],
    device: torch.device,
    repeat: int,
    compile_enabled: bool,
):
    print("\n=== Modified DB get_scores ===")
    print(
        f"{'n_trans':>10}  {'chk':>4}  {'orig(ms)':>10}  {'orig+c(ms)':>10}  {'curr(ms)':>10}  {'curr+c(ms)':>10}  {'spd':>6}  {'spd_c':>6}"
    )
    for n_transitions in sizes:
        model, transitions = _moddb_build_model_and_data(
            n_transitions=n_transitions,
            device=device,
        )

        # correctness (once, not timed)
        orig = _moddb_original_get_scores(
            model, transitions, recalculate_all_logprobs=True
        )
        curr = model.get_scores(transitions)  # type: ignore[arg-type]
        max_abs = (orig - curr).abs().max().item()
        tol = 1e-6
        status = "PASS" if max_abs <= tol else "FAIL"

        orig_times = []
        origc_times = []
        curr_times = []
        currc_times = []
        for _ in range(repeat):
            t_orig, t_origc = _run_with_compile_variants(
                eager_fn=lambda: _moddb_original_get_scores(
                    model, transitions, recalculate_all_logprobs=True
                ),
                compile_enabled=compile_enabled,
            )
            t_curr, t_currc = _run_with_compile_variants(
                eager_fn=lambda: model.get_scores(transitions),  # type: ignore[arg-type]
                compile_enabled=compile_enabled,
            )
            orig_times.append(t_orig)
            curr_times.append(t_curr)
            if t_origc is not None:
                origc_times.append(t_origc)
            if t_currc is not None:
                currc_times.append(t_currc)

        t_orig_ms = torch.tensor(orig_times).median().item() * 1e3
        t_origc_ms = (
            torch.tensor(origc_times).median().item() * 1e3 if origc_times else None
        )
        t_curr_ms = torch.tensor(curr_times).median().item() * 1e3
        t_currc_ms = (
            torch.tensor(currc_times).median().item() * 1e3 if currc_times else None
        )
        speedup = t_orig_ms / t_curr_ms if t_curr_ms > 0 else float("inf")
        speedup_c = (
            t_orig_ms / t_currc_ms if t_currc_ms and t_currc_ms > 0 else float("inf")
        )
        print(
            f"{n_transitions:10d}  {status:>4}  {_format_ms(t_orig_ms)}  {_format_ms(t_origc_ms)}  {_format_ms(t_curr_ms)}  {_format_ms(t_currc_ms)}  {speedup:6.2f}  {speedup_c:6.2f}"
        )


# -------------------------
# Main
# -------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cpu", help="Device to run on (cpu, mps, cuda)."
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Floating dtype for TB/LPV (others use fp32).",
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to all embedded base sizes.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each size multiple times; medians are reported.",
    )
    parser.add_argument(
        "--compile",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable torch.compile for both original and current functions.",
    )
    parser.add_argument(
        "--forward-looking",
        action="store_true",
        help="Enable forward-looking reward path for DB benchmark.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    dtype = _select_dtype(args.dtype, device)

    # Base sizes
    base_tb_sizes = [10240, 40960, 163840]
    base_T = 64
    base_subtb_sizes = [(80, 640), (160, 1280), (320, 2560)]
    base_db_sizes = [65536, 131072, 262144]
    base_moddb_sizes = [65536, 131072, 262144]

    tb_sizes = [_scale_int(n, args.size_scale) for n in base_tb_sizes]
    subtb_sizes = [_scale_pair(p, args.size_scale) for p in base_subtb_sizes]
    db_sizes = [_scale_int(n, args.size_scale) for n in base_db_sizes]
    moddb_sizes = [_scale_int(n, args.size_scale) for n in base_moddb_sizes]

    print("Benchmarking TB/LPV/SubTB/DB/ModDB in sequence")
    print(f"torch version: {torch.__version__}")
    print(f"device: {device}")
    print(f"dtype (TB/LPV): {dtype}")
    print(f"num threads: {torch.get_num_threads()}")
    print(f"size-scale: {args.size_scale}")
    print(f"compile: {args.compile}")
    print(f"repeat: {args.repeat}")
    print(f"forward-looking (DB): {args.forward_looking}")
    print()
    print(
        "Columns: original, original+compile, current, current+compile, speedup vs original (eager and compiled)."
    )

    _run_tb_or_lpv(
        variant="tb",
        sizes=tb_sizes,
        T=base_T,
        device=device,
        dtype=dtype,
        repeat=args.repeat,
        compile_enabled=args.compile,
    )
    _run_tb_or_lpv(
        variant="lpv",
        sizes=tb_sizes,
        T=base_T,
        device=device,
        dtype=dtype,
        repeat=args.repeat,
        compile_enabled=args.compile,
    )
    _run_subtb(
        sizes=subtb_sizes,
        device=device,
        repeat=args.repeat,
        compile_enabled=args.compile,
    )
    _run_db(
        sizes=db_sizes,
        device=device,
        repeat=args.repeat,
        compile_enabled=args.compile,
        forward_looking=args.forward_looking,
    )
    _run_moddb(
        sizes=moddb_sizes,
        device=device,
        repeat=args.repeat,
        compile_enabled=args.compile,
    )


if __name__ == "__main__":
    main()
