"""
Micro-benchmark for DBGFlowNet.get_scores (detailed balance) baseline vs
an optimized version. This mirrors the structure of
`tutorials/misc/bench_subtb_get_scores.py` but isolates the transition-based
DB path.
"""

from __future__ import annotations

import argparse
from types import MethodType
from typing import Any, Callable, Tuple

import torch
from torch.utils import benchmark

from gfn.gflownet.detailed_balance import DBGFlowNet


class _DummyStates:
    """Minimal stand-in for States; keeps only what get_scores touches."""

    def __init__(self, tensor: torch.Tensor, is_sink_state: torch.Tensor | None = None):
        self.tensor = tensor
        self.is_sink_state = (
            is_sink_state
            if is_sink_state is not None
            else torch.zeros(tensor.shape[0], dtype=torch.bool, device=tensor.device)
        )

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx) -> "_DummyStates":
        # Preserve sink-state bookkeeping under boolean or slice indexing.
        return _DummyStates(self.tensor[idx], self.is_sink_state[idx])

    @property
    def batch_shape(self) -> torch.Size:
        # Matches the check used in check_compatibility when debug is enabled.
        return self.tensor.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.tensor.device


class _DummyActions:
    """Minimal stand-in for Actions; only batch_shape and tensor are needed here."""

    # Keep an exit_action attribute for future compatibility (e.g., Modified DBG).
    exit_action = torch.tensor(0, dtype=torch.long)

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.is_exit = torch.zeros_like(tensor, dtype=torch.bool)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx) -> "_DummyActions":
        return _DummyActions(self.tensor[idx])

    @property
    def batch_shape(self) -> torch.Size:
        return self.tensor.shape


class _DummyTransitions:
    """Carries the attributes touched by DBGFlowNet.get_scores."""

    def __init__(
        self,
        states: _DummyStates,
        next_states: _DummyStates,
        actions: _DummyActions,
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


class _DummyEnv:
    """Lightweight env wrapper to supply log_reward."""

    def __init__(self, log_reward_fn: Callable[[Any, Any | None], torch.Tensor]):
        self._log_reward_fn = log_reward_fn

    def log_reward(self, states: Any, conditions: Any | None = None) -> torch.Tensor:
        return self._log_reward_fn(states, conditions)


def build_model_and_data(
    n_transitions: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
    forward_looking: bool = False,
) -> Tuple[DBGFlowNet, _DummyEnv, _DummyTransitions]:
    """Set up a minimal DBGFlowNet + transitions for benchmarking."""
    torch.manual_seed(seed)
    device = torch.device(device)

    # Synthetic data sized to stress memory without extra allocations in the hot path.
    states_tensor = torch.randn(n_transitions, 4, device=device)
    next_states_tensor = torch.randn(n_transitions, 4, device=device)
    is_sink_state = torch.zeros(n_transitions, dtype=torch.bool, device=device)
    states = _DummyStates(states_tensor, is_sink_state=is_sink_state)
    next_states = _DummyStates(next_states_tensor, is_sink_state=is_sink_state.clone())

    # Ensure a mix of terminating and intermediate transitions to exercise both branches.
    is_terminating = torch.zeros(n_transitions, dtype=torch.bool, device=device)
    is_terminating[::3] = True
    actions = _DummyActions(torch.zeros(n_transitions, dtype=torch.long, device=device))
    log_rewards = torch.randn(n_transitions, device=device)

    transitions = _DummyTransitions(
        states=states,
        next_states=next_states,
        actions=actions,
        is_terminating=is_terminating,
        log_rewards=log_rewards,
        conditions=None,
    )

    # Precompute tensors so each benchmark iteration avoids fresh allocations.
    log_pf = torch.randn(n_transitions, device=device)
    log_pb = torch.randn(n_transitions, device=device)
    logF_states = torch.randn(n_transitions, 1, device=device)
    logF_next = torch.randn(n_transitions, 1, device=device)
    log_reward_states = torch.randn(n_transitions, device=device)
    log_reward_next = torch.randn(n_transitions, device=device)

    def get_pfs_and_pbs_stub(_self, _transitions, recalculate_all_logprobs: bool = True):
        # Fixed tensors keep the timing focused on get_scores compute and masking.
        return log_pf, log_pb

    def logF_stub(_self, s, _conditions=None):
        # Return shape (..., 1) so the squeeze(-1) in get_scores matches real behavior.
        length = len(s)
        if length == n_transitions:
            return logF_states
        return logF_next[:length]

    def log_reward_stub(_states, _conditions=None):
        # Forward-looking uses both current and next states; size guides which buffer to use.
        length = len(_states)
        if length == n_transitions:
            return log_reward_states
        return log_reward_next[:length]

    env = _DummyEnv(log_reward_stub)

    model = DBGFlowNet.__new__(DBGFlowNet)
    torch.nn.Module.__init__(model)
    # Minimal attribute set; we bypass __init__ to avoid heavyweight estimator setup.
    model.debug = False
    model.forward_looking = forward_looking
    model.log_reward_clip_min = -float("inf")
    model.get_pfs_and_pbs = MethodType(get_pfs_and_pbs_stub, model)
    model.logF = MethodType(logF_stub, model)

    return model, env, transitions


def original_get_scores(
    model: DBGFlowNet,
    env: _DummyEnv,
    transitions: _DummyTransitions,
    recalculate_all_logprobs: bool = True,
) -> torch.Tensor:
    """Copy of the current DBGFlowNet.get_scores for baseline timing."""
    # Guard bad inputs under debug to avoid graph breaks in torch.compile.
    if model.debug and transitions.is_backward:
        raise ValueError("Backward transitions are not supported")

    states = transitions.states
    actions = transitions.actions

    if len(states) == 0:
        return torch.tensor(0.0, device=transitions.device)

    if model.debug:
        from gfn.gflownet.detailed_balance import check_compatibility

        check_compatibility(states, actions, transitions)  # type: ignore[arg-type]
        assert (
            not transitions.states.is_sink_state.any()
        ), "Transition from sink state is not allowed. This is a bug."

    # Compute log_pf and log_pb
    log_pf, log_pb = model.get_pfs_and_pbs(
        transitions, recalculate_all_logprobs=recalculate_all_logprobs  # type: ignore[arg-type]
    )

    # Compute log_F_s
    # LogF is potentially a conditional computation.
    if transitions.conditions is not None:
        from gfn.utils.handlers import has_conditions_exception_handler

        with has_conditions_exception_handler("logF", model.logF):
            log_F_s = model.logF(states, transitions.conditions).squeeze(-1)
    else:
        from gfn.utils.handlers import no_conditions_exception_handler

        with no_conditions_exception_handler("logF", model.logF):
            log_F_s = model.logF(states).squeeze(-1)

    # Compute log_F_s_next
    log_F_s_next = torch.zeros_like(log_F_s)
    is_terminating = transitions.is_terminating
    is_intermediate = ~is_terminating

    # Assign log_F_s_next for intermediate next states
    interm_next_states = transitions.next_states[is_intermediate]
    # log_F is potentially a conditional computation.
    if transitions.conditions is not None:
        from gfn.utils.handlers import has_conditions_exception_handler

        with has_conditions_exception_handler("logF", model.logF):
            log_F_s_next[is_intermediate] = model.logF(
                interm_next_states,
                transitions.conditions[is_intermediate],
            ).squeeze(-1)
    else:
        from gfn.utils.handlers import no_conditions_exception_handler

        with no_conditions_exception_handler("logF", model.logF):
            log_F_s_next[is_intermediate] = model.logF(interm_next_states).squeeze(-1)

    # Apply forward-looking if applicable
    if model.forward_looking:
        # Reward calculation can also be conditional.
        if transitions.conditions is not None:
            log_rewards_state = env.log_reward(states, transitions.conditions)  # type: ignore
            log_rewards_next = env.log_reward(
                interm_next_states, transitions.conditions[is_intermediate]  # type: ignore
            )
        else:
            log_rewards_state = env.log_reward(states)
            log_rewards_next = env.log_reward(interm_next_states)

        log_rewards_state = log_rewards_state.clamp_min(model.log_reward_clip_min)
        log_rewards_next = log_rewards_next.clamp_min(model.log_reward_clip_min)

        log_F_s = log_F_s + log_rewards_state
        log_F_s_next[is_intermediate] = log_F_s_next[is_intermediate] + log_rewards_next

    # Assign log_F_s_next for terminating transitions as log_rewards
    log_rewards = transitions.log_rewards
    assert log_rewards is not None
    log_rewards = log_rewards.clamp_min(model.log_reward_clip_min)
    log_F_s_next[is_terminating] = log_rewards[is_terminating]

    # Compute scores
    preds = log_pf + log_F_s
    targets = log_pb + log_F_s_next
    scores = preds - targets
    assert scores.shape == (transitions.n_transitions,)
    return scores


def run_once(
    mode: str,
    n_transitions: int,
    forward_looking: bool,
    use_compile: bool = False,
    device: str | torch.device = "cpu",
) -> float:
    """Return median time (seconds) for the chosen mode."""
    model, env, transitions = build_model_and_data(
        n_transitions=n_transitions,
        forward_looking=forward_looking,
        device=device,
    )

    bench: Callable[[], Any]
    compiled_get_scores: Callable | None = None

    if mode == "original":
        # Use the in-file copy of the current implementation to keep a fixed baseline.
        def bench_original():
            return original_get_scores(
                model, env, transitions, recalculate_all_logprobs=True
            )

        bench = bench_original
    elif mode == "current":
        # Benchmarks the method on the model; once optimized, this reflects new code.
        if use_compile:
            compiled_get_scores = torch.compile(
                model.get_scores,
                fullgraph=False,
                dynamic=False,
                mode="reduce-overhead",
            )

        def bench_current():
            fn = (
                compiled_get_scores
                if compiled_get_scores is not None
                else model.get_scores
            )
            return fn(env, transitions)  # type: ignore[arg-type]

        bench = bench_current
    else:
        raise ValueError(mode)

    t = benchmark.Timer(
        stmt="bench()",
        globals={"bench": bench},
        setup="",
        num_threads=torch.get_num_threads(),
    ).blocked_autorange(min_run_time=0.5)
    return t.median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["65536", "131072", "262144"],
        help="Number of transitions per batch to benchmark (larger to surface runtime differences).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for the current/optimized get_scores.",
    )
    parser.add_argument(
        "--forward-looking",
        action="store_true",
        help="Enable forward-looking reward path in the benchmark.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (e.g., cpu, mps, cuda).",
    )
    args = parser.parse_args()

    print("Benchmarking DBGFlowNet.get_scores (Detailed Balance)")
    print(f"torch version: {torch.__version__}")
    print(f"num threads: {torch.get_num_threads()}")
    print(f"forward-looking: {args.forward_looking}")
    print()
    print(f"{'n_trans':>10}  {'orig (ms)':>12}  {'curr (ms)':>12}  {'speedup':>8}")

    for size in args.sizes:
        n_transitions = int(size)
        t_orig = (
            run_once(
                "original",
                n_transitions,
                forward_looking=args.forward_looking,
                device=args.device,
            )
            * 1e3
        )
        t_curr = (
            run_once(
                "current",
                n_transitions,
                forward_looking=args.forward_looking,
                use_compile=args.compile,
                device=args.device,
            )
            * 1e3
        )
        speedup = t_orig / t_curr if t_curr > 0 else float("inf")
        print(f"{n_transitions:10d}  {t_orig:12.3f}  {t_curr:12.3f}  {speedup:8.2f}x")


if __name__ == "__main__":
    main()
