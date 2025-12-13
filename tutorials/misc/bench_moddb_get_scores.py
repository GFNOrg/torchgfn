"""
Micro-benchmark for ModifiedDBGFlowNet.get_scores baseline vs optimized.
Modeled after bench_db_get_scores.py but targets the modified DB path.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Tuple

import torch
from torch.utils import benchmark

from gfn.gflownet.detailed_balance import ModifiedDBGFlowNet
from gfn.utils.handlers import (
    has_conditions_exception_handler,
    no_conditions_exception_handler,
)


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
        return _DummyStates(self.tensor[idx], self.is_sink_state[idx])

    @property
    def device(self) -> torch.device:
        return self.tensor.device


class _DummyActions:
    """Minimal stand-in for Actions; only tensor and is_exit are needed here."""

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

    def __getitem__(self, idx) -> "_DummyActions":
        return _DummyActions(self.tensor[idx], self.is_exit[idx])


class _DummyTransitions:
    """Carries the attributes touched by ModifiedDBGFlowNet.get_scores."""

    def __init__(
        self,
        states: _DummyStates,
        next_states: _DummyStates,
        actions: _DummyActions,
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

    def __getitem__(self, idx) -> "_DummyTransitions":
        return _DummyTransitions(
            self.states[idx],
            self.next_states[idx],
            self.actions[idx],
            self.all_log_rewards[idx],
            self.is_backward,
            self.log_probs[idx] if self.log_probs is not None else None,
            self.has_log_probs,
            self.conditions[idx] if self.conditions is not None else None,
        )


class _FakeDist:
    """Simple distribution wrapper returning preset log-probs."""

    def __init__(self, log_action: torch.Tensor, log_exit: torch.Tensor):
        self._log_action = log_action
        self._log_exit = log_exit

    def log_prob(self, action_tensor: torch.Tensor) -> torch.Tensor:
        # Match shape to input; ignore actual action values to focus on timing.
        n = action_tensor.shape[0]
        # Broadcasting to match shape; slicing guards shorter inputs (next_states path).
        if action_tensor.shape == self._log_exit.shape:
            return self._log_exit
        return self._log_action[:n]


class _DummyEstimator:
    """Estimator stub providing to_probability_distribution and call signature."""

    def __init__(
        self,
        log_action: torch.Tensor,
        log_exit: torch.Tensor,
    ):
        self._log_action = log_action
        self._log_exit = log_exit

    def __call__(self, states: _DummyStates, conditions=None):
        # Return a placeholder; not used by FakeDist.
        return None

    def to_probability_distribution(self, states: _DummyStates, module_output=None):
        # Provide a fresh FakeDist per call to mirror API shape; uses preset tensors.
        return _FakeDist(self._log_action, self._log_exit)


def build_model_and_data(
    n_transitions: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Tuple[ModifiedDBGFlowNet, _DummyTransitions]:
    """Set up a minimal ModifiedDBGFlowNet + transitions for benchmarking."""
    torch.manual_seed(seed)
    device = torch.device(device)

    states_tensor = torch.randn(n_transitions, 4, device=device)
    next_states_tensor = torch.randn(n_transitions, 4, device=device)

    # Mix of sink/non-sink next states to exercise masking.
    is_sink_state = torch.zeros(n_transitions, dtype=torch.bool, device=device)
    is_sink_state[::4] = True
    states = _DummyStates(states_tensor, is_sink_state=torch.zeros_like(is_sink_state))
    next_states = _DummyStates(next_states_tensor, is_sink_state=is_sink_state)

    # Actions and exits.
    actions_tensor = torch.randint(0, 5, (n_transitions,), device=device)
    is_exit = torch.zeros_like(actions_tensor, dtype=torch.bool)
    actions = _DummyActions(actions_tensor, is_exit=is_exit)

    # Rewards for (state, next_state) pairs as expected by ModifiedDB.
    all_log_rewards = torch.randn(n_transitions, 2, device=device)

    transitions = _DummyTransitions(
        states=states,
        next_states=next_states,
        actions=actions,
        all_log_rewards=all_log_rewards,
        has_log_probs=False,
        log_probs=None,
        conditions=None,
    )

    # Precomputed log-probs for pf/pb distributions.
    # Keep same length as non-sink count to align with mask slices.
    non_sink_count = int((~is_sink_state).sum().item())
    log_pf_action = torch.randn(non_sink_count, device=device)
    log_pf_exit = torch.randn(non_sink_count, device=device)
    log_pf_exit_next = torch.randn(non_sink_count, device=device)
    log_pb_action = torch.randn(non_sink_count, device=device)

    pf_estimator = _DummyEstimator(log_pf_action, log_pf_exit)
    pb_estimator = _DummyEstimator(log_pb_action, log_pf_exit_next)

    model = ModifiedDBGFlowNet.__new__(ModifiedDBGFlowNet)
    torch.nn.Module.__init__(model)
    # Minimal attribute set; bypass __init__ to avoid heavy setup.
    model.debug = False
    model.constant_pb = False
    model.pf = pf_estimator
    model.pb = pb_estimator
    model.log_reward_clip_min = -float("inf")

    return model, transitions


def original_get_scores(
    model: ModifiedDBGFlowNet,
    transitions: _DummyTransitions,
    recalculate_all_logprobs: bool = True,
) -> torch.Tensor:
    """Copy of ModifiedDBGFlowNet.get_scores for baseline timing."""
    if model.debug and transitions.is_backward:
        raise ValueError("Backward transitions are not supported")

    if len(transitions) == 0:
        return torch.tensor(0.0, device=transitions.device)

    mask = ~transitions.next_states.is_sink_state
    states = transitions.states[mask]
    valid_next_states = transitions.next_states[mask]
    actions = transitions.actions[mask]
    all_log_rewards = transitions.all_log_rewards[mask]

    if model.debug:
        from gfn.gflownet.detailed_balance import check_compatibility

        check_compatibility(states, actions, transitions)  # type: ignore[arg-type]

    if transitions.conditions is not None:
        with has_conditions_exception_handler("pf", model.pf):  # type: ignore[name-defined]
            module_output = model.pf(states, transitions.conditions[mask])
    else:
        with no_conditions_exception_handler("pf", model.pf):  # type: ignore[name-defined]
            module_output = model.pf(states)

    if len(states) == 0:
        return torch.tensor(0.0, device=transitions.device)

    pf_dist = model.pf.to_probability_distribution(states, module_output)  # type: ignore[arg-type]

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
        with has_conditions_exception_handler("pf", model.pf):  # type: ignore[name-defined]
            module_output = model.pf(valid_next_states, transitions.conditions[mask])
    else:
        with no_conditions_exception_handler("pf", model.pf):  # type: ignore[name-defined]
            module_output = model.pf(valid_next_states)

    valid_log_pf_s_prime_exit = model.pf.to_probability_distribution(
        valid_next_states, module_output  # type: ignore[arg-type]
    ).log_prob(exit_action_tensor[: len(valid_next_states)])

    non_exit_actions = actions[~actions.is_exit]

    if model.pb is not None:
        if transitions.conditions is not None:
            with has_conditions_exception_handler("pb", model.pb):  # type: ignore[name-defined]
                module_output = model.pb(valid_next_states, transitions.conditions[mask])
        else:
            with no_conditions_exception_handler("pb", model.pb):  # type: ignore[name-defined]
                module_output = model.pb(valid_next_states)

        valid_log_pb_actions = model.pb.to_probability_distribution(
            valid_next_states, module_output  # type: ignore[arg-type]
        ).log_prob(non_exit_actions.tensor)
    else:
        valid_log_pb_actions = torch.zeros_like(valid_log_pf_s_exit)

    preds = all_log_rewards[:, 0] + valid_log_pf_actions + valid_log_pf_s_prime_exit
    targets = all_log_rewards[:, 1] + valid_log_pb_actions + valid_log_pf_s_exit

    scores = preds - targets
    if model.debug and torch.any(torch.isinf(scores)):
        raise ValueError("scores contains inf")

    return scores


def run_once(
    mode: str,
    n_transitions: int,
    use_compile: bool = False,
    device: str | torch.device = "cpu",
) -> float:
    """Return median time (seconds) for the chosen mode."""
    model, transitions = build_model_and_data(
        n_transitions=n_transitions,
        device=device,
    )

    bench: Callable[[], Any]
    compiled_get_scores: Callable | None = None

    if mode == "original":

        def bench_original():
            return original_get_scores(model, transitions, recalculate_all_logprobs=True)

        bench = bench_original
    elif mode == "current":
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
            return fn(transitions)  # type: ignore[arg-type]

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
        "--device",
        default="cpu",
        help="Device to run on (e.g., cpu, mps, cuda).",
    )
    args = parser.parse_args()

    print("Benchmarking ModifiedDBGFlowNet.get_scores (Modified DB)")
    print(f"torch version: {torch.__version__}")
    print(f"num threads: {torch.get_num_threads()}")
    print()
    print(f"{'n_trans':>10}  {'orig (ms)':>12}  {'curr (ms)':>12}  {'speedup':>8}")

    for size in args.sizes:
        n_transitions = int(size)
        t_orig = (
            run_once(
                "original",
                n_transitions,
                device=args.device,
            )
            * 1e3
        )
        t_curr = (
            run_once(
                "current",
                n_transitions,
                use_compile=args.compile,
                device=args.device,
            )
            * 1e3
        )
        speedup = t_orig / t_curr if t_curr > 0 else float("inf")
        print(f"{n_transitions:10d}  {t_orig:12.3f}  {t_curr:12.3f}  {speedup:8.2f}x")


if __name__ == "__main__":
    main()
