from contextlib import nullcontext
from typing import Any, Tuple, cast

import torch

from gfn.containers import Trajectories, Transitions
from gfn.estimators import (
    Estimator,
    PolicyEstimatorProtocol,
    RecurrentDiscretePolicyEstimator,
    RecurrentPolicyMixin,
    validate_policy_estimator,
)

# ------------
# Trajectories
# ------------


def get_trajectory_pfs_and_pbs(
    pf: Estimator,
    pb: Estimator | None,
    trajectories: Trajectories,
    recalculate_all_logprobs: bool = True,
    **policy_kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate PF and PB log‑probabilities for trajectories.

    Delegates to ``get_trajectory_pfs`` and ``get_trajectory_pbs`` while
    forwarding policy kwargs.

    Args:
        pf: Forward policy estimator.
        pb: Backward policy estimator, or ``None`` for trees (PB=1).
        trajectories: Trajectories to evaluate.
        recalculate_all_logprobs: If True, recompute PF even if cached.
        **policy_kwargs: Extra kwargs for ``to_probability_distribution``.

    Returns:
        ``(log_pf[T,N], log_pb[T,N])``
    """

    log_pf_trajectories = get_trajectory_pfs(
        pf,
        trajectories,
        recalculate_all_logprobs=recalculate_all_logprobs,
        **policy_kwargs,
    )
    log_pb_trajectories = get_trajectory_pbs(
        pb,
        trajectories,
        **policy_kwargs,
    )

    return log_pf_trajectories, log_pb_trajectories


def _is_kv_cached_recurrent(estimator: Any) -> bool:
    """True for a recurrent estimator sampling through the ``no_grad`` KV-cache."""
    return (
        isinstance(estimator, RecurrentDiscretePolicyEstimator)
        and estimator.use_kv_cache
    )


def _pf_valid_masks(
    trajectories: Trajectories,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masks selecting the valid PF steps: non-sink states and non-dummy actions.

    Returns ``(state_mask (T+1, N), action_mask (T, N))``; boolean indexing with these
    flattens the batch row-major, which is how the vectorized/teacher-forced paths line
    logits up with actions.
    """
    state_mask = ~trajectories.states.is_sink_state
    action_mask = ~trajectories.actions.is_dummy
    assert (
        state_mask[:-1] == action_mask
    ).all(), "state/action mask misalignment in PF evaluation"
    return state_mask, action_mask


def _scatter_pf(action_mask: torch.Tensor, valid_log_pf: torch.Tensor) -> torch.Tensor:
    """Scatter per-step log-probs back into a dense ``(T, N)`` tensor (0 where masked)."""
    out = torch.full(
        action_mask.shape,
        0.0,
        dtype=torch.get_default_dtype(),
        device=valid_log_pf.device,
    )
    out[action_mask] = valid_log_pf.to(out.dtype)
    return out


def _recurrent_teacher_forced_pfs(
    policy_pf: Any,  # a RecurrentDiscretePolicyEstimator (Any avoids protected-access noise)
    trajectories: Trajectories,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Teacher-forced (parallel) PF log-probs for a recurrent estimator.

    Instead of the per-step recompute (which re-runs the sequence model once per
    timestep, re-paying the sequential rollout cost inside the loss), this runs a
    **single** causal forward over the whole committed trajectory
    ``[BOS, w_0, ..., w_{L-1}]`` and gathers the per-step action log-probs. Because
    the sequence model's parallel forward is equal to its incremental decode
    (proven by the module tests), the result matches the per-step path to numerical
    tolerance — but with one grad-bearing forward and a shallow autograd graph.

    Only **fixed-length** trajectories are supported (every trajectory terminates at
    the same step); variable-length teacher forcing is out of scope (Gap 2).

    Args:
        policy_pf: The recurrent forward policy estimator.
        trajectories: Forward trajectories to score.
        **policy_kwargs: Forwarded to ``to_probability_distribution``.

    Returns:
        ``log_pf`` of shape ``(T, N)``.
    """
    T = trajectories.max_length
    N = trajectories.batch_size

    # Fixed-length guard: a sink state may appear only at the final state index.
    if bool(trajectories.states.is_sink_state[:-1].any()):
        raise NotImplementedError(
            "Teacher-forced recompute (teacher_forced_loss=True) supports only "
            "fixed-length trajectories; got variable-length trajectories. See the "
            "Gap-2 TODO in this module for the variable-length follow-up."
        )

    # Reuse the estimator's own full-prefix forward so the teacher-forcing input is
    # built identically to sampling (same [BOS, w_0, ...] construction) rather than
    # re-derived here. The last committed state (index -2; -1 is the sink) holds the
    # whole word sequence, and _forward_full_prefix returns the full-sequence logits.
    terminal = trajectories.states[-2]
    tokens = terminal.tensor.long()
    lengths = (terminal.tensor >= 0).sum(dim=-1)
    logits, _ = policy_pf._forward_full_prefix(tokens, lengths)  # (N, T, n_actions)
    # Transpose (N, T, ·) -> (T, N, ·) then row-major flatten, matching the flattened
    # valid states/actions below (which also flatten (T, N) row-major).
    logits = logits.transpose(0, 1).reshape(T * N, -1)

    state_mask, action_mask = _pf_valid_masks(trajectories)
    valid_states = trajectories.states[state_mask]
    valid_actions = trajectories.actions[action_mask]

    assert len(valid_states) == logits.shape[0], (
        f"teacher-forced logits ({logits.shape[0]}) do not line up with the valid "
        f"states ({len(valid_states)})"
    )

    # Mirror PolicyMixin.compute_dist: a callable temperature is a schedule that must
    # be resolved against (states, estimator_outputs) before building the dist. We
    # bypass compute_dist here (it would re-run the estimator per step), so the
    # resolution has to be repeated.
    if callable(policy_kwargs.get("temperature")):
        policy_kwargs = dict(policy_kwargs)
        policy_kwargs["temperature"] = policy_kwargs["temperature"](valid_states, logits)

    dist = policy_pf.to_probability_distribution(valid_states, logits, **policy_kwargs)
    valid_log_pf = dist.log_prob(valid_actions.tensor)  # (T*N,)
    return _scatter_pf(action_mask, valid_log_pf)


def _per_step_pfs(
    policy_pf: Any,
    trajectories: Trajectories,
    recalculate_all_logprobs: bool,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Per-step PF recompute: re-run the estimator once per timestep.

    The general fallback for estimators that cannot be evaluated on a flattened batch
    (``is_vectorized=False``), notably recurrent policies, whose carry must be threaded
    step by step.

    Args:
        policy_pf: The forward policy estimator.
        trajectories: Forward trajectories to score.
        recalculate_all_logprobs: If False, reuse cached per-step estimator outputs
            when the trajectories carry them.
        **policy_kwargs: Forwarded to ``compute_dist``.

    Returns:
        ``log_pf`` of shape ``(T, N)``.
    """
    N = trajectories.batch_size
    device = trajectories.states.device

    if trajectories.states.conditions is not None:
        cond = trajectories.states.conditions[0]  # shape (N, cond_dim)
    else:
        cond = None

    ctx = policy_pf.init_context(int(N), device, cond)

    T = trajectories.max_length
    log_pf_trajectories = torch.full(
        (T, N),
        fill_value=0.0,
        dtype=torch.get_default_dtype(),
        device=device,
    )

    for t in range(T):
        step_states = trajectories.states[t]
        step_actions = trajectories.actions[t]

        assert (step_states.is_sink_state == step_actions.is_dummy).all()
        step_mask = ~step_states.is_sink_state

        valid_step_states = step_states[step_mask]
        valid_step_actions = step_actions[step_mask]

        if not torch.any(step_mask):
            continue

        # Optimization: forward cached estimator outputs when available
        if trajectories.estimator_outputs is not None and not recalculate_all_logprobs:
            ctx.current_estimator_output = trajectories.estimator_outputs[t][step_mask]
        else:
            # Ensure we do not accidentally reuse estimator outputs from a
            # previous time step. Precomputed outputs must be provided
            # explicitly for the current step.
            ctx.current_estimator_output = None

        # Build distribution for active rows and compute step log-probs
        # TODO: masking ctx with step_mask outside of compute_dist and log_probs,
        # i.e., implement __getitem__ for ctx. (maybe we should contain only the
        # tensors, and not additional metadata like the batch size, device, etc.)
        dist, ctx = policy_pf.compute_dist(
            valid_step_states, ctx, step_mask, **policy_kwargs
        )
        step_log_probs, ctx = policy_pf.log_probs(
            valid_step_actions.tensor, dist, ctx, step_mask, vectorized=False
        )

        # Store in trajectory-level tensor.
        log_pf_trajectories[t] = step_log_probs

    return log_pf_trajectories


def get_trajectory_pfs(
    pf: Estimator,
    trajectories: Trajectories,
    recalculate_all_logprobs: bool = True,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Calculate PF log‑probabilities for trajectories.

    Non‑vectorized (per‑step) evaluation with masks
    ``~is_sink_state[t] & ~is_dummy[t]`` & no action‑id indexing is supported when
    specifically needed (estimator.is_vectorized=False).

    Args:
        pf: Forward policy estimator.
        trajectories: Trajectories to evaluate.
        recalculate_all_logprobs: If True, recompute PF even if cached. Useful for
            off-policy training.
        **policy_kwargs: Extra kwargs for ``to_probability_distribution``.

    Returns:
        ``log_pf`` of shape ``(T, N)``.

    Raises:
        ValueError: If backward trajectories are provided.
    """
    # TODO: Ensure that the estimator is policy-capable here.
    if not hasattr(pf, "init_context"):
        raise TypeError("Estimator is not policy-capable (missing PolicyMixin)")

    if trajectories.is_backward:
        raise ValueError("Backward trajectories are not supported")

    if trajectories.has_log_probs and not recalculate_all_logprobs:
        log_pf_trajectories = trajectories.log_probs
        assert log_pf_trajectories is not None
        # Guard the KV-cache footgun: cached sampling runs under no_grad, so its saved
        # log-probs are detached. Reusing them under an active graph would make
        # loss.backward() fail with an opaque autograd error; fail fast instead.
        #
        # Scoped deliberately narrowly. Detached PF log-probs are legitimate whenever
        # PF is not being trained -- e.g. TBGFlowNet with a frozen PF, learning only PB
        # and logZ -- so the guard fires only for a KV-cached PF that still has
        # trainable parameters, which is the one configuration where reuse is always a
        # bug rather than a choice.
        if (
            torch.is_grad_enabled()
            and not log_pf_trajectories.requires_grad
            and _is_kv_cached_recurrent(pf)
            and any(p.requires_grad for p in pf.parameters())
        ):
            raise RuntimeError(
                "Reusing saved log-probs that are detached from the autograd graph "
                "(sampling ran under no_grad because use_kv_cache=True), so backprop "
                "would not reach the forward policy. Pass "
                "recalculate_all_logprobs=True to recompute them (the estimator's "
                "teacher-forced loss recomputes them efficiently), or freeze the "
                "forward policy if PF is intentionally not being trained."
            )
    else:

        # Decide vectorized vs non-vectorized based on estimator capability
        # Tell the type-checker we expect the Policy mixin surface here.
        policy_pf = cast(PolicyEstimatorProtocol, pf)
        validate_policy_estimator(policy_pf, "pf")
        is_vectorized = bool(getattr(policy_pf, "is_vectorized", True))

        # Opt-in teacher-forced (parallel) recompute: one causal forward over the whole
        # trajectory instead of the per-step loop. Must be tested BEFORE the per-step
        # branch, since recurrent estimators always report is_vectorized=False.
        use_teacher_forcing = (
            isinstance(policy_pf, RecurrentDiscretePolicyEstimator)
            and policy_pf.teacher_forced_loss
        )

        if use_teacher_forcing:
            log_pf_trajectories = _recurrent_teacher_forced_pfs(
                policy_pf, trajectories, **policy_kwargs
            )
        elif not is_vectorized:
            # Per-step path. A KV-cached estimator samples under no_grad, so re-running
            # it here as-is would return detached logits and silently zero the policy
            # gradients; force the grad-bearing full-prefix forward for the recompute.
            with (
                policy_pf.grad_bearing_forward()  # type: ignore[attr-defined]
                if _is_kv_cached_recurrent(policy_pf)
                else nullcontext()
            ):
                log_pf_trajectories = _per_step_pfs(
                    policy_pf,
                    trajectories,
                    recalculate_all_logprobs=recalculate_all_logprobs,
                    **policy_kwargs,
                )

        else:
            # Vectorized path.
            state_mask, action_mask = _pf_valid_masks(trajectories)
            valid_states = trajectories.states[state_mask]
            valid_actions = trajectories.actions[action_mask]

            if len(valid_states) == 0:
                return torch.full(
                    action_mask.shape,
                    0.0,
                    dtype=torch.get_default_dtype(),
                    device=trajectories.states.device,
                )

            # Build conditions per-step shape to align with valid_states
            masked_cond = None
            if trajectories.states.conditions is not None:
                # trajectories.states.conditions shape: (T, N, cond_dim)
                masked_cond = trajectories.states.conditions[state_mask]

            ctx_v = policy_pf.init_context(
                int(len(valid_states)),
                trajectories.states.device,
                conditions=masked_cond,
            )

            # Optional estimator output cache reuse.
            if (
                trajectories.estimator_outputs is not None
                and not recalculate_all_logprobs
            ):
                estimator_outputs = trajectories.estimator_outputs[action_mask]
                ctx_v.current_estimator_output = estimator_outputs

            # Build distribution and compute vectorized log-probs
            dist, ctx_v = policy_pf.compute_dist(
                valid_states, ctx_v, step_mask=None, **policy_kwargs
            )
            valid_log_pf_actions, _ = policy_pf.log_probs(
                valid_actions.tensor, dist, ctx_v, step_mask=None, vectorized=True
            )

            # Pad back to full batch size.
            log_pf_trajectories = _scatter_pf(action_mask, valid_log_pf_actions)

    assert log_pf_trajectories.shape == (
        trajectories.max_length,
        trajectories.batch_size,
    )

    return log_pf_trajectories


def get_trajectory_pbs(
    pb: Estimator | None,
    trajectories: Trajectories,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Calculate PB log‑probabilities for trajectories.

    Non‑vectorized (per‑step) evaluation with with alignment
        (action at ``t`` with state at ``t+1``) and mask
        ``~is_sink_state[t+1] & ~is_initial_state[t+1] & ~is_dummy[t] & ~is_exit[t]``;
        skip ``t==0``. is supported when specifically needed
        (estimator.is_vectorized=False).

    Args:
        pb: Backward policy estimator, or ``None`` for trees (PB=1).
        trajectories: Trajectories to evaluate.
        **policy_kwargs: Extra kwargs for ``to_probability_distribution``.

    Returns:
        ``log_pb`` of shape ``(T, N)``.

    Raises:
        ValueError: If backward trajectories are provided.
    """
    if trajectories.is_backward:
        raise ValueError("Backward trajectories are not supported")

    # Allocate log_pb explicitly as floating point to avoid silent int casts.
    log_pb_trajectories = torch.full(
        trajectories.actions.tensor[..., 0].shape,
        fill_value=0.0,
        dtype=torch.get_default_dtype(),
        device=trajectories.states.device,
    )

    # Note the different mask for valid states and actions compared to the pf case.
    state_mask = (
        ~trajectories.states.is_sink_state & ~trajectories.states.is_initial_state
    )
    # We can't calculate the PB of the first state, even it's not an initial state.
    state_mask[0, :] = False
    action_mask = ~trajectories.actions.is_dummy & ~trajectories.actions.is_exit

    valid_states = trajectories.states[state_mask]
    valid_actions = trajectories.actions[action_mask]

    if valid_states.batch_shape != valid_actions.batch_shape:
        raise AssertionError("Something wrong happening with log_pf evaluations")

    if len(valid_states) == 0:
        return log_pb_trajectories

    # There is no backward policy in this case.
    if pb is None:
        # If pb is None, we assume that the gflownet DAG is a tree, and therefore
        # the backward policy probability is always 1 (log probs are 0).
        valid_log_pb_actions = torch.zeros_like(
            valid_actions.tensor, dtype=torch.get_default_dtype()
        )
        valid_log_pb_actions = valid_log_pb_actions.squeeze(-1)  # no padding.
        log_pb_trajectories[action_mask] = valid_log_pb_actions.to(
            log_pb_trajectories.dtype, copy=False
        )

        assert log_pb_trajectories.shape == (
            trajectories.max_length,
            trajectories.batch_size,
        )

        return log_pb_trajectories

    # There is a backward policy.
    policy_pb = cast(PolicyEstimatorProtocol, pb)
    validate_policy_estimator(policy_pb, "pb")
    is_vectorized = bool(getattr(policy_pb, "is_vectorized", True))

    if not is_vectorized:
        # Per-step pb evaluation (state at t+1, action at t)
        N = trajectories.batch_size
        device = trajectories.states.device

        if trajectories.states.conditions is not None:
            # Assumption: a trajectory has only one condition.
            # Here we use the condition of the initial state of the trajectory.
            cond = trajectories.states.conditions[0]
        else:
            cond = None
        ctx = policy_pb.init_context(int(N), device, conditions=cond)

        # Iterate per-step with masking (state at t+1, action at t)
        for t in range(trajectories.max_length):
            # TODO: these checks are curious - I think one of them is never needed
            # because for now we do not support reversed trajectories.
            next_state_isnt_sink = ~trajectories.states.is_sink_state[t + 1]
            next_state_isnt_initial = ~trajectories.states.is_initial_state[t + 1]
            state_ok = next_state_isnt_sink & next_state_isnt_initial
            if t == 0:
                # log PB is always zero for the transition s1 -> s0.
                state_ok = torch.zeros_like(state_ok, dtype=torch.bool)

            action_ok = (~trajectories.actions.is_dummy[t]) & (
                ~trajectories.actions.is_exit[t]
            )
            step_mask = state_ok & action_ok

            if not torch.any(step_mask):
                continue

            step_states = trajectories.states[t + 1][step_mask]
            step_actions = trajectories.actions.tensor[t][step_mask]

            # Prevent reusing last step's estimator output (batch size may differ,
            # and estimator output caching isn't needed for PB).
            ctx.current_estimator_output = None
            dist, ctx = policy_pb.compute_dist(
                step_states, ctx, step_mask, **policy_kwargs
            )
            step_lp, ctx = policy_pb.log_probs(
                step_actions, dist, ctx, step_mask, vectorized=False
            )

            padded = torch.full((N,), 0.0, device=device, dtype=step_lp.dtype)
            padded[step_mask] = step_lp[step_mask]
            log_pb_trajectories[t] = padded

    # The backward policy supports vectorized evaluation.
    else:
        masked_cond = None
        if trajectories.states.conditions is not None:
            masked_cond = trajectories.states.conditions[state_mask]

        ctx_v = policy_pb.init_context(
            int(len(valid_states)), trajectories.states.device, conditions=masked_cond
        )
        dist, ctx_v = policy_pb.compute_dist(
            valid_states,
            ctx_v,
            step_mask=None,
            **policy_kwargs,
        )
        valid_log_pb_actions, _ = policy_pb.log_probs(
            valid_actions.tensor, dist, ctx_v, step_mask=None, vectorized=True
        )
        log_pb_trajectories[action_mask] = valid_log_pb_actions.to(
            log_pb_trajectories.dtype, copy=False
        )

    assert log_pb_trajectories.shape == (
        trajectories.max_length,
        trajectories.batch_size,
    )

    return log_pb_trajectories


# -----------
# Transitions
# -----------


def get_transition_pfs_and_pbs(
    pf: Estimator,
    pb: Estimator | None,
    transitions: Transitions,
    recalculate_all_logprobs: bool = True,
    **policy_kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate PF and PB log‑probabilities for transitions.

    Args:
        pf: Forward policy estimator.
        pb: Backward policy estimator, or ``None`` for trees (PB=1).
        transitions: Transitions to evaluate.
        recalculate_all_logprobs: If True, recompute PF even if cached. Useful for
            off-policy training.
        **policy_kwargs: Extra kwargs for ``to_probability_distribution``.

    Returns:
        ``(log_pf[M], log_pb[M])``.

    Raises:
        ValueError: If backward transitions are provided.
    """
    if transitions.is_backward:
        raise ValueError("Backward transitions are not supported")

    log_pf_transitions = get_transition_pfs(
        pf, transitions, recalculate_all_logprobs, **policy_kwargs
    )
    log_pb_transitions = get_transition_pbs(pb, transitions, **policy_kwargs)

    assert log_pf_transitions.shape == (transitions.n_transitions,)
    assert log_pb_transitions.shape == (transitions.n_transitions,)

    return log_pf_transitions, log_pb_transitions


def get_transition_pfs(
    pf: Estimator,
    transitions: Transitions,
    recalculate_all_logprobs: bool = True,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Calculate PF log‑probabilities for transitions.

    Args:
        pf: Forward policy estimator.
        transitions: Transitions to evaluate.
        recalculate_all_logprobs: If True, recompute PF even if cached. Useful for
            off-policy training.
        **policy_kwargs: Extra kwargs for ``to_probability_distribution``.

    Returns:
        ``log_pf`` of shape ``(M,)``.
    """
    states = transitions.states
    actions = transitions.actions

    if transitions.has_log_probs and not recalculate_all_logprobs:
        log_pf_actions = transitions.log_probs
        assert log_pf_actions is not None
    else:

        if isinstance(pf, RecurrentPolicyMixin):
            raise TypeError("RecurrentPolicyMixin is only supported for Trajectories")

        N = transitions.n_transitions
        device = transitions.states.device
        cond = states.conditions

        # For static typing, cast to the policy protocol before calling mixin methods.
        policy_pf = cast(PolicyEstimatorProtocol, pf)
        validate_policy_estimator(policy_pf, "pf")
        ctx = policy_pf.init_context(int(N), device, cond)
        mask = torch.ones(N, dtype=torch.bool, device=device)

        # Evaluate the log PF of the actions
        # TODO: Inefficient duplication in case of tempered policy
        # The Transitions container should then have some
        # estimator_outputs attribute as well, to avoid duplication here ?
        # See (#156).
        dist, ctx = policy_pf.compute_dist(states[mask], ctx, mask, **policy_kwargs)
        log_pf_actions, _ = policy_pf.log_probs(
            actions.tensor[mask], dist, ctx, mask, vectorized=False
        )

    return log_pf_actions


def get_transition_pbs(
    pb: Estimator | None,
    transitions: Transitions,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Calculate PB log‑probabilities for transitions.

    Args:
        pb: Backward policy estimator, or ``None`` for trees (PB=1).
        transitions: Transitions to evaluate.
        **policy_kwargs: Extra kwargs for ``to_probability_distribution``.

    Returns:
        ``log_pb`` of shape ``(M,)``.
    """

    log_pb_actions = torch.zeros(
        (transitions.n_transitions,), device=transitions.states.device
    )

    # If pb is None, we assume that the gflownet DAG is a tree, and therefore
    # the backward policy probability is always 1 (log probs are 0).
    if pb is None:
        return log_pb_actions

    if not hasattr(pb, "init_context"):
        raise TypeError("Estimator is not policy-capable (missing PolicyMixin)")

    if isinstance(pb, RecurrentPolicyMixin):
        raise TypeError("RecurrentPolicyMixin is only supported for Trajectories")

    # For static typing, cast to the policy protocol before calling mixin methods.
    policy_pb = cast(PolicyEstimatorProtocol, pb)
    validate_policy_estimator(policy_pb, "pb")
    ctx = policy_pb.init_context(
        int(transitions.n_transitions),
        transitions.states.device,
        transitions.states.conditions,
    )

    # Legacy-complete masking for PB on transitions:
    # require non-terminating next_states and non-exit actions simultaneously
    # automatically removes invalid transitions (i.e. s_f -> s_f)
    mask = ~transitions.is_terminating & ~transitions.actions.is_exit

    if not torch.any(mask):
        return log_pb_actions

    dist, ctx = policy_pb.compute_dist(
        transitions.next_states[mask], ctx, mask, **policy_kwargs
    )
    step_lp, _ = policy_pb.log_probs(
        transitions.actions.tensor[mask], dist, ctx, mask, vectorized=False
    )
    log_pb_actions[mask] = step_lp[mask]

    return log_pb_actions
