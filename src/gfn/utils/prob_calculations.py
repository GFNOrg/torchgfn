from typing import Any, Tuple, cast

import torch

from gfn.containers import Trajectories, Transitions
from gfn.estimators import Estimator, PolicyEstimatorProtocol, RecurrentPolicyMixin

# ------------
# Trajectories
# ------------


def get_trajectory_pfs_and_pbs(
    pf: Estimator,
    pb: Estimator | None,
    trajectories: Trajectories,
    fill_value: float = 0.0,
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
        fill_value: Value used to pad invalid positions.
        recalculate_all_logprobs: If True, recompute PF even if cached.
        **policy_kwargs: Extra kwargs for ``to_probability_distribution``.

    Returns:
        ``(log_pf[T,N], log_pb[T,N])``
    """

    log_pf_trajectories = get_trajectory_pfs(
        pf,
        trajectories,
        fill_value=fill_value,
        recalculate_all_logprobs=recalculate_all_logprobs,
        **policy_kwargs,
    )
    log_pb_trajectories = get_trajectory_pbs(
        pb,
        trajectories,
        fill_value=fill_value,
        **policy_kwargs,
    )

    return log_pf_trajectories, log_pb_trajectories


def get_trajectory_pfs(
    pf: Estimator,
    trajectories: Trajectories,
    fill_value: float = 0.0,
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
        fill_value: Value used to pad invalid positions.
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
    else:

        # Decide vectorized vs non-vectorized based on estimator capability
        # Tell the type-checker we expect the Policy mixin surface here.
        policy_pf = cast(PolicyEstimatorProtocol, pf)
        # Runtime guard: ensure the estimator actually implements the required protocol
        # method and raises an error when a non‑policy estimator is supplied.
        for required in ("init_context", "compute_dist", "log_probs"):
            if not hasattr(policy_pf, required):
                raise TypeError(
                    f"Estimator is not policy-capable (missing PolicyMixin method: {required})"
                )
        is_vectorized = bool(getattr(policy_pf, "is_vectorized", True))

        if not is_vectorized:
            # Per-step path.
            N = trajectories.n_trajectories
            device = trajectories.states.device
            cond = trajectories.conditions  # shape (N, cond_dim)

            ctx = policy_pf.init_context(int(N), device, cond)

            T = trajectories.max_length
            log_pf_trajectories = torch.full(
                (T, N),
                fill_value=fill_value,
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
                if (
                    trajectories.estimator_outputs is not None
                    and not recalculate_all_logprobs
                ):
                    ctx.current_estimator_output = trajectories.estimator_outputs[t][
                        step_mask
                    ]
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

        else:
            state_mask = ~trajectories.states.is_sink_state
            action_mask = ~trajectories.actions.is_dummy
            assert (state_mask[:-1] == action_mask).all()  # state_mask[-1] is all False

            valid_states = trajectories.states[state_mask]
            valid_actions = trajectories.actions[action_mask]

            # Vectorized path.
            log_pf_trajectories = torch.full_like(
                trajectories.actions.tensor[..., 0],
                fill_value=fill_value,
                dtype=torch.get_default_dtype(),
            )

            if len(valid_states) == 0:
                return log_pf_trajectories

            # Build conditions per-step shape to align with valid_states
            masked_cond = None
            if trajectories.conditions is not None:
                # trajectories.conditions shape: (N, cond_dim)
                # Repeat it to (T, N, cond_dim) and then mask it with the state_mask
                T = trajectories.max_length + 1
                masked_cond = trajectories.conditions.repeat(T, 1, 1)
                masked_cond = masked_cond[state_mask]

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
            log_pf_trajectories[action_mask] = valid_log_pf_actions.to(
                log_pf_trajectories.dtype, copy=False
            )

    assert log_pf_trajectories.shape == (
        trajectories.max_length,
        trajectories.n_trajectories,
    )

    return log_pf_trajectories


def get_trajectory_pbs(
    pb: Estimator | None,
    trajectories: Trajectories,
    fill_value: float = 0.0,
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
        fill_value: Value used to pad invalid positions.
        **policy_kwargs: Extra kwargs for ``to_probability_distribution``.

    Returns:
        ``log_pb`` of shape ``(T, N)``.

    Raises:
        ValueError: If backward trajectories are provided.
    """
    if trajectories.is_backward:
        raise ValueError("Backward trajectories are not supported")

    log_pb_trajectories = torch.full_like(
        trajectories.actions.tensor[..., 0],
        fill_value=fill_value,
        dtype=torch.get_default_dtype(),  # Floating point dtype.
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

    # Using all non-initial states, calculate the backward policy, and the logprobs
    # of those actions.
    masked_cond = None
    cond = trajectories.conditions
    if cond is not None:
        T = trajectories.states.tensor.shape[0]
        if cond.shape[0] == T:
            masked_cond = cond[state_mask]
        else:
            masked_cond = cond.unsqueeze(0).expand((T,) + cond.shape)[state_mask]

    # There is no backward policy in this case.
    if pb is None:
        # If pb is None, we assume that the gflownet DAG is a tree, and therefore
        # the backward policy probability is always 1 (log probs are 0).
        valid_log_pb_actions = torch.zeros_like(valid_actions.tensor)
        valid_log_pb_actions = valid_log_pb_actions.squeeze(-1)  # no padding.
        log_pb_trajectories[action_mask] = valid_log_pb_actions.to(
            log_pb_trajectories.dtype, copy=False
        )

        assert log_pb_trajectories.shape == (
            trajectories.max_length,
            trajectories.n_trajectories,
        )

        return log_pb_trajectories

    # There is a backward policy.
    policy_pb = cast(PolicyEstimatorProtocol, pb)
    # Runtime guard: ensure the estimator actually implements the required protocol
    # method and raises an error when a non‑policy estimator is supplied.
    for required in ("init_context", "compute_dist", "log_probs"):
        if not hasattr(policy_pb, required):
            raise TypeError(
                f"Estimator is not policy-capable (missing PolicyMixin method: {required})"
            )
    is_vectorized = bool(getattr(policy_pb, "is_vectorized", True))

    if not is_vectorized:
        # Per-step pb evaluation (state at t+1, action at t)
        N = trajectories.n_trajectories
        device = trajectories.states.device
        cond = trajectories.conditions  # shape (N, cond_dim)
        ctx = policy_pb.init_context(int(N), device, cond)

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

            padded = torch.full((N,), fill_value, device=device, dtype=step_lp.dtype)
            padded[step_mask] = step_lp[step_mask]
            log_pb_trajectories[t] = padded

    # The backward policy supports vectorized evaluation.
    else:
        ctx_v = policy_pb.init_context(
            int(len(valid_states)), trajectories.states.device, conditions=masked_cond  # type: ignore[arg-type]
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
        trajectories.n_trajectories,
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
        cond = transitions.conditions

        # For static typing, cast to the policy protocol before calling mixin methods.
        policy_pf = cast(PolicyEstimatorProtocol, pf)
        # Runtime guard: ensure the estimator actually implements the required protocol
        # method and raises an error when a non‑policy estimator is supplied.
        for required in ("init_context", "compute_dist", "log_probs"):
            if not hasattr(policy_pf, required):
                raise TypeError(
                    f"Estimator is not policy-capable (missing PolicyMixin method: {required})"
                )
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

    # TODO: We support a fill_value for trajectories, but not for transitions.
    # Should we add it here, or remove it for trajectories?
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
    # Runtime guard: ensure the estimator actually implements the required protocol
    # method and raises an error when a non‑policy estimator is supplied.
    for required in ("init_context", "compute_dist", "log_probs"):
        if not hasattr(policy_pb, required):
            raise TypeError(
                f"Estimator is not policy-capable (missing PolicyMixin method: {required})"
            )
    ctx = policy_pb.init_context(
        int(transitions.n_transitions),
        transitions.states.device,
        transitions.conditions,
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
