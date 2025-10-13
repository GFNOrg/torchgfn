import warnings
from typing import Any, Tuple

import torch

from gfn.containers import Trajectories, Transitions
from gfn.estimators import Estimator
from gfn.utils.handlers import check_cond_forward

# ------------
# Trajectories
# ------------


def get_trajectory_pfs_and_pbs(
    pf: Estimator,
    pb: Estimator | None,
    trajectories: Trajectories,
    fill_value: float = 0.0,
    recalculate_all_logprobs: bool = True,
    pf_adapter: Any | None = None,
    pb_adapter: Any | None = None,
    **policy_kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate PF and PB log-probabilities for trajectories.

    This function delegates to :func:`get_trajectory_pfs` and
    :func:`get_trajectory_pbs`, forwarding optional adapter(s) and policy kwargs.

    Vectorized vs non-vectorized
    - If the adapter is None or ``adapter.is_vectorized is True``, the legacy
      vectorized path is used (fast path, strict parity with legacy code).
    - If ``adapter.is_vectorized is False`` (e.g., recurrent), a non‑vectorized
      per‑step path is used with legacy-accurate masks and alignment.

    Args:
        pf: Forward policy estimator.
        pb: Backward policy estimator, or ``None`` if the DAG is a tree (PB=1).
        trajectories: Trajectories container to evaluate.
        fill_value: Fill used for invalid states (e.g., sink state positions).
        recalculate_all_logprobs: If ``True``, recompute PF even if cached.
        pf_adapter: Adapter for PF (vectorized vs non‑vectorized decision).
        pb_adapter: OAdapter for PB (vectorized vs non‑vectorized decision).
        **policy_kwargs: Extra kwargs passed to estimator's
            ``to_probability_distribution`` (e.g., temperature, epsilon, sf_bias).

    Returns:
        Tuple[Tensor, Tensor]:
            - PF log-probs with shape ``(T, N)``
            - PB log-probs with shape ``(T, N)``
    """
    # fill value is the value used for invalid states (sink state usually)

    # uncomment next line for debugging
    # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions.is_dummy)

    if pb_adapter is not None and not isinstance(pb_adapter, type(pf_adapter)):
        warnings.warn(
            (
                "type(pb_adapter)={} and type(pf_adapter)={}, this is probably not what you want "
                "unless you explicitly want to use different sampling logic for the two policies "
                "(with different estimator architectures). This is very uncommon."
            ).format(type(pb_adapter), type(pf_adapter))
        )

    log_pf_trajectories = get_trajectory_pfs(
        pf,
        trajectories,
        fill_value=fill_value,
        recalculate_all_logprobs=recalculate_all_logprobs,
        adapter=pf_adapter,
        **policy_kwargs,
    )
    log_pb_trajectories = get_trajectory_pbs(
        pb,
        trajectories,
        fill_value=fill_value,
        adapter=pb_adapter,
        **policy_kwargs,
    )

    return log_pf_trajectories, log_pb_trajectories


def get_trajectory_pfs(
    pf: Estimator,
    trajectories: Trajectories,
    fill_value: float = 0.0,
    recalculate_all_logprobs: bool = True,
    adapter: Any | None = None,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Calculate PF log-probabilities for trajectories.

    Vectorized vs non-vectorized
    - Vectorized when ``adapter is None`` or ``adapter.is_vectorized is True``:
      uses the legacy vectorized implementation (strict parity with reference).
    - Non‑vectorized when ``adapter.is_vectorized is False``: evaluates per‑step
      using legacy masks (PF: ``~states.is_sink_state[t] & ~actions.is_dummy[t]``),
      passing the active subset to the adapter without any action‑id mask indexing.

    Args:
        pf: Forward policy estimator.
        trajectories: Trajectories container to evaluate.
        fill_value: Fill used for invalid states (e.g., sink state positions).
        recalculate_all_logprobs: If ``True``, recompute PF even if cached.
        adapter: Optional adapter controlling vectorized vs non‑vectorized path.
        **policy_kwargs: Extra kwargs passed to
            ``to_probability_distribution`` (e.g., temperature, epsilon).

    Returns:
        Tensor of shape ``(T, N)`` containing PF log-probabilities.

    Raises:
        ValueError: If backward trajectories are provided.
    """
    if trajectories.is_backward:
        raise ValueError("Backward trajectories are not supported")

    state_mask = ~trajectories.states.is_sink_state
    action_mask = ~trajectories.actions.is_dummy

    valid_states = trajectories.states[state_mask]
    valid_actions = trajectories.actions[action_mask]

    if valid_states.batch_shape != valid_actions.batch_shape:
        raise AssertionError("Something wrong happening with log_pf evaluations")

    if trajectories.has_log_probs and not recalculate_all_logprobs:
        log_pf_trajectories = trajectories.log_probs
        assert log_pf_trajectories is not None
    else:
        # Decide vectorized (legacy) vs non-vectorized (adapter per-step)
        vectorized = adapter is None or getattr(adapter, "is_vectorized", True)

        if not vectorized:
            # Adapter-driven path
            N = trajectories.n_trajectories
            device = trajectories.states.device
            cond = trajectories.conditioning
            if cond is not None and len(cond.shape) >= 2:
                cond = cond[0]
            ctx = adapter.init_context(int(N), device, cond)  # type: ignore[arg-type]

            T = trajectories.max_length
            log_pf_trajectories = torch.full(
                (T, N),
                fill_value=fill_value,
                dtype=torch.get_default_dtype(),
                device=device,
            )

            for t in range(T):
                state_ok = ~trajectories.states.is_sink_state[t]
                action_ok = ~trajectories.actions.is_dummy[t]
                step_mask = state_ok & action_ok

                if not torch.any(step_mask):
                    continue

                step_states = trajectories.states[t][step_mask]
                step_actions = trajectories.actions.tensor[t][step_mask]

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

                # Calculate the log-probabilities of the actions.
                step_lp, ctx = adapter.log_prob_of_actions(  # type: ignore[union-attr]
                    step_states, step_actions, ctx, step_mask, **policy_kwargs
                )
                if fill_value != 0.0:
                    padded = torch.full(
                        (N,), fill_value, device=device, dtype=step_lp.dtype
                    )
                    padded[step_mask] = step_lp[step_mask]
                    step_lp = padded
                log_pf_trajectories[t] = step_lp
        else:
            # Vectorized path.
            log_pf_trajectories = torch.full_like(
                trajectories.actions.tensor[..., 0],
                fill_value=fill_value,
                dtype=torch.get_default_dtype(),
            )

            if len(valid_states) == 0:
                return log_pf_trajectories

            if (
                trajectories.estimator_outputs is not None
                and not recalculate_all_logprobs
            ):
                # Reuse cached outputs to build the distribution
                est_out = trajectories.estimator_outputs[action_mask]
                dist = pf.to_probability_distribution(
                    valid_states, est_out, **policy_kwargs
                )
                valid_log_pf_actions = dist.log_prob(valid_actions.tensor)
            else:
                # Build conditioning per-step shape to align with valid_states
                masked_cond = None
                if trajectories.conditioning is not None:
                    cond_dim = (-1,) * len(trajectories.conditioning.shape)
                    traj_len = trajectories.states.tensor.shape[0]
                    masked_cond = trajectories.conditioning.unsqueeze(0).expand(
                        (traj_len,) + cond_dim
                    )[state_mask]
                est_out = check_cond_forward(pf, "pf", valid_states, masked_cond)
                valid_log_pf_actions = pf.to_probability_distribution(
                    valid_states, est_out, **policy_kwargs
                ).log_prob(valid_actions.tensor)

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
    adapter: Any | None = None,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Calculate PB log-probabilities for trajectories.

    Vectorized vs non-vectorized
    - Vectorized when ``adapter is None`` or ``adapter.is_vectorized is True``:
      uses the legacy vectorized implementation (strict parity with reference).
    - Non‑vectorized when ``adapter.is_vectorized is False``: evaluates per‑step
      using legacy masks/alignment:
      PB aligns actions at time ``t`` with states at time ``t+1`` and uses
      ``~states.is_sink_state[t+1] & ~states.is_initial_state[t+1]
        & ~actions.is_dummy[t] & ~actions.is_exit[t]``, skipping ``t==0``.

    Args:
        pb: Backward policy estimator, or ``None`` for tree DAGs (PB=1).
        trajectories: Trajectories container to evaluate.
        fill_value: Fill used for invalid states (e.g., sink state positions).
        adapter: Optional adapter controlling vectorized vs non‑vectorized path.
        **policy_kwargs: Extra kwargs passed to
            ``to_probability_distribution``.

    Returns:
        Tensor of shape ``(T, N)`` containing PB log-probabilities.

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
    if trajectories.conditioning is not None:
        # We need to index the conditioning vector to broadcast over the states.
        # The conditioning tensor has shape (max_length, n_trajectories, 1)
        # We need to index it with the state_mask to get the valid states
        masked_cond = trajectories.conditioning[state_mask]

    # Recurrent adapters are only valid for trajectories and never require pb
    from gfn.samplers import RecurrentEstimatorAdapter  # type: ignore

    if adapter is not None:
        is_recurrent = isinstance(adapter, RecurrentEstimatorAdapter)
    else:
        is_recurrent = False

    if is_recurrent or pb is None:
        # With recurrent adapter, pb *must* be None (tree DAG); return zeros.
        assert pb is None, "When using a RecurrentEstimatorAdapter, pb must be None."
        # If pb is None, we assume that the gflownet DAG is a tree, and therefore
        # the backward policy probability is always 1 (log probs are 0).
        valid_log_pb_actions = torch.zeros_like(valid_actions.tensor)
        valid_log_pb_actions = valid_log_pb_actions.squeeze(-1)  # no padding.

        # TODO: Add logging in follow up PR.
        # if os.getenv("GFN_DEBUG_REC_PB") == "1":
        #     print(
        #         "[DBG] pb=None path: valid_actions.shape=",
        #         tuple(valid_actions.tensor.shape),
        #         "valid_log_pb_actions.shape=",
        #         tuple(valid_log_pb_actions.shape),
        #         "target_len=",
        #         int(action_mask.sum().item()),
        #     )

    elif pb is not None:
        # Choose vectorized (legacy) vs non-vectorized (adapter per-step)
        # Vectorized path is used by default via DefaultEstimatorAdapter.
        vectorized = adapter is None or getattr(adapter, "is_vectorized", True)
        if adapter is None:
            from gfn.samplers import DefaultEstimatorAdapter  # Avoids circular import.

            adapter = DefaultEstimatorAdapter(pb)

        if not vectorized:
            # Adapter-driven pb evaluation (non-recurrent)
            N = trajectories.n_trajectories
            device = trajectories.states.device
            cond = trajectories.conditioning

            if cond is not None and len(cond.shape) >= 2:
                cond = cond[0]
            ctx = adapter.init_context(int(N), device, cond)  # type: ignore[arg-type]

            T = trajectories.max_length
            # Iterate per-step with legacy-complete masking (state at t+1, action at t)
            for t in range(T):
                state_ok = (~trajectories.states.is_sink_state[t + 1]) & (
                    ~trajectories.states.is_initial_state[t + 1]
                )
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
                step_lp, ctx = adapter.log_prob_of_actions(
                    step_states, step_actions, ctx, step_mask, **policy_kwargs
                )

                padded = torch.full((N,), fill_value, device=device, dtype=step_lp.dtype)
                padded[step_mask] = step_lp[step_mask]
                log_pb_trajectories[t] = padded

            return log_pb_trajectories

        else:
            # Legacy vectorized path
            estimator_outputs = check_cond_forward(pb, "pb", valid_states, masked_cond)
            valid_log_pb_actions = pb.to_probability_distribution(
                valid_states, estimator_outputs
            ).log_prob(valid_actions.tensor)
            log_pb_trajectories[action_mask] = valid_log_pb_actions.to(
                log_pb_trajectories.dtype, copy=False
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
    pf_adapter: Any | None = None,
    pb_adapter: Any | None = None,
    **policy_kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate PF and PB log-probabilities for transitions.

    Vectorized vs non-vectorized mirrors trajectories:
    - Vectorized (adapter is None or ``is_vectorized=True``): legacy vectorized path.
    - Non‑vectorized (``is_vectorized=False``): per‑batch adapter call with legacy
      masks; no action‑id mask indexing.

    Args:
        pf: Forward policy estimator.
        pb: Backward policy estimator, or ``None`` for tree DAGs (PB=1).
        transitions: Transitions container to evaluate.
        recalculate_all_logprobs: If ``True``, recompute PF even if cached.
        pf_adapter: Optional adapter for PF.
        pb_adapter: Optional adapter for PB.
        **policy_kwargs: Extra kwargs passed to
            ``to_probability_distribution``.

    Returns:
        Tuple[Tensor, Tensor]: PF and PB log-probabilities of shape ``(M,)``.

    Raises:
        ValueError: If backward transitions are provided.
    """
    if transitions.is_backward:
        raise ValueError("Backward transitions are not supported")

    if pb_adapter is not None and not isinstance(pb_adapter, type(pf_adapter)):
        warnings.warn(
            (
                "type(pb_adapter)={} and type(pf_adapter)={}, this is probably not what you want "
                "unless you explicitly want to use different sampling logic for the two policies "
                "(with different estimator architectures). This is very uncommon."
            ).format(type(pb_adapter), type(pf_adapter))
        )

    log_pf_transitions = get_transition_pfs(
        pf, transitions, recalculate_all_logprobs, adapter=pf_adapter, **policy_kwargs
    )
    log_pb_transitions = get_transition_pbs(
        pb, transitions, adapter=pb_adapter, **policy_kwargs
    )

    assert log_pf_transitions.shape == (transitions.n_transitions,)
    assert log_pb_transitions.shape == (transitions.n_transitions,)

    return log_pf_transitions, log_pb_transitions


def get_transition_pfs(
    pf: Estimator,
    transitions: Transitions,
    recalculate_all_logprobs: bool = True,
    adapter: Any | None = None,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Calculate PF log-probabilities for transitions.

    Vectorized vs non-vectorized
    - Vectorized when ``adapter is None`` or ``adapter.is_vectorized is True``:
      legacy vectorized path.
    - Non‑vectorized when ``adapter.is_vectorized is False``: single adapter call
      with legacy masks and no action‑id indexing.

    Args:
        pf: Forward policy estimator.
        transitions: Transitions container to evaluate.
        recalculate_all_logprobs: If ``True``, recompute PF even if cached.
        adapter: Optional adapter controlling vectorized vs non‑vectorized path.
        **policy_kwargs: Extra kwargs passed to
            ``to_probability_distribution``.

    Returns:
        Tensor of shape ``(M,)`` containing PF log-probabilities.
    """
    states = transitions.states
    actions = transitions.actions

    if transitions.has_log_probs and not recalculate_all_logprobs:
        log_pf_actions = transitions.log_probs
        assert log_pf_actions is not None
    else:
        if adapter is not None or True:
            from gfn.samplers import RecurrentEstimatorAdapter  # type: ignore

            if adapter is None:
                from gfn.samplers import DefaultEstimatorAdapter  # type: ignore

                adapter = DefaultEstimatorAdapter(pf)
            elif isinstance(adapter, RecurrentEstimatorAdapter):
                raise TypeError(
                    "RecurrentEstimatorAdapter is only supported for Trajectories"
                )
            assert adapter is not None

            N = transitions.n_transitions
            device = transitions.states.device
            cond = transitions.conditioning
            ctx = adapter.init_context(int(N), device, cond)
            mask = torch.ones(N, dtype=torch.bool, device=device)

            # Evaluate the log PF of the actions, with optional conditioning.
            # TODO: Inefficient duplication in case of tempered policy
            # The Transitions container should then have some
            # estimator_outputs attribute as well, to avoid duplication here ?
            # See (#156).
            step_lp, _ = adapter.log_prob_of_actions(
                states[mask],
                actions.tensor[mask],
                ctx,
                mask,
                **policy_kwargs,
            )
            log_pf_actions = step_lp

    return log_pf_actions


def get_transition_pbs(
    pb: Estimator | None,
    transitions: Transitions,
    adapter: Any | None = None,
    **policy_kwargs: Any,
) -> torch.Tensor:
    """Calculate PB log-probabilities for transitions.

    Vectorized vs non-vectorized
    - Vectorized when ``adapter is None`` or ``adapter.is_vectorized is True``:
      legacy vectorized path.
    - Non‑vectorized when ``adapter.is_vectorized is False``: single adapter call
      with legacy masks and no action‑id indexing.

    Args:
        pb: Backward policy estimator, or ``None`` for tree DAGs (PB=1).
        transitions: Transitions container to evaluate.
        adapter: Optional adapter controlling vectorized vs non‑vectorized path.
        **policy_kwargs: Extra kwargs passed to
            ``to_probability_distribution``.

    Returns:
        Tensor of shape ``(M,)`` containing PB log-probabilities.
    """
    # # automatically removes invalid transitions (i.e. s_f -> s_f)
    # valid_next_states = transitions.next_states[~transitions.is_terminating]
    # non_exit_actions = transitions.actions[~transitions.actions.is_exit]

    # # Evaluate the log PB of the actions, with optional conditioning.
    # masked_cond = (
    #     transitions.conditioning[~transitions.is_terminating]
    #     if transitions.conditioning is not None
    #     else None
    # )

    # TODO: We support a fill_value for trajectories, but not for transitions.
    # Should we add it here, or remove it for trajectories?
    log_pb_actions = torch.zeros(
        (transitions.n_transitions,), device=transitions.states.device
    )

    if adapter is not None or True:
        from gfn.samplers import RecurrentEstimatorAdapter  # type: ignore

        if adapter is None and pb is not None:
            from gfn.samplers import DefaultEstimatorAdapter  # type: ignore

            adapter = DefaultEstimatorAdapter(pb)
        elif isinstance(adapter, RecurrentEstimatorAdapter):
            raise TypeError(
                "RecurrentEstimatorAdapter is only supported for Trajectories"
            )
        assert adapter is not None

        # If pb is None, we assume that the gflownet DAG is a tree, and therefore
        # the backward policy probability is always 1 (log probs are 0).
        if pb is None:
            return log_pb_actions

        N = transitions.n_transitions
        device = transitions.states.device
        cond = transitions.conditioning
        ctx = adapter.init_context(int(N), device, cond)
        # Legacy-complete masking for PB on transitions:
        # require non-terminating next_states and non-exit actions simultaneously
        # automatically removes invalid transitions (i.e. s_f -> s_f)
        state_ok = ~transitions.is_terminating
        action_ok = ~transitions.actions.is_exit
        mask = state_ok & action_ok

        if not torch.any(mask):
            return log_pb_actions

        step_lp, _ = adapter.log_prob_of_actions(
            transitions.next_states[mask],
            transitions.actions.tensor[mask],
            ctx,
            mask,
            **policy_kwargs,
        )
        log_pb_actions[mask] = step_lp[mask]

    return log_pb_actions
