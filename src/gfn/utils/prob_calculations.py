from typing import Tuple

import torch

from gfn.containers import Trajectories, Transitions
from gfn.estimators import Estimator
from gfn.states import States
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)


def check_cond_forward(
    module: Estimator,
    module_name: str,
    states: States,
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Checks if conditioning is passed and calls the module's forward method accordingly.

    Args:
        module: The GFN module to call.
        module_name: The name of the module (for error messages).
        states: The states to pass to the module.
        condition: Optional conditioning tensor to pass to the module.

    Returns:
        The output of the module's forward method.

    Raises:
        TypeError: If conditioning is passed but the module does not accept it, or vice-versa.
    """
    if condition is not None:
        with has_conditioning_exception_handler(module_name, module):
            return module(states, condition)
    else:
        with no_conditioning_exception_handler(module_name, module):
            return module(states)


# ------------
# Trajectories
# ------------


def get_trajectory_pfs_and_pbs(
    pf: Estimator,
    pb: Estimator,
    trajectories: Trajectories,
    fill_value: float = 0.0,
    recalculate_all_logprobs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the log probabilities of forward and backward trajectories.

    Args:
        pf: The forward policy GFN module.
        pb: The backward policy GFN module.
        trajectories: The trajectories to calculate probabilities for.
        fill_value: The value to fill for invalid states (e.g., sink states).
        recalculate_all_logprobs: Whether to recalculate log probabilities even if they
            already exist in the trajectories object.

    Returns:
        A tuple containing two tensors: log_pf_trajectories and log_pb_trajectories.
    """
    # fill value is the value used for invalid states (sink state usually)

    # uncomment next line for debugging
    # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions.is_dummy)

    log_pf_trajectories = get_trajectory_pfs(
        pf,
        trajectories,
        fill_value=fill_value,
        recalculate_all_logprobs=recalculate_all_logprobs,
    )
    log_pb_trajectories = get_trajectory_pbs(pb, trajectories, fill_value=fill_value)

    return log_pf_trajectories, log_pb_trajectories


def get_trajectory_pfs(
    pf: Estimator,
    trajectories: Trajectories,
    fill_value: float = 0.0,
    recalculate_all_logprobs: bool = True,
) -> torch.Tensor:
    """Calculates the log probabilities of forward trajectories.

    Args:
        pf: The forward policy GFN module.
        trajectories: The trajectories to calculate probabilities for.
        fill_value: The value to fill for invalid states (e.g., sink states).
        recalculate_all_logprobs: Whether to recalculate log probabilities even if they
            already exist in the trajectories object.

    Returns:
        A tensor containing the log probabilities of the forward trajectories.

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
        log_pf_trajectories = torch.full_like(
            trajectories.actions.tensor[..., 0],
            fill_value=fill_value,
            dtype=torch.float,
        )

        if len(valid_states) == 0:
            return log_pf_trajectories

        if trajectories.estimator_outputs is not None and not recalculate_all_logprobs:
            estimator_outputs = trajectories.estimator_outputs[action_mask]
        else:
            masked_cond = None
            if trajectories.conditioning is not None:
                cond_dim = (-1,) * len(trajectories.conditioning.shape)
                traj_len = trajectories.states.tensor.shape[0]
                masked_cond = trajectories.conditioning.unsqueeze(0).expand(
                    (traj_len,) + cond_dim
                )[state_mask]

            estimator_outputs = check_cond_forward(pf, "pf", valid_states, masked_cond)

        # Calculates the log PF of the actions sampled off policy.
        valid_log_pf_actions = pf.to_probability_distribution(
            valid_states, estimator_outputs
        ).log_prob(
            valid_actions.tensor
        )  # Using the actions sampled off-policy.

        log_pf_trajectories[action_mask] = valid_log_pf_actions

    assert log_pf_trajectories.shape == (
        trajectories.max_length,
        trajectories.n_trajectories,
    )

    return log_pf_trajectories


def get_trajectory_pbs(
    pb: Estimator, trajectories: Trajectories, fill_value: float = 0.0
) -> torch.Tensor:
    """Calculates the log probabilities of backward trajectories.

    Args:
        pb: The backward policy GFN module.
        trajectories: The trajectories to calculate probabilities for.
        fill_value: The value to fill for invalid states (e.g., sink states).

    Returns:
        A tensor containing the log probabilities of the backward trajectories.

    Raises:
        ValueError: If backward trajectories are provided.
    """
    if trajectories.is_backward:
        raise ValueError("Backward trajectories are not supported")

    log_pb_trajectories = torch.full_like(
        trajectories.actions.tensor[..., 0],
        fill_value=fill_value,
        dtype=torch.float,
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
        cond_dim = (-1,) * len(trajectories.conditioning.shape)
        traj_len = trajectories.states.tensor.shape[0]
        masked_cond = trajectories.conditioning.unsqueeze(0).expand(
            (traj_len,) + cond_dim
        )[state_mask]

    estimator_outputs = check_cond_forward(pb, "pb", valid_states, masked_cond)

    valid_log_pb_actions = pb.to_probability_distribution(
        valid_states, estimator_outputs
    ).log_prob(valid_actions.tensor)

    log_pb_trajectories[action_mask] = valid_log_pb_actions

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
    pb: Estimator,
    transitions: Transitions,
    recalculate_all_logprobs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the log probabilities of forward and backward transitions.

    Args:
        pf: The forward policy GFN module.
        pb: The backward policy GFN module.
        transitions: The transitions to calculate probabilities for.
        recalculate_all_logprobs: Whether to recalculate log probabilities even if they
            already exist in the transitions object.

    Returns:
        A tuple containing two tensors: log_pf_transitions and log_pb_transitions.

    Raises:
        ValueError: If backward transitions are provided.
    """
    if transitions.is_backward:
        raise ValueError("Backward transitions are not supported")

    log_pf_transitions = get_transition_pfs(pf, transitions, recalculate_all_logprobs)
    log_pb_transitions = get_transition_pbs(pb, transitions)

    assert log_pf_transitions.shape == (transitions.n_transitions,)
    assert log_pb_transitions.shape == (transitions.n_transitions,)

    return log_pf_transitions, log_pb_transitions


def get_transition_pfs(
    pf: Estimator, transitions: Transitions, recalculate_all_logprobs: bool = True
) -> torch.Tensor:
    """Calculates the log probabilities of forward transitions.

    Args:
        pf: The forward policy GFN module.
        transitions: The transitions to calculate probabilities for.
        recalculate_all_logprobs: Whether to recalculate log probabilities even if they
            already exist in the transitions object.

    Returns:
        A tensor containing the log probabilities of the forward transitions.
    """
    states = transitions.states
    actions = transitions.actions

    if transitions.has_log_probs and not recalculate_all_logprobs:
        log_pf_actions = transitions.log_probs
        assert log_pf_actions is not None
    else:
        # Evaluate the log PF of the actions, with optional conditioning.
        # TODO: Inefficient duplication in case of tempered policy
        # The Transitions container should then have some
        # estimator_outputs attribute as well, to avoid duplication here ?
        # See (#156).
        estimator_outputs = check_cond_forward(
            pf, "pf", states, transitions.conditioning
        )

        log_pf_actions = pf.to_probability_distribution(
            states, estimator_outputs
        ).log_prob(actions.tensor)

    return log_pf_actions


def get_transition_pbs(pb: Estimator, transitions: Transitions) -> torch.Tensor:
    # automatically removes invalid transitions (i.e. s_f -> s_f)
    valid_next_states = transitions.next_states[~transitions.is_terminating]
    non_exit_actions = transitions.actions[~transitions.actions.is_exit]

    # Evaluate the log PB of the actions, with optional conditioning.
    masked_cond = (
        transitions.conditioning[~transitions.is_terminating]
        if transitions.conditioning is not None
        else None
    )
    estimator_outputs = check_cond_forward(pb, "pb", valid_next_states, masked_cond)

    # Evaluate the log PB of the actions.
    log_pb_actions = torch.zeros(
        (transitions.n_transitions,),
        dtype=torch.float,
        device=transitions.states.device,
    )

    if len(valid_next_states) != 0:
        valid_log_pb_actions = pb.to_probability_distribution(
            valid_next_states, estimator_outputs
        ).log_prob(non_exit_actions.tensor)

        log_pb_actions[~transitions.is_terminating] = valid_log_pb_actions

    return log_pb_actions
