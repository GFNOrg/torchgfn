import warnings
from typing import Dict, Iterable, Optional, Tuple

import torch
from tqdm import trange

from gfn.containers import ReplayBuffer
from gfn.env import DiscreteEnv, Env
from gfn.gflownet import GFlowNet
from gfn.gflownet.base import PFBasedGFlowNet
from gfn.samplers import Trajectories
from gfn.states import DiscreteStates


def get_terminating_state_dist(
    env: DiscreteEnv,
    states: DiscreteStates,
) -> torch.Tensor:
    """[DEPRECATED] Use `env.get_terminating_state_dist(states)` instead."""
    warnings.warn(
        "gfn.utils.training.get_terminating_state_dist is deprecated; "
        "use DiscreteEnv.get_terminating_state_dist(states) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return env.get_terminating_state_dist(states)


def validate(
    env: DiscreteEnv,
    gflownet: GFlowNet,
    n_validation_samples: int = 1000,
    visited_terminating_states: Optional[DiscreteStates] = None,
) -> Tuple[Dict[str, float], DiscreteStates | None]:
    """[DEPRECATED] Use `env.validate(gflownet, ...)` instead."""
    warnings.warn(
        "gfn.utils.training.validate is deprecated; use DiscreteEnv.validate(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return env.validate(
        gflownet=gflownet,
        n_validation_samples=n_validation_samples,
        visited_terminating_states=visited_terminating_states,
    )


def states_actions_tns_to_traj(
    states_tns: torch.Tensor,
    actions_tns: torch.Tensor,
    env: DiscreteEnv,
    conditions: torch.Tensor | None = None,
) -> Trajectories:
    """Converts raw state and action tensors into a Trajectories object.

    This utility function helps integrate external data (e.g., expert demonstrations)
    into the GFlowNet framework by converting raw tensors into proper Trajectories objects.
    The downstream GFN needs to be capable of recalculating all logprobs (e.g., PFBasedGFlowNets).

    Args:
        states_tns: A tensor of shape `[traj_len, *state_shape]` containing states for a single trajectory.
        actions_tns: A tensor of shape `[traj_len]` containing discrete action indices.
        env: The discrete environment that defines the state/action spaces.
        conditions: An optional tensor of shape `[traj_len, *conditions_shape]`
            containing condition information for a single trajectory.

    Returns:
        A `Trajectories` object containing the converted states and actions.

    Raises:
        ValueError: If tensor shapes are invalid or inconsistent.
    """

    if states_tns.shape[1:] != env.state_shape:
        raise ValueError(
            f"states_tns state dimensions must match env.state_shape {env.state_shape}, "
            f"got shape {states_tns.shape[1:]}"
        )
    if len(actions_tns.shape) != 1:
        raise ValueError(f"actions_tns must be 1D, got batch_shape {actions_tns.shape}")
    if states_tns.shape[0] != actions_tns.shape[0] + 1:
        raise ValueError(
            f"states and actions must have same trajectory length, got "
            f"states: {states_tns.shape[0]}, actions: {actions_tns.shape[0]}"
        )

    states = [env.states_from_tensor(s.unsqueeze(0)) for s in states_tns]
    actions = [env.actions_from_tensor(a.unsqueeze(0).unsqueeze(0)) for a in actions_tns]

    # stack is a class method, so actions[0] is just to access a class instance and is not particularly relevant
    actions = actions[0].stack(actions)
    log_rewards = env.log_reward(states[-2])
    states = states[0].stack(states)
    terminating_idx = torch.tensor([len(states_tns) - 1])
    if conditions is not None:
        states.conditions = conditions.unsqueeze(1)  # dim 1 for batch dimension
    else:
        states.conditions = None

    log_probs = None
    estimator_outputs = None

    trajectory = Trajectories(
        env,
        states,
        actions,
        log_rewards=log_rewards,
        terminating_idx=terminating_idx,
        log_probs=log_probs,
        estimator_outputs=estimator_outputs,
    )
    return trajectory


def warm_up(
    replay_buf: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    gflownet: GFlowNet,
    env: Env,
    n_epochs: int,
    batch_size: int,
    recalculate_all_logprobs: bool = True,
):
    """Performs a warm-up training phase for a GFlowNet agent.

    This utility function provides an example implementation of pre-training for
    GFlowNet agents.

    Args:
        replay_buf: The replay buffer, which collects Trajectories.
        optimizer: Any `torch.optim` optimizer (e.g., Adam, SGD).
        gflownet: The GFlowNet instance to train.
        env: The environment instance.
        n_epochs: The number of epochs for the warm-up phase.
        batch_size: The number of trajectories to sample from the replay buffer per step.
        recalculate_all_logprobs: For `PFBasedGFlowNets` only, forces recalculation of
            all log probabilities. Useful when trajectories do not already have log
            probabilities.

    Returns:
        The trained GFlowNet instance.
    """
    t = trange(n_epochs, desc="Bar desc", leave=True)
    for epoch in t:
        training_trajs = replay_buf.sample(batch_size)
        optimizer.zero_grad()
        if isinstance(gflownet, PFBasedGFlowNet):
            loss = gflownet.loss(
                env,
                training_trajs,
                recalculate_all_logprobs=recalculate_all_logprobs,
            )
        else:
            loss = gflownet.loss(env, training_trajs)

        loss.backward()
        optimizer.step()
        t.set_description(f"{epoch=}, {loss=}")

    optimizer.zero_grad()
    return gflownet


def grad_norm(params: Iterable[torch.nn.Parameter], p: float = 2) -> float:
    """
    Returns the p-norm of all gradients in ``params`` (ignores params with no grad).
    Example: grad_norm(model.parameters())               # total L2 norm
             grad_norm(model.parameters(), p=float('inf'))  # max-grad
    """
    grads = [p_.grad for p_ in params if p_.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm(p) for g in grads]), p).item()


def param_norm(params: Iterable[torch.nn.Parameter], p: float = 2) -> float:
    """
    Total p-norm of a collection of parameters.
    Example:
        model_pnorm = param_norm(model.parameters())        # L2 norm
        max_abs     = param_norm(model.parameters(), p=float('inf'))
    """
    with torch.no_grad():  # no grad tracking needed
        norms = [p_.data.norm(p) for p_ in params]
    return torch.norm(torch.stack(norms), p).item() if norms else 0.0


def lr_grad_ratio(optimizer: torch.optim.Optimizer) -> list[float]:
    """Return (lr·‖g‖₂)/‖θ‖₂ for each param group."""
    out = []
    for group in optimizer.param_groups:
        lr = group["lr"]
        g_norm = grad_norm(group["params"])
        p_norm = param_norm(group["params"])
        out.append((lr * g_norm) / p_norm if p_norm else 0.0)

    return out
