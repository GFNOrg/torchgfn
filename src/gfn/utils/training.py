from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

import torch
from tqdm import trange

from gfn.containers import ReplayBuffer
from gfn.env import DiscreteEnv, Env
from gfn.gflownet import GFlowNet, TBGFlowNet
from gfn.gflownet.base import PFBasedGFlowNet
from gfn.samplers import Trajectories
from gfn.states import DiscreteStates
from gfn.utils.common import parse_dtype


def get_terminating_state_dist_pmf(
    env: DiscreteEnv,
    states: DiscreteStates,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes the empirical distribution of the terminating states.

    Args:
        env: The environment.
        states: The states to compute the distribution of.
        dtype: The dtype of the PMF.

    Returns:
        The empirical distribution of the terminating states as a tensor of shape
        (n_terminating_states,).
    """
    states_indices = env.get_terminating_states_indices(states).cpu().numpy().tolist()
    counter = Counter(str(idx) for idx in states_indices)
    counter_list = [
        counter[str(state_idx)] if str(state_idx) in counter else 0
        for state_idx in range(env.n_terminating_states)
    ]

    return torch.tensor(counter_list, dtype=parse_dtype(dtype)) / len(states_indices)


def validate(
    env: DiscreteEnv,
    gflownet: GFlowNet,
    n_validation_samples: int = 1000,
    visited_terminating_states: Optional[DiscreteStates] = None,
) -> Tuple[Dict[str, float], DiscreteStates | None]:
    """Evaluates the current GFlowNet on the given environment.

    This function is designed for environments with known target reward distributions.
    Validation is performed by computing the L1 distance between the learned empirical
    distribution of terminating states and the true target distribution.

    Args:
        env: The environment to evaluate the GFlowNet on.
        gflownet: The GFlowNet instance to evaluate.
        n_validation_samples: The number of samples to use for evaluating the PMF.
        visited_terminating_states: Optional. If provided, the PMF is obtained from
            the last `n_validation_samples` states from this collection. Otherwise,
            `n_validation_samples` are resampled directly from the GFlowNet for evaluation.

    Returns:
        A tuple containing:
        - dict: A dictionary containing validation metrics. If the GFlowNet is a
            `TBGFlowNet` (i.e., it contains `LogZ`), the absolute difference between
            the learned and target `LogZ` is also included.
        - DiscreteStates or None: The sampled terminating states, or None if not applicable.
    """

    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf
    if isinstance(true_dist_pmf, torch.Tensor):
        true_dist_pmf = true_dist_pmf.cpu()
    else:
        # The environment does not implement a true_dist_pmf property, nor a log_partition property
        # We cannot validate the gflownet
        return {}, visited_terminating_states

    logZ = None
    if isinstance(gflownet, TBGFlowNet):
        assert isinstance(gflownet.logZ, torch.Tensor)
        logZ = gflownet.logZ.item()
    if visited_terminating_states is None:
        sampled_terminating_states = gflownet.sample_terminating_states(
            env, n_validation_samples
        )
        assert isinstance(sampled_terminating_states, DiscreteStates)
    else:
        # Only keep the most recent n_validation_samples states.
        sampled_terminating_states = visited_terminating_states[-n_validation_samples:]

    final_states_dist_pmf = get_terminating_state_dist_pmf(
        env, sampled_terminating_states
    )
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    validation_info = {"l1_dist": l1_dist}
    if logZ is not None:
        validation_info["logZ_diff"] = abs(logZ - true_logZ)

    return (validation_info, sampled_terminating_states)


def states_actions_tns_to_traj(
    states_tns: torch.Tensor,
    actions_tns: torch.Tensor,
    env: DiscreteEnv,
    conditioning: torch.Tensor | None = None,
) -> Trajectories:
    """Converts raw state and action tensors into a Trajectories object.

    This utility function helps integrate external data (e.g., expert demonstrations)
    into the GFlowNet framework by converting raw tensors into proper Trajectories objects.
    The downstream GFN needs to be capable of recalculating all logprobs (e.g., PFBasedGFlowNets).

    Args:
        states_tns: A tensor of shape `[traj_len, *state_shape]` containing states for a single trajectory.
        actions_tns: A tensor of shape `[traj_len]` containing discrete action indices.
        env: The discrete environment that defines the state/action spaces.
        conditioning: An optional tensor of shape `[traj_len, *conditioning_shape]`
            containing conditioning information for a single trajectory.

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

    log_probs = None
    estimator_outputs = None

    trajectory = Trajectories(
        env,
        states,
        conditioning,
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
