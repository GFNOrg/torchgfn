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


def get_terminating_state_dist_pmf(
    env: DiscreteEnv, states: DiscreteStates
) -> torch.Tensor:
    """Computes the empirical distribution of the terminating states.

    Args:
        env: The environment.
        states: The states to compute the distribution of.

    Returns the empirical distribution of the terminating states as a tensor of shape (n_terminating_states,).
    """
    states_indices = env.get_terminating_states_indices(states).cpu().numpy().tolist()
    counter = Counter(str(idx) for idx in states_indices)
    counter_list = [
        counter[str(state_idx)] if str(state_idx) in counter else 0
        for state_idx in range(env.n_terminating_states)
    ]

    return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)


def validate(
    env: DiscreteEnv,
    gflownet: GFlowNet,
    n_validation_samples: int = 1000,
    visited_terminating_states: Optional[DiscreteStates] = None,
) -> Tuple[Dict[str, float], DiscreteStates | None]:
    """Evaluates the current gflownet on the given environment.

    This is for environments with known target reward. The validation is done by
    computing the l1 distance between the learned empirical and the target
    distributions.

    Args:
        env: The environment to evaluate the gflownet on.
        gflownet: The gflownet to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.
        visited_terminating_states: The terminating states visited during training.
            If given, the pmf is obtained from the last n_validation_samples states.
            Otherwise, n_validation_samples are resampled directly from the gflownet for
            evaluation.

    Returns: A dictionary containing the l1 validation metric. If the gflownet
        is a TBGFlowNet, i.e. contains LogZ, then the (absolute) difference
        between the learned and the target LogZ is also returned in the dictionary, and
        the sampled terminating states are returned.
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
    """
    This utility function helps integrate external data (e.g. expert demonstrations)
    into the GFlowNet framework by converting raw tensors into proper Trajectories objects.
    The downstream GFN needs to be capable of recalculating all logprobs (e.g. PFBasedGFlowNets)

    Args:
        states_tns: Tensor of shape [traj_len, *state_shape] containing states for a single trajectory
        actions_tns: Tensor of shape [traj_len] containing discrete action indices
        env: The discrete environment that defines the state/action spaces
        conditioning: Tensor of shape [traj_len, *conditioning_shape] containing states for a single trajectory

    Returns:
        Trajectories: A Trajectories object containing the converted states and actions

    Raises:
        ValueError: If tensor shapes are invalid or inconsistent
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
    """
    This utility function is an example implementation of pre-training for GFlowNets agent.

    Args:
        replay_buf: Replay Buffer, which collects Trajectories
        optimizer: Any torch.optim optimizer (e.g. Adam, SGD)
        gflownet: The GFlowNet to train
        env: The environment instance
        n_epochs: Number of epochs for warmup
        batch_size: Number of trajectories to sample from replay buffer
        recalculate_all_logprobs: For PFBasedGFlowNets only, force recalculating all log probs.
            Useful trajectories do not already have log probs.
    Returns:
        GFlowNet: A trained GFlowNet
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
