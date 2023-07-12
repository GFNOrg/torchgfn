import torch
from torch.distributions import Distribution, Beta
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical, Beta, MixtureSameFamily
from gfn.utils import NeuralNet as NeuralNet2


class Box2:
    """D-dimensional box with lower bound 0 and upper bound 1. A maximum step size 0<delta<1 defines
    the maximum unidimensional step size in each dimension.
    """

    def __init__(
        self,
        dim=2,
        delta=0.1,
        epsilon=1e-4,
        R0=0.1,
        R1=0.5,
        R2=2.0,
        reward_debug=False,
        device_str="cpu",
        verify_actions=False,
    ):
        # Set verify_actions to False to disable action verification for faster step execution.
        self.dim = dim
        self.delta = delta
        self.epsilon = epsilon
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.terminal_action = torch.full((dim,), -float("inf"), device=self.device)
        self.sink_state = torch.full((dim,), -float("inf"), device=self.device)
        self.verify_actions = verify_actions

        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.reward_debug = reward_debug

    def is_actions_valid(self, states, actions):
        """Check if actions are valid: First, verify that no state component is within epsilon distance from the bounds,
        then for each state [x_1, ..., x_d], the action [a_1, ..., a_d] needs to satisfy
        0 <= a_i < min(self.delta_max, 1 - x_i) for all i. Assume all actions are non terminal. Basically, this means
        that if one coordinate is >= 1 - self.epsilon, then the corresponding action should be "exit".
        """
        first_condition = torch.all(
            torch.logical_and(
                states >= 0,
                states <= 1 - self.epsilon,
            )
        )

        second_condition = torch.all(
            torch.logical_and(
                actions >= 0,
                actions
                <= torch.min(
                    torch.full((self.dim,), self.delta, device=self.device),
                    1 - states,
                ),
            )
        )
        out = first_condition and second_condition
        return out

    def is_terminal_action_mask(self, actions):
        """Return a mask of terminal actions."""
        return torch.all(actions == self.terminal_action, dim=-1)

    def step(self, states, actions):
        """Take a step in the environment. The states can include the sink state [-inf, ..., -inf].
        In which case, the corresponding actions are ignored."""
        # First, select the states that are not the sink state.
        non_sink_mask = ~torch.all(states == self.sink_state, dim=-1)
        non_sink_states = states[non_sink_mask]
        non_sink_actions = actions[non_sink_mask]
        # Then, select states and actions not corresponding to terminal actions, for the non sink states and actions.
        non_terminal_mask = ~self.is_terminal_action_mask(non_sink_actions)
        non_terminal_states = non_sink_states[non_terminal_mask]
        non_terminal_actions = non_sink_actions[non_terminal_mask]
        # Then, if verify_actions is True, check if actions are valid.
        if self.verify_actions:
            assert self.is_actions_valid(non_terminal_states, non_terminal_actions)
        # Then, take a step and store that in a new tensor.
        new_states = torch.full_like(states, -float("inf"))
        non_sink_new_states = new_states[non_sink_mask]
        non_sink_new_states[non_terminal_mask] = (
            non_terminal_states + non_terminal_actions
        )
        new_states[non_sink_mask] = non_sink_new_states
        # Finally, return the new states.
        return new_states

    def reward(self, final_states):
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states - 0.5)
        if not self.reward_debug:
            reward = (
                R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
            )
        elif self.reward_debug:
            reward = torch.ones(final_states.shape[0], device=self.device)
            reward[final_states.norm(dim=-1) > self.delta] = 1e-8
        else:
            raise NotImplementedError

        return reward

    @property
    def Z(self):
        if not self.reward_debug:
            return (
                self.R0
                + (2 * 0.25) ** self.dim * self.R1
                + (2 * 0.1) ** self.dim * self.R2
            )
        else:
            if self.dim != 2:
                raise NotImplementedError("Only implemented for dim=2")
            return torch.pi * self.delta**2 / 4.0


class CirclePF2(NeuralNet2):
    def __init__(
        self,
        hidden_dim: int,
        n_hidden_layers: int,
        n_components_s0: int,
        n_components: int,
        beta_min: float,
        beta_max: float,
        **kwargs,
    ):
        self._n_comp_max = max(n_components_s0, n_components)
        self._n_comp_min = min(n_components_s0, n_components)
        self._n_comp_diff = self._n_comp_max - self._n_comp_min
        self.n_components_s0 = n_components_s0
        self.n_components = n_components

        input_dim = 2

        output_dim = 1 + 3 * self.n_components

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            activation_fn="elu",
            **kwargs,
        )
        # Does not include the + 1 to handle the exit probability (which is
        # impossible at t=0).
        self.PFs0 = torch.nn.Parameter(torch.zeros(1, 5 * self.n_components_s0))

        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, preprocessed_states):
        out = super().forward(preprocessed_states)
        pre_sigmoid_exit = out[..., 0]
        mixture_logits = out[..., 1 : 1 + self.n_components]
        log_alpha = out[..., 1 + self.n_components : 1 + 2 * self.n_components]
        log_beta = out[..., 1 + 2 * self.n_components : 1 + 3 * self.n_components]

        exit_proba = torch.sigmoid(pre_sigmoid_exit)
        return (
            exit_proba,
            mixture_logits,
            self.beta_max * torch.sigmoid(log_alpha) + self.beta_min,
            self.beta_max * torch.sigmoid(log_beta) + self.beta_min,
        )

    def to_dist(self, x):
        if torch.all(x[0] == 0.0):
            assert torch.all(x == 0.0)
            alpha_theta = self.PFs0[0, self.n_components_s0 : 2 * self.n_components_s0]
            alpha_theta = self.beta_max * torch.sigmoid(alpha_theta) + self.beta_min
            alpha_r = self.PFs0[0, 3 * self.n_components_s0 : 4 * self.n_components_s0]
            alpha_r = self.beta_max * torch.sigmoid(alpha_r) + self.beta_min
            beta_theta = self.PFs0[
                0, 2 * self.n_components_s0 : 3 * self.n_components_s0
            ]
            beta_theta = self.beta_max * torch.sigmoid(beta_theta) + self.beta_min
            beta_r = self.PFs0[0, 4 * self.n_components_s0 : 5 * self.n_components_s0]
            beta_r = self.beta_max * torch.sigmoid(beta_r) + self.beta_min

            logits = self.PFs0[0, : self.n_components_s0]
            dist_r = MixtureSameFamily(
                Categorical(logits=logits),
                Beta(alpha_r, beta_r),
            )
            dist_theta = MixtureSameFamily(
                Categorical(logits=logits),
                Beta(alpha_theta, beta_theta),
            )
            return dist_r, dist_theta
        else:
            exit_proba, mixture_logits, alpha, beta = self.forward(x)
            dist = MixtureSameFamily(
                Categorical(logits=mixture_logits),
                Beta(alpha, beta),
            )

            return exit_proba, dist


class CirclePB2(NeuralNet2):
    def __init__(
        self,
        hidden_dim: int,
        n_hidden_layers: int,
        n_components: int,
        beta_min=0.1,
        beta_max=5.0,
        uniform=False,
        **kwargs,
    ):
        """Instantiates the neural network.

        Args:
            hidden_dim: the size of each hidden layer.
            n_hidden_layers: the number of hidden layers.
            n_components: the number of output components for each distribution
                parameter.
            **kwargs: passed to the NeuralNet class.
        """
        input_dim = 2
        output_dim = 3 * n_components

        self.uniform = uniform

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            activation_fn="elu",
        )
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_components = n_components

    def forward(self, preprocessed_states):
        out = super().forward(preprocessed_states)

        # Apply sigmoid to all except the dimensions between 0 and self.n_components.
        out[..., self.n_components :] = torch.sigmoid(out[..., self.n_components :])

        mixture_logits = out[:, 0 : self.n_components]
        alpha = out[:, self.n_components : 2 * self.n_components]
        beta = out[:, 2 * self.n_components : 3 * self.n_components]
        return (
            mixture_logits,
            self.beta_max * alpha + self.beta_min,
            self.beta_max * beta + self.beta_min,
        )

    def to_dist(self, x):
        if self.uniform:
            return Beta(torch.ones(x.shape[0]), torch.ones(x.shape[0]))
        mixture_logits, alpha, beta = self.forward(x)
        dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha, beta),
        )
        return dist


def sample_actions(env, model, states):
    # states is a tensor of shape (n, dim)
    batch_size = states.shape[0]
    out = model.to_dist(states)
    if isinstance(out[0], Distribution):  # s0 input
        dist_r, dist_theta = out
        samples_r = dist_r.sample(torch.Size((batch_size,)))
        samples_theta = dist_theta.sample(torch.Size((batch_size,)))

        actions = (
            torch.stack(
                [
                    samples_r * torch.cos(torch.pi / 2.0 * samples_theta),
                    samples_r * torch.sin(torch.pi / 2.0 * samples_theta),
                ],
                dim=1,
            )
            * env.delta
        )
        logprobs = (
            dist_r.log_prob(samples_r)
            + dist_theta.log_prob(samples_theta)
            - torch.log(samples_r * env.delta)
            - np.log(np.pi / 2)
            - np.log(env.delta)  # why ?
        )
    else:
        exit_proba, dist = out

        exit = torch.bernoulli(exit_proba).bool()
        exit[torch.norm(1 - states, dim=1) <= env.delta] = True
        exit[torch.any(states >= 1 - env.epsilon, dim=-1)] = True
        A = torch.where(
            states[:, 0] <= 1 - env.delta,
            0.0,
            2.0 / torch.pi * torch.arccos((1 - states[:, 0]) / env.delta),
        )
        B = torch.where(
            states[:, 1] <= 1 - env.delta,
            1.0,
            2.0 / torch.pi * torch.arcsin((1 - states[:, 1]) / env.delta),
        )
        assert torch.all(
            B[~torch.any(states >= 1 - env.delta, dim=-1)]
            >= A[~torch.any(states >= 1 - env.delta, dim=-1)]
        )
        samples = dist.sample()

        actions = samples * (B - A) + A
        actions *= torch.pi / 2.0
        actions = (
            torch.stack([torch.cos(actions), torch.sin(actions)], dim=1) * env.delta
        )

        logprobs = (
            dist.log_prob(samples)
            + torch.log(1 - exit_proba)
            - np.log(env.delta)
            - np.log(np.pi / 2)
            - torch.log(B - A)
        )

        actions[exit] = -float("inf")
        logprobs[exit] = torch.log(exit_proba[exit])
        logprobs[torch.norm(1 - states, dim=1) <= env.delta] = 0.0
        logprobs[torch.any(states >= 1 - env.epsilon, dim=-1)] = 0.0

    return actions, logprobs


def sample_trajectories(env, model, n_trajectories):
    step = 0
    states = torch.zeros((n_trajectories, env.dim), device=env.device)
    actionss = []
    trajectories = [states]
    trajectories_logprobs = torch.zeros((n_trajectories,), device=env.device)
    all_logprobs = []
    while not torch.all(states == env.sink_state):
        step_logprobs = torch.full((n_trajectories,), 0.0, device=env.device)
        non_terminal_mask = torch.all(states != env.sink_state, dim=-1)
        actions = torch.full((n_trajectories, env.dim), float("inf"), device=env.device)
        non_terminal_actions, logprobs = sample_actions(
            env,
            model,
            states[non_terminal_mask],
        )
        actions[non_terminal_mask] = non_terminal_actions.reshape(-1, env.dim)
        actionss.append(actions)
        states = env.step(states, actions)
        trajectories.append(states)
        trajectories_logprobs[non_terminal_mask] += logprobs
        step_logprobs[non_terminal_mask] = logprobs
        all_logprobs.append(step_logprobs)
        step += 1
    trajectories = torch.stack(trajectories, dim=1)
    actionss = torch.stack(actionss, dim=1)
    all_logprobs = torch.stack(all_logprobs, dim=1)
    return trajectories, actionss, trajectories_logprobs, all_logprobs


def evaluate_backward_logprobs(env, model, trajectories):
    logprobs = torch.zeros((trajectories.shape[0],), device=env.device)
    all_logprobs = []
    for i in range(trajectories.shape[1] - 2, 1, -1):
        all_step_logprobs = torch.full(
            (trajectories.shape[0],), -float("inf"), device=env.device
        )
        non_sink_mask = torch.all(trajectories[:, i] != env.sink_state, dim=-1)
        current_states = trajectories[:, i][non_sink_mask]
        previous_states = trajectories[:, i - 1][non_sink_mask]
        difference_1 = current_states[:, 0] - previous_states[:, 0]
        difference_1.clamp_(
            min=0.0, max=env.delta
        )  # Should be the case already - just to avoid numerical issues
        A = torch.where(
            current_states[:, 0] >= env.delta,
            0.0,
            2.0 / torch.pi * torch.arccos((current_states[:, 0]) / env.delta),
        )
        B = torch.where(
            current_states[:, 1] >= env.delta,
            1.0,
            2.0 / torch.pi * torch.arcsin((current_states[:, 1]) / env.delta),
        )

        dist = model.to_dist(current_states)

        step_logprobs = (
            dist.log_prob(
                (
                    1.0
                    / (B - A)
                    * (2.0 / torch.pi * torch.acos(difference_1 / env.delta) - A)
                ).clamp(1e-4, 1 - 1e-4)
            ).clamp_max(100)
            - np.log(env.delta)
            - np.log(np.pi / 2)
            - torch.log(B - A)
        )

        if torch.any(torch.isnan(step_logprobs)):
            raise ValueError("NaN in backward logprobs")

        if torch.any(torch.isinf(step_logprobs)):
            raise ValueError("Inf in backward logprobs")

        logprobs[non_sink_mask] += step_logprobs
        all_step_logprobs[non_sink_mask] = step_logprobs

        all_logprobs.append(all_step_logprobs)

    all_logprobs.append(torch.zeros((trajectories.shape[0],), device=env.device))
    all_logprobs = torch.stack(all_logprobs, dim=1)

    return logprobs, all_logprobs.flip(1)


def evaluate_state_flows(env, model, trajectories, logZ):
    state_flows = torch.full(
        (trajectories.shape[0], trajectories.shape[1]),
        -float("inf"),
        device=trajectories.device,
    )
    non_sink_mask = torch.all(trajectories != env.sink_state, dim=-1)
    state_flows[non_sink_mask] = model(trajectories[non_sink_mask]).squeeze(-1)
    state_flows[:, 0] = logZ

    return state_flows[:, :-1]


if __name__ == "__main__":
    from model import CirclePF, CirclePB, NeuralNet
    from env import Box, get_last_states

    env = Box(dim=2, delta=0.25)

    model = CirclePF()
    bw_model = CirclePB()

    flow = NeuralNet(output_dim=1)

    logZ = torch.zeros(1, requires_grad=True)

    trajectories, actionss, logprobs, all_logprobs = sample_trajectories(env, model, 5)

    bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(
        env, bw_model, trajectories
    )

    exits = torch.full(
        (trajectories.shape[0], trajectories.shape[1] - 1), -float("inf")
    )
    msk = torch.all(trajectories[:, 1:] != -float("inf"), dim=-1)
    middle_states = trajectories[:, 1:][msk]
    exit_proba, _ = model.to_dist(middle_states)
    true_exit_log_probs = torch.zeros_like(exit_proba)  # type: ignore
    edgy_middle_states_mask = torch.norm(1 - middle_states, dim=-1) <= env.delta
    other_edgy_middle_states_mask = torch.any(middle_states >= 1 - env.epsilon, dim=-1)
    true_exit_log_probs[edgy_middle_states_mask] = 0
    true_exit_log_probs[other_edgy_middle_states_mask] = 0
    true_exit_log_probs[
        ~edgy_middle_states_mask & ~other_edgy_middle_states_mask
    ] = torch.log(
        exit_proba[~edgy_middle_states_mask & ~other_edgy_middle_states_mask]  # type: ignore
    )

    exits[msk] = true_exit_log_probs
    exits = torch.cat([torch.zeros((trajectories.shape[0], 1)), exits], dim=1)
    non_infinity_mask = all_logprobs != -float("inf")
    _, indices = torch.max(non_infinity_mask.flip(1), dim=1)
    indices = all_logprobs.shape[1] - indices - 1
    new_all_logprobs = all_logprobs.scatter(1, indices.unsqueeze(1), -float("inf"))

    all_log_rewards = torch.full(
        (trajectories.shape[0], trajectories.shape[1] - 1), -float("inf")
    )
    log_rewards = env.reward(trajectories[:, 1:][msk]).log()
    all_log_rewards[msk] = log_rewards

    all_log_rewards = torch.cat(
        [logZ * torch.ones((trajectories.shape[0], 1)), all_log_rewards], dim=1
    )
    preds = new_all_logprobs[:, :-1] + exits[:, 1:-1] + all_log_rewards[:, :-2]
    targets = all_bw_logprobs + exits[:, :-2] + all_log_rewards[:, 1:-1]
    flat_preds = preds[preds != -float("inf")]
    flat_targets = targets[targets != -float("inf")]
    loss = torch.mean((flat_preds - flat_targets) ** 2)
