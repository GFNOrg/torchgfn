from typing import Tuple

import torch
from torchtyping import TensorType as TT

from gfn.containers import Trajectories, Transitions
from gfn.env import Env
from gfn.gflownet.base import PFBasedGFlowNet
from gfn.modules import GFNModule, ScalarEstimator


class DBGFlowNet(PFBasedGFlowNet):
    r"""The Detailed Balance GFlowNet.

    Corresponds to $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2 \times
    \mathcal{O}_3$, where $\mathcal{O}_1$ is the set of functions from the internal
    states (no $s_f$) to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the
    non-negativity constraint), and $\mathcal{O}_2$ is the set of forward probability
    functions consistent with the DAG. $\mathcal{O}_3$ is the set of backward
    probability functions consistent with the DAG, or a singleton thereof, if
    `self.logit_PB` is a fixed `DiscretePBEstimator`.

    Attributes:
        logF: a ScalarEstimator instance.
        on_policy: boolean indicating whether we need to reevaluate the log probs.
        forward_looking: whether to implement the forward looking GFN loss.
    """

    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        logF: ScalarEstimator,
        on_policy: bool = False,
        forward_looking: bool = False,
    ):
        super().__init__(pf, pb, on_policy=on_policy)
        self.logF = logF
        self.forward_looking = forward_looking

    def get_scores(
        self, env: Env, transitions: Transitions
    ) -> Tuple[
        TT["n_transitions", float],
        TT["n_transitions", float],
        TT["n_transitions", float],
    ]:
        """Given a batch of transitions, calculate the scores.

        Args:
            transitions: a batch of transitions.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)

        if states.batch_shape != tuple(actions.batch_shape):
            raise ValueError("Something wrong happening with log_pf evaluations")
        if self.on_policy:
            valid_log_pf_actions = transitions.log_probs
        else:
            module_output = self.pf(states)
            valid_log_pf_actions = self.pf.to_probability_distribution(
                states, module_output
            ).log_prob(actions.tensor)

        valid_log_F_s = self.logF(states).squeeze(-1)
        if self.forward_looking:
            log_rewards = env.log_reward(states)  # RM unsqueeze(-1)
            valid_log_F_s = valid_log_F_s + log_rewards

        preds = valid_log_pf_actions + valid_log_F_s
        targets = torch.zeros_like(preds)

        # uncomment next line for debugging
        # assert transitions.next_states.is_sink_state.equal(transitions.is_done)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_done]
        non_exit_actions = actions[~actions.is_exit]

        module_output = self.pb(valid_next_states)
        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        valid_log_F_s_next = self.logF(valid_next_states).squeeze(-1)
        targets[~valid_transitions_is_done] = valid_log_pb_actions
        log_pb_actions = targets.clone()
        targets[~valid_transitions_is_done] += valid_log_F_s_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        targets[valid_transitions_is_done] = valid_transitions_log_rewards[
            valid_transitions_is_done
        ]

        scores = preds - targets

        return (valid_log_pf_actions, log_pb_actions, scores)

    def loss(self, env: Env, transitions: Transitions) -> TT[0, float]:
        """Detailed balance loss.

        The detailed balance loss is described in section
        3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)."""
        _, _, scores = self.get_scores(env, transitions)
        loss = torch.mean(scores**2)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()


class ModifiedDBGFlowNet(PFBasedGFlowNet):
    r"""The Modified Detailed Balance GFlowNet. Only applicable to environments where
    all states are terminating.

    See Bayesian Structure Learning with Generative Flow Networks
    https://arxiv.org/abs/2202.13903 for more details.
    """

    def get_scores(self, transitions: Transitions) -> TT["n_trajectories", torch.float]:
        """DAG-GFN-style detailed balance, when all states are connected to the sink.

        Raises:
            ValueError: when backward transitions are supplied (not supported).
            ValueError: when the computed scores contain `inf`.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")

        mask = ~transitions.next_states.is_sink_state
        states = transitions.states[mask]
        valid_next_states = transitions.next_states[mask]
        actions = transitions.actions[mask]
        all_log_rewards = transitions.all_log_rewards[mask]
        module_output = self.pf(states)
        pf_dist = self.pf.to_probability_distribution(states, module_output)
        if self.on_policy:
            valid_log_pf_actions = transitions[mask].log_probs
        else:
            valid_log_pf_actions = pf_dist.log_prob(actions.tensor)
        valid_log_pf_s_exit = pf_dist.log_prob(
            torch.full_like(actions.tensor, actions.__class__.exit_action[0])
        )

        # The following two lines are slightly inefficient, given that most
        # next_states are also states, for which we already did a forward pass.
        module_output = self.pf(valid_next_states)
        valid_log_pf_s_prime_exit = self.pf.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(torch.full_like(actions.tensor, actions.__class__.exit_action[0]))

        non_exit_actions = actions[~actions.is_exit]
        module_output = self.pb(valid_next_states)
        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

        preds = all_log_rewards[:, 0] + valid_log_pf_actions + valid_log_pf_s_prime_exit
        targets = all_log_rewards[:, 1] + valid_log_pb_actions + valid_log_pf_s_exit

        scores = preds - targets
        if torch.any(torch.isinf(scores)):
            raise ValueError("scores contains inf")

        return scores

    def loss(self, env: Env, transitions: Transitions) -> TT[0, float]:
        """Calculates the modified detailed balance loss."""
        scores = self.get_scores(transitions)
        return torch.mean(scores**2)

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()
