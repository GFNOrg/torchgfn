import torch
from torchtyping import TensorType

from gfn.containers import States
from gfn.envs import Env
from gfn.losses.base import StateDecomposableLoss
from gfn.parametrizations import FMParametrization

# Typing
ScoresTensor = TensorType["n_states", float]
LossTensor = TensorType[0, float]


class FlowMatching(StateDecomposableLoss):
    def __init__(self, parametrization: FMParametrization, env: Env) -> None:
        self.parametrization = parametrization
        self.env = env

    def get_scores(self, states: States) -> ScoresTensor:
        """
        Compute the scores for the given states, defined as the log-sum incoming flows minus log-sum outgoing flows.
        The batch_shape of states should be (n_states,).
        The scores are first computed for the the states that are not s0.
        Then s0 is treated separately, the corresponding F(s0 ->sf), if s0 is terminating, should be equal to R(s0).
        Therefore, there would be as many entries in the ScoresTensor corresponding to s0 as there are s0 in the batch,
        and the score (as a convention) for those entries is defined to be log F(s0 -> sf) - log R(s0).
        """
        assert len(states.batch_shape) == 1
        non_sink_states = states[~states.is_sink_state]

        non_initial_states = non_sink_states[~non_sink_states.is_initial_state]

        non_initial_terminal_mask = non_initial_states.forward_masks[:, -1]
        non_initial_terminal_states = non_initial_states[non_initial_terminal_mask]
        non_initial_terminal_states_rewards = self.env.reward(
            non_initial_terminal_states
        )

        incoming_flows = torch.full_like(
            non_initial_states.backward_masks, -float("inf"), dtype=torch.float
        )
        outgoing_flows = torch.full_like(
            non_initial_states.forward_masks, -float("inf"), dtype=torch.float
        )
        outgoing_flows[non_initial_terminal_mask, -1] = torch.log(
            non_initial_terminal_states_rewards
        )
        for action_idx in range(self.env.n_actions - 1):
            valid_backward_mask = non_initial_states.backward_masks[:, action_idx]
            valid_forward_mask = non_initial_states.forward_masks[:, action_idx]
            valid_backward_states = non_initial_states[valid_backward_mask]
            valid_forward_states = non_initial_states[valid_forward_mask]
            backward_actions = torch.full_like(
                valid_backward_states.backward_masks[:, 0], action_idx, dtype=torch.long
            )

            valid_backward_states_parents = self.env.backward_step(
                valid_backward_states, backward_actions
            )

            incoming_flows[valid_backward_mask, action_idx] = self.parametrization.logF(
                valid_backward_states_parents
            )[:, action_idx]
            outgoing_flows[valid_forward_mask, action_idx] = self.parametrization.logF(
                valid_forward_states
            )[:, action_idx]
        incoming_flows = torch.logsumexp(incoming_flows, dim=1)
        outgoing_flows = torch.logsumexp(outgoing_flows, dim=1)

        # We still need to compute the scores for the initial states that appear in the batch
        initial_states = non_sink_states[non_sink_states.is_initial_state]
        is_terminal = torch.all(initial_states.forward_masks[:, -1])
        if is_terminal:
            initial_states_rewards = self.env.reward(initial_states)
            initial_states_terminating_flows = self.parametrization.logF(
                initial_states
            )[:, -1]
            incoming_flows = torch.cat(
                [incoming_flows, initial_states_terminating_flows]
            )
            outgoing_flows = torch.cat(
                [outgoing_flows, torch.log(initial_states_rewards)]
            )

        return incoming_flows - outgoing_flows

    def __call__(self, states: States) -> LossTensor:
        scores = self.get_scores(states)
        return scores.pow(2).mean()
