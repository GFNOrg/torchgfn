from typing import List, Optional

import torch
from torchtyping import TensorType

from gfn.containers import States, Trajectories
from gfn.envs import Env
from gfn.samplers.actions_samplers import ActionsSampler, BackwardActionsSampler

# Typing
StatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
ActionsTensor = TensorType["n_trajectories", torch.long]
DonesTensor = TensorType["n_trajectories", torch.bool]


class TrajectoriesSampler:
    def __init__(
        self,
        env: Env,
        actions_sampler: ActionsSampler,
        evaluate_log_probabilities: bool = False,
        backward_actions_sampler: BackwardActionsSampler | None = None,
    ):
        """Sample complete trajectories, or completes trajectories from a given batch states, using actions_sampler.

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
            evaluate_log_probabilities (bool, optional): Whether to evaluate log probabilities of actions. Defaults to False. If True, requires backward_actions_sampler to be not None.
            backward_actions_sampler (BackwardActionsSampler, optional): Useful to calculate log_pbs. If None (default), log_pbs will be set to None.
        """
        self.env = env
        self.actions_sampler = actions_sampler
        self.evaluate_log_probabilities = evaluate_log_probabilities
        self.backward_actions_sampler = backward_actions_sampler
        self.is_backward = isinstance(actions_sampler, BackwardActionsSampler)
        if evaluate_log_probabilities and (
            backward_actions_sampler is None or self.is_backward
        ):
            raise ValueError(
                "evaluate_log_probabilities is True but backward_actions_sampler is None or actions_sampler is BackwardActionsSampler"
            )

    def sample_trajectories(
        self,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
    ) -> Trajectories:
        if states is None:
            assert (
                n_trajectories is not None
            ), "Either states or n_trajectories should be specified"
            states = self.env.reset(batch_shape=(n_trajectories,))
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]
        assert states is not None

        dones = states.is_initial_state if self.is_backward else states.is_sink_state

        trajectories_states: List[StatesTensor] = [states.states_tensor]
        trajectories_actions: List[ActionsTensor] = []
        trajectories_dones = torch.zeros(n_trajectories, dtype=torch.long)
        trajectories_rewards = torch.zeros(n_trajectories, dtype=torch.float)
        if self.evaluate_log_probabilities:
            trajectories_log_pfs = torch.zeros(n_trajectories, dtype=torch.float)
            trajectories_log_pbs = torch.zeros(n_trajectories, dtype=torch.float)
        else:
            trajectories_log_pfs = None
            trajectories_log_pbs = None
        step = 0

        while not all(dones):
            actions = torch.full(
                (n_trajectories,),
                fill_value=-1,
                dtype=torch.long,
                device=states.states_tensor.device,
            )
            actions_log_probs, valid_actions = self.actions_sampler.sample(
                states[~dones]
            )
            actions[~dones] = valid_actions
            trajectories_actions += [actions]

            if self.is_backward:
                new_states = self.env.backward_step(states, actions)
            else:
                new_states = self.env.step(states, actions)
            sink_states_mask = new_states.is_sink_state

            if self.evaluate_log_probabilities:
                assert (
                    trajectories_log_pfs is not None
                    and trajectories_log_pbs is not None
                    and self.backward_actions_sampler is not None
                )
                trajectories_log_pfs[~dones] += actions_log_probs
                non_sink_new_states = new_states[~sink_states_mask]
                backward_probs = self.backward_actions_sampler.get_probs(
                    non_sink_new_states
                )
                trajectories_log_pbs[~sink_states_mask] += (
                    backward_probs.gather(-1, actions[~sink_states_mask].unsqueeze(-1))
                    .squeeze(-1)
                    .log()
                )
            step += 1

            new_dones = (
                new_states.is_initial_state if self.is_backward else sink_states_mask
            ) & ~dones
            trajectories_dones[new_dones & ~dones] = step
            trajectories_rewards[new_dones & ~dones] = self.env.reward(
                new_states[new_dones & ~dones]
            )
            states = new_states
            dones = dones | new_dones

            trajectories_states += [states.states_tensor]

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = self.env.States(states_tensor=trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions, dim=0)

        trajectories = Trajectories(
            env=self.env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.is_backward,
            rewards=trajectories_rewards,
            log_pfs=trajectories_log_pfs,
            log_pbs=trajectories_log_pbs,
        )

        return trajectories

    def sample(self, n_objects: int) -> Trajectories:
        return self.sample_trajectories(n_trajectories=n_objects)
