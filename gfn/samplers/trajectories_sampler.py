import torch
from torchtyping import TensorType
from typing import Union

from dataclasses import dataclass
from gfn.envs.env import Env, AbstractStatesBatch
from gfn.samplers.action_samplers import ActionSampler, BackwardsActionSampler, FixedActions

# Typing
# max_length = None
n_trajectories = None
shape = None
Tensor2D = TensorType['max_length', 'n_trajectories', torch.long]
Tensor2D2 = TensorType['n_trajectories', 'shape']
Tensor1D = TensorType['n_trajectories', torch.long]


@dataclass
class Trajectories:
    """Class for keeping track of multiple COMPLETE trajectories."""
    env: Env
    n_trajectories: int
    states: AbstractStatesBatch
    actions: Tensor2D
    # The following field mentions how many actions were taken in each trajectory.
    when_is_done: Tensor1D
    rewards: Tensor1D
    last_states: AbstractStatesBatch

    def __repr__(self) -> str:
        states = self.states.states
        assert states.ndim == 3
        states = states.transpose(0, 1)
        states_repr = '\n'.join(['-> '.join([str(step.numpy()) for step in traj])
                                 for traj in states])
        return f"Trajectories(n_trajectories={self.n_trajectories}, " \
               f"states={states_repr}, actions={self.actions}, " \
               f"when_is_done={self.when_is_done}, rewards={self.rewards})"

    def get_last_states_raw(self) -> Tensor2D2:
        return self.states.states[-1]

    def purge(self, raw_state) -> None:
        """Remove all trajectories that ended in the given state."""
        ndim = self.states.shape[-1]
        mask = (self.get_last_states_raw() == raw_state).sum(1) == ndim
        self.n_trajectories -= mask.sum().item()
        self.states.states = self.states.states[:, ~mask]
        self.states.update_masks()
        self.actions = self.actions[:, ~mask]
        self.when_is_done = self.when_is_done[~mask]
        if self.rewards is not None:
            self.rewards = self.rewards[~mask]

    def sample(self, n_trajectories: int) -> 'Trajectories':
        """Sample a random subset of trajectories."""
        perm = torch.randperm(self.n_trajectories)
        indices = perm[:n_trajectories]

        states_raw = self.states.states[:, indices, ...]
        states = self.env.StatesBatch(states=states_raw)
        states.update_masks()
        actions = self.actions[:, indices]
        when_is_done = self.when_is_done[indices]
        rewards = self.rewards[indices] if self.rewards is not None else None
        last_states_raw = self.last_states.states[indices, ...]
        last_states = self.env.StatesBatch(states=last_states_raw)
        last_states.update_masks()
        return Trajectories(env=self.env, n_trajectories=n_trajectories, states=states, actions=actions,
                            when_is_done=when_is_done, rewards=rewards, last_states=last_states)


class TrajectoriesSampler:
    def __init__(self, env: Env, action_sampler: ActionSampler):
        self.env = env
        self.action_sampler = action_sampler
        self.is_backwards = isinstance(action_sampler, BackwardsActionSampler)

    def sample_trajectories(self, states: Union[None, AbstractStatesBatch] = None) -> Trajectories:
        if states is None:
            states = self.env.reset()
        n_trajectories = states.shape[0]
        trajectories_states = []
        trajectories_actions = []
        trajectories_already_dones = []
        # trajectories_backwards_masks = []
        trajectories_dones = (-1) * \
            torch.ones(n_trajectories, dtype=torch.long)
        step = 0
        while not all(states.already_dones):
            logits, actions = self.action_sampler.sample(states)
            trajectories_states += [states.states]
            trajectories_actions += [actions]
            trajectories_already_dones += [states.already_dones]

            if self.is_backwards:
                new_states, dones = self.env.backward_step(states, actions)
            else:
                # TODO: fix this when states is passed as argument
                new_states, dones = self.env.step(actions)
            trajectories_dones[dones & ~states.already_dones] = step
            states = new_states
            step += 1
            if isinstance(self.action_sampler, FixedActions) and self.action_sampler.step >= self.action_sampler.total_steps:
                states.already_dones = torch.ones_like(
                    states.already_dones, dtype=torch.bool)

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_already_dones = torch.stack(
            trajectories_already_dones, dim=0)
        trajectories_states = self.env.StatesBatch(states=trajectories_states)
        trajectories_states.already_dones = trajectories_already_dones
        trajectories_actions = torch.stack(trajectories_actions, dim=0)

        if self.is_backwards:
            trajectories_rewards = None
        else:
            trajectories_rewards = self.env.reward(states)

        trajectories = Trajectories(env=self.env,
                                    n_trajectories=n_trajectories,
                                    states=trajectories_states,
                                    actions=trajectories_actions,
                                    when_is_done=trajectories_dones,
                                    rewards=trajectories_rewards,
                                    last_states=states)

        return trajectories


if __name__ == '__main__':
    from gfn.envs import HyperGrid
    from gfn.preprocessors import IdentityPreprocessor, OneHotPreprocessor, KHotPreprocessor
    from gfn.samplers.action_samplers import (UniformActionSampler, UniformBackwardsActionSampler,
                                              LogitPBActionSampler,
                                              FixedActions, LogitPFActionSampler, LogEdgeFlowsActionSampler)
    from gfn.estimators import LogitPFEstimator, LogEdgeFlowEstimator, LogitPBEstimator
    from gfn.models import NeuralNet
    n_envs = 5
    env = HyperGrid(n_envs=n_envs)

    print('---Trying Forward sampling of trajectories---')

    print("Trying the Uniform Action Sample with sf_temperature")
    action_sampler = UniformActionSampler(sf_temperature=2.)
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    print(trajectories)

    print("Trying the Fixed Actions Sampler: ")
    action_sampler = FixedActions(torch.tensor([[0, 1, 2, 0],
                                                [1, 1, 1, 2],
                                                [2, 2, 2, 2],
                                                [1, 0, 1, 2],
                                                [1, 0, 2, 1]]))
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    print(trajectories)

    print("Trying the LogitPFActionSampler: ")
    preprocessors = [IdentityPreprocessor(env=env),
                     OneHotPreprocessor(env=env),
                     KHotPreprocessor(env=env)]
    modules = [NeuralNet(input_dim=preprocessor.output_dim,
                         hidden_dim=12,
                         output_dim=env.n_actions)
               for preprocessor in preprocessors]
    logit_pf_estimators = [LogitPFEstimator(preprocessor=preprocessor, env=env, module=module)
                           for (preprocessor, module) in zip(preprocessors, modules)]

    logit_pf_action_samplers = [LogitPFActionSampler(logit_PF=logit_pf_estimator)
                                for logit_pf_estimator in logit_pf_estimators]

    trajectories_samplers = [TrajectoriesSampler(env, logit_pf_action_sampler)
                             for logit_pf_action_sampler in logit_pf_action_samplers]

    for i, trajectories_sampler in enumerate(trajectories_samplers):
        print(i, ": Trying the LogitPFActionSampler with preprocessor {}".format(
            preprocessors[i]))
        trajectories = trajectories_sampler.sample_trajectories()
        print(trajectories)

    print('---Trying Backwards sampling of trajectories---')
    states = env.StatesBatch(random=True)
    states.states[0] = torch.zeros(2)
    states.update_masks()
    states.update_the_dones()
    print('Trying the Uniform Backwards Action Sampler with one of the initial states being s_0')
    action_sampler = UniformBackwardsActionSampler()
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories(states)
    print(trajectories)

    modules = [NeuralNet(input_dim=preprocessor.output_dim,
                         hidden_dim=12,
                         output_dim=env.n_actions - 1)
               for preprocessor in preprocessors]
    logit_pb_estimators = [LogitPBEstimator(preprocessor=preprocessor,
                                            env=env,
                                            module=module)
                           for (preprocessor, module) in zip(preprocessors, modules)]

    logit_pb_action_samplers = [LogitPBActionSampler(logit_PB=logit_pb_estimator)
                                for logit_pb_estimator in logit_pb_estimators]

    trajectories_samplers = [TrajectoriesSampler(env, logit_pb_action_sampler)
                             for logit_pb_action_sampler in logit_pb_action_samplers]

    for i, trajectories_sampler in enumerate(trajectories_samplers):
        print(i, ": Trying the LogitPBActionSampler with preprocessor {}".format(
            preprocessors[i]))
        states = env.StatesBatch(random=True)
        states.update_masks()
        states.update_the_dones()
        trajectories = trajectories_samplers[i].sample_trajectories(states)
        print(trajectories)

    print('---Making Sure Last states are computed correcly---')
    action_sampler = UniformActionSampler(sf_temperature=2.)
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    if trajectories.last_states.states.equal(trajectories.get_last_states_raw()):
        print('Last states are computed correctly')
    else:
        print('WARNING !! Last states are not computed correctly')

    print('---Testing SubSampling---')
    print(
        f"There are {trajectories.n_trajectories} original trajectories, from which we will sample 2")
    print(trajectories)
    sampled_trajectories = trajectories.sample(n_trajectories=2)
    print("The two sampled trajectories are:")
    print(sampled_trajectories)
