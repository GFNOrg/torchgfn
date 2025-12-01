# import pytest
# import torch

# from gfn.containers import StatesContainer, Trajectories, Transitions
# from gfn.env import Env
# from gfn.gym.discrete_ebm import DiscreteEBM

# TODO: these transitions are actually invalid under the envs provided. We
#       should fix that (probably by making a dummy environment). This is kept
#       now as a placeholder.

# def transitions_containers(env: Env):
#     """Creates Transitions containers."""

#     # Create some states
#     states1 = env.states_from_tensor(torch.tensor([[0, 0], [1, 1]]))
#     actions1 = env.actions_from_tensor(torch.tensor([[1], [0]]))
#     is_terminating1 = torch.tensor([False, True])
#     next_states1 = env.states_from_tensor(torch.tensor([[1, 0], [1, 1]]))
#     log_probs1 = torch.tensor([-0.5, -0.3])
#     log_rewards1 = torch.tensor([0.0, -1.0])

#     transitions1 = Transitions(
#         env=env,
#         states=states1,
#         actions=actions1,
#         is_terminating=is_terminating1,
#         next_states=next_states1,
#         log_probs=log_probs1,
#         log_rewards=log_rewards1,
#     )

#     # Create another set of transitions
#     states2 = env.states_from_tensor(torch.tensor([[2, 1], [0, 2]]))
#     actions2 = env.actions_from_tensor(torch.tensor([[0], [1]]))
#     is_terminating2 = torch.tensor([True, False])
#     next_states2 = env.states_from_tensor(torch.tensor([[2, 1], [1, 2]]))
#     log_probs2 = torch.tensor([-0.2, -0.7])
#     log_rewards2 = torch.tensor([0.0, -1.0])

#     transitions2 = Transitions(
#         env=env,
#         states=states2,
#         actions=actions2,
#         is_terminating=is_terminating2,
#         next_states=next_states2,
#         log_probs=log_probs2,
#         log_rewards=log_rewards2,
#     )

#     return transitions1, transitions2


# def state_containers(env: Env):
#     """Creates StateContainer containers."""

#     # Create first set of state pairs.
#     states1 = env.states_from_tensor(torch.tensor([[0, 1], [1, 0]]))
#     terminating_states1 = env.states_from_tensor(torch.tensor([[2, 2], [1, 2]]))
#     states1.extend(terminating_states1)
#     is_terminating1 = torch.tensor([False, False, True, True])
#     log_rewards1 = torch.tensor([torch.inf, torch.inf, -1.0, -2.0])

#     state_pairs1 = StatesContainer(
#         env=env,
#         states=states1,
#         is_terminating=is_terminating1,
#         log_rewards=log_rewards1,
#     )

#     # Create second set of state pairs.
#     states2 = env.states_from_tensor(torch.tensor([[1, 1], [0, 0]]))
#     terminating_states2 = env.states_from_tensor(torch.tensor([[2, 1], [2, 0]]))
#     states2.extend(terminating_states2)
#     is_terminating2 = torch.tensor([False, False, True, True])
#     log_rewards2 = torch.tensor([torch.inf, torch.inf, -1.5, -0.5])

#     state_pairs2 = StatesContainer(
#         env=env,
#         states=states2,
#         is_terminating=is_terminating2,
#         log_rewards=log_rewards2,
#     )

#     return state_pairs1, state_pairs2


# def trajectories_containers(env: Env):
#     """Creates Trajectories containers."""

#     # Create first set of trajectories
#     states1 = env.states_from_tensor(
#         torch.tensor(
#             [
#                 [[0, 0], [0, 0]],  # Initial states
#                 [[1, 0], [0, 1]],  # Step 1
#                 [[2, 0], [0, 2]],  # Step 2
#                 [[2, 0], [0, 2]],  # Padding for trajectory 1
#             ],
#             dtype=torch.int,  # Tests type compatibility with int32.
#         )
#     )

#     actions1 = env.actions_from_tensor(
#         torch.tensor(
#             [
#                 [[0], [1]],  # Step 0
#                 [[0], [1]],  # Step 1
#                 [[2], [2]],  # Exit action for both trajectories
#             ],
#             dtype=torch.int,  # Tests type compatibility with int32.
#         )
#     )

#     terminating_idx1 = torch.tensor([2, 2], dtype=torch.int)
#     log_rewards1 = torch.tensor([-2.0, -2.0])

#     trajectories1 = Trajectories(
#         env=env,
#         states=states1,
#         actions=actions1,
#         terminating_idx=terminating_idx1,
#         log_rewards=log_rewards1,
#     )

#     # Create second set of trajectories
#     states2 = env.states_from_tensor(
#         torch.tensor(
#             [
#                 [[0, 0], [0, 0]],  # Initial states
#                 [[1, 0], [0, 1]],  # Step 1
#                 [[1, 1], [1, 1]],  # Step 2
#                 [[2, 1], [1, 1]],  # Step 3 (only for trajectory 1)
#             ],
#             dtype=torch.long,  # Tests type compatibility with int64.
#         )
#     )

#     actions2 = env.actions_from_tensor(
#         torch.tensor(
#             [
#                 [[0], [1]],  # Step 0
#                 [[1], [1]],  # Step 1
#                 [[2], [2]],  # Exit action for both trajectories
#             ],
#             dtype=torch.long,  # Tests type compatibility with int64.
#         )
#     )

#     terminating_idx2 = torch.tensor(
#         [3, 2], dtype=torch.long
#     )  # Tests type compatibility with int64.
#     log_rewards2 = torch.tensor([-3.0, -2.0])

#     trajectories2 = Trajectories(
#         env=env,
#         states=states2,
#         actions=actions2,
#         terminating_idx=terminating_idx2,
#         log_rewards=log_rewards2,
#     )

#     return trajectories1, trajectories2


# @pytest.mark.parametrize(
#     "container_type", ["transitions", "states_container", "trajectories"]
# )
# def test_containers(container_type: str):
#     env = DiscreteEBM(ndim=2)
#     if container_type == "transitions":
#         container1, container2 = transitions_containers(env)
#     elif container_type == "states_container":
#         container1, container2 = state_containers(env)
#     elif container_type == "trajectories":
#         container1, container2 = trajectories_containers(env)

#     initial_len = len(container1)

#     # Test extending container1 with container2
#     container1.extend(container2)  # type: ignore

#     # Check that the length of container1 is now the sum of both containers
#     assert len(container1) == initial_len + len(container2)

#     # Check that the elements from container2 are correctly added to container1
#     if isinstance(container1, Transitions):
#         for i in range(len(container2)):
#             container1_obj = container1[i + initial_len]
#             container2_obj = container2[i]
#             assert torch.equal(
#                 container1_obj.states.tensor, container2_obj.states.tensor
#             )
#             assert torch.equal(
#                 container1_obj.actions.tensor, container2_obj.actions.tensor
#             )
#             assert torch.equal(
#                 container1_obj.is_terminating, container2_obj.is_terminating
#             )
#             assert torch.equal(
#                 container1_obj.next_states.tensor, container2_obj.next_states.tensor
#             )
#             assert container1_obj.log_probs is not None
#             assert container2_obj.log_probs is not None
#             assert torch.equal(container1_obj.log_probs, container2_obj.log_probs)

#             assert isinstance(container1_obj.log_rewards, torch.Tensor)
#             assert isinstance(container2_obj.log_rewards, torch.Tensor)
#             assert torch.equal(container1_obj.log_rewards, container2_obj.log_rewards)
#     elif isinstance(container1, StatesContainer):
#         for i in range(len(container2)):
#             container1_obj = container1[i + initial_len]
#             container2_obj = container2[i]
#             assert torch.equal(
#                 container1_obj.intermediary_states.tensor,
#                 container2_obj.intermediary_states.tensor,
#             )
#             assert torch.equal(
#                 container1_obj.terminating_states.tensor,
#                 container2_obj.terminating_states.tensor,
#             )
#             assert isinstance(container1_obj.log_rewards, torch.Tensor)
#             assert isinstance(container2_obj.log_rewards, torch.Tensor)
#             assert torch.equal(container1_obj.log_rewards, container2_obj.log_rewards)
#     elif isinstance(container1, Trajectories):
#         for i in range(len(container2)):
#             container1_obj = container1[i + initial_len]
#             container2_obj = container2[i]
#             assert torch.equal(
#                 container1_obj.states.tensor, container2_obj.states.tensor
#             )
#             assert torch.equal(
#                 container1_obj.actions.tensor, container2_obj.actions.tensor
#             )
#             assert torch.equal(
#                 container1_obj.terminating_idx, container2_obj.terminating_idx
#             )
#             assert isinstance(container1_obj.log_rewards, torch.Tensor)
#             assert isinstance(container2_obj.log_rewards, torch.Tensor)
#             assert torch.equal(container1_obj.log_rewards, container2_obj.log_rewards)
