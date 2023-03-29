from typing import ClassVar, Literal, Tuple

import torch
import numpy as np
from gymnasium.spaces import Discrete
from torchtyping import TensorType
from itertools import product, permutations

from gfn.containers.states import States
from gfn.envs.env import Env

# https://oeis.org/A003024
NUM_DAGS = [1, 1, 3, 25, 543, 29281]

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]


class DAGEnv(Env):
    def __init__(
        self,
        num_variables: int,
        device_str: Literal["cpu", "cuda"] = "cpu"
    ):
        self.num_variables = num_variables

        s0 = torch.zeros(
            (num_variables, num_variables),
            dtype=torch.long,
            device=torch.device(device_str)
        )
        sf = torch.ones(
            (num_variables, num_variables),
            dtype=torch.long,
            device=torch.device(device_str)
        )
        preprocessor = None  # TODO

        super().__init__(
            action_space=Discrete(num_variables ** 2 + 1),
            s0=s0,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor
        )

    def make_States_class(self) -> type[States]:
        env = self

        class DAGStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.num_variables, env.num_variables)
            s0 = env.s0
            sf = env.sf

            @property
            def forward_masks(self):
                kwargs = {'dtype': torch.bool, 'device': env.device}

                continue_masks = 1 - (self.states_tensor + self._closure_T)
                continue_masks = continue_masks.view(*self.batch_shape, -1)
                continue_masks = continue_masks.to(dtype=torch.bool)

                # The stop action is always a valid action
                stop_mask = torch.ones(self.batch_shape + (1,), **kwargs)
                return torch.cat((continue_masks, stop_mask), dim=1)

            @forward_masks.setter
            def forward_masks(self, value):
                self._forward_masks = value

            def make_masks(
                self
            ) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                kwargs = {'dtype': torch.bool, 'device': env.device}
                # Initialize the transitive closure of the transpose of the graphs
                self._closure_T = torch.zeros(
                    self.batch_shape + self.state_shape,
                    **kwargs
                )
                self._closure_T.logical_or_(torch.eye(env.num_variables, **kwargs))

                # Initialize the forward masks using `None`. The computation of
                # the forward masks is being done in the `forward_masks` property
                forward_masks = None

                # Initialize the backward masks (no valid action)
                backward_masks = torch.zeros(
                    self.batch_shape + (env.num_variables ** 2,),
                    **kwargs
                )
                return (forward_masks, backward_masks)
            
            def update_masks(self):
                # TODO: The update of the transitive closure of the graphs
                # (required to get the forward masks) depends on the last action
                # applied. Reference: https://github.com/tristandeleu/jax-dag-gflownet/blob/53350bbfba3ab24f9ffa8196a3de42f139d9c5cc/dag_gflownet/env.py#L112-L116
                self._closure_T = self._closure_T

                self.backward_masks = (self.states_tensor != 0)
                self.backward_masks = self.backward_masks.view(*self.batch_shape, -1)

        return DAGStates

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        return actions == self.action_space.n - 1

    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.view(*states.shape[-2], -1).scatter_(
            -1, actions.unsqueeze(-1), 1, reduce="add")

    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.view(*states.shape[-2], -1).scatter_(
            -1, actions.unsqueeze(-1), -1, reduce="add")

    def log_reward(self, final_states: States) -> TensorFloat:
        # TODO: Implement the log-reward function (BGe score & uniform prior)
        return torch.zeros(
            final_states.shape[-2],
            dtype=final_states.dtype,
            device=final_states.device
        )

    @property
    def n_states(self) -> int:
        if self.num_variables < len(NUM_DAGS):
            return NUM_DAGS[self.num_variables]
        else:
            raise NotImplementedError(
                "The environment does not support enumeration of states for "
                f"`num_variables > {len(NUM_DAGS) - 1}` (num_variables = {self.num_variables})."
            )

    @property
    def n_terminating_states(self) -> int:
        # All the states of the environment are terminating
        return self.n_states

    @property
    def all_states(self) -> States:
        if self.num_variables >= len(NUM_DAGS):
            raise NotImplementedError(
                "The environment does not support enumeration of states for "
                f"`num_variables > {len(NUM_DAGS) - 1}` (num_variables = {self.num_variables})."
            )

        # Generate all the DAGs over num_variables nodes
        shape = (self.num_variables, self.num_variables)
        repeat = self.num_variables * (self.num_variables - 1) // 2

        # Generate all the possible binary codes
        codes = list(product([0, 1], repeat=repeat))
        codes = np.asarray(codes)

        # Get upper-triangular indices
        x, y = np.triu_indices(self.num_variables, k=1)

        # Fill the upper-triangular matrices
        trius = np.zeros((len(codes),) + shape, dtype=np.int_)
        trius[:, x, y] = codes

        # Apply permutation, and remove duplicates
        compressed_dags = set()
        for perm in permutations(range(self.num_variables)):
            permuted = trius[:, :, perm][:, perm, :]
            permuted = permuted.reshape(-1, self.num_variables ** 2)
            permuted = np.packbits(permuted, axis=1)
            compressed_dags.update(map(tuple, permuted))
        compressed_dags = sorted(list(compressed_dags))

        # Uncompress the DAGs
        adjacencies = np.unpackbits(compressed_dags, axis=1, count=self.num_variables ** 2)
        adjacencies = adjacencies.reshape(-1, self.num_variables, self.num_variables)

        states = torch.as_tensor(adjacencies, dtype=torch.long, device=self.device)
        return self.States(states)

    @property
    def terminating_states(self) -> States:
        return self.all_states
