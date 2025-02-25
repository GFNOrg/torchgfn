from __future__ import annotations  # This allows to use the class name in type hints

import torch
from typing import Optional

from gfn.env import DiscreteEnv
from gfn.preprocessors import Preprocessor
from gfn.actions import Actions
from gfn.containers import Trajectories

from .bitSequenceStates import BitSequenceStates
from .bitSequence import BitSequence


class BitSequencePlus(BitSequence):
    """
    Prepend-Append version of BitSequence env.
    """

    def __init__(
        self,
        word_size: int = 4,
        seq_size: int = 120,
        n_modes: int = 60,
        temperature: float = 1.0,
        H: Optional[torch.Tensor] = None,
        device_str: str = "cpu",
        preprocessor: Optional[Preprocessor] = None,
    ):
        assert seq_size % word_size == 0, "word_size must divide seq_size."
        self.words_per_seq: int = seq_size // word_size
        self.word_size: int = word_size
        self.seq_size: int = seq_size
        self.n_modes: int = n_modes
        n_actions = 2 ** (word_size + 1) + 1
        s0 = -torch.ones(
            self.words_per_seq, dtype=torch.long, device=torch.device(device_str)
        )
        state_shape = s0.shape
        action_shape = (1,)
        dummy_action = -torch.ones(1, dtype=torch.long)
        exit_action = (n_actions - 1) * torch.ones(1, dtype=torch.long)
        sf = ((n_actions - 1) // 2) * torch.ones(
            self.words_per_seq, dtype=torch.long, device=torch.device(device_str)
        )
        DiscreteEnv.__init__(
            self,
            n_actions,
            s0,
            state_shape,
            action_shape,
            dummy_action,
            exit_action,
            sf,
            device_str,
            preprocessor,
        )
        self.H = H
        self.modes = self.make_modes_set()  # set of modes written as binary
        self.temperature = temperature

    def update_masks(self, states: BitSequenceStates) -> None:

        is_done = states.length == self.words_per_seq
        states.forward_masks = torch.ones_like(
            states.forward_masks, dtype=torch.bool, device=states.__class__.device
        )
        states.forward_masks[is_done, :-1] = False
        states.forward_masks[~is_done, -1] = False

        is_sink = states.is_sink_state

        last_actions = states.tensor[
            ~is_sink, states[~is_sink].length - 1  # pyright:ignore
        ]
        first_actions = states.tensor[~is_sink, 0]
        states.backward_masks = torch.zeros_like(
            states.backward_masks, dtype=torch.bool, device=states.__class__.device
        )
        states.backward_masks[~is_sink, last_actions] = True
        states.backward_masks[~is_sink, first_actions + (self.n_actions - 1) // 2] = (
            True
        )

    def step(self, states: BitSequenceStates, actions: Actions) -> torch.Tensor:
        is_exit = actions.is_exit
        old_tensor = states.tensor.clone()
        append_mask = (actions.tensor < (self.n_actions - 1) // 2).squeeze()
        prepend_mask = ~append_mask
        assert states.length
        old_tensor[append_mask & ~is_exit, states.length[append_mask & ~is_exit]] = (
            actions.tensor[append_mask & ~is_exit].squeeze()  # pyright:ignore
        )

        old_tensor[prepend_mask & ~is_exit, 1:] = old_tensor[
            prepend_mask & ~is_exit, :-1
        ]
        old_tensor[prepend_mask & ~is_exit, 0] = (
            actions.tensor[prepend_mask & ~is_exit].squeeze()
            - (self.n_actions - 1) // 2
        )

        old_tensor[is_exit] = torch.full_like(
            old_tensor[is_exit],
            self.n_actions - 1,
            dtype=torch.long,
            device=old_tensor.device,
        )
        return old_tensor

    def backward_step(
        self, states: BitSequenceStates, actions: Actions
    ) -> torch.Tensor:
        old_tensor = states.tensor.clone()
        remove_end_mask = (actions.tensor < (self.n_actions - 1) // 2).squeeze()
        remove_front_mask = ~remove_end_mask
        assert states.length
        old_tensor[remove_end_mask, states.length[remove_end_mask] - 1] = -1

        old_tensor[remove_front_mask, :-1] = old_tensor[remove_front_mask, 1:]

        old_tensor[remove_front_mask, -1] = -1
        return old_tensor

    def trajectory_from_terminating_states(
        self, terminating_states_tensor: torch.Tensor
    ) -> Trajectories:
        raise NotImplementedError("Only available for append-only BitSequence.")
