from __future__ import annotations  # This allows to use the class name in type hints

import torch
from typing import Optional, ClassVar, Sequence, List

from gfn.states import DiscreteStates


# This environment is the torchgfn implmentation of the bit sequences task presented in :Malkin, Nikolay & Jain, Moksh & Bengio, Emmanuel & Sun, Chen & Bengio, Yoshua. (2022).
# Trajectory Balance: Improved Credit Assignment in GFlowNets. https://arxiv.org/pdf/2201.13259


class BitSequenceStates(DiscreteStates):
    """
    A class representing states of bit sequences in a discrete state space.

    Attributes:
        word_size (ClassVar[int]): The size of each word in the bit sequence.
        tensor (torch.Tensor): The tensor representing the states.
        length (torch.Tensor): The tensor representing the length of each bit sequence.
        forward_masks (Optional[torch.Tensor]): The tensor representing the forward masks.
        backward_masks (Optional[torch.Tensor]): The tensor representing the backward masks.

    Methods:
        __init__(tensor: torch.Tensor, length: Optional[torch.Tensor[int]] = None, forward_masks: Optional[torch.Tensor] = None, backward_masks: Optional[torch.Tensor] = None) -> None:
            Initializes the BitSequencesStates object.

        clone() -> BitSequencesStates:
            Returns a clone of the current BitSequencesStates object.

        __getitem__(index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor) -> BitSequencesStates:
            Returns a subset of the BitSequencesStates object based on the given index.

        __setitem__(index: int | Sequence[int] | Sequence[bool], states: BitSequencesStates) -> None:
            Sets a subset of the BitSequencesStates object based on the given index and states.

        flatten() -> BitSequencesStates:
            Flattens the BitSequencesStates object.

        extend(other: BitSequencesStates) -> None:
            Extends the current BitSequencesStates object with another BitSequencesStates object.

        extend_with_sf(required_first_dim: int) -> None:
            Extends the current BitSequencesStates object with sink states.

        stack_states(cls, states: List[BitSequencesStates]) -> BitSequencesStates:
            Stacks a list of BitSequencesStates objects into a single BitSequencesStates object.

        to_str() -> List[str]:
            Converts the BitSequencesStates object to a list of binary strings.
    """

    word_size: ClassVar[int]

    def __init__(
        self,
        tensor: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        forward_masks: Optional[torch.Tensor] = None,
        backward_masks: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            tensor, forward_masks=forward_masks, backward_masks=backward_masks
        )
        if length is None:
            length = torch.zeros(
                self.batch_shape, dtype=torch.long, device=self.__class__.device
            )
        assert length.dtype == torch.long  # pyright: ignore
        self.length = length

    def clone(self) -> BitSequenceStates:
        return self.__class__(
            self.tensor.detach().clone(),
            self.length,
            self.forward_masks,
            self.backward_masks,
        )

    def _check_both_forward_backward_masks_exist(self):
        assert self.forward_masks is not None and self.backward_masks is not None

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> BitSequenceStates:
        states = self.tensor[index]
        self._check_both_forward_backward_masks_exist()
        length = self.length[index]  # pyright: ignore
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        out = self.__class__(states, length, forward_masks, backward_masks)
        if self._log_rewards is not None:
            log_probs = self._log_rewards[index]
            out.log_rewards = log_probs
        return out

    def __setitem__(
        self, index: int | Sequence[int] | Sequence[bool], states: BitSequenceStates
    ) -> None:
        super().__setitem__(index, states)
        self.length[index] = states.length  # pyright: ignore

    def flatten(self) -> BitSequenceStates:
        states = self.tensor.view(-1, *self.state_shape)
        length = self.length.view(-1, self.length.shape[-1])  # pyright: ignore
        self._check_both_forward_backward_masks_exist()
        forward_masks = self.forward_masks.view(-1, self.forward_masks.shape[-1])
        backward_masks = self.backward_masks.view(-1, self.backward_masks.shape[-1])
        return self.__class__(states, length, forward_masks, backward_masks)

    def extend(self, other: BitSequenceStates) -> None:
        super().extend(other)
        self.length = torch.cat(
            (self.length, other.length), dim=len(self.batch_shape) - 1
        )

    def extend_with_sf(self, required_first_dim: int) -> None:
        super().extend_with_sf(required_first_dim)

        def _extend(masks, first_dim):
            return torch.cat(
                (
                    masks,
                    torch.ones(
                        first_dim - masks.shape[0],
                        *masks.shape[1:],
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ),
                dim=0,
            )

        self.length = _extend(self.length, required_first_dim)

    @classmethod
    def stack_states(cls, states: List[BitSequenceStates]):
        stacked_states: BitSequenceStates = super().stack_states(
            states  # pyright: ignore
        )
        stacked_states.length = torch.stack(
            [s.length for s in states], dim=0  # pyright: ignore
        )
        return stacked_states

    def to_str(self) -> List[str]:
        """
        Converts the tensor to a list of binary strings.

        The tensor is reshaped according to the state shape and then each row is
        converted to a binary string, ignoring entries with a value of -1.

        Returns:
            List[str]: A list of binary strings representing the tensor.
        """
        tensor = self.tensor.view(-1, *self.state_shape)
        mask = tensor != -1

        def row_to_binary_string(row, row_mask):
            valid_entries = row[row_mask]
            return "".join(
                format(x.item(), f"0{self.word_size}b") for x in valid_entries
            )

        return [
            row_to_binary_string(tensor[i], mask[i]) for i in range(tensor.shape[0])
        ]
