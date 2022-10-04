from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Sequence

import torch
from torchtyping import TensorType

# Typing
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
DonesTensor = TensorType["batch_shape", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]


class States(ABC):
    """Base class for states, seen as nodes of the DAG.
    Each environment/task should have its own States subclass.
    In essence, a States object is a container of three tensors:
    - states: a torch tensor of shape (*batch_shape , *state_shape)
    - forward_masks: a torch tensor of shape (*batch_shape , n_actions)
    - backward_masks: a torch tensor of shape (*batch_shape , (n_actions - 1))
    """

    n_actions: ClassVar[int | None] = None
    s_0: ClassVar[OneStateTensor | None] = None  # Represents the state s_0
    s_f: ClassVar[
        OneStateTensor | None
    ] = None  # Represents the state s_f, for example it can be torch.tensor([-1., -1., ...])
    state_shape: ClassVar[tuple[int] | None] = None
    state_ndim: ClassVar[int | None] = None
    device = torch.device(
        "cpu"
    )  # if s_0 is on another device, this will change automatically

    def __init_subclass__(cls, **kwargs) -> None:
        if getattr(cls, "n_actions") is None:
            raise ValueError("n_actions must be specified")
        if not isinstance(getattr(cls, "n_actions"), int):
            raise ValueError("n_actions must be an integer")
        if getattr(cls, "s_0") is None:
            raise ValueError("s_0 must be specified")
        if isinstance(getattr(cls, "s_0"), torch.Tensor):
            state_shape = tuple(getattr(cls, "s_0").shape)
            setattr(cls, "state_shape", state_shape)
            setattr(cls, "state_ndim", len(state_shape))
            setattr(cls, "device", getattr(cls, "s_0").device)
        else:
            raise ValueError("s_0 must be a torch tensor")
        if getattr(cls, "s_f") is None:
            setattr(cls, "s_f", torch.full_like(getattr(cls, "s_0"), -float("inf")))
        elif isinstance(getattr(cls, "s_f"), torch.Tensor):
            if not getattr(cls, "s_0").shape == getattr(cls, "s_f").shape:
                raise TypeError(f"{cls.__name__}' s_0 and s_f must have the same shape")
        else:
            raise ValueError("s_f must be a torch tensor, or unspecified")

        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        states: StatesTensor | None = None,
        random_init: bool = False,
        batch_shape: tuple[int] | None = None,
        **kwargs,
    ) -> None:
        r"""Initialize the states.
        If states is not None,  then random_init and batch_shape are ignored.
        If states is None, then the initialized state is either the
        initial state $s_0$ or random states. In this case, batch_shape cannot be None.
        If random_init is True, then the states are initialized randomly.
        The `forward_mask` and `backward_mask` attributes can be passed as keyword arguments.
        """
        if states is None:
            if batch_shape is None:
                batch_shape = (0,)

            self.batch_shape = batch_shape
            if random_init:
                self.states = self.make_random_states_tensor(batch_shape)
            else:
                self.states = self.make_initial_states_tensor(batch_shape)
        else:
            self.states = states
            shape = tuple(self.states.shape)
            assert shape[-self.__class__.state_ndim :] == self.__class__.state_shape  # type: ignore
            self.batch_shape = shape[: -self.__class__.state_ndim]  # type: ignore
            assert states.dtype == torch.float

        self.device = self.states.device
        assert self.device == self.__class__.device

        self.forward_masks: ForwardMasksTensor
        self.backward_masks: BackwardMasksTensor
        if "forward_masks" in kwargs and "backward_masks" in kwargs:
            self.forward_masks = kwargs["forward_masks"]
            self.backward_masks = kwargs["backward_masks"]
        else:
            self.make_masks()

    def __repr__(self):
        return f"""{self.__class__.__name__} object of batch shape {self.batch_shape}
        and state shape {self.state_shape}"""

    def __getitem__(self, index: int | Sequence[int]) -> States:
        # TODO: add test for this method
        states = self.states[index]
        forward_masks = self.forward_masks[index]
        backward_masks = self.backward_masks[index]
        return self.__class__(
            states, forward_masks=forward_masks, backward_masks=backward_masks
        )

    def flatten(self) -> States:
        """Flatten the batch dimension of the states.
        This is useful for example when extracting individual states from trajectories.
        """
        states = self.states.view(-1, *self.state_shape)
        forward_masks = self.forward_masks.view(-1, self.forward_masks.shape[-1])
        backward_masks = self.backward_masks.view(-1, self.backward_masks.shape[-1])
        return self.__class__(
            states, forward_masks=forward_masks, backward_masks=backward_masks
        )

    def extend(self, other: States) -> None:
        other_batch_shape = other.batch_shape
        if len(other_batch_shape) == len(self.batch_shape) == 1:
            self.batch_shape = (self.batch_shape[0] + other_batch_shape[0],)
            self.states = torch.cat((self.states, other.states), dim=0)
            self.forward_masks = torch.cat(
                (self.forward_masks, other.forward_masks), dim=0
            )
            self.backward_masks = torch.cat(
                (self.backward_masks, other.backward_masks), dim=0
            )
        elif len(other_batch_shape) == len(self.batch_shape) == 2:
            self.extend_with_sf(
                required_first_dim=max(self.batch_shape[0], other_batch_shape[0])
            )
            other.extend_with_sf(
                required_first_dim=max(self.batch_shape[0], other_batch_shape[0])
            )
            self.batch_shape = (
                self.batch_shape[0],
                self.batch_shape[1] + other_batch_shape[1],
            )
            self.states = torch.cat((self.states, other.states), dim=1)
            self.forward_masks = torch.cat(
                (self.forward_masks, other.forward_masks), dim=1
            )
            self.backward_masks = torch.cat(
                (self.backward_masks, other.backward_masks), dim=1
            )
        else:
            raise ValueError(
                f"extend is not implemented for batch shapes {self.batch_shape} and {other_batch_shape}"
            )
        # free memory
        del other

    def extend_with_sf(self, required_first_dim: int) -> None:
        """Takes a two-dimensional batch of states (i.e. of batch_shape (a, b)),
        and extends it to a States object of batch_shape (required_first_dim, b),
        by adding the required number of sink_states. This is useful to extend trajectories
        of different lengths."""
        if len(self.batch_shape) == 2:
            if self.batch_shape[0] >= required_first_dim:
                return
            self.states = torch.cat(
                (
                    self.states,
                    self.__class__.s_f.repeat(  # type: ignore
                        required_first_dim - self.batch_shape[0], self.batch_shape[1], 1
                    ),
                ),
                dim=0,
            )
            self.forward_masks = torch.cat(
                (
                    self.forward_masks,
                    torch.ones(
                        (
                            required_first_dim - self.batch_shape[0],
                            *self.forward_masks.shape[1:],
                        ),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ),
                dim=0,
            )
            self.backward_masks = torch.cat(
                (
                    self.backward_masks,
                    torch.ones(
                        (
                            required_first_dim - self.batch_shape[0],
                            *self.backward_masks.shape[1:],
                        ),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ),
                dim=0,
            )
            self.batch_shape = (required_first_dim, self.batch_shape[1])
        else:
            raise ValueError(
                f"extend_with_sf is not implemented for batch shapes {self.batch_shape}"
            )

    @classmethod
    def make_initial_states_tensor(cls, batch_shape: tuple[int]) -> StatesTensor:
        assert cls.s_0 is not None and cls.state_ndim is not None
        return cls.s_0.repeat(*batch_shape, *((1,) * cls.state_ndim))

    @classmethod
    @abstractmethod
    def make_random_states_tensor(cls, batch_shape: tuple[int]) -> StatesTensor:
        pass

    @property
    def is_initial_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_0$ of the DAG.
        """
        assert self.__class__.state_ndim is not None
        out = self.states == self.make_initial_states_tensor(self.batch_shape)
        for _ in range(self.__class__.state_ndim):
            out = out.all(dim=-1)
        return out

    @property
    def is_sink_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_f$ of the DAG.
        """
        assert self.__class__.state_ndim is not None and self.__class__.s_f is not None
        sink_states = self.__class__.s_f.repeat(
            *self.batch_shape, *((1,) * self.__class__.state_ndim)
        )
        out = self.states == sink_states

        for _ in range(self.__class__.state_ndim):
            out = out.all(dim=-1)
        return out

    def make_masks(self) -> ForwardMasksTensor:
        assert self.__class__.n_actions is not None
        self.forward_masks = torch.ones(
            (*self.batch_shape, self.__class__.n_actions),
            dtype=torch.bool,
            device=self.device,
        )
        self.backward_masks = torch.ones(
            (*self.batch_shape, self.__class__.n_actions - 1),
            dtype=torch.bool,
            device=self.device,
        )
        self.update_masks()

    @abstractmethod
    def update_masks(self) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save(
            {
                "states": self.states,
                "forward_masks": self.forward_masks,
                "backward_masks": self.backward_masks,
            },
            path,
        )

    def load(self, path: str) -> None:
        loaded = torch.load(path)
        self.states = loaded["states"]
        self.forward_masks = loaded["forward_masks"]
        self.backward_masks = loaded["backward_masks"]
        shape = tuple(self.states.shape)
        assert shape[-self.__class__.state_ndim :] == self.__class__.state_shape  # type: ignore
        self.batch_shape = shape[: -self.__class__.state_ndim]  # type: ignore

    def copy(self) -> States:
        return self.__class__(
            self.states.clone(),
            self.forward_masks.clone(),
            self.backward_masks.clone(),
        )


def make_States_class(
    class_name: str,
    n_actions: int,
    s_0: OneStateTensor,
    s_f: OneStateTensor | None,
    make_random_states_tensor: Callable[[Any, tuple[int]], StatesTensor],
    update_masks: Callable[[States], None],
) -> type[States]:
    """
    Creates a States subclass with the given state_shape and forward/backward mask makers.
    """
    return type(
        class_name,
        (States,),
        {
            "n_actions": n_actions,
            "s_0": s_0,
            "s_f": s_f,
            "make_random_states_tensor": make_random_states_tensor,
            "update_masks": update_masks,
        },
    )
