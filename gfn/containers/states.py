from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from torchtyping import TensorType
import torch
from typing import Tuple, Union, ClassVar, Callable, Any


# Typing
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
StatesTensor = TensorType["shape", torch.float]
DonesTensor = TensorType["batch_shape", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]


class StatesMetaClass(ABCMeta):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if "s_0" in attrs:
            cls.s_0 = attrs["s_0"]

    @property
    def state_shape(cls) -> Tuple[int]:
        return tuple(cls.s_0.shape)

    @property
    def state_ndim(cls) -> int:
        return len(cls.state_shape)


class States(metaclass=StatesMetaClass):
    """Base class for states, seen as nodes of the DAG.
    Each environment/task should have its own States subclass.
    States are represented as torch tensors of shape=(*batch_shape , *state_shape).
    """

    n_actions: ClassVar[int]
    s_0: ClassVar[OneStateTensor]  # Represents the state s_0
    s_f: ClassVar[
        OneStateTensor
    ]  # Represents the state s_f, for example it can be torch.tensor([-1., -1., ...])

    def __init__(
        self,
        states: Union[StatesTensor, None],
        random_init: bool = False,
        batch_shape: Union[Tuple[int], None] = None,
        is_initial: Union[DonesTensor, None] = None,
        is_sink: Union[DonesTensor, None] = None,
    ) -> None:
        r"""Initialize the states.
        If states is not None,  then random_init and batch_shape are ignored.
        If states is None, then the initialized state is either the
        initial state $s_0$ or random states. In this case, batch_shape cannot be None.
        If random_init is True, then the states are initialized randomly.
        """
        if states is None:
            assert (
                batch_shape is not None
            ), "batch_shape cannot be None if states is None"
            assert (
                is_initial is None and is_sink is None
            ), "when states is None, is_initial, is_sink cannot be None"

            self.batch_shape = batch_shape
            if random_init:
                self.states = self.make_random_states(batch_shape)
                self.is_initial = self.is_initial_state()
                self.is_sink = self.is_sink_state()
            else:
                self.states = self.make_initial_states(batch_shape)
                self.is_initial = torch.ones(batch_shape, dtype=torch.bool).to(
                    self.__class__.s_0.device
                )
                self.is_sink = torch.zeros(batch_shape, dtype=torch.bool).to(
                    self.__class__.s_0.device
                )
        else:
            self.states = states
            shape = tuple(self.states.shape)
            assert shape[-self.__class__.state_ndim :] == self.__class__.state_shape
            self.batch_shape = shape[: -self.__class__.state_ndim]
            if is_initial is None:
                self.is_initial = self.is_initial_state()
            else:
                self.is_initial = is_initial
            if is_sink is None:
                self.is_sink = self.is_sink_state()
            else:
                self.is_sink = is_sink

        self.device = self.states.device
        assert self.device == self.__class__.s_0.device

        self.forward_masks: ForwardMasksTensor
        self.backward_masks: BackwardMasksTensor
        self.make_masks()

    def __repr__(self):
        return f"""States(states={self.states},\n forward_masks={self.forward_masks},
        \n backward_masks={self.backward_masks})
        """

    @classmethod
    def make_initial_states(cls, batch_shape: Tuple[int]) -> StatesTensor:
        return cls.s_0.repeat(*batch_shape, *((1,) * cls.state_ndim))

    def is_initial_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_0$ of the DAG.
        """
        out = self.states == self.make_initial_states(self.batch_shape)
        for _ in range(self.__class__.state_ndim):
            out = out.all(dim=-1)
        return out

    def is_sink_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_f$ of the DAG.
        """
        sink_states = self.s_f.repeat(
            *self.batch_shape, *((1,) * self.__class__.state_ndim)
        )
        out = self.states == sink_states
        for _ in range(self.__class__.state_ndim):
            out = out.all(dim=-1)
        return out

    @classmethod
    @abstractmethod
    def make_random_states(cls, batch_shape: Tuple[int]) -> StatesTensor:
        pass

    def make_masks(self) -> ForwardMasksTensor:
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


def make_States_class(
    n_actions: int,
    s_0: OneStateTensor,
    s_f: OneStateTensor,
    make_random_states: Callable[[Any, Tuple[int]], StatesTensor],
    update_masks: Callable[[States], None],
) -> StatesMetaClass:
    """
    Creates a States subclass with the given state_shape and forward/backward mask makers.
    """
    return StatesMetaClass(
        "States",
        (States,),
        {
            "n_actions": n_actions,
            "s_0": s_0,
            "s_f": s_f,
            "make_random_states": make_random_states,
            "update_masks": update_masks,
        },
    )
