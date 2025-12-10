import os

import pytest
import torch

from gfn.gym.bitSequence import BitSequence
from gfn.gym.hypergrid import HyperGrid
from gfn.gym.line import Line
from gfn.gym.perfect_tree import PerfectBinaryTree
from gfn.gym.set_addition import SetAddition

RUN_COMPILE_TESTS = os.environ.get("TORCHGFN_ENABLE_COMPILE_TESTS", "0") == "1"
ENABLE_MPS = os.environ.get("TORCHGFN_ENABLE_MPS_COMPILE_TESTS", "0") == "1"

if not RUN_COMPILE_TESTS:
    pytest.skip(
        "Set TORCHGFN_ENABLE_COMPILE_TESTS=1 to run torch.compile smoke tests.",
        allow_module_level=True,
    )

if not hasattr(torch, "compile"):
    pytest.skip(
        "torch.compile is unavailable in this PyTorch build", allow_module_level=True
    )


def _make_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if ENABLE_MPS and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


DEVICES = _make_devices()


def _line_setup(device: torch.device):
    env = Line(mus=[0.0], sigmas=[1.0], init_value=0.0, device=device)
    states = env.reset(batch_shape=1)
    action_tensor = torch.zeros((1, 1), device=device)
    return env, states.tensor, action_tensor


def _set_addition_setup(device: torch.device):
    def reward_fn(x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1).float()

    env = SetAddition(
        n_items=3,
        max_items=2,
        reward_fn=reward_fn,
        fixed_length=False,
        device=device,
    )
    states = env.reset(batch_shape=1)
    action_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
    return env, states.tensor, action_tensor


def _perfect_tree_setup(device: torch.device):
    def reward_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], device=x.device, dtype=torch.float32)

    env = PerfectBinaryTree(reward_fn=reward_fn, depth=3, device=device)
    states = env.reset(batch_shape=1)
    action_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
    return env, states.tensor, action_tensor


def _bit_sequence_setup(device: torch.device):
    env = BitSequence(
        word_size=2,
        seq_size=8,
        n_modes=2,
        temperature=1.0,
        device_str=str(device),
        seed=0,
    )
    states = env.reset(batch_shape=1)
    action_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
    return env, states.tensor, action_tensor


def _hypergrid_setup(device: torch.device):
    env = HyperGrid(ndim=2, height=4, device=device)
    states = env.reset(batch_shape=1)
    action_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
    return env, states.tensor, action_tensor


ENV_SETUP_FNS = [
    ("line", _line_setup),
    ("set_addition", _set_addition_setup),
    ("perfect_binary_tree", _perfect_tree_setup),
    ("bit_sequence", _bit_sequence_setup),
    ("hypergrid", _hypergrid_setup),
]


def _compiled_transition(env, step_fn, state_tensor, action_tensor):
    def fn(in_state, in_action):
        states = env.states_from_tensor(in_state.clone())
        actions = env.actions_from_tensor(in_action.clone())
        return step_fn(states, actions).tensor

    compiled_fn = torch.compile(fn, dynamic=True)
    return compiled_fn(state_tensor, action_tensor)


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: str(d))
@pytest.mark.parametrize("env_name,setup_fn", ENV_SETUP_FNS)
def test_env_steps_are_torch_compilable(env_name, setup_fn, device):
    env, state_tensor, action_tensor = setup_fn(device)

    forward_out = _compiled_transition(env, env._step, state_tensor, action_tensor)
    assert isinstance(forward_out, torch.Tensor)
    assert forward_out.device == state_tensor.device

    backward_out = _compiled_transition(
        env, env._backward_step, forward_out, action_tensor
    )
    assert isinstance(backward_out, torch.Tensor)
    assert backward_out.device == state_tensor.device
