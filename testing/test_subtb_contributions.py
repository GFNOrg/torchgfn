from typing import cast

import pytest
import torch

from gfn.containers import Trajectories
from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet


class DummyTrajectories:
    def __init__(self, terminating_idx: torch.Tensor):
        self.terminating_idx = terminating_idx
        self.max_length = int(torch.max(terminating_idx).item())

    def __len__(self) -> int:
        return self.terminating_idx.numel()


def _reference_contributions(
    lam: float | torch.Tensor,
    terminating_idx: torch.Tensor,
    max_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    lam64 = torch.as_tensor(lam, dtype=torch.float64, device=device)
    t_idx64 = terminating_idx.to(dtype=torch.float64)

    base = lam64.pow(torch.arange(max_len, device=device, dtype=torch.float64))
    base = base.unsqueeze(-1).repeat(1, len(terminating_idx))
    base = base.repeat_interleave(
        torch.arange(max_len, 0, -1, device=device),
        dim=0,
        output_size=int(max_len * (max_len + 1) / 2),
    )

    denom = []
    for n in t_idx64.long().tolist():
        ar = torch.arange(n, device=device, dtype=torch.float64)
        denom.append(((n - ar) * lam64.pow(ar)).sum())
    denom = torch.stack(denom)

    return (base / denom / len(terminating_idx)).to(dtype)


def test_geometric_within_contributions_matches_reference():
    terminating_idx = torch.tensor([1, 2, 3, 5], dtype=torch.int64)
    trajectories = DummyTrajectories(terminating_idx)

    model = object.__new__(SubTBGFlowNet)
    model.lamda = 0.9

    result = model.get_geometric_within_contributions(
        cast(Trajectories, trajectories)  # type: ignore[arg-type]
    )
    reference = _reference_contributions(
        lam=model.lamda,
        terminating_idx=terminating_idx,
        max_len=trajectories.max_length,
        dtype=result.dtype,
        device=terminating_idx.device,
    )

    torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-6)


def test_geometric_within_contributions_near_one_is_stable():
    terminating_idx = torch.tensor([2, 4, 6], dtype=torch.int64)
    trajectories = DummyTrajectories(terminating_idx)

    model_near_one = object.__new__(SubTBGFlowNet)
    model_near_one.lamda = 1.0 - 1e-4

    model_exact_one = object.__new__(SubTBGFlowNet)
    model_exact_one.lamda = 1.0

    result_near_one = model_near_one.get_geometric_within_contributions(
        cast(Trajectories, trajectories)  # type: ignore[arg-type]
    )
    result_exact_one = model_exact_one.get_geometric_within_contributions(
        cast(Trajectories, trajectories)  # type: ignore[arg-type]
    )

    assert torch.isfinite(result_near_one).all()
    assert torch.isfinite(result_exact_one).all()
    torch.testing.assert_close(result_near_one, result_exact_one, rtol=5e-3, atol=5e-5)


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.7, 0.9, 0.999, 1.0])
def test_geometric_within_contributions_matches_bruteforce(lam: float):
    torch.manual_seed(0)
    terminating_idx = torch.randint(1, 7, size=(6,), dtype=torch.int64)
    trajectories = DummyTrajectories(terminating_idx)

    model = object.__new__(SubTBGFlowNet)
    model.lamda = lam

    result = model.get_geometric_within_contributions(
        cast(Trajectories, trajectories)  # type: ignore[arg-type]
    )
    reference = _reference_contributions(
        lam=model.lamda,
        terminating_idx=terminating_idx,
        max_len=trajectories.max_length,
        dtype=result.dtype,
        device=terminating_idx.device,
    )

    torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-6)
