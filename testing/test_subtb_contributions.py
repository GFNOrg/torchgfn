from typing import cast

import pytest
import torch

from gfn.containers import Trajectories
from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet, _assert_normalized


class DummyTrajectories:
    def __init__(self, terminating_idx: torch.Tensor, max_length: int | None = None):
        self.terminating_idx = terminating_idx
        longest = int(torch.max(terminating_idx).item())
        self.max_length = longest if max_length is None else max_length

    def __len__(self) -> int:
        return self.terminating_idx.numel()


class _SubTBStub:
    """Minimal stub exposing the geometric contribution methods without full init."""

    get_geometric_within_contributions = SubTBGFlowNet.get_geometric_within_contributions
    get_geometric_contributions = SubTBGFlowNet.get_geometric_contributions

    def __init__(self, lamda: float):
        self.lamda = lamda


def _within(model: _SubTBStub, trajectories: DummyTrajectories) -> torch.Tensor:
    return model.get_geometric_within_contributions(
        cast(Trajectories, trajectories)  # type: ignore[arg-type]
    )


def _geometric(model: _SubTBStub, trajectories: DummyTrajectories) -> torch.Tensor:
    return model.get_geometric_contributions(
        cast(Trajectories, trajectories)  # type: ignore[arg-type]
    )


def _existing_mask(terminating_idx: torch.Tensor, max_len: int) -> torch.Tensor:
    """True for contribution entries whose sub-trajectory exists (as in get_scores)."""
    return torch.cat(
        [
            ~terminating_idx.lt(torch.arange(i, max_len + 1).unsqueeze(-1))
            for i in range(1, max_len + 1)
        ]
    )


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


def _assert_matches_reference(
    result: torch.Tensor, lam: float, terminating_idx: torch.Tensor, max_len: int
) -> None:
    """Compares the entries loss() keeps; the others are never used."""
    reference = _reference_contributions(
        lam=lam,
        terminating_idx=terminating_idx,
        max_len=max_len,
        dtype=result.dtype,
        device=terminating_idx.device,
    )
    existing = _existing_mask(terminating_idx, max_len)
    torch.testing.assert_close(
        result[existing], reference[existing], rtol=1e-5, atol=1e-7
    )


def test_geometric_within_contributions_matches_reference():
    terminating_idx = torch.tensor([1, 2, 3, 5], dtype=torch.int64)
    trajectories = DummyTrajectories(terminating_idx)

    model = _SubTBStub(lamda=0.9)

    result = _within(model, trajectories)
    _assert_matches_reference(
        result, model.lamda, terminating_idx, trajectories.max_length
    )


def test_geometric_within_contributions_near_one_is_stable():
    terminating_idx = torch.tensor([2, 4, 6], dtype=torch.int64)
    trajectories = DummyTrajectories(terminating_idx)

    model_near_one = _SubTBStub(lamda=1.0 - 1e-4)
    model_exact_one = _SubTBStub(lamda=1.0)

    result_near_one = _within(model_near_one, trajectories)
    result_exact_one = _within(model_exact_one, trajectories)

    assert torch.isfinite(result_near_one).all()
    assert torch.isfinite(result_exact_one).all()
    torch.testing.assert_close(result_near_one, result_exact_one, rtol=5e-3, atol=5e-5)


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.7, 0.9, 0.999, 1.0, 1.001, 1.5, 3.0])
def test_geometric_within_contributions_matches_bruteforce(lam: float):
    torch.manual_seed(0)
    terminating_idx = torch.randint(1, 7, size=(6,), dtype=torch.int64)
    trajectories = DummyTrajectories(terminating_idx)

    result = _within(_SubTBStub(lamda=lam), trajectories)
    _assert_matches_reference(result, lam, terminating_idx, trajectories.max_length)


@pytest.mark.parametrize(
    ("lam", "max_len"), [(1.5, 219), (2.0, 1000), (1.1, 1000), (0.5, 1000)]
)
def test_geometric_within_contributions_long_trajectories(lam: float, max_len: int):
    """lamda ** max_len overflows float32 here; the contributions must not."""
    terminating_idx = torch.tensor([1, 2, max_len // 2, max_len], dtype=torch.int64)
    trajectories = DummyTrajectories(terminating_idx)

    result = _within(_SubTBStub(lamda=lam), trajectories)

    assert torch.isfinite(result).all()
    _assert_matches_reference(result, lam, terminating_idx, max_len)
    existing = _existing_mask(terminating_idx, max_len)
    per_trajectory = (result * existing).sum(0)
    torch.testing.assert_close(
        per_trajectory, torch.full_like(per_trajectory, 1 / len(terminating_idx))
    )


@pytest.mark.parametrize(
    ("lam", "max_len"),
    [(0.0, 8), (0.5, 8), (0.9, 8), (1.0, 8), (1.5, 8), (1.5, 219), (2.0, 1000)],
)
def test_geometric_contributions_match_mean_of_per_length_means(
    lam: float, max_len: int
):
    """'geometric' is the lamda ** k weighted mean of the per-length mean losses."""
    torch.manual_seed(0)
    terminating_idx = torch.randint(1, max_len + 1, size=(7,), dtype=torch.int64)
    terminating_idx[0] = max_len
    trajectories = DummyTrajectories(terminating_idx)
    existing = _existing_mask(terminating_idx, max_len)
    losses = torch.rand(existing.shape, dtype=torch.float64)

    result = _geometric(_SubTBStub(lamda=lam), trajectories)

    assert torch.isfinite(result).all()
    _assert_normalized(result[existing], "geometric contributions")

    # xlogy gives 0 * log(0) = 0, i.e. lamda ** 0 == 1 even for lamda == 0.
    log_weights = torch.xlogy(
        torch.arange(max_len, dtype=torch.float64),
        torch.tensor(lam, dtype=torch.float64),
    )
    weights = torch.softmax(log_weights, 0)
    row_ends = torch.arange(max_len, 0, -1).cumsum(0)
    row_starts = row_ends - torch.arange(max_len, 0, -1)
    per_length_means = torch.stack(
        [
            losses[start:end][existing[start:end]].mean()
            for start, end in zip(row_starts.tolist(), row_ends.tolist())
        ]
    )
    expected = (weights * per_length_means).sum()

    actual = (result[existing].double() * losses[existing]).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("method", [_geometric, _within])
@pytest.mark.parametrize("lam", [0.0, 0.9, 1.0, 1.5])
def test_geometric_contributions_ignore_padding(method, lam: float):
    """Padding the container beyond its longest trajectory changes nothing.

    Lengths beyond the longest trajectory have no sub-trajectories. Before, the
    'geometric' weighting averaged over those empty lengths and returned NaN.
    """
    terminating_idx = torch.tensor([2, 3, 5], dtype=torch.int64)
    model = _SubTBStub(lamda=lam)
    unpadded = method(model, DummyTrajectories(terminating_idx))
    padded = method(model, DummyTrajectories(terminating_idx, max_length=9))

    assert torch.isfinite(padded).all()
    torch.testing.assert_close(
        padded[_existing_mask(terminating_idx, 9)],
        unpadded[_existing_mask(terminating_idx, 5)],
    )


@pytest.mark.parametrize(
    ("dtype", "n"),
    [
        (torch.float32, 10_000),
        (torch.bfloat16, 300),
        (torch.float16, 300),
        # 1 / n underflows to a float16 subnormal, so rounding is absolute.
        (torch.float16, 800_000),
    ],
)
def test_assert_normalized_accepts_rounded_weights(dtype: torch.dtype, n: int):
    """Correct weights rounded to a low-precision dtype pass the debug check."""
    _assert_normalized(torch.full((n,), 1 / n).to(dtype), "uniform weights")
    geometric = torch.softmax(torch.arange(n, dtype=torch.float64) * -1e-3, 0)
    _assert_normalized(geometric.to(dtype), "geometric weights")


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (torch.tensor([0.5, float("nan"), 0.5]), "non-finite"),
        (torch.tensor([0.5, float("inf"), 0.5]), "non-finite"),
        (torch.tensor([1.5, -0.5]), "negative"),
        (torch.tensor([0.5, 0.4]), "sum to"),
        (torch.tensor([0.5, 0.5]).to(torch.bfloat16) * 1.1, "sum to"),
    ],
)
def test_assert_normalized_rejects_invalid_weights(weights: torch.Tensor, match: str):
    with pytest.raises(AssertionError, match=match):
        _assert_normalized(weights, "weights")
