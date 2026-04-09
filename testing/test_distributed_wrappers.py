"""Tests for backend-agnostic collective wrappers in gfn.utils.distributed.

These tests run in a single process without initializing any distributed
backend, so they can only exercise:
  - error paths (unknown backend)
  - the torch backend via a minimal single-process process group
  - basic API contracts (return types, shapes)

Full multi-process / multi-rank tests require MPI or torch.distributed
launchers and are not feasible in standard CI.
"""

import pytest
import torch
import torch.distributed as dist

from gfn.utils.distributed import (
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    get_rank,
    get_world_size,
    recv,
    send,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_pg():
    """Initialize a single-process torch.distributed process group for testing.

    Uses the gloo backend (CPU-only, no GPU required). Torn down after the
    module's tests complete.
    """
    if dist.is_initialized():
        yield None
        return

    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=0,
        world_size=1,
    )
    yield None
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Unknown backend error tests (no distributed init needed)
# ---------------------------------------------------------------------------


class TestUnknownBackend:
    """Every wrapper should raise ValueError for an unrecognized backend."""

    def test_send_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            send(torch.zeros(1), dst_rank=0, backend="nccl_fake")

    def test_recv_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            recv(src_rank=0, backend="nccl_fake")

    def test_barrier_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            barrier(backend="nccl_fake")

    def test_get_rank_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_rank(backend="nccl_fake")

    def test_get_world_size_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_world_size(backend="nccl_fake")

    def test_all_reduce_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            all_reduce(torch.zeros(1), backend="nccl_fake")

    def test_all_gather_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            all_gather([torch.zeros(1)], torch.zeros(1), backend="nccl_fake")

    def test_broadcast_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            broadcast(torch.zeros(1), src=0, backend="nccl_fake")


# ---------------------------------------------------------------------------
# Torch backend single-process tests
# ---------------------------------------------------------------------------


class TestTorchBackendSingleProcess:
    """Tests using a single-process gloo process group (world_size=1)."""

    def test_get_rank(self, torch_pg):
        assert get_rank(backend="torch") == 0

    def test_get_world_size(self, torch_pg):
        assert get_world_size(backend="torch") == 1

    def test_barrier(self, torch_pg):
        # Should not raise.
        barrier(backend="torch")

    def test_all_reduce_sum(self, torch_pg):
        t = torch.tensor([3.0, 4.0])
        all_reduce(t, op="SUM", backend="torch")
        # world_size=1, so all_reduce is identity.
        assert torch.allclose(t, torch.tensor([3.0, 4.0]))

    def test_all_reduce_max(self, torch_pg):
        t = torch.tensor([1.0, 5.0, 3.0])
        all_reduce(t, op="MAX", backend="torch")
        assert torch.allclose(t, torch.tensor([1.0, 5.0, 3.0]))

    def test_all_reduce_min(self, torch_pg):
        t = torch.tensor([1.0, 5.0, 3.0])
        all_reduce(t, op="MIN", backend="torch")
        assert torch.allclose(t, torch.tensor([1.0, 5.0, 3.0]))

    def test_all_gather(self, torch_pg):
        t = torch.tensor([1.0, 2.0])
        output = [torch.zeros(2)]
        all_gather(output, t, backend="torch")
        assert torch.allclose(output[0], t)

    def test_broadcast(self, torch_pg):
        t = torch.tensor([7.0, 8.0])
        broadcast(t, src=0, backend="torch")
        assert torch.allclose(t, torch.tensor([7.0, 8.0]))

    # NOTE: send/recv cannot be tested in a single-process gloo group — gloo
    # does not support self-sends and segfaults when attempted from threads.
    # Full send/recv testing requires a multi-process launcher (e.g. mpirun).


# ---------------------------------------------------------------------------
# MPI4py backend single-process tests
# ---------------------------------------------------------------------------

try:
    import mpi4py  # noqa: F401  # pyright: ignore[reportUnusedImport]

    _mpi4py_available = True
except (ImportError, RuntimeError):
    _mpi4py_available = False


@pytest.mark.skipif(not _mpi4py_available, reason="mpi4py not available")
class TestMPI4pyBackendSingleProcess:
    """Tests using mpi4py with COMM_WORLD (world_size=1 without mpirun)."""

    def test_get_rank(self):
        assert get_rank(backend="mpi") == 0

    def test_get_world_size(self):
        assert get_world_size(backend="mpi") == 1

    def test_barrier(self):
        # Should not raise.
        barrier(backend="mpi")

    def test_all_reduce_sum(self):
        t = torch.tensor([3.0, 4.0])
        all_reduce(t, op="SUM", backend="mpi")
        # world_size=1, so all_reduce is identity.
        assert torch.allclose(t, torch.tensor([3.0, 4.0]))

    def test_all_reduce_max(self):
        t = torch.tensor([1.0, 5.0, 3.0])
        all_reduce(t, op="MAX", backend="mpi")
        assert torch.allclose(t, torch.tensor([1.0, 5.0, 3.0]))

    def test_all_reduce_min(self):
        t = torch.tensor([1.0, 5.0, 3.0])
        all_reduce(t, op="MIN", backend="mpi")
        assert torch.allclose(t, torch.tensor([1.0, 5.0, 3.0]))

    def test_all_gather(self):
        t = torch.tensor([1.0, 2.0])
        output = [torch.zeros(2)]
        all_gather(output, t, backend="mpi")
        assert torch.allclose(output[0], t)

    def test_broadcast(self):
        t = torch.tensor([7.0, 8.0])
        broadcast(t, src=0, backend="mpi")
        assert torch.allclose(t, torch.tensor([7.0, 8.0]))

    # NOTE: send/recv cannot be tested in a single-process MPI communicator —
    # self-sends with blocking Send/Recv will deadlock or behave unexpectedly.
    # Full send/recv testing requires a multi-process launcher (e.g. mpirun).
