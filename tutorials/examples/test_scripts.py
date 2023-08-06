from dataclasses import dataclass
import pytest

from .train_box import main as train_box_main
from .train_discreteebm import main as train_discreteebm_main
from .train_hypergrid import main as train_hypergrid_main


@dataclass
class CommonArgs:
    no_cuda: bool = True
    seed: int = 0
    batch_size: int = 16
    replay_buffer_size: int = 0
    loss: str = "TB"
    subTB_weighting: str = "geometric_within"
    subTB_lambda: float = 0.9
    tabular: bool = False
    uniform_pb: bool = False
    tied: bool = False
    hidden_dim: int = 256
    n_hidden: int = 2
    lr: float = 1e-3
    lr_Z: float = 1e-1
    n_trajectories: int = 32000
    validation_interval: int = 100
    validation_samples: int = 200000
    wandb_project: str = ""


@dataclass
class DiscreteEBMArgs(CommonArgs):
    ndim: int = 4
    alpha: float = 1.0


@dataclass
class HypergridArgs(CommonArgs):
    ndim: int = 2
    height: int = 8
    R0: float = 0.1
    R1: float = 0.5
    R2: float = 2.0


@dataclass
class BoxArgs(CommonArgs):
    delta: float = 0.25
    min_concentration: float = 0.1
    max_concentration: float = 5.1
    n_components: int = 2
    n_components_s0: int = 2
    gamma_scheduler: float = 0.5
    scheduler_milestone: int = 2500
    lr_F: float = 1e-2


@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("height", [8, 16])
def test_hypergrid(ndim: int, height: int):
    n_trajectories = 32000 if ndim == 2 else 16000
    args = HypergridArgs(ndim=ndim, height=height, n_trajectories=n_trajectories)
    final_l1_dist = train_hypergrid_main(args)
    if ndim == 2 and height == 8:
        assert final_l1_dist < 1e-3
    elif ndim == 2 and height == 16:
        assert final_l1_dist < 6e-4
    elif ndim == 4 and height == 8:
        assert final_l1_dist < 2e-4
    elif ndim == 4 and height == 16:
        assert final_l1_dist < 3e-5


@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("alpha", [0.1, 1.0])
def test_discreteebm(ndim: int, alpha: float):
    n_trajectories = 16000
    args = DiscreteEBMArgs(ndim=ndim, alpha=alpha, n_trajectories=n_trajectories)
    final_l1_dist = train_discreteebm_main(args)
    if ndim == 2 and alpha == 0.1:
        assert final_l1_dist < 6e-3
    elif ndim == 2 and alpha == 1.0:
        assert final_l1_dist < 3e-2
    elif ndim == 4 and alpha == 0.1:
        assert final_l1_dist < 9e-3
    elif ndim == 4 and alpha == 1.0:
        assert final_l1_dist < 7e-2


@pytest.mark.parametrize("delta", [0.1, 0.25])
@pytest.mark.parametrize("loss", ["TB", "DB"])
def test_box(delta: float, loss: str):
    n_trajectories = 128128
    validation_interval = 500
    validation_samples = 10000
    args = BoxArgs(
        delta=delta,
        loss=loss,
        n_trajectories=n_trajectories,
        hidden_dim=128,
        n_hidden=4,
        batch_size=128,
        lr_Z=1e-3,
        validation_interval=validation_interval,
        validation_samples=validation_samples,
    )
    print(args)
    final_jsd = train_box_main(args)
    if loss == "TB" and delta == 0.1:
        assert final_jsd < 7e-2
    elif loss == "DB" and delta == 0.1:
        assert final_jsd < 0.2
    if loss == "TB" and delta == 0.25:
        assert final_jsd < 1e-3
    elif loss == "DB" and delta == 0.25:
        assert final_jsd < 4e-2
