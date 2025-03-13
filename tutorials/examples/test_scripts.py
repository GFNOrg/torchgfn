"""Includes tests for some examples in the tutorials folder.
The tests ensure that after a certain number of iterations, the final L1
distance or JSD between the learned distribution and the target distribution is
below a certain threshold.
"""

from dataclasses import dataclass

import numpy as np
import pytest

try:
    from train_box import main as train_box_main
    from train_discreteebm import main as train_discreteebm_main
    from train_hypergrid import main as train_hypergrid_main
except ImportError:
    from .train_box import main as train_box_main
    from .train_discreteebm import main as train_discreteebm_main
    from .train_hypergrid import main as train_hypergrid_main


@dataclass
class CommonArgs:
    batch_size: int = 16
    cutoff_distance: float = 0.1
    hidden_dim: int = 256
    loss: str = "TB"
    lr_Z: float = 1e-1
    lr: float = 1e-3
    n_hidden: int = 2
    n_trajectories: int = 32000
    no_cuda: bool = True
    p_norm_distance: float = 2.0
    replay_buffer_size: int = 0
    seed: int = 1  # We fix the seed for reproducibility.
    subTB_lambda: float = 0.9
    subTB_weighting: str = "geometric_within"
    tabular: bool = False
    tied: bool = False
    uniform_pb: bool = False
    validation_interval: int = 100
    validation_samples: int = 200000
    wandb_project: str = ""


@dataclass
class DiscreteEBMArgs(CommonArgs):
    ndim: int = 4
    alpha: float = 1.0


@dataclass
class HypergridArgs(CommonArgs):
    calculate_all_states: bool = True
    calculate_partition: bool = True
    distributed: bool = False
    diverse_replay_buffer: bool = False
    height: int = 8
    loss: str = "TB"
    ndim: int = 2
    profile: bool = False
    R0: float = 0.1
    R1: float = 0.5
    R2: float = 2.0
    replay_buffer_size: int = 0


@dataclass
class BoxArgs(CommonArgs):
    delta: float = 0.25
    gamma_scheduler: float = 0.5
    lr_F: float = 1e-2
    max_concentration: float = 5.1
    min_concentration: float = 0.1
    n_components_s0: int = 4
    n_components: int = 2
    scheduler_milestone: int = 2500
    use_local_search: bool = False


@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("height", [8, 16])
def test_hypergrid(ndim: int, height: int):
    n_trajectories = 64000  # if ndim == 2 else 16000
    args = HypergridArgs(ndim=ndim, height=height, n_trajectories=n_trajectories)
    final_l1_dist = train_hypergrid_main(args)
    print(final_l1_dist)

    if ndim == 2 and height == 8:
        assert np.isclose(
            final_l1_dist, 8.78e-4, atol=1e-3
        ), f"Final L1 distance: {final_l1_dist}"
    elif ndim == 2 and height == 16:
        assert np.isclose(
            final_l1_dist, 2.62e-4, atol=1e-3
        ), f"Final L1 distance: {final_l1_dist}"
    elif ndim == 4 and height == 8:
        assert np.isclose(
            final_l1_dist, 1.6e-4, atol=1e-3
        ), f"Final L1 distance: {final_l1_dist}"
    elif ndim == 4 and height == 16:
        assert np.isclose(
            final_l1_dist, 6.89e-6, atol=1e-5
        ), f"Final L1 distance: {final_l1_dist}"


@pytest.mark.parametrize("loss", ["FM", "TB", "DB", "SubTB", "ZVar", "ModifiedDB"])
@pytest.mark.parametrize("replay_buffer_size", [0, 100, 10000])
def test_hypergrid_losses_and_replay_buffer(loss: str, replay_buffer_size: int):
    args = HypergridArgs(
        ndim=2,
        height=8,
        n_trajectories=1000,
        loss=loss,
        replay_buffer_size=replay_buffer_size,
        diverse_replay_buffer=False,
    )
    final_l1_dist = train_hypergrid_main(args)
    if loss == "TB" and replay_buffer_size == 0:
        assert final_l1_dist > 0  # This is a sanity check that the script is running


# TODO: Most tests all fail, it appears the models do not train properly.
@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("alpha", [0.1, 1.0])
def test_discreteebm(ndim: int, alpha: float):
    args = DiscreteEBMArgs(ndim=ndim, alpha=alpha, n_trajectories=16000)
    final_l1_dist = train_discreteebm_main(args)
    if ndim == 2 and alpha == 0.1:
        assert np.isclose(
            final_l1_dist, 2.97e-3, atol=1e-2
        ), f"Final L1 distance: {final_l1_dist}"
    elif ndim == 2 and alpha == 1.0:
        assert np.isclose(
            final_l1_dist, 0.017, atol=1e-2
        ), f"Final L1 distance: {final_l1_dist}"
    elif ndim == 4 and alpha == 0.1:
        assert np.isclose(
            final_l1_dist, 0.009, atol=1e-2
        ), f"Final L1 distance: {final_l1_dist}"
    elif ndim == 4 and alpha == 1.0:
        assert np.isclose(
            final_l1_dist, 0.062, atol=1e-2
        ), f"Final L1 distance: {final_l1_dist}"


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
        # TODO: This value seems to be machine dependent. Either that or is is
        #       an issue with no seeding properly. Need to investigate.
        assert np.isclose(final_jsd, 0.1, atol=1e-2) or np.isclose(
            final_jsd, 3.81e-2, atol=1e-2
        ), f"Final JSD: {final_jsd}"
    elif loss == "DB" and delta == 0.1:
        assert np.isclose(final_jsd, 0.134, atol=1e-1), f"Final JSD: {final_jsd}"
    if loss == "TB" and delta == 0.25:
        assert np.isclose(final_jsd, 0.0411, atol=1e-1), f"Final JSD: {final_jsd}"
    elif loss == "DB" and delta == 0.25:
        assert np.isclose(final_jsd, 0.0142, atol=1e-2), f"Final JSD: {final_jsd}"


if __name__ == "__main__":
    test_box(0.1, "TB")
