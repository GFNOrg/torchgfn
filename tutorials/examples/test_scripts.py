"""Includes tests for some examples in the tutorials folder.
The tests ensure that after a certain number of iterations, the final L1
distance or JSD between the learned distribution and the target distribution is
below a certain threshold.
"""

from argparse import Namespace
from dataclasses import asdict, dataclass

import numpy as np
import pytest

from .train_bayesian_structure import main as train_bayesian_structure_main
from .train_bit_sequences import main as train_bitsequence_main
from .train_box import main as train_box_main
from .train_discreteebm import main as train_discreteebm_main
from .train_graph_ring import main as train_graph_ring_main
from .train_hypergrid import main as train_hypergrid_main
from .train_hypergrid_local_search import main as train_hypergrid_local_search_main
from .train_hypergrid_simple import main as train_hypergrid_simple_main
from .train_ising import main as train_ising_main
from .train_line import main as train_line_main


@dataclass
class CommonArgs:
    batch_size: int = 128
    cutoff_distance: float = 0.1
    hidden_dim: int = 256
    loss: str = "TB"
    lr_Z: float = 1e-1
    lr: float = 1e-3
    n_hidden: int = 2
    n_trajectories: int = 32000
    no_cuda: bool = True
    p_norm_distance: float = 2.0
    replay_buffer_prioritized: bool = False
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
    back_ratio: float = 0.5
    calculate_all_states: bool = True
    calculate_partition: bool = True
    distributed: bool = False
    diverse_replay_buffer: bool = False
    epsilon: float = 0.1
    height: int = 8
    loss: str = "TB"
    lr_logz: float = 1e-3
    n_iterations: int = 10
    n_local_search_loops: int = 3
    n_threads: int = 1
    ndim: int = 2
    plot: bool = False
    profile: bool = False
    R0: float = 0.1
    R1: float = 0.5
    R2: float = 2.0
    replay_buffer_size: int = 0
    use_metropolis_hastings: bool = True


@dataclass
class IsingArgs(CommonArgs):
    wandb_project: str = ""
    n_threads: int = 1
    L: float = 6
    J: float = 0.44
    n_iterations: int = 10
    device: str = "cpu"
    batch_size: int = 128


@dataclass
class LineArgs(CommonArgs):
    device: str = "cpu"
    exploration_var_starting_val: float = 1.0
    gradient_clip_value: float = 1.0
    lr_base: float = 1e-3
    lr_logz: float = 1e-3
    n_threads: int = 1
    n_trajectories: int = 10
    plot: bool = False
    wandb_project: str = ""


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


@dataclass
class BitSequenceArgs(CommonArgs):
    n_iterations: int = 1000
    word_size: int = 1
    seq_size: int = 4
    n_modes: int = 2


@dataclass
class GraphRingArgs(CommonArgs):
    action_type_epsilon: float = 0.0
    batch_size: int = 128
    device: str = "cpu"
    directed: bool = True
    edge_index_epsilon: float = 0.0
    embedding_dim: int = 128
    hidden_dim: int = 64
    lr_Z: float = 0.1
    lr: float = 0.001
    n_hidden: int = 1
    n_iterations: int = 4
    n_nodes: int = 4
    n_trajectories: int = 3  # Small number for smoke test
    num_conv_layers: int = 1
    plot: bool = False
    use_buffer: bool = False
    use_gnn: bool = True


@dataclass
class BayesianStructureArgs(CommonArgs):
    num_nodes: int = 3
    num_edges: int = 3
    num_samples: int = 100
    graph_name: str = "erdos_renyi_lingauss"
    prior_name: str = "uniform"
    node_names: list[str] | None = None
    num_samples_posterior: int = 1000
    num_layers: int = 1
    embedding_dim: int = 32
    module: str = "gnn_v2"
    max_epsilon: float = 0.9
    min_epsilon: float = 0.1
    use_buffer: bool = True
    buffer_capacity: int = 1000
    prefill: int = 5
    sampling_batch_size: int = 32
    lr: float = 0.001
    lr_Z: float = 1.0
    n_iterations: int = 10
    batch_size: int = 32
    n_steps_per_iteration: int = 1
    seed: int = 0
    use_cuda: bool = False


@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("height", [8, 16])
@pytest.mark.parametrize("replay_buffer_size", [0, 1000])
def test_hypergrid_tb(ndim: int, height: int, replay_buffer_size: int):
    args = HypergridArgs(
        ndim=ndim,
        height=height,
        replay_buffer_size=replay_buffer_size,
    )
    final_l1_dist = train_hypergrid_main(args)
    assert final_l1_dist is not None

    if ndim == 2 and height == 8:
        if replay_buffer_size == 0:
            tgt = 2.975e-3  # 8.78e-4
            atol = 1e-3
            assert np.isclose(
                final_l1_dist, tgt, atol=atol
            ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
        else:
            tgt = 3.1364e-3  # 6.68e-4
            atol = 1e-3
            assert np.isclose(
                final_l1_dist, tgt, atol=atol
            ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
    elif ndim == 2 and height == 16:
        tgt = 1.224e-3  # 2.62e-4
        atol = 1e-3
        if replay_buffer_size != 0:
            pytest.skip("Skipping test for replay buffer size != 0")
        assert np.isclose(
            final_l1_dist, tgt, atol=atol
        ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
    elif ndim == 4 and height == 8:
        tgt = 1.6e-4
        atol = 1e-4
        if replay_buffer_size == 0:
            assert np.isclose(
                final_l1_dist, tgt, atol=atol
            ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
        else:
            tgt = 1.7123e-4  # 6.65e-05
            atol = 1e-4
            assert np.isclose(
                final_l1_dist, tgt, atol=atol
            ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
    elif ndim == 4 and height == 16:
        if replay_buffer_size != 0:
            pytest.skip("Skipping test for replay buffer size != 0")
        tgt = 2.224e-05  # 6.89e-6
        atol = 1e-5
        assert np.isclose(
            final_l1_dist, tgt, atol=atol
        ), f"final_l1_dist: {final_l1_dist} vs {tgt}"


@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("replay_buffer_size", [0, 100])
def test_hypergrid_fm(ndim: int, replay_buffer_size: int):
    args = HypergridArgs(
        loss="FM",
        ndim=ndim,
        height=8,
        replay_buffer_size=replay_buffer_size,
    )
    final_l1_dist = train_hypergrid_main(args)
    if ndim == 2:
        if replay_buffer_size == 0:
            tgt = 5.024e-3  # 5.1e-4
            atol = 1e-3
            assert np.isclose(
                final_l1_dist, tgt, atol=atol
            ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
        else:
            tgt = 1.376e-2  # 9.85e-4
            atol = 1e-2
            assert np.isclose(
                final_l1_dist, tgt, atol=atol
            ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
    elif ndim == 4:
        if replay_buffer_size == 0:
            tgt = 2.3227e-4  # 6.28e-5
            atol = 1e-4
            assert np.isclose(
                final_l1_dist, tgt, atol=atol
            ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
        else:
            tgt = 3.2208e-4  # 9.47e-5
            atol = 1e-4
            assert np.isclose(
                final_l1_dist, tgt, atol=atol
            ), f"final_l1_dist: {final_l1_dist} vs {tgt}"


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


@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("alpha", [0.1, 1.0])
def test_discreteebm(ndim: int, alpha: float):
    args = DiscreteEBMArgs(ndim=ndim, alpha=alpha, n_trajectories=16000)
    final_l1_dist = train_discreteebm_main(args)
    if ndim == 2 and alpha == 0.1:
        tgt = 2.6972e-2  # 2.97e-3
        atol = 1e-1  # TODO: this tolerance is very suspicious.
        assert np.isclose(
            final_l1_dist, tgt, atol=atol
        ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
    elif ndim == 2 and alpha == 1.0:
        tgt = 1.3159e-1  # 0.017
        atol = 1e-1
        assert np.isclose(
            final_l1_dist, tgt, atol=atol
        ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
    elif ndim == 4 and alpha == 0.1:
        tgt = 2.46e-2  # 0.009
        atol = 1e-2
        assert np.isclose(
            final_l1_dist, tgt, atol=atol
        ), f"final_l1_dist: {final_l1_dist} vs {tgt}"
    elif ndim == 4 and alpha == 1.0:
        tgt = 8.675e-2  # 0.062
        atol = 1e-2
        assert np.isclose(
            final_l1_dist, tgt, atol=atol
        ), f"final_l1_dist: {final_l1_dist} vs {tgt}"


@pytest.mark.parametrize("delta", [0.1, 0.25])
@pytest.mark.parametrize("loss", ["TB", "DB"])
def test_box(delta: float, loss: str):
    args = BoxArgs(
        delta=delta,
        loss=loss,
        hidden_dim=128,
        n_hidden=4,
        batch_size=128,
        lr_Z=1e-3,
        validation_interval=500,
        validation_samples=10000,
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    final_jsd = train_box_main(namespace_args)

    if loss == "TB" and delta == 0.1:
        # TODO: This value seems to be machine dependent. Either that or is is
        #       an issue with no seeding properly. Need to investigate.
        tgt_1 = 0.285
        tgt_2 = 3.81e-2
        atol = 1e-2
        test_1 = np.isclose(final_jsd, tgt_1, atol=atol)
        test_2 = np.isclose(final_jsd, tgt_2, atol=atol)
        assert test_1 or test_2, f"final_jsd: {final_jsd} vs {tgt_1} or {tgt_2}"

    elif loss == "DB" and delta == 0.1:
        tgt = 0.2757
        atol = 1e-2
        assert np.isclose(final_jsd, tgt, atol=atol), f"final_jsd: {final_jsd} vs {tgt}"
    if loss == "TB" and delta == 0.25:
        tgt = 0.1492
        atol = 1e-2
        assert np.isclose(final_jsd, tgt, atol=atol), f"final_jsd: {final_jsd} vs {tgt}"
    elif loss == "DB" and delta == 0.25:
        tgt = 0.1427
        atol = 1e-2
        assert np.isclose(final_jsd, tgt, atol=atol), f"final_jsd: {final_jsd} vs {tgt}"


def test_graph_ring_smoke():
    """Smoke test for the graph ring training script."""
    args = GraphRingArgs()
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_graph_ring_main(namespace_args)  # Just ensure it runs without errors.


def test_bayesian_structure_smoke():
    """Smoke test for the Bayesian structure learning training script."""
    args = BayesianStructureArgs()
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_bayesian_structure_main(namespace_args)  # Just ensure it runs without errors.


def test_hypergrid_simple_smoke():
    """Smoke test for the simple hypergrid training script."""
    args = HypergridArgs(
        batch_size=4,
        hidden_dim=64,
        n_hidden=1,
        n_trajectories=10,  # Small number for smoke test
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_hypergrid_simple_main(namespace_args)  # Just ensure it runs without errors.


def test_hypergrid_simple_ls_smoke():
    """Smoke test for the simple hypergrid with local search training script."""
    args = HypergridArgs(
        batch_size=4,
        hidden_dim=64,
        n_hidden=1,
        n_trajectories=10,  # Small number for smoke test
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_hypergrid_local_search_main(
        namespace_args
    )  # Just ensure it runs without errors.


def test_ising_smoke():
    """Smoke test for the Ising model training script."""
    args = IsingArgs(
        n_iterations=10,  # Small number for smoke test
        batch_size=4,
        hidden_dim=64,
        n_hidden=1,
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_ising_main(namespace_args)  # Just ensure it runs without errors.


def test_line_smoke():
    """Smoke test for the line training script."""
    args = LineArgs()
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_line_main(namespace_args)  # Just ensure it runs without errors.


@pytest.mark.parametrize("seq_size", [4, 8])
@pytest.mark.parametrize("n_modes", [2, 4])
def test_bitsequence(seq_size: int, n_modes: int):
    n_iterations = 1000
    args = BitSequenceArgs(
        seq_size=seq_size, n_modes=n_modes, n_iterations=n_iterations, seed=0
    )
    final_l1_dist = train_bitsequence_main(args)
    assert final_l1_dist is not None
    if seq_size == 4 and n_modes == 2:
        assert final_l1_dist <= 1e-4
    if seq_size == 4 and n_modes == 4:
        assert final_l1_dist <= 1e-4
    if seq_size == 8 and n_modes == 2:
        assert final_l1_dist <= 1e-3
    if seq_size == 8 and n_modes == 4:
        assert final_l1_dist <= 1e-3
