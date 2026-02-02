"""Includes tests for some examples in the tutorials folder.
The tests ensure that after a certain number of iterations, the final L1
distance or JSD between the learned distribution and the target distribution is
below a certain threshold.
"""

import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

# Make repo root importable so `tutorials` can be found without installing the package.
# flake8: noqa: E402
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Ensure we run with Python debug mode enabled (no -O) so envs use debug guards.
assert __debug__, "Tests must run without -O so __debug__ stays True."

from tutorials.examples.train_bayesian_structure import (
    main as train_bayesian_structure_main,
)
from tutorials.examples.train_bit_sequences import main as train_bitsequence_main
from tutorials.examples.train_bitsequence_recurrent import (
    main as train_bitsequence_recurrent_main,
)
from tutorials.examples.train_box import main as train_box_main
from tutorials.examples.train_conditional import main as train_conditional_main
from tutorials.examples.train_diffusion_sampler import (
    main as train_diffusion_sampler_main,
)
from tutorials.examples.train_discreteebm import main as train_discreteebm_main
from tutorials.examples.train_graph_ring import main as train_graph_ring_main
from tutorials.examples.train_graph_triangle import main as train_graph_triangle_main
from tutorials.examples.train_hypergrid import main as train_hypergrid_main
from tutorials.examples.train_hypergrid_buffer import main as train_hypergrid_buffer_main
from tutorials.examples.train_hypergrid_exploration_examples import (
    main as train_hypergrid_exploration_main,
)
from tutorials.examples.train_hypergrid_gafn import main as train_hypergrid_gafn_main
from tutorials.examples.train_hypergrid_local_search import (
    main as train_hypergrid_local_search_main,
)
from tutorials.examples.train_hypergrid_simple import main as train_hypergrid_simple_main
from tutorials.examples.train_ising import main as train_ising_main
from tutorials.examples.train_line import main as train_line_main
from tutorials.examples.train_with_example_modes import (
    main as train_with_example_modes_main,
)


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
    store_all_states: bool = True
    calculate_partition: bool = True
    distributed: bool = False
    diverse_replay_buffer: bool = False
    epsilon: float = 0.1
    height: int = 8
    loss: str = "TB"
    lr_logz: float = 1e-3
    n_iterations: int = 10
    n_threads: int = 1
    ndim: int = 2
    plot: bool = False
    profile: bool = False
    R0: float = 0.1
    R1: float = 0.5
    R2: float = 2.0
    replay_buffer_size: int = 0
    timing: bool = True
    half_precision: bool = False
    remote_buffer_freq = 1
    validate_environment: bool = True


@dataclass
class HypergridBufferArgs(HypergridArgs):
    buffer_type: str = "terminating_state"
    buffer_capacity: int = 100
    prefill: int = 0
    prioritized_capacity: bool = False
    prioritized_sampling: bool = False


@dataclass
class HypergridExplorationArgs(HypergridArgs):
    validation_interval: int = 5
    validation_samples: int = 100
    n_seeds: int = 3
    plot: bool = False


@dataclass
class HypergridGAFNArgs(HypergridArgs):
    use_edge_ri: bool = True
    lr_rnd: float = 1e-3
    rnd_reward_scale: float = 0.1
    rnd_loss_scale: float = 1.0
    rnd_hidden_dim: int = 256
    rnd_s_latent_dim: int = 128


@dataclass
class HypergridLocalSearchArgs(HypergridArgs):
    n_local_search_loops: int = 3
    use_metropolis_hastings: bool = True
    half_precision: bool = False


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
class DiffusionSamplerArgs:
    no_cuda: bool = True
    seed: int = 0
    target: str = "gmm2"
    dim: int | None = None
    num_components: int | None = None
    target_seed: int = 2
    num_steps: int = 8
    sigma: float = 5.0
    harmonics_dim: int = 16
    t_emb_dim: int = 16
    s_emb_dim: int = 16
    hidden_dim: int = 32
    joint_layers: int = 1
    zero_init: bool = False
    n_iterations: int = 3
    batch_size: int = 16
    lr: float = 1e-3
    lr_logz: float = 1e-1
    eval_interval: int = 10
    eval_n: int = 100
    eval_batch_size: int = 100
    vis_interval: int = 10
    vis_n: int = 100
    visualize: bool = False


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
    n_iterations: int = 5000
    word_size: int = 1
    seq_size: int = 4
    n_modes: int = 2
    temperature: float = 1.0
    lr: 1e-4
    lr_Z: 1e-2
    seed: int = 0
    batch_size: int = 32


@dataclass
class GraphRingArgs(CommonArgs):
    action_type_epsilon: float = 0.0
    batch_size: int = 16
    device: str = "cpu"
    directed: bool = True
    edge_index_epsilon: float = 0.0
    embedding_dim: int = 32
    hidden_dim: int = 64
    lr_Z: float = 0.1
    lr: float = 0.001
    n_hidden: int = 1
    n_iterations: int = 4
    n_nodes: int = 3
    n_trajectories: int = 3  # Small number for smoke test
    num_conv_layers: int = 1
    plot: bool = False
    use_buffer: bool = False
    use_gnn: bool = True


@dataclass
class GraphTriangleArgs(CommonArgs):
    device: str = "cpu"
    batch_size: int = 32
    n_iterations: int = 4
    embedding_dim: int = 32
    num_conv_layers: int = 1
    use_buffer: bool = False
    plot: bool = False
    lr_Z: float = 0.1
    lr: float = 0.001
    epsilon_action_type: float = 0.0
    epsilon_node_class: float = 0.0
    epsilon_edge_class: float = 0.0
    epsilon_edge_index: float = 0.0


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


@dataclass
class ConditionalArgs(CommonArgs):
    gflownet: str = "tb"
    ndim: int = 5
    height: int = 2
    n_iterations: int = 10
    batch_size: int = 1000
    seed: int = 4444
    lr: float = 1e-3
    lr_logz: float = 1e-2
    epsilon: float = 0.0
    validation_interval: int = 100
    validation_samples: int = 200000
    n_eval_samples: int = 10000
    no_cuda: bool = True  # Disable CUDA for tests


@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("height", [8, 16])
@pytest.mark.parametrize("replay_buffer_size", [0, 1000])
def test_hypergrid_tb(ndim: int, height: int, replay_buffer_size: int):
    args = HypergridArgs(
        ndim=ndim,
        height=height,
        replay_buffer_size=replay_buffer_size,
    )
    logs = train_hypergrid_main(args)
    assert "l1_dist" in logs
    final_l1_dist = logs["l1_dist"]
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
        # TODO: Why is this skipped?
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
        # TODO: Why is this skipped?
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
    logs = train_hypergrid_main(args)
    assert "l1_dist" in logs
    final_l1_dist = logs["l1_dist"]
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
    logs = train_hypergrid_main(args)

    if loss == "TB" and replay_buffer_size == 0:
        assert "l1_dist" in logs
        final_l1_dist = logs["l1_dist"]
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
        tgt1 = 8.675e-2  # 0.062
        tgt2 = 6.2e-2
        atol = 1e-2
        test_1 = np.isclose(final_l1_dist, tgt1, atol=atol)
        test_2 = np.isclose(final_l1_dist, tgt2, atol=atol)

        assert (
            test_1 or test_2
        ), f"final_l1_dist: {final_l1_dist} not close to [{tgt1}, {tgt2}]"


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
        tgt1 = 0.1
        tgt2 = 0.285
        tgt3 = 3.81e-2
        tgt4 = 5.67e-2
        test_1 = np.isclose(final_jsd, tgt1, atol=1e-2)
        test_2 = np.isclose(final_jsd, tgt2, atol=1e-2)
        test_3 = np.isclose(final_jsd, tgt3, atol=1e-2)
        test_4 = np.isclose(final_jsd, tgt4, atol=1e-2)
        assert (
            test_1 or test_2 or test_3 or test_4
        ), f"final_jsd: {final_jsd} not close to [{tgt1}, {tgt2}, {tgt3}, {tgt4}]"

    elif loss == "DB" and delta == 0.1:
        tgt1 = 0.2757
        tgt2 = 0.2878
        atol = 1e-2
        test_1 = np.isclose(final_jsd, tgt1, atol=atol)
        test_2 = np.isclose(final_jsd, tgt2, atol=atol)
        assert test_1 or test_2, f"final_jsd: {final_jsd} not close to [{tgt1}, {tgt2}]"
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


def test_graph_triangle_smoke():
    """Smoke test for the graph triangle training script."""
    args = GraphTriangleArgs()
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_graph_triangle_main(namespace_args)  # Just ensure it runs without errors.


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


def test_hypergrid_simple_smoke_fp64():
    """Smoke test for the simple hypergrid training script at fp64 precision."""
    torch.set_default_dtype(torch.float64)
    args = HypergridArgs(
        batch_size=4,
        hidden_dim=64,
        n_hidden=1,
        n_trajectories=10,  # Small number for smoke test
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_hypergrid_simple_main(namespace_args)  # Just ensure it runs without errors.


def test_hypergrid_buffer_smoke():
    """Smoke test for the hypergrid buffer training script."""
    args = HypergridBufferArgs(
        batch_size=4,
        hidden_dim=64,
        n_hidden=1,
        n_trajectories=10,  # Small number for smoke test
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_hypergrid_buffer_main(namespace_args)  # Just ensure it runs without errors.


def test_hypergrid_gafn_smoke():
    """Smoke test for the GAFN training script."""
    args = HypergridGAFNArgs(
        batch_size=4,
        hidden_dim=64,
        n_hidden=1,
        n_trajectories=10,  # Small number for smoke test
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_hypergrid_gafn_main(namespace_args)  # Just ensure it runs without errors.


def test_hypergrid_simple_ls_smoke():
    """Smoke test for the simple hypergrid with local search training script."""
    args = HypergridLocalSearchArgs(
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


def test_ising_smoke_fp64():
    """Smoke test for the Ising model training script at fp64 precision."""
    torch.set_default_dtype(torch.float64)
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


def test_line_smoke_fp64():
    """Smoke test for the line training script at fp64 precision."""
    torch.set_default_dtype(torch.float64)
    args = LineArgs()
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_line_main(namespace_args)  # Just ensure it runs without errors.


def test_diffusion_sampler_smoke():
    """Smoke test for the diffusion sampler training script."""
    args = DiffusionSamplerArgs()
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_diffusion_sampler_main(namespace_args)  # Runs without errors.


@pytest.mark.parametrize("seq_size", [4, 8])
@pytest.mark.parametrize("n_modes", [2, 4])
def test_bitsequence(seq_size: int, n_modes: int):
    args = BitSequenceArgs(seq_size=seq_size, n_modes=n_modes)
    final_l1_dist = train_bitsequence_main(args)
    assert final_l1_dist is not None
    # print(f"[DEBUG] BitSequence seq_size={seq_size}, n_modes={n_modes}, l1={final_l1_dist}")
    if seq_size == 4 and n_modes == 2:
        assert final_l1_dist <= 9e-5
    if seq_size == 4 and n_modes == 4:
        assert final_l1_dist <= 1e-4
    if seq_size == 8 and n_modes == 2:
        assert final_l1_dist <= 1e-3
    if seq_size == 8 and n_modes == 4:
        assert final_l1_dist <= 2e-4


@pytest.mark.parametrize("gflownet", ["tb", "db", "subtb", "fm"])
def test_conditional_basic(gflownet: str):
    """Test basic conditional training with different GFlowNet types."""
    args = ConditionalArgs(
        gflownet=gflownet,
        n_iterations=5,
        batch_size=100,
        validation_interval=10,
        validation_samples=100,
    )
    args_dict = asdict(args)
    # Don't set evaluate flag - defaults to False (no action="store_true" triggered)
    namespace_args = Namespace(**args_dict)
    final_loss = train_conditional_main(namespace_args)
    assert final_loss is not None
    assert final_loss > 0  # Loss should be positive


def test_conditional_all_gflownets():
    """Test conditional training with all GFlowNet types sequentially."""
    args = ConditionalArgs(
        gflownet="all",
        n_iterations=3,
        batch_size=50,
        validation_interval=10,
        validation_samples=50,
    )
    args_dict = asdict(args)
    # Don't set evaluate flag - defaults to False (no action="store_true" triggered)
    namespace_args = Namespace(**args_dict)
    final_loss = train_conditional_main(namespace_args)
    assert final_loss is not None
    assert final_loss > 0  # Average loss should be positive


def test_conditional_convergence():
    """Test that conditional GFlowNet training converges to reasonable loss values."""
    args = ConditionalArgs(
        gflownet="tb",
        ndim=2,
        height=8,  # Small environment for quick convergence
        n_iterations=20,  # Small but enough to see improvement
        batch_size=100,
        validation_interval=10,
        validation_samples=500,  # Small but enough for rough L1 estimate
        lr=1e-3,
        lr_logz=1e-2,
        epsilon=0.0,
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    final_loss = train_conditional_main(namespace_args)

    assert final_loss is not None
    assert final_loss > 0  # Loss should be positive
    assert final_loss < 100  # Loss should be reasonable, not exploded


@pytest.mark.parametrize("gflownet", ["tb", "db"])
def test_conditional_different_dims(gflownet: str):
    """Test conditional training with different environment dimensions."""
    for ndim in [2, 3]:
        args = ConditionalArgs(
            gflownet=gflownet,
            ndim=ndim,
            height=8,
            n_iterations=10,
            batch_size=64,
            validation_interval=5,
            validation_samples=100,
        )
        args_dict = asdict(args)
        namespace_args = Namespace(**args_dict)
        final_loss = train_conditional_main(namespace_args)

        assert final_loss is not None
        assert 0 < final_loss < 1000  # Reasonable loss range


def test_conditional_with_exploration():
    """Test conditional training with exploration (epsilon > 0)."""
    args = ConditionalArgs(
        gflownet="tb",
        ndim=2,
        height=8,
        n_iterations=10,
        batch_size=100,
        epsilon=0.1,  # Enable exploration
        validation_interval=10,
        validation_samples=200,
    )
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    final_loss = train_conditional_main(namespace_args)

    assert final_loss is not None
    assert final_loss > 0


def test_conditional_loss_types():
    """Test that different GFlowNet loss types work with conditions."""
    loss_types = ["tb", "db", "subtb", "fm"]
    losses = []

    for loss_type in loss_types:
        args = ConditionalArgs(
            gflownet=loss_type,
            ndim=2,
            height=8,
            n_iterations=5,
            batch_size=50,
            validation_interval=10,
            validation_samples=50,
        )
        args_dict = asdict(args)
        namespace_args = Namespace(**args_dict)
        final_loss = train_conditional_main(namespace_args)

        assert final_loss is not None, f"Loss type {loss_type} returned None"
        assert final_loss > 0, f"Loss type {loss_type} returned non-positive loss"
        losses.append(final_loss)

    # All loss types should produce finite losses
    assert all(0 < loss < float("inf") for loss in losses)


def test_hypergrid_exploration_smoke():
    """Smoke test for the hypergrid exploration training script."""
    args = HypergridExplorationArgs()
    args_dict = asdict(args)
    namespace_args = Namespace(**args_dict)
    train_hypergrid_exploration_main(namespace_args)  # Runs without errors.


def test_bitsequence_recurrent_smoke():
    """Smoke test for the recurrent BitSequence training script."""
    args = BitSequenceArgs(
        n_iterations=50,
        word_size=1,
        seq_size=4,
        n_modes=2,
        seed=0,
        batch_size=4,
    )
    args_dict = asdict(args)
    # Added (not needed for non-recurrent script).
    args_dict.update(
        embedding_dim=4,
        hidden_size=8,
        num_layers=2,
        rnn_type="gru",
        dropout=0.1,
        lr_logz=1e-1,
        lr=1e-3,
        epsilon=0.1,
    )

    train_bitsequence_recurrent_main(Namespace(**args_dict))


def test_with_example_modes_smoke():
    """Smoke test for example modes graph-building script."""
    # Keep small for speed.
    args = Namespace(
        n_nodes=4,
        max_rings=20,
        device="cpu",
        seed=0,
        embedding_dim=16,
        num_conv_layers=1,
        lr=1e-3,
        lr_Z=1e-3,
        use_lr_scheduler=False,
        n_iterations=2,
        replay_buffer_max_size=50,
        use_expert_data=False,
        action_type_epsilon=0.0,
        edge_index_epsilon=0.0,
        batch_size=2,
        plot=False,
    )
    train_with_example_modes_main(args)  # Runs without errors.


if __name__ == "__main__":
    test_conditional_basic("tb")
