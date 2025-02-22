import pytest
import torch
from test_samplers_and_trajectories import trajectory_sampling_with_return

from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import Box, DiscreteEBM, HyperGrid
from gfn.gym.helpers.box_utils import (
    BoxPBEstimator,
    BoxPBMLP,
    BoxPBUniform,
    BoxPFEstimator,
    BoxPFMLP,
)
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.utils.modules import MLP, DiscreteUniform, Tabular

N = 10  # Number of trajectories from sample_trajectories (changes tests globally).


@pytest.mark.parametrize(
    "module_name",
    ["MLP", "Tabular"],
)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
def test_FM(env_name: str, ndim: int, module_name: str):
    if env_name == "HyperGrid":
        env = HyperGrid(
            ndim=ndim, preprocessor_name="Enum" if module_name == "Tabular" else "KHot"
        )
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(
            ndim=ndim,
            preprocessor_name="Enum" if module_name == "Tabular" else "Identity",
        )
    else:
        raise ValueError("Unknown environment name")

    if module_name == "MLP":
        module = MLP(input_dim=env.preprocessor.output_dim, output_dim=env.n_actions)
    elif module_name == "Tabular":
        module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
    else:
        raise ValueError("Unknown module name")

    log_F_edge = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )

    gflownet = FMGFlowNet(log_F_edge)  # forward looking by default.
    trajectories = gflownet.sample_trajectories(env, n=N, save_logprobs=True)
    states_tuple = trajectories.to_state_pairs()
    loss = gflownet.loss(env, states_tuple)
    assert loss >= 0


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_pfs_and_pbs(env_name: str, preprocessor_name: str):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = trajectory_sampling_with_return(
        env_name,
        preprocessor_name,  # pyright: ignore
        delta=0.1,
        n_components=1,
        n_components_s0=1,  # pyright: ignore
    )
    gflownet_on = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    gflownet_off = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)

    log_pfs_on, log_pbs_on = gflownet_on.get_pfs_and_pbs(trajectories)
    log_pfs_off, log_pbs_off = gflownet_off.get_pfs_and_pbs(
        trajectories, recalculate_all_logprobs=True
    )


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_scores(env_name: str, preprocessor_name: str):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = trajectory_sampling_with_return(
        env_name,
        preprocessor_name,  # pyright: ignore
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )
    gflownet_on = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    gflownet_off = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    scores_on = gflownet_on.get_trajectories_scores(trajectories)
    scores_off = gflownet_off.get_trajectories_scores(
        trajectories, recalculate_all_logprobs=True
    )
    assert all(
        [
            torch.all(torch.abs(scores_on[i] - scores_off[i]) < 1e-4)
            for i in range(len(scores_on))
        ]
    )


def PFBasedGFlowNet_with_return(
    env_name: str,
    ndim: int,
    module_name: str,
    tie_pb_to_pf: bool,
    gflownet_name: str,
    sub_tb_weighting: str,
    forward_looking: bool,
    zero_logF: bool,
):
    if env_name == "HyperGrid":
        env = HyperGrid(
            ndim=ndim,
            height=4,
            preprocessor_name="Enum" if module_name == "Tabular" else "KHot",
        )
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(
            ndim=ndim,
            preprocessor_name="Enum" if module_name == "Tabular" else "Identity",
        )
    elif env_name == "Box":
        if module_name == "Tabular":
            pytest.skip("Tabular module impossible for Box")
        env = Box(delta=1.0 / ndim)
    else:
        raise ValueError("Unknown environment name")

    if module_name == "Tabular":
        # Cannot be the Box environment
        pf_module = Tabular(
            n_states=env.n_states, output_dim=env.n_actions  # pyright: ignore
        )  # pyright: ignore
        pb_module = Tabular(
            n_states=env.n_states, output_dim=env.n_actions - 1  # pyright: ignore
        )  # pyright: ignore
        logF_module = Tabular(n_states=env.n_states, output_dim=1)  # pyright: ignore
    else:
        if env_name == "Box":
            pf_module = BoxPFMLP(
                hidden_dim=32,
                n_hidden_layers=2,
                n_components=ndim,
                n_components_s0=ndim - 1,
            )

        else:
            pf_module = MLP(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,  # pyright: ignore
            )

        if module_name == "MLP" and env_name == "Box":
            pb_module = BoxPBMLP(
                hidden_dim=32,
                n_hidden_layers=2,
                n_components=ndim + 1,
                trunk=pf_module.trunk if tie_pb_to_pf else None,
            )
        elif module_name == "MLP" and env_name != "Box":
            pb_module = MLP(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions - 1,  # pyright: ignore
            )
        elif module_name == "Uniform" and env_name != "Box":
            pb_module = DiscreteUniform(output_dim=env.n_actions - 1)  # pyright: ignore
        else:
            # Uniform with Box environment
            pb_module = BoxPBUniform()
        if zero_logF:
            logF_module = DiscreteUniform(output_dim=1)
        else:
            logF_module = MLP(input_dim=env.preprocessor.output_dim, output_dim=1)

    if env_name == "Box":
        pf = BoxPFEstimator(
            env,  # pyright: ignore
            pf_module,
            n_components_s0=ndim - 1,
            n_components=ndim,
        )
        pb = BoxPBEstimator(
            env,  # pyright: ignore
            pb_module,
            n_components=ndim + 1 if module_name != "Uniform" else 1,
        )
    else:
        pf = DiscretePolicyEstimator(
            pf_module, env.n_actions, preprocessor=env.preprocessor  # pyright: ignore
        )
        pb = DiscretePolicyEstimator(
            pb_module,
            env.n_actions,  # pyright: ignore
            preprocessor=env.preprocessor,
            is_backward=True,  # pyright: ignore
        )
    logF = ScalarEstimator(module=logF_module, preprocessor=env.preprocessor)

    if gflownet_name == "DB":
        gflownet = DBGFlowNet(
            logF=logF,
            forward_looking=forward_looking,
            pf=pf,
            pb=pb,
        )
    elif gflownet_name == "ModifiedDB":
        gflownet = ModifiedDBGFlowNet(pf=pf, pb=pb)
    elif gflownet_name == "TB":
        gflownet = TBGFlowNet(pf=pf, pb=pb)
    elif gflownet_name == "ZVar":
        gflownet = LogPartitionVarianceGFlowNet(pf=pf, pb=pb)
    elif gflownet_name == "SubTB":
        gflownet = SubTBGFlowNet(
            logF=logF,
            weighting=sub_tb_weighting,  # pyright: ignore
            pf=pf,
            pb=pb,
        )
    else:
        raise ValueError(f"Unknown gflownet {gflownet_name}")

    trajectories = gflownet.sample_trajectories(env, n=N, save_logprobs=True)
    training_objects = gflownet.to_training_samples(trajectories)

    _ = gflownet.loss(env, training_objects)  # pyright: ignore

    if gflownet_name == "TB":
        assert torch.all(
            torch.abs(
                gflownet.get_pfs_and_pbs(training_objects)[0]  # pyright: ignore
                - training_objects.log_probs
            )
            < 1e-5
        )

    return env, pf, pb, logF, gflownet


@pytest.mark.parametrize(
    ("module_name", "tie_pb_to_pf"),
    [("MLP", False), ("MLP", True), ("Uniform", False), ("Tabular", False)],
)
@pytest.mark.parametrize(
    ("gflownet_name", "sub_tb_weighting"),
    [
        ("DB", None),
        ("ModifiedDB", None),
        ("TB", None),
        ("ZVar", None),
        ("SubTB", "DB"),
        ("SubTB", "TB"),
        ("SubTB", "ModifiedDB"),
        ("SubTB", "equal"),
        ("SubTB", "equal_within"),
        ("SubTB", "geometric"),
        ("SubTB", "geometric_within"),
    ],
)
@pytest.mark.parametrize("forward_looking", [True, False])
@pytest.mark.parametrize("zero_logF", [True, False])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_PFBasedGFlowNet(
    env_name: str,
    ndim: int,
    module_name: str,
    tie_pb_to_pf: bool,
    gflownet_name: str,
    sub_tb_weighting: str,
    forward_looking: bool,
    zero_logF: bool,
):
    if env_name == "Box" and module_name == "Tabular":
        pytest.skip("Tabular module impossible for Box")
    if env_name != "HyperGrid" and gflownet_name == "ModifiedDB":
        pytest.skip("ModifiedDB not implemented for DiscreteEBM or Box")

    env, pf, pb, logF, gflownet = PFBasedGFlowNet_with_return(
        env_name,
        ndim,
        module_name,
        tie_pb_to_pf,
        gflownet_name,
        sub_tb_weighting,
        forward_looking,
        zero_logF,
    )


@pytest.mark.parametrize(
    ("module_name", "tie_pb_to_pf"),
    [("MLP", False), ("MLP", True), ("Uniform", False), ("Tabular", False)],
)
@pytest.mark.parametrize(
    "weighting", ["equal", "TB", "DB", "geometric", "equal_within", "geometric_within"]
)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_subTB_vs_TB(
    env_name: str,
    ndim: int,
    module_name: str,
    tie_pb_to_pf: bool,
    weighting: str,
):
    if env_name == "Box" and module_name == "Tabular":
        pytest.skip("Tabular module impossible for Box")
    env, pf, pb, logF, gflownet = PFBasedGFlowNet_with_return(
        env_name=env_name,
        ndim=ndim,
        module_name=module_name,
        tie_pb_to_pf=tie_pb_to_pf,
        gflownet_name="SubTB",
        sub_tb_weighting=weighting,
        forward_looking=False,
        zero_logF=True,
    )

    trajectories = gflownet.sample_trajectories(env, n=N, save_logprobs=True)
    subtb_loss = gflownet.loss(env, trajectories)  # pyright: ignore

    if weighting == "TB":
        tb_loss = TBGFlowNet(pf=pf, pb=pb).loss(
            env, trajectories
        )  # LogZ is default 0.0.
        assert (tb_loss - subtb_loss).abs() < 1e-4


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_flow_matching_state_pairs(env_name: str, ndim: int):
    """Test that flow matching correctly processes state pairs from trajectories."""
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim, preprocessor_name="KHot")
    else:
        env = DiscreteEBM(ndim=ndim, preprocessor_name="Identity")

    module = MLP(input_dim=env.preprocessor.output_dim, output_dim=env.n_actions)
    log_F_edge = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )

    gflownet = FMGFlowNet(log_F_edge)
    trajectories = gflownet.sample_trajectories(env, n=N, save_logprobs=True)
    states_pairs = trajectories.to_state_pairs()
    loss = gflownet.loss(env, states_pairs)
    assert loss >= 0
