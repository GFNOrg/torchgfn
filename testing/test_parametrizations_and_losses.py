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
    BoxPBNeuralNet,
    BoxPBUniform,
    BoxPFEstimator,
    BoxPFNeuralNet,
)
from gfn.modules import DiscretePolicyEstimator, GFNModule, ScalarEstimator
from gfn.utils.modules import DiscreteUniform, NeuralNet, Tabular


@pytest.mark.parametrize(
    "module_name",
    ["NeuralNet", "Tabular"],
)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
def test_FM(env_name: int, ndim: int, module_name: str):
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

    if module_name == "NeuralNet":
        module = NeuralNet(
            input_dim=env.preprocessor.output_dim, output_dim=env.n_actions
        )
    elif module_name == "Tabular":
        module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
    else:
        raise ValueError("Unknown module name")

    log_F_edge = DiscretePolicyEstimator(
        env=env,
        module=module,
        forward=True,
    )

    gflownet = FMGFlowNet(log_F_edge)  # forward looking by default.
    trajectories = gflownet.sample_trajectories(n_samples=10)
    states_tuple = trajectories.to_non_initial_intermediary_and_terminating_states()
    loss = gflownet.loss(states_tuple)
    assert loss >= 0


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_pfs_and_pbs(env_name: str, preprocessor_name: str):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = trajectory_sampling_with_return(
        env_name, preprocessor_name, delta=0.1, n_components=1, n_components_s0=1
    )
    gflownet_on = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, on_policy=True)
    gflownet_off = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, on_policy=False)

    log_pfs_on, log_pbs_on = gflownet_on.get_pfs_and_pbs(trajectories)
    log_pfs_off, log_pbs_off = gflownet_off.get_pfs_and_pbs(trajectories)


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_scores(env_name: str, preprocessor_name: str):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = trajectory_sampling_with_return(
        env_name, preprocessor_name, delta=0.1, n_components=1, n_components_s0=1
    )
    gflownet_on = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, on_policy=True)
    gflownet_off = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, on_policy=False)
    scores_on = gflownet_on.get_trajectories_scores(trajectories)
    scores_off = gflownet_off.get_trajectories_scores(trajectories)
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
        pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
        logF_module = Tabular(n_states=env.n_states, output_dim=1)
    else:
        if env_name == "Box":
            pf_module = BoxPFNeuralNet(
                hidden_dim=32,
                n_hidden_layers=2,
                n_components=ndim,
                n_components_s0=ndim - 1,
            )

        else:
            pf_module = NeuralNet(
                input_dim=env.preprocessor.output_dim, output_dim=env.n_actions
            )

        if module_name == "NeuralNet" and env_name == "Box":
            pb_module = BoxPBNeuralNet(
                hidden_dim=32,
                n_hidden_layers=2,
                n_components=ndim + 1,
                torso=pf_module.torso if tie_pb_to_pf else None,
            )
        elif module_name == "NeuralNet" and env_name != "Box":
            pb_module = NeuralNet(
                input_dim=env.preprocessor.output_dim, output_dim=env.n_actions - 1
            )
        elif module_name == "Uniform" and env_name != "Box":
            pb_module = DiscreteUniform(output_dim=env.n_actions - 1)
        else:
            # Uniform with Box environment
            pb_module = BoxPBUniform()
        if zero_logF:
            logF_module = DiscreteUniform(output_dim=1)
        else:
            logF_module = NeuralNet(input_dim=env.preprocessor.output_dim, output_dim=1)

    if env_name == "Box":
        pf = BoxPFEstimator(env, pf_module, n_components_s0=ndim - 1, n_components=ndim)
        pb = BoxPBEstimator(
            env,
            pb_module,
            n_components=ndim + 1 if module_name != "Uniform" else 1,
        )
    else:
        pf = DiscretePolicyEstimator(env, pf_module, forward=True)
        pb = DiscretePolicyEstimator(env, pb_module, forward=False)

    logF = ScalarEstimator(env, module=logF_module)

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
            weighting=sub_tb_weighting,
            pf=pf,
            pb=pb,
        )
    else:
        raise ValueError(f"Unknown gflownet {gflownet_name}")

    trajectories = gflownet.sample_trajectories(10)
    training_objects = gflownet.to_training_samples(trajectories)

    _ = gflownet.loss(training_objects)

    if gflownet_name == "TB":
        assert torch.all(
            torch.abs(
                gflownet.get_pfs_and_pbs(training_objects)[0]
                - training_objects.log_probs
            )
            < 1e-5
        )

    return env, pf, pb, logF, gflownet


@pytest.mark.parametrize(
    ("module_name", "tie_pb_to_pf"),
    [("NeuralNet", False), ("NeuralNet", True), ("Uniform", False), ("Tabular", False)],
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
    [("NeuralNet", False), ("NeuralNet", True), ("Uniform", False), ("Tabular", False)],
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

    trajectories = gflownet.sample_trajectories(10)
    subtb_loss = gflownet.loss(trajectories)

    if weighting == "TB":
        tb_loss = TBGFlowNet(pf=pf, pb=pb).loss(trajectories)  # LogZ is default 0.0.
        assert (tb_loss - subtb_loss).abs() < 1e-4
