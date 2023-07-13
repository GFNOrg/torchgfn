import pytest
import torch
from test_samplers_and_trajectories import test_trajectory_sampling

from gfn.estimators import LogEdgeFlowEstimator, LogStateFlowEstimator, LogZEstimator
from gfn.gym import BoxEnv, DiscreteEBMEnv, HyperGrid
from gfn.gym.helpers.box_utils import (
    BoxPBEstimator,
    BoxPBNeuralNet,
    BoxPBUniform,
    BoxPFEstimator,
    BoxPFNeuralNet,
)
from gfn.losses import (
    DBParametrization,
    FMParametrization,
    LogPartitionVarianceParametrization,
    SubTBParametrization,
    TBParametrization,
)
from gfn.utils.estimators import DiscretePBEstimator, DiscretePFEstimator
from gfn.utils.modules import DiscreteUniform, NeuralNet, Tabular


@pytest.mark.parametrize(
    "module_name",
    ["NeuralNet", "Tabular"],
)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
def test_FM(env_name: int, ndim: int, module_name: str):
    # TODO: once the flow matching loss implemented, add a test for it here, as done for the other parametrizations
    if env_name == "HyperGrid":
        env = HyperGrid(
            ndim=ndim, preprocessor_name="Enum" if module_name == "Tabular" else "KHot"
        )
    elif env_name == "DiscreteEBM":
        env = DiscreteEBMEnv(
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

    log_F_edge = LogEdgeFlowEstimator(env=env, module=module)
    parametrization = FMParametrization(log_F_edge)

    trajectories = parametrization.sample_trajectories(n_samples=10)

    states_tuple = trajectories.to_non_initial_intermediary_and_terminating_states()

    loss = parametrization.loss(states_tuple)
    assert loss >= 0


test_FM("DiscreteEBM", 2, "NeuralNet")


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_pfs_and_pbs(env_name: str, preprocessor_name: str):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = test_trajectory_sampling(
        env_name, preprocessor_name, delta=0.1
    )
    logZ = LogZEstimator(torch.tensor(0.0))
    parametrization_on = TBParametrization(
        pf_estimator, pb_estimator, on_policy=True, logZ=logZ
    )
    parametrization_off = TBParametrization(
        pf_estimator, pb_estimator, on_policy=False, logZ=logZ
    )

    try:
        log_pfs_on, log_pbs_on = parametrization_on.get_pfs_and_pbs(trajectories)
        log_pfs_off, log_pbs_off = parametrization_off.get_pfs_and_pbs(trajectories)
    except Exception as e:
        raise ValueError("Error in get_pfs_and_pbs") from e


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_scores(env_name: str, preprocessor_name: str):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = test_trajectory_sampling(
        env_name, preprocessor_name, delta=0.1
    )
    logZ = LogZEstimator(torch.tensor(0.0))
    parametrization_on = TBParametrization(
        pf_estimator, pb_estimator, on_policy=True, logZ=logZ
    )
    parametrization_off = TBParametrization(
        pf_estimator, pb_estimator, on_policy=False, logZ=logZ
    )
    scores_on = parametrization_on.get_trajectories_scores(trajectories)
    scores_off = parametrization_off.get_trajectories_scores(trajectories)
    assert all(
        [
            torch.all(torch.abs(scores_on[i] - scores_off[i]) < 1e-4)
            for i in range(len(scores_on))
        ]
    )


@pytest.mark.parametrize(
    ("module_name", "tie_pb_to_pf"),
    [("NeuralNet", False), ("NeuralNet", True), ("Uniform", False), ("Tabular", False)],
)
@pytest.mark.parametrize(
    ("parametrization_name", "sub_tb_weighing"),
    [
        ("DB", None),
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
def test_PFBasedParametrization(
    env_name: str,
    ndim: int,
    module_name: str,
    tie_pb_to_pf: bool,
    parametrization_name: str,
    sub_tb_weighing: str,
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
        env = DiscreteEBMEnv(
            ndim=ndim,
            preprocessor_name="Enum" if module_name == "Tabular" else "Identity",
        )
    elif env_name == "Box":
        if module_name == "Tabular":
            pytest.skip("Tabular module impossible for Box")
        env = BoxEnv(delta=1.0 / ndim)
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
                n_components=3,
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
            n_components=ndim + 1,
        )
    else:
        pf = DiscretePFEstimator(env, pf_module)
        pb = DiscretePBEstimator(env, pb_module)

    logF = LogStateFlowEstimator(
        env, module=logF_module, forward_looking=forward_looking
    )
    logZ = LogZEstimator(torch.tensor(0.0))

    if parametrization_name == "DB":
        parametrization = DBParametrization(pf, pb, logF=logF)
    elif parametrization_name == "TB":
        parametrization = TBParametrization(pf, pb, logZ=logZ)
    elif parametrization_name == "ZVar":
        parametrization = LogPartitionVarianceParametrization(pf, pb)
    elif parametrization_name == "SubTB":
        parametrization = SubTBParametrization(
            pf, pb, logF=logF, weighing=sub_tb_weighing
        )
    else:
        raise ValueError(f"Unknown parametrization {parametrization_name}")

    trajectories = parametrization.sample_trajectories(10)
    if parametrization_name == "DB":
        training_objects = trajectories.to_transitions()
    else:
        training_objects = trajectories
    try:
        loss = parametrization.loss(training_objects)
    except Exception as e:
        raise ValueError("Loss computation failed") from e

    if parametrization_name == "TB":
        assert torch.all(
            torch.abs(
                parametrization.get_pfs_and_pbs(training_objects)[0]
                - training_objects.log_probs
            )
            < 1e-5
        )

    return env, pf, pb, logF, logZ, parametrization


@pytest.mark.parametrize(
    ("module_name", "tie_pb_to_pf"),
    [("NeuralNet", False), ("NeuralNet", True), ("Uniform", False), ("Tabular", False)],
)
@pytest.mark.parametrize(
    "weighing", ["equal", "TB", "DB", "geometric", "equal_within", "geometric_within"]
)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_subTB_vs_TB(
    env_name: str,
    ndim: int,
    module_name: str,
    tie_pb_to_pf: bool,
    weighing: str,
):
    env, pf, pb, logF, logZ, parametrization = test_PFBasedParametrization(
        env_name=env_name,
        ndim=ndim,
        module_name=module_name,
        tie_pb_to_pf=tie_pb_to_pf,
        parametrization_name="SubTB",
        sub_tb_weighing=weighing,
        forward_looking=False,
        zero_logF=True,
    )

    trajectories = parametrization.sample_trajectories(10)
    subtb_loss = parametrization.loss(trajectories)

    if weighing == "TB":
        tb_loss = TBParametrization(pf, pb, logZ=logZ).loss(trajectories)
        assert (tb_loss - subtb_loss).abs() < 1e-4
