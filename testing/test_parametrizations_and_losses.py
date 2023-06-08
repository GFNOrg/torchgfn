import pytest
import torch
from test_samplers_and_trajectories import test_trajectory_sampling

from gfn.envs import BoxEnv, DiscreteEBMEnv, HyperGrid
from gfn.estimators import (
    LogEdgeFlowEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
    ProbabilityEstimator,
)
from gfn.examples.box_utils import (
    BoxPBEstimator,
    BoxPBNeuralNet,
    BoxPBUniform,
    BoxPFEStimator,
    BoxPFNeuralNet,
)
from gfn.losses import (
    DBParametrization,
    DetailedBalance,
    FlowMatching,
    FMParametrization,
    LogPartitionVarianceLoss,
    PFBasedParametrization,
    SubTBParametrization,
    SubTrajectoryBalance,
    TBParametrization,
    TrajectoryBalance,
)
from gfn.samplers import ActionsSampler, TrajectoriesSampler
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

    print(parametrization.sample_trajectories(n_samples=10))
    print(parametrization.sample_terminating_states(n_samples=10))
    print(parametrization.parameters.keys())

    trajectories = parametrization.sample_trajectories(n_samples=10)

    states_tuple = trajectories.to_non_initial_intermediary_and_terminating_states()

    loss = FlowMatching(parametrization)
    print(loss(states_tuple))


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_pfs_and_pbs(env_name: str, preprocessor_name: str):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = test_trajectory_sampling(
        env_name, preprocessor_name, delta=0.1
    )
    logZ = LogZEstimator(torch.tensor(0.0))
    parametrization = TBParametrization(pf_estimator, pb_estimator, logZ)
    loss_on = TrajectoryBalance(parametrization, on_policy=True)
    loss_off = TrajectoryBalance(parametrization, on_policy=False)
    log_pfs_on, log_pbs_on = loss_on.get_pfs_and_pbs(trajectories)
    log_pfs_off, log_pbs_off = loss_off.get_pfs_and_pbs(trajectories)
    print(log_pfs_on, log_pbs_on, log_pfs_off, log_pbs_off)


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_scores(env_name: str, preprocessor_name: str):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = test_trajectory_sampling(
        env_name, preprocessor_name, delta=0.1
    )
    logZ = LogZEstimator(torch.tensor(0.0))
    parametrization = TBParametrization(pf_estimator, pb_estimator, logZ)
    loss_on = TrajectoryBalance(parametrization, on_policy=True)
    loss_off = TrajectoryBalance(parametrization, on_policy=False)
    scores_on = loss_on.get_trajectories_scores(trajectories)
    scores_off = loss_off.get_trajectories_scores(trajectories)
    print(scores_on)
    print(scores_off)
    assert all(
        [
            torch.all(torch.abs(scores_on[i] - scores_off[i]) < 1e-4)
            for i in range(len(scores_on))
        ]
    )


# test_get_scores("Box", "Identity")


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
        pf = BoxPFEStimator(env, pf_module, n_components_s0=ndim - 1, n_components=ndim)
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

    loss_kwargs = {}
    if parametrization_name == "DB":
        parametrization = DBParametrization(pf, pb, logF)
        loss_cls = DetailedBalance
    elif parametrization_name == "TB":
        parametrization = TBParametrization(pf, pb, logZ)
        loss_cls = TrajectoryBalance
    elif parametrization_name == "ZVar":
        parametrization = PFBasedParametrization(pf, pb)
        loss_cls = LogPartitionVarianceLoss
    elif parametrization_name == "SubTB":
        parametrization = SubTBParametrization(pf, pb, logF)
        loss_cls = SubTrajectoryBalance
        loss_kwargs = {"weighing": sub_tb_weighing}
    else:
        raise ValueError(f"Unknown parametrization {parametrization_name}")
    print(parametrization.sample_trajectories(10))

    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    trajectories = parametrization.sample_trajectories(10)
    if parametrization_name == "DB":
        training_objects = trajectories.to_transitions()
    else:
        training_objects = trajectories
    loss_fn = loss_cls(parametrization, **loss_kwargs)
    loss = loss_fn(training_objects)

    if parametrization_name == "TB":
        assert torch.all(
            torch.abs(
                loss_fn.get_pfs_and_pbs(training_objects)[0]
                - training_objects.log_probs
            )
            < 1e-5
        )

    print(loss)
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
    subtb_los_fn = SubTrajectoryBalance(parametrization, weighing=weighing)
    subtb_loss = subtb_los_fn(trajectories)

    if weighing == "TB":
        tb_loss = TrajectoryBalance(TBParametrization(pf, pb, logZ))(trajectories)
        print("TB loss", tb_loss)
        print("SubTB loss", subtb_loss)
        assert (tb_loss - subtb_loss).abs() < 1e-4
