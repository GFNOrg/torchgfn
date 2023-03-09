import pytest
import torch

from gfn.envs import DiscreteEBMEnv, HyperGrid
from gfn.estimators import (
    LogEdgeFlowEstimator,
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.losses.detailed_balance import DBParametrization, DetailedBalance
from gfn.losses.flow_matching import FlowMatching, FMParametrization
from gfn.losses.sub_trajectory_balance import SubTBParametrization, SubTrajectoryBalance
from gfn.losses.trajectory_balance import (
    LogPartitionVarianceLoss,
    PFBasedParametrization,
    TBParametrization,
    TrajectoryBalance,
)
from gfn.samplers.actions_samplers import DiscreteActionsSampler
from gfn.samplers.trajectories_sampler import TrajectoriesSampler


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "module_name",
    ["NeuralNet", "Zero", "Tabular"],
)
def test_FM(env_name: int, ndim: int, module_name: str):
    # TODO: once the flow matching loss implemented, add a test for it here, as done for the other parametrizations
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBMEnv(ndim=ndim)
    else:
        raise ValueError("Unknown environment name")

    log_F_edge = LogEdgeFlowEstimator(env=env, module_name=module_name)
    parametrization = FMParametrization(log_F_edge)

    print(parametrization.Pi(env, n_samples=10).sample())
    print(parametrization.parameters.keys())

    actions_sampler = DiscreteActionsSampler(log_F_edge)
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=10)
    states_tuple = trajectories.to_non_initial_intermediary_and_terminating_states()

    loss = FlowMatching(parametrization)
    print(loss(states_tuple))


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("ndim", [2, 3])
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
def test_PFBasedParametrization(
    env_name: str,
    ndim: int,
    module_name: str,
    tie_pb_to_pf: bool,
    parametrization_name: str,
    sub_tb_weighing: str,
    forward_looking: bool,
):
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim, height=4)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBMEnv(ndim=ndim)
    else:
        raise ValueError("Unknown environment name")

    logit_PF = LogitPFEstimator(env, module_name=module_name)
    logit_PB = LogitPBEstimator(env, module_name=module_name)
    if tie_pb_to_pf:
        logit_PB.module.torso = logit_PF.module.torso
    logF = LogStateFlowEstimator(
        env,
        forward_looking=forward_looking,
        module_name=module_name if module_name != "Uniform" else "Zero",
    )
    logZ = LogZEstimator(torch.tensor(0.0))

    actions_sampler = DiscreteActionsSampler(estimator=logit_PF)

    trajectories_sampler = TrajectoriesSampler(
        env=env,
        actions_sampler=actions_sampler,
    )

    loss_kwargs = {}
    if parametrization_name == "DB":
        parametrization = DBParametrization(logit_PF, logit_PB, logF)
        loss_cls = DetailedBalance
    elif parametrization_name == "TB":
        parametrization = TBParametrization(logit_PF, logit_PB, logZ)
        loss_cls = TrajectoryBalance
    elif parametrization_name == "ZVar":
        parametrization = PFBasedParametrization(logit_PF, logit_PB)
        loss_cls = LogPartitionVarianceLoss
    elif parametrization_name == "SubTB":
        parametrization = SubTBParametrization(logit_PF, logit_PB, logF)
        loss_cls = SubTrajectoryBalance
        loss_kwargs = {"weighing": sub_tb_weighing}
    else:
        raise ValueError(f"Unknown parametrization {parametrization_name}")
    print(parametrization.Pi(env, n_samples=10).sample())

    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    trajectories = trajectories_sampler.sample(n_trajectories=10)
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


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "preprocessor_name",
    ["KHot", "OneHot", "Identity"],
)
@pytest.mark.parametrize("module_name", ["NeuralNet", "Uniform", "Tabular"])
@pytest.mark.parametrize("weighing", ["equal", "TB", "DB", "geometric"])
def test_subTB_vs_TB(
    env_name: str,
    ndim: int,
    preprocessor_name: str,
    module_name: str,
    weighing: str,
):
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim, height=7, preprocessor_name=preprocessor_name)
    elif env_name == "DiscreteEBM":
        if preprocessor_name != "Identity":
            pytest.skip("Preprocessor not supported")
        env = DiscreteEBMEnv(ndim=ndim)
    else:
        raise ValueError("Unknown environment name")

    env = HyperGrid(ndim=ndim, height=7, preprocessor_name=preprocessor_name)

    logit_PF = LogitPFEstimator(env, module_name=module_name)
    logit_PB = LogitPBEstimator(env, module_name=module_name)
    logF = LogStateFlowEstimator(env, forward_looking=False, module_name="Zero")
    logZ = LogZEstimator(torch.tensor(0.0))
    actions_sampler = DiscreteActionsSampler(estimator=logit_PF)
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)

    subtb_loss = SubTrajectoryBalance(
        SubTBParametrization(logit_PF, logit_PB, logF), weighing=weighing
    )(trajectories)

    if weighing == "TB":
        tb_loss = TrajectoryBalance(TBParametrization(logit_PF, logit_PB, logZ))(
            trajectories
        )
        print("TB loss", tb_loss)
        print("SubTB loss", subtb_loss)
        assert (tb_loss - subtb_loss).abs() < 1e-4
