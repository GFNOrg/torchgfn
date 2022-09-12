import pytest
import torch

from gfn.envs import HyperGrid
from gfn.estimators import (
    LogEdgeFlowEstimator,
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.losses.detailed_balance import DetailedBalance
from gfn.losses.sub_trajectory_balance import SubTrajectoryBalance
from gfn.losses.trajectory_balance import TrajectoryBalance
from gfn.parametrizations import (
    DBParametrization,
    FMParametrization,
    SubTBParametrization,
    TBParametrization,
)
from gfn.samplers.actions_samplers import FixedActionsSampler, LogitPFActionsSampler
from gfn.samplers.trajectories_sampler import TrajectoriesSampler
from gfn.samplers.transitions_sampler import TransitionsSampler


@pytest.mark.parametrize("ndim", [2, 3])
def test_FM_hypergrid(ndim: int):
    # TODO: once the flow matching loss implemented, add a test for it here, as done for the other parametrizations
    env = HyperGrid(ndim=ndim)

    log_F_edge = LogEdgeFlowEstimator(env=env, module_name="NeuralNet")
    parametrization = FMParametrization(log_F_edge)

    print(parametrization.Pi(env, n_samples=10).sample())
    print(parametrization.parameters.keys())


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
        ("SubTB", "DB"),
        ("SubTB", "TB"),
        ("SubTB", "ModifiedDB"),
        ("SubTB", "equal"),
        ("SubTB", "equal_within"),
        ("SubTB", "geometric"),
        ("SubTB", "geometric_within"),
    ],
)
def test_PFBasedParametrization_hypergrid(
    ndim: int,
    module_name: str,
    parametrization_name: str,
    tie_pb_to_pf: bool,
    sub_tb_weighing: str,
):
    env = HyperGrid(ndim=ndim, height=4)

    print("\nTrying the DB parametrization... with learnable logit_PB")

    logit_PF = LogitPFEstimator(env, module_name)
    logit_PB = LogitPBEstimator(env, module_name)
    logF = LogStateFlowEstimator(
        env, module_name if module_name != "Uniform" else "Zero"
    )
    logZ = LogZEstimator(torch.tensor(0.0))

    actions_sampler = LogitPFActionsSampler(estimator=logit_PF)

    loss_kwargs = {}
    if parametrization_name == "DB":
        parametrization = DBParametrization(logit_PF, logit_PB, logF)
        training_sampler_cls = TransitionsSampler
        loss_cls = DetailedBalance
    elif parametrization_name == "TB":
        parametrization = TBParametrization(logit_PF, logit_PB, logZ)
        training_sampler_cls = TrajectoriesSampler
        loss_cls = TrajectoryBalance
    elif parametrization_name == "SubTB":
        parametrization = SubTBParametrization(logit_PF, logit_PB, logF)
        training_sampler_cls = TrajectoriesSampler
        loss_cls = SubTrajectoryBalance
        loss_kwargs = {"weighing": sub_tb_weighing}
    else:
        raise ValueError(f"Unknown parametrization {parametrization_name}")
    print(parametrization.Pi(env, n_samples=10).sample())

    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    training_sampler = training_sampler_cls(env=env, actions_sampler=actions_sampler)

    training_objects = training_sampler.sample(n_objects=10)
    loss_fn = loss_cls(parametrization, **loss_kwargs)
    loss = loss_fn(training_objects)

    print(loss)

    if (
        ndim == 2
        and parametrization_name in ("TB", "SubTB")
        and module_name == "Uniform"
    ):
        print("Evaluating the TB loss on 5 trajectories with manually chosen actions")
        actions_sampler = FixedActionsSampler(
            torch.tensor(
                [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 2, 2, 1]]
            )
        )
        trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
        trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
        # sanity check, by hand, we should get the following loss
        pbs = torch.tensor([0.5, 1, 1, 0.25, 1.0])
        pfs = torch.tensor(
            [
                1.0 / (3**3),
                1.0 / (3**3) * 0.5,
                1.0 / 3,
                1.0 / (3**4),
                1.0 / (3**2),
            ]
        )
        true_losses_exp = torch.exp(logZ.tensor) * pfs / (pbs * trajectories.rewards)
        true_loss = torch.log(true_losses_exp).pow(2).mean()

        loss = loss_fn(trajectories)

        print(loss)
        if parametrization_name == "TB":
            if true_loss == loss:
                print("OK - TB LOSS PROBABLY OK")
            else:
                raise ValueError("TB LOSS NOT PROPERLY CALCULATED")


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "preprocessor_name",
    ["KHot", "OneHot", "Identity"],
)
@pytest.mark.parametrize("module_name", ["NeuralNet", "Uniform", "Tabular"])
@pytest.mark.parametrize("weighing", ["equal", "TB", "DB", "geometric"])
def test_subTB_vs_TB(
    ndim: int,
    preprocessor_name: str,
    module_name: str,
    weighing: str,
):
    env = HyperGrid(ndim=ndim, height=7, preprocessor_name=preprocessor_name)

    logit_PF = LogitPFEstimator(env, module_name=module_name)
    logit_PB = LogitPBEstimator(env, module_name=module_name)
    logF = LogStateFlowEstimator(env, module_name="Zero")
    logZ = LogZEstimator(torch.tensor(0.0))
    actions_sampler = LogitPFActionsSampler(estimator=logit_PF)
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
