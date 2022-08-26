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
from gfn.losses.trajectory_balance import TrajectoryBalance
from gfn.modules import NeuralNet, Tabular, Uniform
from gfn.parametrizations import DBParametrization, FMParametrization, TBParametrization
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor, OneHotPreprocessor
from gfn.samplers.actions_samplers import FixedActionsSampler, LogitPFActionsSampler
from gfn.samplers.trajectories_sampler import TrajectoriesSampler
from gfn.samplers.transitions_sampler import TransitionsSampler


@pytest.mark.parametrize("ndim", [2, 3])
def test_FM_hypergrid(ndim: int):
    # TODO: once the flow matching loss implemented, add a test for it here, as done for the other parametrizations
    env = HyperGrid(ndim=ndim)

    preprocessor = IdentityPreprocessor(env)
    module = NeuralNet(
        input_dim=preprocessor.output_dim,
        n_hidden_layers=1,
        hidden_dim=16,
        output_dim=env.n_actions - 1,
    )
    log_F_edge = LogEdgeFlowEstimator(preprocessor, module)
    parametrization = FMParametrization(log_F_edge)

    print(parametrization.Pi(env, n_samples=10).sample())
    print(parametrization.parameters.keys())


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "preprocessor_cls", [IdentityPreprocessor, OneHotPreprocessor, KHotPreprocessor]
)
@pytest.mark.parametrize("module_cls", [NeuralNet, Uniform, Tabular])
@pytest.mark.parametrize("parametrization_name", ["DB", "TB"])
@pytest.mark.parametrize("tie_pb_to_pf", [True, False])
def test_PFBasedParametrization_hypergrid(
    ndim: int,
    preprocessor_cls: type,
    module_cls: type,
    parametrization_name: str,
    tie_pb_to_pf: bool,
):
    env = HyperGrid(ndim=ndim, height=4)

    preprocessor = preprocessor_cls(env)

    print("\nTrying the DB parametrization... with learnable logit_PB")

    if module_cls == Tabular and preprocessor_cls != IdentityPreprocessor:
        pytest.skip("Tabular only supports IdentityPreprocessor")
    if tie_pb_to_pf and module_cls != NeuralNet:
        pytest.skip("Tying PB to PF only works with NeuralNet")

    pf_module = module_cls(
        input_dim=preprocessor.output_dim,
        n_hidden_layers=1,
        hidden_dim=16,
        output_dim=env.n_actions,
        env=env,
    )
    pb_module = module_cls(
        input_dim=preprocessor.output_dim,
        n_hidden_layers=1,
        hidden_dim=16,
        output_dim=env.n_actions - 1,
        env=env,
        torso=pf_module.torso if tie_pb_to_pf else None,
    )
    f_module = module_cls(
        input_dim=preprocessor.output_dim,
        n_hidden_layers=1,
        hidden_dim=16,
        output_dim=1,
        env=env,
    )
    logit_PF = LogitPFEstimator(preprocessor, pf_module)
    logit_PB = LogitPBEstimator(preprocessor, pb_module)
    logF = LogStateFlowEstimator(preprocessor, f_module)
    logZ = LogZEstimator(torch.tensor(0.0))

    actions_sampler = LogitPFActionsSampler(estimator=logit_PF)

    if parametrization_name == "DB":
        parametrization = DBParametrization(logit_PF, logit_PB, logF)
        training_sampler_cls = TransitionsSampler
        loss_cls = DetailedBalance
    else:
        parametrization = TBParametrization(logit_PF, logit_PB, logZ)
        training_sampler_cls = TrajectoriesSampler
        loss_cls = TrajectoryBalance
    print(parametrization.Pi(env, n_samples=10).sample())

    print(parametrization.parameters.keys())
    print(len(set(parametrization.parameters.values())))

    training_sampler = training_sampler_cls(env=env, actions_sampler=actions_sampler)

    training_objects = training_sampler.sample(n_objects=10)
    loss_fn = loss_cls(parametrization)
    loss = loss_fn(training_objects)

    print(loss)

    if ndim == 2 and parametrization_name == "TB" and module_cls == Uniform:
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

        if true_loss == loss_fn(trajectories):
            print("OK - TB LOSS PROBABLY OK")
        else:
            raise ValueError("TB LOSS NOT PROPERLY CALCULATED")
