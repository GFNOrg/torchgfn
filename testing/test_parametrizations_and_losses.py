from typing import Literal, cast

import pytest
import torch
from test_samplers_and_trajectories import trajectory_sampling_with_return

from gfn.containers import Trajectories
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    GFlowNet,
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
from gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor, KHotPreprocessor
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
        env = HyperGrid(ndim=ndim)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(ndim=ndim)
    else:
        raise ValueError("Unknown environment name")

    if module_name == "Tabular":
        preprocessor = EnumPreprocessor(env.get_states_indices)
    else:
        if env_name == "HyperGrid":
            preprocessor = KHotPreprocessor(env.height, env.ndim)
        else:
            preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])

    if module_name == "MLP":
        input_dim = (
            preprocessor.output_dim
            if preprocessor.output_dim is not None
            else env.state_shape[-1]
        )
        module = MLP(input_dim=input_dim, output_dim=env.n_actions)
    elif module_name == "Tabular":
        module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
    else:
        raise ValueError("Unknown module name")

    log_F_edge = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )

    gflownet = FMGFlowNet(log_F_edge)  # forward looking by default.
    trajectories = gflownet.sample_trajectories(env, n=N, save_logprobs=True)
    states_container = trajectories.to_states_container()
    loss = gflownet.loss(env, states_container, recalculate_all_logprobs=False)
    assert loss >= 0


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_pfs_and_pbs(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box"],
    preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"],
):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = trajectory_sampling_with_return(
        env_name,
        preprocessor_name,
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )
    gflownet_on = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    gflownet_off = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)

    _ = gflownet_on.get_pfs_and_pbs(trajectories, recalculate_all_logprobs=False)
    _ = gflownet_off.get_pfs_and_pbs(trajectories, recalculate_all_logprobs=True)


@pytest.mark.parametrize("preprocessor_name", ["Identity", "KHot"])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_get_scores(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box"],
    preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"],
):
    if preprocessor_name == "KHot" and env_name != "HyperGrid":
        pytest.skip("KHot preprocessor only implemented for HyperGrid")
    trajectories, _, pf_estimator, pb_estimator = trajectory_sampling_with_return(
        env_name,
        preprocessor_name,
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )
    gflownet_on = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    gflownet_off = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    scores_on = gflownet_on.get_scores(trajectories, recalculate_all_logprobs=False)
    scores_off = gflownet_off.get_scores(trajectories, recalculate_all_logprobs=True)
    assert torch.all(torch.abs(scores_on - scores_off) < 1e-4)


def PFBasedGFlowNet_with_return(
    env_name: str,
    ndim: int,
    module_name: str,
    tie_pb_to_pf: bool,
    gflownet_name: str,
    sub_tb_weighting: Literal[
        "DB",
        "TB",
        "ModifiedDB",
        "equal",
        "equal_within",
        "geometric",
        "geometric_within",
    ],
    forward_looking: bool,
    zero_logF: bool,
):
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim, height=4)
        preprocessor = KHotPreprocessor(env.height, env.ndim)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(ndim=ndim)
        preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])
    elif env_name == "Box":
        if module_name == "Tabular":
            pytest.skip("Tabular module impossible for Box")
        env = Box(delta=1.0 / ndim)
        preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])

    else:
        raise ValueError("Unknown environment name")

    assert isinstance(preprocessor.output_dim, int)

    if module_name == "Tabular":
        # Cannot be the Box environment
        assert isinstance(env, HyperGrid) or isinstance(env, DiscreteEBM)
        preprocessor = EnumPreprocessor(env.get_states_indices)
        pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
        logF_module = Tabular(n_states=env.n_states, output_dim=1)
    else:
        if isinstance(env, Box):
            pf_module = BoxPFMLP(
                hidden_dim=32,
                n_hidden_layers=2,
                n_components=ndim,
                n_components_s0=ndim - 1,
            )

        else:
            pf_module = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions,
            )

        if module_name == "MLP" and env_name == "Box":
            pb_module = BoxPBMLP(
                hidden_dim=32,
                n_hidden_layers=2,
                n_components=ndim + 1,
                trunk=pf_module.trunk if tie_pb_to_pf else None,
            )
        elif module_name == "MLP" and not isinstance(env, Box):
            pb_module = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
            )
        elif module_name == "Uniform" and not isinstance(env, Box):
            pb_module = DiscreteUniform(output_dim=env.n_actions - 1)
        else:
            # Uniform with Box environment
            pb_module = BoxPBUniform()
        if zero_logF:
            logF_module = DiscreteUniform(output_dim=1)
        else:
            logF_module = MLP(input_dim=preprocessor.output_dim, output_dim=1)

    if isinstance(env, Box):
        pf = BoxPFEstimator(
            env,
            pf_module,
            n_components_s0=ndim - 1,
            n_components=ndim,
        )
        pb = BoxPBEstimator(
            env,
            pb_module,
            n_components=ndim + 1 if module_name != "Uniform" else 1,
        )
    else:
        pf = DiscretePolicyEstimator(pf_module, env.n_actions, preprocessor=preprocessor)
        pb = DiscretePolicyEstimator(
            pb_module,
            env.n_actions,
            preprocessor=preprocessor,
            is_backward=True,
        )
    logF = ScalarEstimator(module=logF_module, preprocessor=preprocessor)

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

    trajectories = gflownet.sample_trajectories(env, n=N, save_logprobs=True)
    training_objects = gflownet.to_training_samples(trajectories)
    gflownet = cast(GFlowNet, gflownet)
    _ = gflownet.loss(env, training_objects, recalculate_all_logprobs=False)

    if isinstance(gflownet, TBGFlowNet):
        assert isinstance(training_objects, Trajectories)
        assert training_objects.log_probs is not None
        assert torch.all(
            torch.abs(
                gflownet.get_pfs_and_pbs(
                    training_objects, recalculate_all_logprobs=False
                )[0]
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
    sub_tb_weighting: Literal[
        "DB",
        "TB",
        "ModifiedDB",
        "equal",
        "equal_within",
        "geometric",
        "geometric_within",
    ],
    forward_looking: bool,
    zero_logF: bool,
):
    if env_name == "Box" and module_name == "Tabular":
        pytest.skip("Tabular module impossible for Box")
    if env_name != "HyperGrid" and gflownet_name == "ModifiedDB":
        pytest.skip("ModifiedDB not implemented for DiscreteEBM or Box")

    # Test that function can be called without errors
    # Variables not used as we're only testing initialization
    _ = PFBasedGFlowNet_with_return(
        env_name=env_name,
        ndim=ndim,
        module_name=module_name,
        tie_pb_to_pf=tie_pb_to_pf,
        gflownet_name=gflownet_name,
        sub_tb_weighting=sub_tb_weighting,
        forward_looking=forward_looking,
        zero_logF=zero_logF,
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
    weighting: Literal[
        "equal", "TB", "DB", "geometric", "equal_within", "geometric_within"
    ],
):
    if env_name == "Box" and module_name == "Tabular":
        pytest.skip("Tabular module impossible for Box")
    env, pf, pb, _, gflownet = PFBasedGFlowNet_with_return(
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
    subtb_loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)

    if weighting == "TB":
        tb_loss = TBGFlowNet(pf=pf, pb=pb).loss(
            env, trajectories, recalculate_all_logprobs=False
        )  # LogZ is default 0.0.
        assert (tb_loss - subtb_loss).abs() < 1e-4


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_flow_matching_states_container(env_name: str, ndim: int):
    """Test that flow matching correctly processes state pairs from trajectories."""
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim)
        preprocessor = KHotPreprocessor(env.height, env.ndim)
    else:
        env = DiscreteEBM(ndim=ndim)
        preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])

    input_dim = (
        preprocessor.output_dim
        if preprocessor.output_dim is not None
        else env.state_shape[-1]
    )
    module = MLP(input_dim=input_dim, output_dim=env.n_actions)
    log_F_edge = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )

    gflownet = FMGFlowNet(log_F_edge)
    trajectories = gflownet.sample_trajectories(env, n=N, save_logprobs=True)
    states_pairs = trajectories.to_states_container()
    loss = gflownet.loss(env, states_pairs)
    assert loss >= 0
