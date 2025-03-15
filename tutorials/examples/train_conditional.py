#!/usr/bin/env python
from argparse import ArgumentParser
from typing import Tuple

import torch
from torch.optim import Adam
from tqdm import tqdm

from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.modules import (
    ConditionalDiscretePolicyEstimator,
    ConditionalScalarEstimator,
    ScalarEstimator,
)
from gfn.utils.modules import MLP

DEFAULT_SEED: int = 4444


def build_conditional_pf_pb(
    env: HyperGrid,
) -> Tuple[ConditionalDiscretePolicyEstimator, ConditionalDiscretePolicyEstimator]:
    """Build conditional policy forward and backward estimators.

    Args:
        env: The HyperGrid environment

    Returns:
        A tuple of (forward policy estimator, backward policy estimator)
    """
    CONCAT_SIZE = 16
    module_PF = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_PB = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        trunk=module_PF.trunk,
    )

    # Encoder for the Conditioning information.
    module_cond = MLP(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )

    # Modules post-concatenation.
    module_final_PF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions,
    )
    module_final_PB = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions - 1,
        trunk=module_final_PF.trunk,
    )

    pf_estimator = ConditionalDiscretePolicyEstimator(
        module_PF,
        module_cond,
        module_final_PF,
        env.n_actions,
        is_backward=False,
        preprocessor=env.preprocessor,
    )
    pb_estimator = ConditionalDiscretePolicyEstimator(
        module_PB,
        module_cond,
        module_final_PB,
        env.n_actions,
        is_backward=True,
        preprocessor=env.preprocessor,
    )

    return pf_estimator, pb_estimator


def build_conditional_logF_scalar_estimator(
    env: HyperGrid,
) -> ConditionalScalarEstimator:
    """Build conditional log flow estimator.

    Args:
        env: The HyperGrid environment

    Returns:
        A conditional scalar estimator for log flow
    """
    CONCAT_SIZE = 16
    module_state_logF = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        n_hidden_layers=1,
    )
    module_conditioning_logF = MLP(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        n_hidden_layers=1,
    )
    module_final_logF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=1,
        hidden_dim=256,
        n_hidden_layers=1,
    )

    logF_estimator = ConditionalScalarEstimator(
        module_state_logF,
        module_conditioning_logF,
        module_final_logF,
        preprocessor=env.preprocessor,
    )

    return logF_estimator


# Build the GFlowNet -- Modules pre-concatenation.
def build_tb_gflownet(env: HyperGrid) -> TBGFlowNet:
    """Build a Trajectory Balance GFlowNet.

    Args:
        env: The HyperGrid environment

    Returns:
        A TBGFlowNet instance
    """
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)

    module_logZ = MLP(
        input_dim=1,
        output_dim=1,
        hidden_dim=16,
        n_hidden_layers=2,
    )

    logZ_estimator = ScalarEstimator(module_logZ)
    gflownet = TBGFlowNet(logZ=logZ_estimator, pf=pf_estimator, pb=pb_estimator)

    return gflownet


def build_db_gflownet(env):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)
    logF_estimator = build_conditional_logF_scalar_estimator(env)
    gflownet = DBGFlowNet(logF=logF_estimator, pf=pf_estimator, pb=pb_estimator)

    return gflownet


def build_db_mod_gflownet(env):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)
    gflownet = ModifiedDBGFlowNet(pf=pf_estimator, pb=pb_estimator)

    return gflownet


def build_fm_gflownet(env):
    CONCAT_SIZE = 16
    module_logF = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_cond = MLP(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_final_logF = MLP(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions,
    )
    logF_estimator = ConditionalDiscretePolicyEstimator(
        module_logF,
        module_cond,
        module_final_logF,
        env.n_actions,
        is_backward=False,
        preprocessor=env.preprocessor,
    )

    gflownet = FMGFlowNet(logF=logF_estimator)

    return gflownet


def build_subTB_gflownet(env):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)
    logF_estimator = build_conditional_logF_scalar_estimator(env)
    gflownet = SubTBGFlowNet(logF=logF_estimator, pf=pf_estimator, pb=pb_estimator)

    return gflownet


def train(env, gflownet, seed):
    torch.manual_seed(seed)
    exploration_rate = 0.5
    lr = 0.0005

    # Move the gflownet to the GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gflownet = gflownet.to(device)

    # Policy parameters and logZ/logF get independent LRs (logF/Z typically higher).
    if type(gflownet) is TBGFlowNet:
        optimizer = Adam(gflownet.pf_pb_parameters(), lr=lr)
        optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": lr * 100})
    elif type(gflownet) is DBGFlowNet or type(gflownet) is SubTBGFlowNet:
        optimizer = Adam(gflownet.pf_pb_parameters(), lr=lr)
        optimizer.add_param_group({"params": gflownet.logF_parameters(), "lr": lr * 100})
    elif type(gflownet) is FMGFlowNet or type(gflownet) is ModifiedDBGFlowNet:
        optimizer = Adam(gflownet.parameters(), lr=lr)
    else:
        print("What is this gflownet? {}".format(type(gflownet)))

    n_iterations = int(10)  # 1e4)
    batch_size = int(1e4)

    print("+ Training Conditional {}!".format(type(gflownet)))
    for _ in (pbar := tqdm(range(n_iterations))):
        conditioning = torch.rand((batch_size, 1), device=device)
        conditioning = (conditioning > 0.5).to(torch.float)  # Randomly 1 and zero.

        trajectories = gflownet.sample_trajectories(
            env,
            n=batch_size,
            conditioning=conditioning,
            save_logprobs=False,
            save_estimator_outputs=True,
            epsilon=exploration_rate,
        )
        optimizer.zero_grad()
        loss = gflownet.loss_from_trajectories(
            env, trajectories, recalculate_all_logprobs=False
        )
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

    print("+ Training complete!")


GFN_FNS = {
    "tb": build_tb_gflownet,
    "db": build_db_gflownet,
    "db_mod": build_db_mod_gflownet,
    "subtb": build_subTB_gflownet,
    "fm": build_fm_gflownet,
}


def main(args):
    environment = HyperGrid(
        ndim=5,
        height=2,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
    )

    seed = int(args.seed) if args.seed is not None else DEFAULT_SEED

    if args.gflownet == "all":
        for fn in GFN_FNS.values():
            gflownet = fn(environment)
            train(environment, gflownet, seed)
    else:
        assert args.gflownet in GFN_FNS, "invalid gflownet name\n{}".format(GFN_FNS)
        gflownet = GFN_FNS[args.gflownet](environment)
        train(environment, gflownet, seed)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, if not set, then {} is used".format(DEFAULT_SEED),
    )
    parser.add_argument(
        "--gflownet",
        "-g",
        type=str,
        default="all",
        help="Name of the gflownet. From {}".format(list(GFN_FNS.keys())),
    )

    args = parser.parse_args()
    main(args)
