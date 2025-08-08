#!/usr/bin/env python
from argparse import ArgumentParser
from typing import Tuple

import torch
from torch.optim import Adam
from tqdm import tqdm

from gfn.estimators import (
    ConditionalDiscretePolicyEstimator,
    ConditionalScalarEstimator,
    ScalarEstimator,
)
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
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
    # Create preprocessor for the environment
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    CONCAT_SIZE = 16
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_PB = MLP(
        input_dim=preprocessor.output_dim,
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
        preprocessor=preprocessor,
        is_backward=False,
    )
    pb_estimator = ConditionalDiscretePolicyEstimator(
        module_PB,
        module_cond,
        module_final_PB,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=True,
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
    # Create preprocessor for the environment
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    CONCAT_SIZE = 16
    module_state_logF = MLP(
        input_dim=preprocessor.output_dim,
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
        preprocessor=preprocessor,
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

    # Create conditional logZ estimator
    # Use the same preprocessor as the policy estimators
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    module_logZ_state = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=16,
        hidden_dim=16,
        n_hidden_layers=2,
    )
    module_logZ_cond = MLP(
        input_dim=1,
        output_dim=16,
        hidden_dim=16,
        n_hidden_layers=2,
    )
    module_logZ_final = MLP(
        input_dim=32,  # 16 + 16
        output_dim=1,
        hidden_dim=16,
        n_hidden_layers=2,
    )

    # Create a wrapper class to make ConditionalScalarEstimator compatible with TBGFlowNet
    class ConditionalLogZWrapper(ScalarEstimator):
        def __init__(self, conditional_estimator, env):
            super().__init__(
                conditional_estimator.module, conditional_estimator.preprocessor
            )
            self.conditional_estimator = conditional_estimator
            self.env = env

        def forward(self, conditioning):
            # Create dummy states for the conditional estimator
            # The conditional estimator expects states, but we only have conditioning
            # We'll create dummy states with the same batch shape as conditioning
            batch_shape = (
                conditioning.shape[:-1]
                if len(conditioning.shape) > 1
                else conditioning.shape
            )
            dummy_states = self.env.reset(batch_shape)
            return self.conditional_estimator(dummy_states, conditioning)

    conditional_logZ = ConditionalScalarEstimator(
        module_logZ_state,
        module_logZ_cond,
        module_logZ_final,
        preprocessor=preprocessor,
    )
    logZ_estimator = ConditionalLogZWrapper(conditional_logZ, env)
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
    # Create preprocessor for the environment
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    CONCAT_SIZE = 16
    module_logF = MLP(
        input_dim=preprocessor.output_dim,
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
        preprocessor=preprocessor,
        is_backward=False,
    )

    gflownet = FMGFlowNet(logF=logF_estimator)

    return gflownet


def build_subTB_gflownet(env):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)
    logF_estimator = build_conditional_logF_scalar_estimator(env)
    gflownet = SubTBGFlowNet(logF=logF_estimator, pf=pf_estimator, pb=pb_estimator)

    return gflownet


def train(env, gflownet, seed, device, n_iterations=10, batch_size=1000):
    torch.manual_seed(seed)
    exploration_rate = 0.5
    lr = 0.0005

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
        print("unknown gflownet type: {}".format(type(gflownet)))

    print("+ Training Conditional {}!".format(type(gflownet)))
    final_loss = None
    for _ in (pbar := tqdm(range(n_iterations))):
        conditioning = torch.rand((batch_size,)).to(device)
        conditioning = (conditioning > 0.5).to(
            torch.get_default_dtype()
        )  # Randomly 1 and zero.
        conditioning = conditioning.unsqueeze(-1)  # Add feature dimension for MLP

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
        final_loss = loss.item()
        pbar.set_postfix({"loss": final_loss})

    print("+ Training complete!")
    return final_loss


GFN_FNS = {
    "tb": build_tb_gflownet,
    "db": build_db_gflownet,
    "db_mod": build_db_mod_gflownet,
    "subtb": build_subTB_gflownet,
    "fm": build_fm_gflownet,
}


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    environment = HyperGrid(
        ndim=args.ndim,
        height=args.height,
        device=device,
    )
    seed = int(args.seed) if args.seed is not None else DEFAULT_SEED
    n_iterations = args.n_iterations
    batch_size = args.batch_size

    if args.gflownet == "all":
        final_losses = []
        for fn in GFN_FNS.values():
            gflownet = fn(environment)
            gflownet = gflownet.to(device)
            final_loss = train(
                environment, gflownet, seed, device, n_iterations, batch_size
            )
            final_losses.append(final_loss)
        return sum(final_losses) / len(final_losses)  # Return average loss
    else:
        assert args.gflownet in GFN_FNS, "invalid gflownet name\n{}".format(GFN_FNS)
        gflownet = GFN_FNS[args.gflownet](environment)
        gflownet = gflownet.to(device)
        final_loss = train(environment, gflownet, seed, device, n_iterations, batch_size)
        return final_loss


if __name__ == "__main__":
    parser = ArgumentParser()

    # Machine settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, if not set, then {} is used".format(DEFAULT_SEED),
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Prevent CUDA usage",
    )

    # Environment settings
    parser.add_argument(
        "--ndim",
        type=int,
        default=5,
        help="Number of dimensions in the environment",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2,
        help="Height of the environment",
    )

    # Training settings
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )

    # GFlowNet settings
    parser.add_argument(
        "--gflownet",
        "-g",
        type=str,
        default="tb",
        help="Name of the gflownet. From {}".format(list(GFN_FNS.keys())),
    )

    args = parser.parse_args()
    main(args)
