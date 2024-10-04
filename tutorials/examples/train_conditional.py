#!/usr/bin/env python
import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet, DBGFlowNet, FMGFlowNet, SubTBGFlowNet, ModifiedDBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import ConditionalDiscretePolicyEstimator, ScalarEstimator, ConditionalScalarEstimator
from gfn.utils import NeuralNet


def build_conditional_pf_pb(env):
    CONCAT_SIZE = 16
    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_PB = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        torso=module_PF.torso,
    )

    # Encoder for the Conditioning information.
    module_cond = NeuralNet(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )

    # Modules post-concatenation.
    module_final_PF = NeuralNet(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions,
    )
    module_final_PB = NeuralNet(
        input_dim=CONCAT_SIZE * 2,
        output_dim=env.n_actions - 1,
        torso=module_final_PF.torso,
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


def build_conditional_logF_scalar_estimator(env):
    CONCAT_SIZE = 16
    module_state_logF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        n_hidden_layers=1,
    )
    module_conditioning_logF = NeuralNet(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
        n_hidden_layers=1,
    )
    module_final_logF = NeuralNet(
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
def build_tb_gflownet(env):
    pf_estimator, pb_estimator = build_conditional_pf_pb(env)

    module_logZ = NeuralNet(
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
    module_logF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_cond = NeuralNet(
        input_dim=1,
        output_dim=CONCAT_SIZE,
        hidden_dim=256,
    )
    module_final_logF = NeuralNet(
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


def train(env, gflownet):

    torch.manual_seed(0)
    exploration_rate = 0.5
    lr = 0.0005

    # Move the gflownet to the GPU.
    if torch.cuda.is_available():
        gflownet = gflownet.to("cuda")

    # Policy parameters and logZ/logF get independent LRs (logF/Z typically higher).
    if type(gflownet) is TBGFlowNet:
        optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=lr)
        optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": lr * 100})
    elif type(gflownet) is DBGFlowNet or type(gflownet) is SubTBGFlowNet:
        optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=lr)
        optimizer.add_param_group({"params": gflownet.logF_parameters(), "lr": lr * 100})
    elif type(gflownet) is FMGFlowNet or type(gflownet) is ModifiedDBGFlowNet:
        optimizer = torch.optim.Adam(gflownet.parameters(), lr=lr)
    else:
        print("What is this gflownet? {}".format(type(gflownet)))

    n_iterations = int(10)  #1e4)
    batch_size = int(1e4)

    print("+ Training Conditional {}!".format(type(gflownet)))
    for i in (pbar := tqdm(range(n_iterations))):
        conditioning = torch.rand((batch_size, 1))
        conditioning = (conditioning > 0.5).to(torch.float)  # Randomly 1 and zero.

        trajectories = gflownet.sample_trajectories(
            env,
            n=batch_size,
            conditioning=conditioning,
            save_logprobs=False,
            save_estimator_outputs=True,
            epsilon=exploration_rate,
        )
        training_samples = gflownet.to_training_samples(trajectories)
        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

    print("+ Training complete!")


def main():
    environment = HyperGrid(
        ndim=5,
        height=2,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
    )

    gflownet = build_tb_gflownet(environment)
    train(environment, gflownet)

    gflownet = build_db_gflownet(environment)
    train(environment, gflownet)

    gflownet = build_db_mod_gflownet(environment)
    train(environment, gflownet)

    gflownet = build_subTB_gflownet(environment)
    train(environment, gflownet)

    gflownet = build_fm_gflownet(environment)
    train(environment, gflownet)


if __name__ == "__main__":
    main()
