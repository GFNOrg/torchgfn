import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet, SubTBGFlowNet
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet  # NeuralNet is a simple multi-layer perceptron (MLP)

if __name__ == "__main__":

    # 1 - We define the environment.
    env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

    # 2 - We define the needed modules (neural networks).
    # The environment has a preprocessor attribute, which is used to preprocess the state before feeding it to the policy estimator
    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions
    )  # Neural network for the forward policy, with as many outputs as there are actions

    module_PB = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        torso=module_PF.torso  # We share all the parameters of P_F and P_B, except for the last layer
    )

    # 3 - We define the estimators.
    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)

    # 4 - We define the GFlowNet.
    use_tb = False
    if use_tb:
        gfn = TBGFlowNet(init_logZ=0., pf=pf_estimator, pb=pb_estimator)  # We initialize logZ to 0
    else:
        # import IPython; IPython.embed()
        logF = DiscretePolicyEstimator(module=module_PF, n_actions=env.n_actions, preprocessor=env.preprocessor)
        gfn = SubTBGFlowNet(pf=pf_estimator, pb=pb_estimator, logF=logF, lamda=0.9)


    # 5 - We define the sampler and the optimizer.
    sampler = Sampler(estimator=pf_estimator)  # We use an on-policy sampler, based on the forward policy

    # Policy parameters have their own LR.
    if use_tb:
        non_logz_params = [v for k, v in dict(gfn.named_parameters()).items() if k != "logZ"]
        optimizer = torch.optim.Adam(non_logz_params, lr=1e-3)

        # Log Z gets dedicated learning rate (typically higher).
        logz_params = [dict(gfn.named_parameters())["logZ"]]
        optimizer.add_param_group({"params": logz_params, "lr": 1e-1})
    else:
        import IPython; IPython.embed()
        non_logz_params = [v for k, v in dict(gfn.named_parameters()).items() if k != "logF"]
        optimizer = torch.optim.Adam(non_logz_params, lr=1e-3)

        # Log Z gets dedicated learning rate (typically higher).
        logz_params = [dict(gfn.named_parameters())["logF"]]
        optimizer.add_param_group({"params": logz_params, "lr": 1e-1})

    # 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration
    for i in (pbar := tqdm(range(1000))):
        trajectories = sampler.sample_trajectories(env=env, n_trajectories=16)
        optimizer.zero_grad()
        loss = gfn.loss(env, trajectories)
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            pbar.set_postfix({"loss": loss.item()})
