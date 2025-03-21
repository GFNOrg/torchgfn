import torch
from tqdm import tqdm
from argparse import ArgumentParser
from typing import cast

from gfn.gflownet import TBGFlowNet, DBGFlowNet, FMGFlowNet, GFlowNet
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.utils.modules import MLP
from gfn.gym import BitSequence
from gfn.utils.common import set_seed

DEFAULT_SEED = 4444


def estimated_dist_pmf(gflownet, env):
    states = env.terminating_states
    trajectories = env.trajectory_from_terminating_states(states.tensor)
    scores = gflownet.get_trajectories_scores(
        trajectories=trajectories, 
        recalculate_all_logprobs=True
    )[0]
    pf = torch.exp(scores)
    return pf    
    

def main(args):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device = torch.device(args.device)
    H = torch.randint(0, 2, (args.n_modes, args.seq_size), dtype=torch.int64, device=device)
    env = BitSequence(args.word_size, args.seq_size, args.n_modes, device_str=args.device, H=H)

    
    if args.loss == 'TB':
        pf = MLP(env.words_per_seq,env.n_actions)
        pb = MLP(env.words_per_seq, env.n_actions - 1, trunk=pf.trunk)


        pf_estimator = DiscretePolicyEstimator(pf, n_actions=env.n_actions)
        pb_estimator = DiscretePolicyEstimator(pb, n_actions=env.n_actions, is_backward=True)

        gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.).to(device)
        non_logz_params = [v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"]
        optimizer = torch.optim.Adam(non_logz_params, lr=args.lr)
        logz_params = [dict(gflownet.named_parameters())["logZ"]]
        optimizer.add_param_group({"params": logz_params, "lr": args.lr_Z})

    if args.loss == 'FM':
        logF = MLP(env.words_per_seq,env.n_actions)
        estimator = DiscretePolicyEstimator(
            module=logF,
            n_actions=env.n_actions,
            preprocessor=env.preprocessor,
        )
        gflownet = FMGFlowNet(estimator)
        optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)

    if args.loss == 'DB':
        
        pf = MLP(env.words_per_seq,env.n_actions)
        pb = MLP(env.words_per_seq, env.n_actions - 1, trunk=pf.trunk)
        logF = MLP(
                    input_dim=env.words_per_seq,
                    output_dim=1,
                    trunk=pf.trunk,
                )

        pf_estimator = DiscretePolicyEstimator(pf, n_actions=env.n_actions)
        pb_estimator = DiscretePolicyEstimator(pb, n_actions=env.n_actions, is_backward=True)

        logF_estimator = ScalarEstimator(
                module=logF, preprocessor=env.preprocessor
            )
        gflownet = DBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                )
        optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)

    gflownet = cast(GFlowNet, gflownet)
    for iteration in tqdm(range(args.n_iterations), desc="Training"):
        trajectories = gflownet.sample_trajectories(
            env,
            n=args.batch_size,
            save_estimator_outputs=False,
        )
        training_samples = gflownet.to_training_samples(trajectories)
        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples)
        loss.backward()
        optimizer.step()

    try:
        return torch.abs(estimated_dist_pmf(gflownet, env) - env.true_dist_pmf).mean().item()
    except:
            print("Training was completed succesfully. However computing the L1 distance is only implemented for TB for now.")


    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--loss", type=str, default="TB", help="Loss to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--n_iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_Z", type=float, default=1e-1, help="Learning rate for Z")
    parser.add_argument("--word_size", type=int, default=1, help="Word size")
    parser.add_argument("--seq_size", type=int, default=4, help="Sequence size")
    parser.add_argument("--n_modes", type=int, default=2, help="Number of modes")

    args = parser.parse_args()

    print(main(args))


