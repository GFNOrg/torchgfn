import argparse
import logging
from datetime import datetime

import numpy as np
import torch
from env import CodonDesignEnv
from evaluate import evaluate
from preprocessor import CodonSequencePreprocessor
from tqdm import tqdm
from utils import compute_reward, load_config, plot_training_curves

from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = load_config(args.config_path)

    # 1. Create the environment and preprocessor
    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(
        env.seq_length, embedding_dim=args.embedding_dim, device=device
    )

    # Build the GFlowNet.
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
    )

    module_PB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        trunk=module_PF.trunk if args.tied else None,
    )

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    # 2. Create the gflownet.
    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0).to(device)
    sampler = Sampler(estimator=pf_estimator)

    # 3. Create the optimizer and Lr scheduler
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    loss_history = []
    reward_history = []
    reward_components_history = []

    # 4. Train the gflownet.
    logging.info("Starting training loop...")

    for it in tqdm(range(args.n_iterations), dynamic_ncols=True):

        # Sample new reward weights from a Dirichlet distribution
        weights = (np.random.dirichlet([1, 1, 1])).tolist()
        env.set_weights(weights)

        # Sample trajectories using the current policy
        trajectories = sampler.sample_trajectories(
            env, args.batch_size, save_logprobs=True, epsilon=args.epsilon
        )

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # Evaluate final states
        with torch.no_grad():
            final_states = trajectories.terminating_states.tensor.to(device)
            rewards, components = [], []

            for state in final_states:
                state = state.to(device)
                r, c = compute_reward(state, env.codon_gc_counts, env.weights)
                rewards.append(r)
                components.append(c)

        avg_reward = torch.mean(torch.tensor(rewards)).item()
        reward_history.append(avg_reward)
        reward_components_history.extend(components)
        loss_history.append(loss.item())

    plot_training_curves(loss_history, reward_components_history)

    # Sample final sequences using fixed reward weights (eg, 30% GC, 30% MFE, 40% CAI) to evaluate policy performance
    with torch.no_grad():
        samples, gc_list, mfe_list, cai_list = evaluate(
            env, sampler, weights=torch.tensor([0.3, 0.3, 0.4]), n_samples=args.n_samples
        )

    torch.save(
        {
            "model_state": gflownet.state_dict(),
            "logZ": gflownet.logZ,
            "training_history": {"loss": loss_history, "reward": reward_history},
        },
        "trained_gflownet.pth",
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"generated_sequences_{timestamp}.txt"

    sorted_samples = sorted(samples.items(), key=lambda x: x[1][0], reverse=True)

    with open(filename, "w") as f:
        for i, (seq, reward) in enumerate(sorted_samples):
            f.write(
                f"Sequence {i+1}: {seq}, "
                f"Reward: {reward[0]:.2f}, "
                f"GC Content: {reward[1][0]:.2f}, "
                f"MFE: {reward[1][1]:.2f}, "
                f"CAI: {reward[1][2]:.2f}\n"
            )

    logging.info(f"Saving generated sequences to {filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lr_logz", type=float, default=1e-1, help="Learning rate for logZ"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=10, help="Number of training iterations"
    )
    parser.add_argument(
        "--n_samples", type=int, default=20, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon for sampler exploration"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=32, help="Codon embedding dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="MLP hidden layer dimension"
    )
    parser.add_argument(
        "--n_hidden", type=int, default=2, help="Number of MLP hidden layers"
    )
    parser.add_argument(
        "--tied", action="store_true", help="Tie PF and PB network trunks"
    )
    parser.add_argument(
        "--lr_patience", type=int, default=10, help="LR scheduler patience"
    )
    parser.add_argument(
        "--config_path", type=str, default="config.yaml", help="Path to config YAML"
    )

    args = parser.parse_args()
    config = load_config(args.config_path)

    main(args)
