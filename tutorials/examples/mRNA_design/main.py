import argparse
import logging
import time
from datetime import datetime

import torch
from env import CodonDesignEnv
from evaluate import evaluate
from plots import plot_training_curves
from preprocessor import CodonSequencePreprocessor
from train import train
from utils import load_config

from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    start_time = time.time()

    # 1. Create the environment.
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
    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)

    # Feed pf to the sampler.
    sampler = Sampler(estimator=pf_estimator)
    gflownet = gflownet.to(env.device)

    # 3. Create the optimizer.
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    loss_history = []
    reward_history = []

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    # 4. Train the gflownet.
    logging.info("Starting training loop...")

    start_time = time.time()

    loss_history, reward_history, reward_components, _ = train(
        args, env, gflownet, sampler, optimizer, scheduler, device
    )

    total_time = time.time() - start_time

    logging.info(f"Training completed in {total_time:.2f} seconds.")

    plot_training_curves(loss_history, reward_components)

    start_inference_time = time.time()

    # Sample final sequences
    with torch.no_grad():

        samples, gc_list, mfe_list, cai_list = evaluate(
            env, sampler, weights=torch.tensor([0.3, 0.3, 0.4]), n_samples=args.n_samples
        )

    inference_time = time.time() - start_inference_time
    avg_time_per_seq = inference_time / args.n_samples

    logging.info("Saving trained model and metrics...")
    torch.save(
        {
            "model_state": gflownet.state_dict(),
            "logZ": gflownet.logZ,
            "training_history": {"loss": loss_history, "reward": reward_history},
        },
        "trained_gflownet.pth",
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"outputs/generated_sequences_{timestamp}.txt"

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
    logging.info(f"Average time per generated sequence {avg_time_per_seq}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_logz",
        type=float,
        default=1e-1,
        help="Learning rate for the logZ parameter",
    )

    parser.add_argument(
        "--n_iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--n_samples", type=int, default=20, help="Number of samples to generate"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon for the sampler"
    )

    parser.add_argument(
        "--embedding_dim", type=int, default=32, help="Dimension of codon embeddings"
    )

    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension of the networks"
    )
    parser.add_argument(
        "--n_hidden", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--tied", action="store_true", help="Whether to tie the parameters of PF and PB"
    )

    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="Patience for learning rate scheduler",
    )

    parser.add_argument("--config_path", type=str, default="config.yaml")

    args = parser.parse_args()
    config = load_config(args.config_path)

    main(args)
