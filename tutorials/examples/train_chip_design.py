import argparse
import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym.chip_design import ChipDesign
from gfn.estimators import DiscretePolicyEstimator
from gfn.utils.modules import MLP
from gfn.preprocessors import Preprocessor
import torch.nn as nn

class ChipDesignPreprocessor(Preprocessor):
    def __init__(self, env, embedding_dim=64):
        super().__init__(output_dim=env.n_macros * embedding_dim)
        self.embedding = nn.Embedding(env.n_grid_cells + 2, embedding_dim) # +2 for -1 and -2
        self.n_macros = env.n_macros
        self.embedding_dim = embedding_dim

    def preprocess(self, states):
        # states.tensor is (batch_size, n_macros) with values from -2 to n_grid_cells-1
        # We add 2 to make them non-negative for embedding.
        preprocessed_states = states.tensor + 2
        embedded = self.embedding(preprocessed_states)
        # embedded shape: (batch_size, n_macros, embedding_dim)
        # flatten it
        return embedded.view(-1, self.n_macros * self.embedding_dim)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    env = ChipDesign(device=str(device))

    preprocessor = ChipDesignPreprocessor(env, embedding_dim=args.embedding_dim)

    module_pf = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
    )
    module_pb = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        trunk=module_pf.trunk,
    )

    pf_estimator = DiscretePolicyEstimator(module_pf, env.n_actions, preprocessor=preprocessor)
    pb_estimator = DiscretePolicyEstimator(module_pb, env.n_actions, preprocessor=preprocessor, is_backward=True)

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0).to(device)

    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)

    for i in tqdm(range(args.n_iterations)):
        trajectories = gflownet.sample_trajectories(env, n=args.batch_size)
        training_samples = gflownet.to_training_samples(trajectories)
        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}, Loss: {loss.item()}")

    print("Training finished.")
    # Sample some final states and print them
    final_states = gflownet.sample_terminating_states(env, n=5)
    print("Sampled final placements (macro locations):")
    print(final_states.tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_hidden", type=int, default=2)
    args = parser.parse_args()
    main(args)
