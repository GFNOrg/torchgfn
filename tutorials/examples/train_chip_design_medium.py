"""Train a GFlowNet on a medium-sized chip placement problem.

Uses ~15 hard macros on an 8x8 grid (64 cells) with a replay buffer
for off-policy training. Logs wirelength and density costs per iteration.
"""

import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from gfn.containers.replay_buffer import ReplayBuffer
from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import TBGFlowNet
from gfn.gym.chip_design import ChipDesign, ChipDesignStates
from gfn.gym.helpers.chip_design import MEDIUM_INIT_PLACEMENT, MEDIUM_NETLIST_FILE
from gfn.preprocessors import Preprocessor
from gfn.utils.modules import MLP


class ChipDesignPreprocessor(Preprocessor):
    def __init__(self, env: ChipDesign, embedding_dim: int = 64):
        super().__init__(output_dim=env.n_macros * embedding_dim)
        self.embedding = nn.Embedding(env.n_grid_cells + 2, embedding_dim)
        self.n_macros = env.n_macros
        self.embedding_dim = embedding_dim

    def preprocess(self, states: ChipDesignStates) -> torch.Tensor:
        preprocessed_states = states.tensor + 2  # shift so -2 -> 0, -1 -> 1
        embedded = self.embedding(preprocessed_states)
        return embedded.view(-1, self.n_macros * self.embedding_dim)


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    env = ChipDesign(
        netlist_file=MEDIUM_NETLIST_FILE,
        init_placement=MEDIUM_INIT_PLACEMENT,
        device=str(device),
    )
    print(
        f"Environment: {env.n_macros} macros, "
        f"{env.n_grid_cells} grid cells, "
        f"{env.n_actions} actions"
    )

    preprocessor = ChipDesignPreprocessor(env, embedding_dim=args.embedding_dim)
    output_dim = preprocessor.output_dim
    assert output_dim is not None

    module_pf = MLP(
        input_dim=output_dim,
        output_dim=env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
    )
    module_pb = MLP(
        input_dim=output_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        trunk=module_pf.trunk,
    )

    pf_estimator = DiscretePolicyEstimator(
        module_pf, env.n_actions, preprocessor=preprocessor
    )
    pb_estimator = DiscretePolicyEstimator(
        module_pb, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0).to(device)
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)

    replay_buffer = ReplayBuffer(env, capacity=args.replay_buffer_size)

    for i in tqdm(range(args.n_iterations)):
        trajectories = gflownet.sample_trajectories(env, n=args.batch_size)

        if args.replay_buffer_size > 0:
            training_samples = gflownet.to_training_samples(trajectories)
            replay_buffer.add(training_samples)
            if len(replay_buffer) >= args.batch_size:
                training_samples = replay_buffer.sample(n_samples=args.batch_size)
        else:
            training_samples = gflownet.to_training_samples(trajectories)

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples)  # type: ignore[arg-type]
        loss.backward()
        optimizer.step()

        if (i + 1) % args.log_every == 0:
            with torch.no_grad():
                eval_states = gflownet.sample_terminating_states(
                    env, n=args.eval_batch_size
                )
                assert isinstance(eval_states, ChipDesignStates)
                log_rewards = env.log_reward(eval_states)
                mean_cost = -log_rewards.mean().item()
                best_cost = -log_rewards.max().item()
            print(
                f"Iter {i+1} | Loss: {loss.item():.4f} | "
                f"Mean cost: {mean_cost:.4f} | Best cost: {best_cost:.4f} | "
                f"logZ: {gflownet.logZ.item():.4f} | "  # type: ignore[operator]
                f"Replay: {len(replay_buffer)}"
            )

    print("\nTraining finished.")
    with torch.no_grad():
        final_states = gflownet.sample_terminating_states(env, n=args.eval_batch_size)
        assert isinstance(final_states, ChipDesignStates)
        log_rewards = env.log_reward(final_states)
    print(f"Final mean cost: {-log_rewards.mean().item():.4f}")
    print(f"Final best cost: {-log_rewards.max().item():.4f}")
    best_idx = int(log_rewards.argmax())
    print("Best placement:", final_states.tensor[best_idx].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_iterations", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden", type=int, default=3)
    parser.add_argument("--replay_buffer_size", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()
    main(args)
