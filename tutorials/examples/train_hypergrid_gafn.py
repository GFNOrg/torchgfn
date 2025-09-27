#!/usr/bin/env python
r"""
A version of GFlowNet training that implements Generative Augmented Flow Networks (GAFN,
https://arxiv.org/abs/2210.03308). It is a variant of GFlowNet that introduces intrinsic
rewards to the GFlowNet training. It relies on the Random Network Distillation (RND,
https://arxiv.org/abs/1810.12894) to define intrinsic rewards.

Example usage:
python train_hypergrid_gafn.py --ndim 2 --height 8 --rnd_reward_scale 0.005

Key features:
- Implements GAFN training
- Uses RND to define intrinsic rewards
- Based on TB loss like the train_hypergrid_simple.py example
"""

import argparse
from typing import cast

import torch
import torch.nn as nn
from tqdm import tqdm

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.estimators import DiscretePolicyEstimator, Estimator, ScalarEstimator
from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor, Preprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform
from gfn.utils.training import validate


class RND(nn.Module):
    """
    Random Network Distillation (RND) module. It is a module that predicts the random
    target net from the state.
    """

    def __init__(
        self,
        state_dim: int,
        preprocessor: Preprocessor,
        reward_scale: float = 0.1,
        loss_scale: float = 0.1,
        hidden_dim: int = 256,
        s_latent_dim: int = 128,
    ) -> None:
        """
        Random Network Distillation (RND) module. It is a module that predicts the
        random target net from the state.

        Args:
            state_dim: The dimension of the state space.
            preprocessor: The preprocessor for the state space.
            reward_scale: The scale of the reward.
            loss_scale: The scale of the loss.
            hidden_dim: The dimension of the hidden layer.
            s_latent_dim: The dimension of the latent state.
        """
        super().__init__()

        self.preprocessor = preprocessor
        self.reward_scale = reward_scale
        self.loss_scale = loss_scale

        self.random_target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim),
        )

        # Note the same architecture as the random target net
        self.predictor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim),
        )

        self.reward_scale = reward_scale

    def forward(self, states: States) -> torch.Tensor:
        l2_error = torch.zeros(states.batch_shape, device=states.device)

        valid_states = states[~states.is_sink_state]

        states_tensor = self.preprocessor(valid_states).to(l2_error.dtype)
        random_target_feature = self.random_target_net(states_tensor).detach()
        predictor_feature = self.predictor_net(states_tensor)

        l2_error[~states.is_sink_state] = torch.norm(
            random_target_feature - predictor_feature, dim=-1, p=2
        )
        return l2_error

    def compute_intrinsic_reward(self, states: States) -> torch.Tensor:
        l2_error = self(states)
        return l2_error.detach() * self.reward_scale

    def compute_rnd_loss(self, states: States) -> torch.Tensor:
        l2_error = self(states)
        l2_error = l2_error.sum(dim=0) / (~states.is_sink_state).sum(dim=0)
        return l2_error.mean() * self.loss_scale


class TBGAFN(TBGFlowNet):
    """
    Generative Augmented Flow Networks based on the Trajectory Balance loss.

    Args:
        pf: The forward policy estimator.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator,
        rnd: RND,
        logZ: nn.Parameter | ScalarEstimator | None = None,
        init_logZ: float = 0.0,
        use_edge_ri: bool = False,
        flow_estimator: ScalarEstimator | None = None,
        log_reward_clip_min: float = -float("inf"),
    ):
        """Initializes a TBGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator.
            rnd: The RND module.
            logZ: A learnable parameter or a ScalarEstimator instance (for
                conditional GFNs).
            init_logZ: The initial value for the logZ parameter (used if logZ is None).
            use_edge_ri: Whether to use edge-based intrinsic rewards.
            flow_estimator: The flow estimator, required if use_edge_ri is True.
            log_reward_clip_min: If finite, clips log rewards to this value.
        """
        super().__init__(pf, pb, logZ, init_logZ, log_reward_clip_min)
        self.rnd = rnd
        self.use_edge_ri = use_edge_ri
        if use_edge_ri and flow_estimator is None:
            raise ValueError("flow_estimator is required if use_edge_ri is True")
        self.flow_estimator = flow_estimator

    def rnd_parameters(self) -> list[torch.Tensor]:
        return list(self.rnd.parameters())

    def flow_parameters(self) -> list[torch.Tensor]:
        if self.flow_estimator is None:
            return []
        return list(
            [v for k, v in self.flow_estimator.named_parameters() if "trunk" not in k]
        )

    def get_scores(
        self, trajectories: Trajectories, recalculate_all_logprobs: bool = True
    ) -> torch.Tensor:
        """Computes Trajectory Balance scores with intrinsic rewards for a batch of
        trajectories.

        Args:
            trajectories: The Trajectories object to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns:
            A tensor of shape (n_trajectories,) containing the scores for each trajectory.
        """
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        log_rewards = trajectories.log_rewards
        assert log_rewards is not None
        if self.use_edge_ri:
            # Use the edge-based intrinsic rewards.
            # Note that the original implementation uses the edge-based intrinsic
            # rewards (eq. (4) of the original paper) by default rather than the joint
            # intermediate reward (eq. (6)).
            # Also, they define r(s_t \to s_{t+1}) = r(s_{t+1}) and here we follow
            # this definition.
            assert self.flow_estimator is not None

            # Note: These two computations below may require a lot of memory if trajectory
            # length is very long, because this operation is performed on the entire
            # batch of trajectories, i.e., batch_size becomes O(T x B).
            # Consider mini-batching if this is a problem for your use case.
            log_state_flows = torch.zeros(
                trajectories.states.batch_shape, device=trajectories.states.device
            )
            log_state_flows[~trajectories.states.is_sink_state] = self.flow_estimator(
                trajectories.states[~trajectories.states.is_sink_state]
            ).squeeze(-1)
            edge_ri = torch.zeros(
                trajectories.states.batch_shape, device=trajectories.states.device
            )
            edge_ri[~trajectories.states.is_sink_state] = (
                self.rnd.compute_intrinsic_reward(
                    trajectories.states[~trajectories.states.is_sink_state]
                )
            )

            terminal_ri = edge_ri[
                trajectories.terminating_idx - 1,
                torch.arange(trajectories.n_trajectories, device=edge_ri.device),
            ]  # shape: (n_trajectories,)
            _terminal_part = torch.stack(
                [log_rewards, terminal_ri.log()], dim=0
            ).logsumexp(dim=0)
            _interm_part = torch.stack(
                [log_pb_trajectories, edge_ri[1:].log() - log_state_flows[1:]], dim=0
            ).logsumexp(dim=0)
            log_target = _terminal_part + _interm_part.sum(dim=0)
        else:
            # Use the state-based intrinsic rewards.
            # We use only the terminal states to compute the intrinsic rewards.
            # This isn't from the original paper, but it's a special case of the joint
            # intermediate reward (eq. (6)) with the edge-based intrinsic rewards
            # being zero, i.e., r(s_t \to s_{t+1}) = 0 for all t.
            terminal_ri = self.rnd.compute_intrinsic_reward(
                trajectories.terminating_states
            )  # shape: (n_trajectories,)
            _terminal_part = torch.stack(
                [log_rewards, terminal_ri.log()], dim=0
            ).logsumexp(dim=0)
            log_target = _terminal_part + log_pb_trajectories.sum(dim=0)

        scores = log_pf_trajectories.sum(dim=0) - log_target
        return scores

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        loss = super().loss(env, trajectories, recalculate_all_logprobs, reduction)
        rnd_loss = self.rnd.compute_rnd_loss(trajectories.states)
        return loss + rnd_loss


def main(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Setup the Environment.
    env = HyperGrid(
        ndim=args.ndim,
        height=args.height,
        reward_fn_str="sparse",
        device=device,
        calculate_partition=True,
        store_all_states=True,
    )
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    # Build the GFlowNet.
    input_dim = (
        preprocessor.output_dim
        if preprocessor.output_dim is not None
        else env.state_shape[-1]
    )
    module_PF = MLP(
        input_dim=input_dim,
        output_dim=env.n_actions,
        activation_fn="leaky_relu",  # use leaky relu as in the original GAFN paper
    )
    if not args.uniform_pb:
        module_PB = MLP(
            input_dim=input_dim,
            output_dim=env.n_actions - 1,
            trunk=module_PF.trunk,
        )
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)
    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    rnd = RND(
        state_dim=input_dim,
        preprocessor=preprocessor,
        reward_scale=args.rnd_reward_scale,
        loss_scale=args.rnd_loss_scale,
        hidden_dim=args.rnd_hidden_dim,
        s_latent_dim=args.rnd_s_latent_dim,
    )
    flow_estimator = None
    if args.use_edge_ri:
        flow_estimator = ScalarEstimator(
            module=MLP(
                input_dim=input_dim,
                output_dim=1,
                trunk=module_PF.trunk,
            ),
            preprocessor=preprocessor,
        )
    gflownet = TBGAFN(
        pf=pf_estimator,
        pb=pb_estimator,
        init_logZ=0.0,
        rnd=rnd,
        use_edge_ri=args.use_edge_ri,
        flow_estimator=flow_estimator,
    )

    # Feed pf to the sampler.
    sampler = Sampler(estimator=pf_estimator)

    # Move the gflownet to the GPU.
    gflownet = gflownet.to(device)

    # Policy parameters have their own LR. Log Z gets dedicated learning rate
    # (typically higher).
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})
    optimizer.add_param_group({"params": gflownet.rnd_parameters(), "lr": args.lr_rnd})
    if args.use_edge_ri:
        optimizer.add_param_group({"params": gflownet.flow_parameters(), "lr": args.lr})

    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
        )
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        loss.backward()
        optimizer.step()
        if (it + 1) % args.validation_interval == 0:
            validation_info, _ = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )
            print(f"Iter {it + 1}: L1 distance {validation_info['l1_dist']:.8f}")
            print(
                f"RND loss: {gflownet.rnd.compute_rnd_loss(trajectories.states).item()}"
            )

        pbar.set_postfix({"loss": loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=8, help="Height of the environment"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
        "--uniform_pb", action="store_true", help="Use a uniform backward policy"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument(
        "--validation_interval", type=int, default=100, help="Validation interval"
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=100000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    # GAFN & RND parameters.
    parser.add_argument(
        "--use_edge_ri",
        action="store_true",
        help="Use edge-based intrinsic rewards",
    )
    parser.add_argument(
        "--lr_rnd",
        type=float,
        default=1e-3,
        help="Learning rate for the RND module",
    )
    parser.add_argument(
        "--rnd_reward_scale",
        type=float,
        default=0.1,
        help="The scale of the reward for the RND module",
    )
    parser.add_argument(
        "--rnd_loss_scale",
        type=float,
        default=1.0,
        help="The scale of the loss for the RND module",
    )
    parser.add_argument(
        "--rnd_hidden_dim",
        type=int,
        default=256,
        help="The dimension of the hidden layer for the RND module",
    )
    parser.add_argument(
        "--rnd_s_latent_dim",
        type=int,
        default=128,
        help="The dimension of the latent state for the RND module",
    )
    args = parser.parse_args()

    main(args)
