#!/usr/bin/env python
"""
Example script for training a diffusion sampler with a GFlowNet loss (Trajectory Balance)
with SimpleGaussianMixtureTarget as the target unnormalized distribution.

Here, we use the pinned Brownian motion as the reference process; see https://arxiv.org/abs/2402.05098
for more details, and see https://arxiv.org/abs/2302.13834 or https://arxiv.org/abs/2211.01364 for
Ornstein-Uhlenbeck process as the reference process.

Reference: https://github.com/GFNOrg/gfn-diffusion
"""

import argparse
import math
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Distribution
from tqdm import tqdm

from gfn.estimators import Estimator, PolicyMixin
from gfn.gflownet import TBGFlowNet
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.common import set_seed

LOGTWOPI: float = math.log(2 * math.pi)


##############################
### Neural Network Modules ###
##############################


class PISTimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
        )
        self.register_buffer(
            "pe", torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            t: torch.Tensor
        """
        t_sin = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).sin()  # type: ignore
        t_cos = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).cos()  # type: ignore
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class PISStateEncoding(nn.Module):
    def __init__(self, x_dim: int, s_emb_dim: int) -> None:
        super().__init__()

        self.s_model = nn.Linear(x_dim, s_emb_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.s_model(s)


class PISJointPolicy(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.GELU(),  # Because this model accepts embeddings (linear projections).
            nn.Linear(s_emb_dim, hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(1e-8)  # type: ignore
            self.model[-1].bias.data.fill_(0.0)  # type: ignore

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.model(s_emb + t_emb)


class PISGradNetForward(nn.Module):  # TODO: support Learnable Backward policy
    """PISGradNet from https://arxiv.org/abs/2111.15141.

    Attributes:
    """

    def __init__(
        self,
        s_dim: int,  # dimension of states (== target.dim)
        harmonics_dim: int = 64,
        t_emb_dim: int = 64,
        s_emb_dim: int = 64,
        hidden_dim: int = 64,
        joint_layers: int = 2,
        zero_init: bool = False,
        # predict_flow: bool,  # TODO: support predict flow for db or subtb
        # share_embeddings: bool = False,
        # flow_harmonics_dim: int = 64,
        # flow_t_emb_dim: int = 64,
        # flow_s_emb_dim: int = 64,
        # flow_hidden_dim: int = 64,
        # flow_layers: int = 2,
        # lp: bool,  # TODO: support Langevin parameterization
        # lp_layers: int = 3,
        # lp_scaling_per_dimension: bool = True,
        # clipping: bool = False,  # TODO: support clipping
        # out_clip: float = 1e4,
        # lp_clip: float = 1e2,
        # learn_variance: bool = True,  # TODO: support learnable variance
        # log_var_range: float = 4.0,
    ):
        """Initialize the PISGradNetForward.

        Args:
            s_dim: The dimension of the states.
            harmonics_dim: The dimension of the harmonics.
            t_emb_dim: The dimension of the time embedding.
            s_emb_dim: The dimension of the state embedding.
            hidden_dim: The dimension of the hidden layers.
        """
        super().__init__()
        self.s_dim = s_dim
        self.input_dim = s_dim + 1  # + 1 for time, for the default IdentityPreprocessor
        self.harmonics_dim = harmonics_dim
        self.t_emb_dim = t_emb_dim
        self.s_emb_dim = s_emb_dim
        self.hidden_dim = hidden_dim
        self.joint_layers = joint_layers
        self.zero_init = zero_init
        self.out_dim = s_dim  # 2 * out_dim if learn_variance is True

        assert (
            self.s_emb_dim == self.t_emb_dim
        ), "Dimensionality of state embedding and time embedding should be the same!"

        self.t_model = PISTimeEncoding(
            self.harmonics_dim, self.t_emb_dim, self.hidden_dim
        )
        self.s_model = PISStateEncoding(self.s_dim, self.s_emb_dim)
        self.joint_model = PISJointPolicy(
            self.s_emb_dim,
            self.hidden_dim,
            self.out_dim,
            self.joint_layers,
            self.zero_init,
        )

    def forward(
        self,
        preprocessed_states: torch.Tensor,
        # grad_logr_fn: Callable,  # TODO: grad_logr_fn for lp
    ) -> torch.Tensor:
        s = preprocessed_states[..., :-1]
        t = preprocessed_states[..., -1]
        s_emb = self.s_model(s)
        t_emb = self.t_model(t)
        out = self.joint_model(s_emb, t_emb)

        # TODO: learn variance, lp, clipping, ...
        if torch.isnan(out).any():
            print("+ out has {} nans".format(torch.isnan(out).sum()))
            out = torch.nan_to_num(out)

        return out


class FixedBackwardModule(nn.Module):
    def __init__(self, s_dim: int):
        super().__init__()
        self.input_dim = s_dim + 1  # + 1 for time, for the default IdentityPreprocessor

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(preprocessed_states[..., :-1])


###########################################
### Distribution Wrapper and Estimators ###
###########################################


class IsotropicGaussianWithTime(Distribution):
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        actions_detach: bool = True,
    ):
        """
        Initialize the IsotropicGaussianWithTime distribution.

        Args:
            loc: The mean of the Gaussian distribution (shape: (*batch_shape, s_dim))
            scale: The scale of the Gaussian distribution (shape: (*batch_shape, 1))
            actions_detach: Whether to detach the actions from the graph.
        """
        super().__init__()
        self.loc = loc  # shape: (*batch_shape, s_dim)
        self.scale = scale  # shape: (*batch_shape, 1)
        self.actions_detach = actions_detach

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        noise = torch.randn(sample_shape + self.loc.shape, device=self.loc.device)
        actions = self.loc + self.scale * noise
        if self.actions_detach:
            actions = actions.detach()
        return actions

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        noise = (actions - self.loc) / self.scale
        scale_squeezed = self.scale.squeeze(-1)
        logprobs = torch.where(
            (
                (scale_squeezed.abs() < 1e-6)  # Exit case for backward sampling
                | (actions[..., 0].isinf())  # Exit case for forward sampling
            ),
            torch.zeros_like(scale_squeezed),
            -0.5 * (LOGTWOPI + 2 * torch.log(self.scale) + noise**2).sum(dim=-1),
        )
        return logprobs


class PinnedBrownianMotionForward(PolicyMixin, Estimator):
    def __init__(
        self,
        s_dim: int,
        pf_module: nn.Module,
        sigma: float,
        num_discretization_steps: int,
    ):
        """Initialize the PinnedBrownianMotionForward.

        Args:
            s_dim: The dimension of the states.
            pf_module: The neural network module to use for the forward policy.
            sigma: The diffusion coefficient parameter for the pinned Brownian motion.
            num_discretization_steps: The number of discretization steps.
        """
        self.s_dim = s_dim
        super().__init__(
            module=pf_module,
            preprocessor=None,  # Use the IdentityPreprocessor
            is_backward=False,
        )

        # Pinned-Brownian Motion related
        self.sigma = sigma
        self.num_discretization_steps = num_discretization_steps
        self.dt = 1.0 / self.num_discretization_steps

    def forward(self, input: States) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            input: The input to the module as states.

        Returns:
            The output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        s_curr = input.tensor[:, :-1]
        t_curr = input.tensor[:, [-1]]

        out = torch.where(
            (1.0 - t_curr) < self.dt * 1e-2,  # sf case; t_curr is >= 0.
            torch.full_like(s_curr, -float("inf")),
            self.module(self.preprocessor(input)),
        )

        if self.expected_output_dim is not None:
            assert out.shape[-1] == self.expected_output_dim, (
                f"Module output shape {out.shape} does not match expected output "
                f"dimension {self.expected_output_dim}"
            )
        return out

    @property
    def expected_output_dim(self) -> int:
        return self.s_dim

    def to_probability_distribution(
        self,
        states: States,
        module_output: torch.Tensor,
        **policy_kwargs: Any,
        # TODO: add epsilon-noisy exploration
    ) -> IsotropicGaussianWithTime:
        assert len(states.batch_shape) == 1, "States must have a batch_shape of length 1"
        fwd_mean = self.dt * module_output
        fwd_std = torch.tensor(self.sigma * self.dt**0.5, device=fwd_mean.device)
        fwd_std = fwd_std.repeat(fwd_mean.shape[0], 1)
        return IsotropicGaussianWithTime(fwd_mean, fwd_std)


class PinnedBrownianMotionBackward(PolicyMixin, Estimator):
    def __init__(
        self,
        s_dim: int,
        pb_module: nn.Module,
        sigma: float,
        num_discretization_steps: int,
    ):
        """Initialize the PinnedBrownianMotionForward.

        Args:
            s_dim: The dimension of the states.
            pb_module: The neural network module to use for the backward policy.
            sigma: The diffusion coefficient parameter for the pinned Brownian motion.
            num_discretization_steps: The number of discretization steps.
        """
        self.s_dim = s_dim
        super().__init__(
            module=pb_module,
            preprocessor=None,  # Use the IdentityPreprocessor
            is_backward=True,
        )

        # Pinned-Brownian Motion related
        self.sigma = sigma
        self.dt = 1.0 / num_discretization_steps

    def forward(self, input: States) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            input: The input to the module as states.

        Returns:
            The output of the module, as a tensor of shape (*batch_shape, output_dim).
        """
        s_curr = input.tensor[:, :-1]
        t_curr = input.tensor[:, [-1]]  # shape: (*batch_shape,)

        out = torch.where(
            (t_curr - self.dt) < self.dt * 1e-2,  # s0 case; t_curr and dt are >= 0.
            torch.zeros_like(s_curr),
            self.module(self.preprocessor(input)),
        )

        if self.expected_output_dim is not None:
            assert out.shape[-1] == self.expected_output_dim, (
                f"Module output shape {out.shape} does not match expected output "
                f"dimension {self.expected_output_dim}"
            )
        return out

    @property
    def expected_output_dim(self) -> int:
        return self.s_dim

    def to_probability_distribution(
        self,
        states: States,
        module_output: torch.Tensor,  # TODO: support learnable backward mean and var
        **policy_kwargs: Any,
        # TODO: add epsilon-noisy exploration
    ) -> IsotropicGaussianWithTime:
        assert len(states.batch_shape) == 1, "States must have a batch_shape of length 1"
        s_curr = states.tensor[:, :-1]
        t_curr = states.tensor[:, [-1]]

        bwd_mean = s_curr * self.dt / t_curr
        bwd_std = self.sigma * (self.dt * (t_curr - self.dt) / t_curr).sqrt()
        return IsotropicGaussianWithTime(bwd_mean, bwd_std)


###########
### Run ###
###########


def main(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Pass command-line arguments to the target class for the diffusion sampling
    # environment.
    target_kwargs = {"seed": args.target_seed}
    if args.dim is not None:
        target_kwargs["dim"] = args.dim
    if args.num_components is not None:
        target_kwargs["num_components"] = args.num_components

    # Set up environment.
    env = DiffusionSampling(
        target_str=args.target,
        target_kwargs=target_kwargs,
        num_discretization_steps=args.num_steps,
        device=device,
        check_action_validity=False,
    )

    # Build forward/backward modules and estimators
    s_dim = env.dim
    pf_module = PISGradNetForward(
        s_dim=s_dim,
        harmonics_dim=args.harmonics_dim,
        t_emb_dim=args.t_emb_dim,
        s_emb_dim=args.s_emb_dim,
        hidden_dim=args.hidden_dim,
        joint_layers=args.joint_layers,
        zero_init=args.zero_init,
    )
    pb_module = FixedBackwardModule(s_dim=s_dim)

    pf_estimator = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=args.sigma,
        num_discretization_steps=args.num_steps,
    )
    pb_estimator = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=args.sigma,
        num_discretization_steps=args.num_steps,
    )

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
    gflownet = gflownet.to(device)

    sampler = Sampler(estimator=pf_estimator)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        # On-policy sampling
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
        )

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        loss.backward()
        optimizer.step()

        if args.visualize:
            if (
                it == 0
                or (it + 1) % args.vis_interval == 0
                or it == args.n_iterations - 1
            ):
                with torch.no_grad():
                    samples_states = gflownet.sample_terminating_states(env, args.vis_n)
                    xs = samples_states.tensor[:, :-1]
                    env.target.visualize(
                        samples=xs,
                        prefix=f"it{it}_",
                        show=False,
                    )  # type: ignore[attr-defined]

        pbar.set_postfix({"loss": float(loss.item())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # System
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Target / environment
    parser.add_argument("--target", type=str, default="gmm_2", help="Target name")
    parser.add_argument(
        "--dim", type=int, default=None, help="State dimension override for the target"
    )
    parser.add_argument(
        "--num_components", type=int, default=None, help="Mixture components"
    )
    parser.add_argument("--target_seed", type=int, default=2, help="Target RNG seed")

    # Discretization / diffusion params
    parser.add_argument(
        "--num_steps", type=int, default=32, help="number of discretization steps"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,  # 5.0 is for simple_gmm target, you might need to change this for other targets
        help="diffusion coefficient for the pinned Brownian motion",
    )

    # Model (PISGradNet)
    parser.add_argument("--harmonics_dim", type=int, default=64)
    parser.add_argument("--t_emb_dim", type=int, default=64)
    parser.add_argument("--s_emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--joint_layers", type=int, default=2)
    parser.add_argument("--zero_init", action="store_true")

    # Training
    parser.add_argument("--n_iterations", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_logz", type=float, default=1e-1)

    # Visualization
    parser.add_argument("--vis_interval", type=int, default=200)
    parser.add_argument("--vis_n", type=int, default=2000)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    main(args)
