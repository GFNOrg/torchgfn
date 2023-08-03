r"""
The goal of this script is to reproduce some of the published results on the Box
environment. Run one of the following commands to reproduce some of the results in
[A theory of continuous generative flow networks](https://arxiv.org/abs/2301.12594)


python train_box.py --delta {0.1, 0.25} --tied {--uniform_pb} --loss {TB, DB}
"""

from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from tqdm import tqdm, trange

from gfn.gflownet import (
    DBGFlowNet,
    LogPartitionVarianceGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import Box
from gfn.gym.helpers.box_utils import (
    BoxPBEstimator,
    BoxPBNeuralNet,
    BoxPBUniform,
    BoxPFEstimator,
    BoxPFNeuralNet,
    BoxStateFlowModule,
)
from gfn.modules import ScalarEstimator


def sample_from_reward(env: Box, n_samples: int):
    """Samples states from the true reward distribution

    Implement rejection sampling, with proposal being uniform distribution in [0, 1]^2
    Returns:
        A numpy array of shape (n_samples, 2) containing the sampled states
    """
    samples = []
    while len(samples) < n_samples:
        sample = env.reset(batch_shape=(n_samples,), random=True)
        rewards = env.reward(sample)
        rand_n = torch.rand(n_samples).to(env.device)
        mask = rand_n * (env.R0 + max(env.R1, env.R2)) < rewards
        true_samples = sample[mask]
        samples.extend(true_samples[-(n_samples - len(samples)) :].tensor.cpu().numpy())
    return np.array(samples)


def get_test_states(n=100, maxi=1.0):
    """Create a list of states from [0, 1]^2 by discretizing it into n x n grid.

    Returns:
        A numpy array of shape (n^2, 2) containing the test states,
    """
    x = np.linspace(0.001, maxi, n)
    y = np.linspace(0.001, maxi, n)
    xx, yy = np.meshgrid(x, y)
    test_states = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    return test_states


def estimate_jsd(kde1, kde2):
    """Estimate Jensen-Shannon divergence between two distributions defined by KDEs

    Returns:
        A float value of the estimated JSD
    """
    test_states = get_test_states()
    log_dens1 = kde1.score_samples(test_states)
    log_dens1 = log_dens1 - logsumexp(log_dens1)
    log_dens2 = kde2.score_samples(test_states)
    log_dens2 = log_dens2 - logsumexp(log_dens2)
    log_dens = np.log(0.5 * np.exp(log_dens1) + 0.5 * np.exp(log_dens2))
    jsd = np.sum(np.exp(log_dens1) * (log_dens1 - log_dens))
    jsd += np.sum(np.exp(log_dens2) * (log_dens2 - log_dens))
    return jsd / 2.0


if __name__ == "__main__":  # noqa: C901
    parser = ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument(
        "--delta",
        type=float,
        default=0.25,
        help="maximum distance between two successive states",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed, if 0 then a random seed is used",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["TB", "DB", "SubTB", "ZVar"],
        default="TB",
        help="Loss function to use",
    )
    parser.add_argument(
        "--subTB_weighting",
        type=str,
        default="geometric_within",
        help="weighting scheme for SubTB",
    )
    parser.add_argument(
        "--subTB_lambda", type=float, default=0.9, help="Lambda parameter for SubTB"
    )

    parser.add_argument(
        "--min_concentration",
        type=float,
        default=0.1,
        help="minimal value for the Beta concentration parameters",
    )

    parser.add_argument(
        "--max_concentration",
        type=float,
        default=5.1,
        help="maximal value for the Beta concentration parameters",
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of Beta distributions for P_F(s'|s)",
    )
    parser.add_argument(
        "--n_components_s0",
        type=int,
        default=4,
        help="Number of Beta distributions for P_F(s'|s_0)",
    )

    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform PB")
    parser.add_argument(
        "--tied",
        action="store_true",
        help="Tie the parameters of PF, PB. F is never tied.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of the estimators' neural network modules.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=4,
        help="Number of hidden layers (of size `hidden_dim`) in the estimators'"
        + " neural network modules",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_Z",
        type=float,
        default=1e-3,
        help="Specific learning rate for logZ",
    )
    parser.add_argument(
        "--lr_F",
        type=float,
        default=1e-2,
        help="Specific learning rate for the state flow function (only used for DB and SubTB losses)",
    )
    parser.add_argument(
        "--gamma_scheduler",
        type=float,
        default=0.5,
        help="Every scheduler_milestone steps, multiply the learning rate by gamma_scheduler",
    )
    parser.add_argument(
        "--scheduler_milestone",
        type=int,
        default=2500,
        help="Every scheduler_milestone steps, multiply the learning rate by gamma_scheduler",
    )

    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(3e6),
        help="Total budget of trajectories to train on. "
        + "Training iterations = n_trajectories // batch_size",
    )

    parser.add_argument(
        "--validation_interval",
        type=int,
        default=500,
        help="How often (in training steps) to validate the gflownet",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=10000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )

    args = parser.parse_args()

    seed = args.seed if args.seed != 0 else torch.randint(int(10e10), (1,))[0].item()
    torch.manual_seed(seed)

    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    use_wandb = len(args.wandb_project) > 0
    if use_wandb:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    n_iterations = args.n_trajectories // args.batch_size

    # 1. Create the environment
    env = Box(delta=args.delta, epsilon=1e-10, device_str=device_str)

    # 2. Create the gflownet.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    gflownet = None
    pf_module = BoxPFNeuralNet(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        n_components=args.n_components,
        n_components_s0=args.n_components_s0,
    )
    if args.uniform_pb:
        pb_module = BoxPBUniform()
    else:
        pb_module = BoxPBNeuralNet(
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            n_components=args.n_components,
            torso=pf_module.torso if args.tied else None,
        )

    pf_estimator = BoxPFEstimator(
        env,
        pf_module,
        n_components_s0=args.n_components_s0,
        n_components=args.n_components,
        min_concentration=args.min_concentration,
        max_concentration=args.max_concentration,
    )
    pb_estimator = BoxPBEstimator(
        env,
        pb_module,
        n_components=args.n_components if not args.uniform_pb else 1,
        min_concentration=args.min_concentration,
        max_concentration=args.max_concentration,
    )
    module = None
    logZ = None
    if args.loss in ("DB", "SubTB"):
        # We always need a LogZEstimator
        logZ = torch.tensor(0.0, device=env.device, requires_grad=True)
        # We need a LogStateFlowEstimator

        module = BoxStateFlowModule(
            input_dim=env.preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            torso=None,  # We do not tie the parameters of the flow function to PF
            logZ_value=logZ,
        )
        logF_estimator = ScalarEstimator(env=env, module=module)

        if args.loss == "DB":
            gflownet = DBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                logF=logF_estimator,
                on_policy=True,
            )
        else:
            gflownet = SubTBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                logF=logF_estimator,
                on_policy=True,
                weighting=args.subTB_weighting,
                lamda=args.subTB_lambda,
            )
    elif args.loss == "TB":
        gflownet = TBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            on_policy=True,
        )
    elif args.loss == "ZVar":
        gflownet = LogPartitionVarianceGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            on_policy=True,
        )

    assert gflownet is not None, f"No gflownet for loss {args.loss}"

    # 3. Create the optimizer and scheduler

    optimizer = torch.optim.Adam(pf_module.parameters(), lr=args.lr)
    if not args.uniform_pb:
        optimizer.add_param_group(
            {
                "params": pb_module.last_layer.parameters()
                if args.tied
                else pb_module.parameters(),
                "lr": args.lr,
            }
        )
    if args.loss in ("DB", "SubTB"):
        assert module is not None
        optimizer.add_param_group(
            {
                "params": module.parameters(),
                "lr": args.lr_F,
            }
        )
    if "logZ" in dict(gflownet.named_parameters()):
        logZ = dict(gflownet.named_parameters())["logZ"]
    if args.loss != "ZVar":
        assert logZ is not None
        optimizer.add_param_group({"params": [logZ], "lr": args.lr_Z})

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            i * args.scheduler_milestone
            for i in range(1, 1 + int(n_iterations / args.scheduler_milestone))
        ],
        gamma=args.gamma_scheduler,
    )

    # 4. Sample from the true reward distribution, and fit a KDE to the samples
    samples_from_reward = sample_from_reward(env, n_samples=args.validation_samples)
    true_kde = KernelDensity(kernel="exponential", bandwidth=0.1).fit(
        samples_from_reward
    )

    states_visited = 0

    jsd = float("inf")
    for iteration in trange(n_iterations):
        if iteration % 1000 == 0:
            print(f"current optimizer LR: {optimizer.param_groups[0]['lr']}")

        trajectories = gflownet.sample_trajectories(n_samples=args.batch_size)

        training_samples = gflownet.to_training_samples(trajectories)

        optimizer.zero_grad()
        loss = gflownet.loss(training_samples)

        loss.backward()
        for p in gflownet.parameters():
            p.grad.data.clamp_(-10, 10).nan_to_num_(0.0)
        optimizer.step()
        scheduler.step()

        states_visited += len(trajectories)

        to_log = {"loss": loss.item(), "states_visited": states_visited}
        logZ_info = ""
        if args.loss != "ZVar":
            assert logZ is not None
            to_log.update({"logZdiff": env.log_partition - logZ.item()})
            logZ_info = f"logZ: {logZ.item():.2f}, "
        if use_wandb:
            wandb.log(to_log, step=iteration)
        if iteration % (args.validation_interval // 5) == 0:
            tqdm.write(
                f"States: {states_visited}, Loss: {loss.item():.3f}, {logZ_info}true logZ: {env.log_partition:.2f}, JSD: {jsd:.4f}"
            )

        if iteration % args.validation_interval == 0:
            validation_samples = gflownet.sample_terminating_states(
                args.validation_samples
            )
            kde = KernelDensity(kernel="exponential", bandwidth=0.1).fit(
                validation_samples.tensor.detach().cpu().numpy()
            )
            jsd = estimate_jsd(kde, true_kde)

            if use_wandb:
                wandb.log({"JSD": jsd}, step=iteration)

            to_log.update({"JSD": jsd})
