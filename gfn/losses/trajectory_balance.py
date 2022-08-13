import torch
from torchtyping import TensorType
from gfn.containers import Trajectories
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.samplers.action_samplers import LogitPBActionSampler, LogitPFActionSampler
from gfn.parametrizations import ForwardBackwardTransitionParametrizationWithZ


class TrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        o: ForwardBackwardTransitionParametrizationWithZ,
        reward_clip_min: float = 1e-5,
    ):
        self.o = o
        self.reward_clip_min = reward_clip_min
        self.action_sampler = LogitPFActionSampler(o.logit_PF)
        self.backward_action_sampler = LogitPBActionSampler(o.logit_PB)

    def __call__(self, trajectories: Trajectories) -> TensorType[0]:
        pf_logits = self.action_sampler.get_logits(trajectories.states)
        log_pf_all = pf_logits.log_softmax(dim=-1)
        log_pf_actions = torch.gather(
            log_pf_all, dim=-1, index=trajectories.actions.unsqueeze(-1)
        ).squeeze(-1)
        forward_mask = (
            torch.arange(log_pf_actions.shape[0])
            .unsqueeze(1)
            .lt(1 + trajectories.when_is_done.unsqueeze(0))
            .float()
        )
        log_pf_actions *= forward_mask
        log_pf_trajectories = torch.sum(log_pf_actions, dim=0)  # should be 1D

        pb_logits = self.backward_action_sampler.get_logits(trajectories.states)[1:]
        log_pb_all = pb_logits.log_softmax(dim=-1)
        # log_pb_al is now a 3D tensor of shape [n_steps - 1, n_trajectories, n_actions - 1]
        # in order to be able to use "gather", we need to add a dimension to the last dimension
        # we can fill it with anything we want, given that it will be masked out later
        new_log_pb_all = torch.cat(
            [log_pb_all, 5 * torch.ones(log_pb_all.shape[:-1]).unsqueeze(-1)], -1
        )
        # the following tensor should be of shape [n_steps - 1, n_trajectories]
        log_pb_actions = torch.gather(
            new_log_pb_all, dim=-1, index=trajectories.actions[:-1].unsqueeze(-1)
        ).squeeze(-1)
        backward_mask = (
            torch.arange(new_log_pb_all.shape[0])
            .unsqueeze(1)
            .lt(trajectories.when_is_done.unsqueeze(0))
            .float()
        )
        log_pb_actions.nan_to_num_()  # TODO: this seems necessary, but I don't like it
        log_pb_actions *= backward_mask
        log_pb_trajectories = torch.sum(log_pb_actions, dim=0)  # should be 1D

        preds = log_pf_trajectories - log_pb_trajectories + self.o.logZ.logZ

        targets = torch.log(trajectories.rewards.clamp_min(self.reward_clip_min))

        loss = torch.nn.MSELoss()(preds, targets)
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss


if __name__ == "__main__":
    from gfn.envs import HyperGrid
    from gfn.preprocessors import KHotPreprocessor
    from gfn.samplers.action_samplers import FixedActions
    from gfn.models import NeuralNet, Uniform
    from gfn.samplers.trajectories_sampler import TrajectoriesSampler
    from gfn.estimators import LogitPFEstimator, LogitPBEstimator, LogZEstimator

    n_envs = 5
    height = 4
    ndim = 2
    env = HyperGrid(n_envs=n_envs, height=height, ndim=ndim)

    print("Evaluating the loss on 5 trajectories with manually chosen actions")
    action_sampler = FixedActions(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 2, 2, 1]]
        )
    )
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    print(trajectories)

    preprocessor = KHotPreprocessor(env)
    pf = Uniform(env.n_actions)
    pb = Uniform(env.n_actions - 1)

    logZ = torch.tensor(0.0)
    logit_PF = LogitPFEstimator(preprocessor, env, pf)
    logit_PB = LogitPBEstimator(preprocessor, env, pb)
    logZ = LogZEstimator(logZ)

    o = ForwardBackwardTransitionParametrizationWithZ(logit_PF, logZ, logit_PB)

    loss = TrajectoryBalance(o)
    print(loss(trajectories))
    # sanity check, by hand, we should get the following loss
    pbs = torch.tensor([0.5, 1, 1, 0.25, 1.0])
    pfs = torch.tensor(
        [1.0 / (3**3), 1.0 / (3**3) * 0.5, 1.0 / 3, 1.0 / (3**4), 1.0 / (3**2)]
    )
    true_losses_exp = torch.exp(logZ.logZ) * pfs / (pbs * trajectories.rewards)
    true_loss = torch.log(true_losses_exp).pow(2).mean()

    if true_loss == loss(trajectories):
        print("OK - TB LOSS PROBABLY OK")
    else:
        raise ValueError("TB LOSS NOT PROPERLY CALCULATED")

    print(
        "Evaluating the loss on 5 trajectories with randomly chosen actions (P_F and P_B as modules)"
    )

    preprocessor = KHotPreprocessor(env)
    pf = NeuralNet(
        input_dim=preprocessor.output_dim, output_dim=env.n_actions, hidden_dim=16
    )

    pb = NeuralNet(
        input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1, hidden_dim=16
    )

    print("Now with LogitPFActionSampler")
    logit_PF = LogitPFEstimator(preprocessor, env, pf)
    action_sampler = LogitPFActionSampler(logit_PF=logit_PF)
    trajectories_sampler = TrajectoriesSampler(env, action_sampler)
    trajectories = trajectories_sampler.sample_trajectories()
    print(trajectories)

    logZ = torch.tensor(-5.0)

    logit_PB = LogitPBEstimator(preprocessor, env, pb)
    logZ = LogZEstimator(logZ)

    o = ForwardBackwardTransitionParametrizationWithZ(logit_PF, logZ, logit_PB)

    loss = TrajectoryBalance(o)
    print(loss(trajectories))
