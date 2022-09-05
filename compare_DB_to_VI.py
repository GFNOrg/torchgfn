import torch
from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import encode
from tqdm import tqdm, trange

import wandb
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.envs import HyperGrid
from gfn.estimators import (
    LogitPBEstimator,
    LogitPFEstimator,
    LogZEstimator,
    LogStateFlowEstimator,
)
from gfn.losses.detailed_balance import DetailedBalance
from gfn.modules import NeuralNet, Tabular, Uniform
from gfn.parametrizations import DBParametrization
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor, OneHotPreprocessor
from gfn.samplers import LogitPFActionsSampler, TransitionsSampler
from gfn.validate import validate

parser = ArgumentParser()
parser.add_argument("--ndim", type=int, default=2)
parser.add_argument("--height", type=int, default=8)
parser.add_argument("--R0", type=float, default=0.1)
parser.add_argument(
    "--preprocessor", type=str, choices=["Identity", "KHot", "OneHot"], default="KHot"
)
parser.add_argument("--tabular", action="store_true")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_iterations", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--schedule", type=float, default=1.0)
parser.add_argument("--learn_PB", action="store_true")
parser.add_argument("--tie_PB", action="store_true")
parser.add_argument("--replay_buffer_size", type=int, default=0)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--use_db", action="store_true", default=False)
parser.add_argument("--use_baseline", action="store_true", default=False)
parser.add_argument("--wandb", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--validation_samples",
    type=int,
    default=200000,
    help="Number of validation samples to use to evaluate the pmf.",
)
parser.add_argument("--validation_interval", type=int, default=100)
parser.add_argument(
    "--validate_with_training_examples",
    action="store_true",
    default=False,
    help="If true, the pmf is obtained from the latest visited terminating states",
)

args = parser.parse_args()
print(encode(args))

torch.manual_seed(args.seed)
if args.no_cuda:
    device_str = "cpu"
else:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"


env = HyperGrid(args.ndim, args.height, R0=args.R0)
if args.preprocessor == "Identity":
    preprocessor = IdentityPreprocessor(env)
elif args.preprocessor == "OneHot":
    preprocessor = OneHotPreprocessor(env)
else:
    preprocessor = KHotPreprocessor(env)

if args.tabular:
    logit_PF = Tabular(env, output_dim=env.n_actions)
else:
    logit_PF = NeuralNet(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=256,
        n_hidden_layers=2,
    )
if args.learn_PB:
    if args.tabular:
        logit_PB = Tabular(env, output_dim=env.n_actions - 1)
    else:
        logit_PB = NeuralNet(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=256,
            n_hidden_layers=2,
            torso=logit_PF.torso if args.tie_PB else None,
        )
else:
    logit_PB = Uniform(env=env, output_dim=env.n_actions - 1)

logF = NeuralNet(
    input_dim=preprocessor.output_dim,
    output_dim=1,
    hidden_dim=256,
    n_hidden_layers=2,
    torso=logit_PF.torso if args.tie_PB else None,
)

logit_PF = LogitPFEstimator(preprocessor=preprocessor, module=logit_PF)
logit_PB = LogitPBEstimator(preprocessor=preprocessor, module=logit_PB)
logF = LogStateFlowEstimator(preprocessor=preprocessor, module=logF)
parametrization = DBParametrization(logit_PF, logit_PB, logF)


actions_sampler = LogitPFActionsSampler(estimator=logit_PF)
transitions_sampler = TransitionsSampler(env=env, actions_sampler=actions_sampler)

loss_fn = DetailedBalance(parametrization=parametrization)


params = [
    {
        "params": parametrization.parameters.values(),
        "lr": args.lr,
    }
]
optimizer = torch.optim.Adam(params)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=list(range(2000, 100000, 2000)), gamma=args.schedule
)


use_replay_buffer = args.replay_buffer_size > 0
if args.replay_buffer_size > 0:
    use_replay_buffer = True
    replay_buffer = ReplayBuffer(
        env, capacity=args.replay_buffer_size, objects="transitions"
    )


use_wandb = len(args.wandb) > 0


if use_wandb:
    wandb.init(project=args.wandb)
    wandb.config.update(encode(args))
    run_name = ("DB_") if args.use_db else ("VI_")
    run_name += f"lr{args.lr}_"
    run_name += "baseline_" if args.use_baseline else ""
    run_name += f"_{args.ndim}_{args.height}_{args.R0}_{args.seed}_"
    wandb.run.name = run_name + wandb.run.name.split("-")[-1]  # type: ignore


visited_terminating_states = (
    env.States() if args.validate_with_training_examples else None
)


with trange(args.n_iterations, desc="Training") as pbar:
    postfix = None
    for i in pbar:
        transitions = transitions_sampler.sample(n_objects=args.batch_size)
        if use_replay_buffer:
            replay_buffer.add(transitions)  # type: ignore
            training_objects = replay_buffer.sample(n_objects=args.batch_size)  # type: ignore
        else:
            training_objects = transitions

        optimizer.zero_grad()
        if args.use_db:
            logPF_actions, logPB_actions, scores = loss_fn.get_scores(training_objects)
            loss = (scores).pow(2)
            loss = loss.mean()

        else:  # DOESN'T WORK BECAUSE IT'S NOT FULLY ONLINE !
            states_raw = transitions.states.states.repeat(5, 1)
            forward_masks = transitions.states.forward_masks.repeat(5, 1)
            backward_masks = transitions.states.backward_masks.repeat(5, 1)
            states = env.States(states_raw, forward_masks, backward_masks)
            new_transitions = transitions_sampler.sample_transitions(states)
            new_logPF_actions, new_logPB_actions, new_scores = loss_fn.get_scores(
                new_transitions
            )
            new_scores = new_scores.view(5, -1).transpose(0, 1)
            new_logPF_actions = new_logPF_actions.view(5, -1).transpose(0, 1)
            new_logPB_actions = new_logPB_actions.view(5, -1).transpose(0, 1)

            if args.use_baseline:
                new_scores = (
                    new_scores - torch.mean(new_scores, 1).unsqueeze(1).detach()
                )

            # V1: should be ok
            loss = torch.mean(new_scores**2)
            # V2:
            # loss = new_logPF_actions * (new_scores).detach()
            # loss += new_logPB_actions
            # loss += new_scores - new_logPF_actions + new_logPB_actions
            # loss = loss.mean()
        loss.backward()

        optimizer.step()
        scheduler.step()
        if args.validate_with_training_examples:
            visited_terminating_states.extend(training_objects.last_states)  # type: ignore
        to_log = {"loss": loss.item(), "states_visited": (i + 1) * args.batch_size}
        if use_wandb:
            wandb.log(to_log, step=i)
        pbar.set_postfix(postfix)
        if i % args.validation_interval == 0:
            validation_info = validate(
                env,
                parametrization,
                args.validation_samples,
                visited_terminating_states,
            )
            if use_wandb:
                wandb.log(validation_info, step=i)
            to_log.update(validation_info)
            postfix = to_log.copy()
