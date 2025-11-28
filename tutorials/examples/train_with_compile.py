#!/usr/bin/env python
r"""
Benchmark the runtime impact of different `torch.compile` strategies on several
GFlowNet losses (Trajectory Balance, Detailed Balance, SubTB) and environments.

Four compile modes are compared for each (env, loss) pair:

0) Pure eager execution
1) Compile loss only
2) Compile estimator modules only (`try_compile_gflownet`)
3) Compile both the loss wrapper and estimator modules

The script reuses the components defined in the example training scripts:
`train_hypergrid.py`, `train_line.py`, `train_bitsequence_recurrent.py`,
`train_bit_sequences.py`, `train_graph_ring.py`, and `train_diffusion_sampler.py`.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal

import torch
from tqdm import tqdm

try:  # pragma: no cover - older PyTorch versions may lack torch._dynamo
    import torch._dynamo as _torch_dynamo

    _torch_dynamo.config.capture_scalar_outputs = True
except Exception:  # pragma: no cover
    _torch_dynamo = None

from gfn.actions import GraphActions
from gfn.env import Env
from gfn.estimators import (
    DiscreteGraphPolicyEstimator,
    DiscretePolicyEstimator,
    PinnedBrownianMotionBackward,
    PinnedBrownianMotionForward,
    RecurrentDiscretePolicyEstimator,
    ScalarEstimator,
)
from gfn.gflownet import ModifiedDBGFlowNet, PFBasedGFlowNet, SubTBGFlowNet
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.gym.bitSequence import BitSequence
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.gym.graph_building import GraphBuildingOnEdges
from gfn.gym.line import Line
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.common import set_seed
from gfn.utils.compile import try_compile_gflownet
from gfn.utils.modules import (
    MLP,
    DiffusionFixedBackwardModule,
    DiffusionPISGradNetForward,
    GraphActionGNN,
    GraphEdgeActionMLP,
    GraphScalarMLP,
    RecurrentDiscreteSequenceModel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tutorials.examples.train_graph_ring import RingReward  # noqa: E402
from tutorials.examples.train_line import GaussianStepMLP, StepEstimator  # noqa: E402

# ---------------------------------------------------------------------------
# Dataclasses and shared configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowVariant:
    key: Literal["tb", "modified_dbg", "subtb"]
    label: str
    description: str
    requires_logf: bool


FLOW_VARIANTS: dict[str, FlowVariant] = {
    "tb": FlowVariant(
        key="tb",
        label="Trajectory Balance",
        description="Standard TB loss with learnable log Z.",
        requires_logf=False,
    ),
    "modified_dbg": FlowVariant(
        key="modified_dbg",
        label="Modified Detailed Balance",
        description="Modified detailed balance variant without learned log-state flows.",
        requires_logf=False,
    ),
    "subtb": FlowVariant(
        key="subtb",
        label="Sub-trajectory Balance",
        description="SubTB with configurable weighting scheme.",
        requires_logf=True,
    ),
}
DEFAULT_FLOW_ORDER = ["tb", "modified_dbg", "subtb"]


@dataclass(frozen=True)
class CompileMode:
    key: Literal["eager", "loss", "estimators", "both"]
    label: str
    description: str
    compile_loss: bool
    compile_estimators: bool


COMPILE_MODES: dict[str, CompileMode] = {
    "eager": CompileMode(
        key="eager",
        label="Eager",
        description="No compilation; reference runtime.",
        compile_loss=False,
        compile_estimators=False,
    ),
    "loss": CompileMode(
        key="loss",
        label="Loss Only",
        description="Compile the loss wrapper while leaving estimators eager.",
        compile_loss=True,
        compile_estimators=False,
    ),
    "estimators": CompileMode(
        key="estimators",
        label="Estimators Only",
        description="Compile estimator modules via try_compile_gflownet.",
        compile_loss=False,
        compile_estimators=True,
    ),
    "both": CompileMode(
        key="both",
        label="Loss + Estimators",
        description="Compile estimator modules and the loss wrapper.",
        compile_loss=True,
        compile_estimators=True,
    ),
}
DEFAULT_COMPILE_ORDER = ["eager", "loss", "estimators", "both"]
COMPILE_MODE_COLORS: dict[str, str] = {
    "eager": "#000000",
    "loss": "#1f77b4",
    "estimators": "#d62728",
    "both": "#ff7f0e",
}


@dataclass
class TrainingComponents:
    env: Env
    gflownet: PFBasedGFlowNet
    optimizer: torch.optim.Optimizer
    sampler: Sampler | None
    use_training_samples: bool = False
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    recalc_logprobs: bool = True
    notes: str = ""


@dataclass
class EnvironmentBenchmark:
    key: Literal[
        "hypergrid", "line", "bitseq_recurrent", "bitseq_mlp", "diffusion", "graph_ring"
    ]
    label: str
    description: str
    color: str
    builder: Callable[
        [argparse.Namespace, torch.device, FlowVariant], TrainingComponents
    ]
    supported_flows: list[str]


# Default hypergrid parameters (shared across builders + CLI defaults).
HYPERGRID_DEFAULTS: Dict[str, Any] = {
    "ndim": 2,
    "height": 32,
    "reward_fn_str": "original",
    "reward_fn_kwargs": {"R0": 0.1, "R1": 0.5, "R2": 2.0},
    "calculate_partition": False,
    "store_all_states": False,
    "check_action_validity": __debug__,
}


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _normalize_keys(
    requested: list[str], valid: Dict[str, Any], label: str
) -> list[str]:
    normalized: list[str] = []
    for key in requested:
        alias = key.lower()
        if alias not in valid:
            raise ValueError(
                f"Unsupported {label} '{key}'. Choose from {', '.join(sorted(valid))}."
            )
        if alias not in normalized:
            normalized.append(alias)
    return normalized


def _mps_backend_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_backend_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "mps" and not _mps_backend_available():
        raise RuntimeError("MPS requested but not available.")
    return device


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and _mps_backend_available() and hasattr(torch, "mps"):
        torch.mps.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compile strategies across multiple GFlowNet workloads."
    )
    parser.add_argument("--n-iterations", type=int, default=50, dest="n_iterations")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--warmup-iters", type=int, default=5, dest="warmup_iters")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Grad clip value.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    parser.add_argument(
        "--lr-logz", type=float, default=1e-1, dest="lr_logz", help="LogZ learning rate."
    )
    parser.add_argument(
        "--lr-logf", type=float, default=1e-3, dest="lr_logf", help="LogF learning rate."
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device preference.",
    )

    parser.add_argument(
        "--environments",
        nargs="+",
        default=[
            "hypergrid",
            "line",
            "bitseq_recurrent",
            "bitseq_mlp",
            "diffusion",
            "graph_ring",
        ],
        help="Subset of environments to benchmark.",
    )
    parser.add_argument(
        "--gflownets",
        nargs="+",
        default=DEFAULT_FLOW_ORDER,
        help="Loss variants to benchmark (tb, modified_dbg, subtb).",
    )
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        default=DEFAULT_COMPILE_ORDER,
        help="Compile modes to evaluate (eager, loss, estimators, both).",
    )

    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=str(Path.home() / "compile_benchmark.png"),
        help="Path to save the optional benchmark plot.",
    )
    parser.add_argument("--skip-plot", action="store_true", help="Disable plotting.")
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default="reduce-overhead",
        help="Mode passed to torch.compile.",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Request `fullgraph=True` when compiling the loss wrapper.",
    )
    parser.add_argument(
        "--large-models",
        action="store_true",
        help="Double estimator widths/depths to stress-test large model builds.",
    )

    # Hypergrid knobs
    parser.add_argument("--hypergrid-ndim", type=int, default=HYPERGRID_DEFAULTS["ndim"])
    parser.add_argument(
        "--hypergrid-height", type=int, default=HYPERGRID_DEFAULTS["height"]
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden dim.")
    parser.add_argument("--n-hidden", type=int, default=2, help="#hidden layers.")

    # Line environment knobs
    parser.add_argument("--line-n-steps", type=int, default=5)
    parser.add_argument("--line-hidden-dim", type=int, default=64)
    parser.add_argument("--line-n-hidden", type=int, default=2)
    parser.add_argument("--line-std-min", type=float, default=0.1)
    parser.add_argument("--line-std-max", type=float, default=1.0)

    # Bit sequence knobs
    parser.add_argument("--bitseq-word-size", type=int, default=3)
    parser.add_argument("--bitseq-seq-size", type=int, default=9)
    parser.add_argument("--bitseq-n-modes", type=int, default=5)
    parser.add_argument("--bitseq-temperature", type=float, default=1.0)
    parser.add_argument("--bitseq-embedding-dim", type=int, default=64)
    parser.add_argument("--bitseq-hidden-size", type=int, default=128)
    parser.add_argument("--bitseq-num-layers", type=int, default=2)
    parser.add_argument("--bitseq-dropout", type=float, default=0.0)
    parser.add_argument("--bitseq-mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--bitseq-mlp-n-hidden", type=int, default=2)

    # Diffusion knobs
    parser.add_argument("--diffusion-target", type=str, default="gmm2")
    parser.add_argument("--diffusion-num-steps", type=int, default=32)
    parser.add_argument("--diffusion-sigma", type=float, default=5.0)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-joint-layers", type=int, default=2)
    parser.add_argument("--diffusion-harmonics-dim", type=int, default=64)
    parser.add_argument("--diffusion-t-emb-dim", type=int, default=64)
    parser.add_argument("--diffusion-s-emb-dim", type=int, default=64)
    parser.add_argument("--diffusion-zero-init", action="store_true")
    parser.add_argument("--diffusion-dim", type=int, default=None)
    parser.add_argument("--diffusion-num-components", type=int, default=None)
    parser.add_argument("--diffusion-target-seed", type=int, default=2)

    # Graph ring knobs
    parser.add_argument("--ring-n-nodes", type=int, default=4)
    parser.add_argument("--ring-directed", action="store_true", default=True)
    parser.add_argument("--ring-use-gnn", action="store_true", default=True)
    parser.add_argument("--ring-embedding-dim", type=int, default=128)
    parser.add_argument("--ring-num-conv-layers", type=int, default=1)
    parser.add_argument("--ring-num-edge-classes", type=int, default=2)
    parser.add_argument("--ring-reward", type=float, default=100.0)
    parser.add_argument("--ring-reward-eps", type=float, default=1e-6)

    # SubTB knobs
    parser.add_argument(
        "--subtb-weighting",
        type=str,
        default="geometric_within",
        help="Weighting scheme for SubTB.",
    )
    parser.add_argument(
        "--subtb-lambda",
        type=float,
        default=0.9,
        dest="subtb_lamda",
        help="Lambda parameter for SubTB.",
    )

    return parser.parse_args()


LARGE_MODEL_FIELDS = [
    "hidden_dim",
    "n_hidden",
    "line_hidden_dim",
    "line_n_hidden",
    "bitseq_embedding_dim",
    "bitseq_hidden_size",
    "bitseq_num_layers",
    "bitseq_mlp_hidden_dim",
    "bitseq_mlp_n_hidden",
    "diffusion_hidden_dim",
    "diffusion_joint_layers",
    "diffusion_harmonics_dim",
    "diffusion_t_emb_dim",
    "diffusion_s_emb_dim",
    "ring_embedding_dim",
    "ring_num_conv_layers",
]


def _apply_large_model_scaling(args: argparse.Namespace) -> None:
    if not getattr(args, "large_models", False):
        return
    for attr in LARGE_MODEL_FIELDS:
        value = getattr(args, attr, None)
        if value is None:
            continue
        if isinstance(value, int):
            setattr(args, attr, max(1, value * 2))
    print("[config] Enabled large model mode: doubled estimator sizes.")


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------


def _build_hypergrid_components(
    args: argparse.Namespace, device: torch.device, variant: FlowVariant
) -> TrainingComponents:
    kwargs = dict(HYPERGRID_DEFAULTS)
    kwargs["ndim"] = args.hypergrid_ndim
    kwargs["height"] = args.hypergrid_height
    kwargs["device"] = device
    env = HyperGrid(**kwargs)

    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    pf_module = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
    )
    pb_module = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        trunk=pf_module.trunk,
    )

    pf = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    pb = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )

    logF: ScalarEstimator | None = None
    if variant.requires_logf:
        logF_module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
        )
        logF = ScalarEstimator(module=logF_module, preprocessor=preprocessor)

    if variant.key == "tb":
        gflownet = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0)
    elif variant.key == "modified_dbg":
        gflownet = ModifiedDBGFlowNet(pf=pf, pb=pb)
    elif variant.key == "subtb":
        assert logF is not None
        gflownet = SubTBGFlowNet(
            pf=pf,
            pb=pb,
            logF=logF,
            weighting=args.subtb_weighting,
            lamda=args.subtb_lamda,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported FlowVariant '{variant.key}'")

    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)

    logz_params = getattr(gflownet, "logz_parameters", None)
    if callable(logz_params):
        params = logz_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logz})
    logf_params = getattr(gflownet, "logF_parameters", None)
    if callable(logf_params):
        params = logf_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logf})

    sampler = Sampler(estimator=pf)
    sampler_kwargs = {
        "save_logprobs": False,
        "save_estimator_outputs": False,
        "epsilon": 0.05,
    }
    use_training_samples = variant.key in {"modified_dbg", "subtb"}
    return TrainingComponents(
        env=env,
        gflownet=gflownet,
        optimizer=optimizer,
        sampler=sampler,
        sampler_kwargs=sampler_kwargs,
        recalc_logprobs=True,
        use_training_samples=use_training_samples,
    )


def _build_line_components(
    args: argparse.Namespace, device: torch.device, variant: FlowVariant
) -> TrainingComponents:
    env = Line(
        mus=[2, 5],
        sigmas=[0.5, 0.5],
        init_value=0,
        n_sd=4.5,
        n_steps_per_trajectory=args.line_n_steps,
        device=device,
    )

    pf_module = GaussianStepMLP(
        hidden_dim=args.line_hidden_dim,
        n_hidden_layers=args.line_n_hidden,
        policy_std_min=args.line_std_min,
        policy_std_max=args.line_std_max,
    )
    pb_module = GaussianStepMLP(
        hidden_dim=args.line_hidden_dim,
        n_hidden_layers=args.line_n_hidden,
        policy_std_min=args.line_std_min,
        policy_std_max=args.line_std_max,
    )

    pf = StepEstimator(env, pf_module, backward=False)
    pb = StepEstimator(env, pb_module, backward=True)

    logF: ScalarEstimator | None = None
    if variant.requires_logf:
        logF_module = MLP(
            input_dim=2,  # [position, step counter]
            output_dim=1,
            hidden_dim=args.line_hidden_dim,
            n_hidden_layers=args.line_n_hidden,
        )
        logF = ScalarEstimator(module=logF_module)

    if variant.key == "tb":
        gflownet = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0)
    elif variant.key == "modified_dbg":
        gflownet = ModifiedDBGFlowNet(pf=pf, pb=pb)
    elif variant.key == "subtb":
        assert logF is not None
        gflownet = SubTBGFlowNet(
            pf=pf,
            pb=pb,
            logF=logF,
            weighting=args.subtb_weighting,
            lamda=args.subtb_lamda,
        )
    else:
        raise ValueError(f"Unsupported FlowVariant '{variant.key}'")

    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    logz_params = getattr(gflownet, "logz_parameters", None)
    if callable(logz_params):
        params = logz_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logz})
    logf_params = getattr(gflownet, "logF_parameters", None)
    if callable(logf_params):
        params = logf_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logf})

    sampler = Sampler(estimator=pf)
    sampler_kwargs = {
        "save_logprobs": False,
        "save_estimator_outputs": False,
    }
    use_training_samples = variant.key in {"modified_dbg", "subtb"}
    return TrainingComponents(
        env=env,
        gflownet=gflownet,
        optimizer=optimizer,
        sampler=sampler,
        sampler_kwargs=sampler_kwargs,
        recalc_logprobs=True,
        use_training_samples=use_training_samples,
    )


def _build_bitsequence_recurrent_components(
    args: argparse.Namespace, device: torch.device, variant: FlowVariant
) -> TrainingComponents:
    H = torch.randint(
        0,
        2,
        (args.bitseq_n_modes, args.bitseq_seq_size),
        dtype=torch.long,
        device=device,
    )
    env = BitSequence(
        word_size=args.bitseq_word_size,
        seq_size=args.bitseq_seq_size,
        n_modes=args.bitseq_n_modes,
        temperature=args.bitseq_temperature,
        H=H,
        device_str=str(device),
        seed=args.seed,
        check_action_validity=__debug__,
    )

    pf_module = RecurrentDiscreteSequenceModel(
        vocab_size=env.n_actions,
        embedding_dim=args.bitseq_embedding_dim,
        hidden_size=args.bitseq_hidden_size,
        num_layers=args.bitseq_num_layers,
        dropout=args.bitseq_dropout,
    ).to(device)
    pf = RecurrentDiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        is_backward=False,
    )

    if variant.key != "tb":
        raise ValueError(
            "BitSequence benchmark currently supports Trajectory Balance only."
        )

    gflownet = TBGFlowNet(
        pf=pf,
        pb=None,
        init_logZ=0.0,
        constant_pb=True,
    ).to(device)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    sampler_kwargs = {
        "n": args.batch_size,
        "save_logprobs": True,
        "save_estimator_outputs": False,
        "epsilon": 0.05,
    }
    return TrainingComponents(
        env=env,
        gflownet=gflownet,
        optimizer=optimizer,
        sampler=None,  # Use gflownet.sample_trajectories for recurrent adapter support.
        sampler_kwargs=sampler_kwargs,
        recalc_logprobs=False,
    )


def _build_bitsequence_mlp_components(
    args: argparse.Namespace, device: torch.device, variant: FlowVariant
) -> TrainingComponents:
    H = torch.randint(
        0,
        2,
        (args.bitseq_n_modes, args.bitseq_seq_size),
        dtype=torch.long,
        device=device,
    )
    env = BitSequence(
        word_size=args.bitseq_word_size,
        seq_size=args.bitseq_seq_size,
        n_modes=args.bitseq_n_modes,
        temperature=args.bitseq_temperature,
        H=H,
        device_str=str(device),
        seed=args.seed,
        check_action_validity=__debug__,
    )

    pf_module = MLP(
        input_dim=env.words_per_seq,
        output_dim=env.n_actions,
        hidden_dim=args.bitseq_mlp_hidden_dim,
        n_hidden_layers=args.bitseq_mlp_n_hidden,
    )
    pb_module = MLP(
        input_dim=env.words_per_seq,
        output_dim=env.n_actions - 1,
        hidden_dim=args.bitseq_mlp_hidden_dim,
        n_hidden_layers=args.bitseq_mlp_n_hidden,
        trunk=pf_module.trunk,
    )
    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
    )
    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
    )

    logF: ScalarEstimator | None = None
    if variant.requires_logf:
        logF_module = MLP(
            input_dim=env.words_per_seq,
            output_dim=1,
            hidden_dim=args.bitseq_mlp_hidden_dim,
            n_hidden_layers=args.bitseq_mlp_n_hidden,
            trunk=pf_module.trunk,
        )
        logF = ScalarEstimator(module=logF_module)

    if variant.key == "tb":
        gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
    elif variant.key == "modified_dbg":
        gflownet = ModifiedDBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    elif variant.key == "subtb":
        assert logF is not None
        gflownet = SubTBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            logF=logF,
            weighting=args.subtb_weighting,
            lamda=args.subtb_lamda,
        )
    else:
        raise ValueError(f"Unsupported FlowVariant '{variant.key}' for bitseq_mlp")

    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)

    logz_params = getattr(gflownet, "logz_parameters", None)
    if callable(logz_params):
        params = logz_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logz})
    logf_params = getattr(gflownet, "logF_parameters", None)
    if callable(logf_params):
        params = logf_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logf})

    sampler = Sampler(estimator=pf_estimator)
    sampler_kwargs = {
        "save_logprobs": False,
        "save_estimator_outputs": False,
        "epsilon": 0.05,
    }
    use_training_samples = variant.key in {"modified_dbg", "subtb"}
    return TrainingComponents(
        env=env,
        gflownet=gflownet,
        optimizer=optimizer,
        sampler=sampler,
        sampler_kwargs=sampler_kwargs,
        recalc_logprobs=True,
        use_training_samples=use_training_samples,
    )


def _build_diffusion_components(
    args: argparse.Namespace, device: torch.device, variant: FlowVariant
) -> TrainingComponents:
    target_kwargs: Dict[str, Any] = {"seed": args.diffusion_target_seed}
    if args.diffusion_dim is not None:
        target_kwargs["dim"] = args.diffusion_dim
    if args.diffusion_num_components is not None:
        target_kwargs["num_components"] = args.diffusion_num_components

    env = DiffusionSampling(
        target_str=args.diffusion_target,
        target_kwargs=target_kwargs,
        num_discretization_steps=args.diffusion_num_steps,
        device=device,
        check_action_validity=False,
    )

    pf_module = DiffusionPISGradNetForward(
        s_dim=env.dim,
        harmonics_dim=args.diffusion_harmonics_dim,
        t_emb_dim=args.diffusion_t_emb_dim,
        s_emb_dim=args.diffusion_s_emb_dim,
        hidden_dim=args.diffusion_hidden_dim,
        joint_layers=args.diffusion_joint_layers,
        zero_init=args.diffusion_zero_init,
    )
    pb_module = DiffusionFixedBackwardModule(s_dim=env.dim)

    pf = PinnedBrownianMotionForward(
        s_dim=env.dim,
        pf_module=pf_module,
        sigma=args.diffusion_sigma,
        num_discretization_steps=args.diffusion_num_steps,
    )
    pb = PinnedBrownianMotionBackward(
        s_dim=env.dim,
        pb_module=pb_module,
        sigma=args.diffusion_sigma,
        num_discretization_steps=args.diffusion_num_steps,
    )

    logF: ScalarEstimator | None = None
    if variant.requires_logf:
        logF_module = MLP(
            input_dim=env.state_shape[-1],
            output_dim=1,
            hidden_dim=args.diffusion_hidden_dim,
            n_hidden_layers=args.diffusion_joint_layers,
        )
        preproc = IdentityPreprocessor(output_dim=env.state_shape[-1])
        logF = ScalarEstimator(module=logF_module, preprocessor=preproc)

    if variant.key == "tb":
        gflownet = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0)
    elif variant.key == "modified_dbg":
        gflownet = ModifiedDBGFlowNet(pf=pf, pb=pb)
    elif variant.key == "subtb":
        assert logF is not None
        gflownet = SubTBGFlowNet(
            pf=pf,
            pb=pb,
            logF=logF,
            weighting=args.subtb_weighting,
            lamda=args.subtb_lamda,
        )
    else:
        raise ValueError(f"Unsupported FlowVariant '{variant.key}'")

    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)

    logz_params = getattr(gflownet, "logz_parameters", None)
    if callable(logz_params):
        params = logz_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logz})
    logf_params = getattr(gflownet, "logF_parameters", None)
    if callable(logf_params):
        params = logf_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logf})

    sampler = Sampler(estimator=pf)
    sampler_kwargs = {
        "save_logprobs": True,
        "save_estimator_outputs": False,
        "n": args.batch_size,
    }
    use_training_samples = variant.key in {"modified_dbg", "subtb"}
    return TrainingComponents(
        env=env,
        gflownet=gflownet,
        optimizer=optimizer,
        sampler=sampler,
        sampler_kwargs=sampler_kwargs,
        recalc_logprobs=False,
        use_training_samples=use_training_samples,
    )


def _build_graph_ring_components(
    args: argparse.Namespace, device: torch.device, variant: FlowVariant
) -> TrainingComponents:
    state_evaluator = RingReward(
        directed=args.ring_directed,
        reward_val=args.ring_reward,
        eps_val=args.ring_reward_eps,
        device=device,
    )
    env = GraphBuildingOnEdges(
        n_nodes=args.ring_n_nodes,
        state_evaluator=state_evaluator,
        directed=args.ring_directed,
        device=device,
    )

    num_node_classes = getattr(env, "num_node_classes", args.ring_n_nodes)
    num_edge_classes = getattr(env, "num_edge_classes", args.ring_num_edge_classes)

    if args.ring_use_gnn:
        module_pf = GraphActionGNN(
            num_node_classes=num_node_classes,
            directed=args.ring_directed,
            num_conv_layers=args.ring_num_conv_layers,
            num_edge_classes=num_edge_classes,
            embedding_dim=args.ring_embedding_dim,
        )
        module_pb = GraphActionGNN(
            num_node_classes=num_node_classes,
            directed=args.ring_directed,
            is_backward=True,
            num_conv_layers=args.ring_num_conv_layers,
            num_edge_classes=num_edge_classes,
            embedding_dim=args.ring_embedding_dim,
        )
    else:
        module_pf = GraphEdgeActionMLP(
            args.ring_n_nodes,
            args.ring_directed,
            num_node_classes=num_node_classes,
            num_edge_classes=num_edge_classes,
            embedding_dim=args.ring_embedding_dim,
        )
        module_pb = GraphEdgeActionMLP(
            args.ring_n_nodes,
            args.ring_directed,
            is_backward=True,
            num_node_classes=num_node_classes,
            num_edge_classes=num_edge_classes,
            embedding_dim=args.ring_embedding_dim,
        )

    pf_estimator = DiscreteGraphPolicyEstimator(module=module_pf)
    pb_estimator = DiscreteGraphPolicyEstimator(module=module_pb, is_backward=True)

    logF: ScalarEstimator | None = None
    if variant.requires_logf:
        logF_module = GraphScalarMLP(
            n_nodes=args.ring_n_nodes,
            directed=args.ring_directed,
            embedding_dim=args.ring_embedding_dim,
            n_outputs=1,
            n_hidden_layers=2,
        )
        logF = ScalarEstimator(module=logF_module)

    if variant.key == "tb":
        gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
    elif variant.key == "modified_dbg":
        gflownet = ModifiedDBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    elif variant.key == "subtb":
        assert logF is not None
        gflownet = SubTBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            logF=logF,
            weighting=args.subtb_weighting,
            lamda=args.subtb_lamda,
        )
    else:
        raise ValueError(f"Unsupported FlowVariant '{variant.key}' for graph ring")

    gflownet = gflownet.to(device)
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    logz_params = getattr(gflownet, "logz_parameters", None)
    if callable(logz_params):
        params = logz_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logz})
    logf_params = getattr(gflownet, "logF_parameters", None)
    if callable(logf_params):
        params = logf_params()
        if params:
            optimizer.add_param_group({"params": params, "lr": args.lr_logf})

    sampler = Sampler(estimator=pf_estimator)
    epsilon_dict = {
        GraphActions.ACTION_TYPE_KEY: 0.05,
        GraphActions.EDGE_INDEX_KEY: 0.05,
        GraphActions.NODE_CLASS_KEY: 0.05,
        GraphActions.NODE_INDEX_KEY: 0.05,
        GraphActions.EDGE_CLASS_KEY: 0.05,
    }
    sampler_kwargs = {
        "save_logprobs": False,
        "save_estimator_outputs": False,
        "epsilon": epsilon_dict,
    }
    use_training_samples = variant.key in {"modified_dbg", "subtb"}
    return TrainingComponents(
        env=env,
        gflownet=gflownet,
        optimizer=optimizer,
        sampler=sampler,
        sampler_kwargs=sampler_kwargs,
        recalc_logprobs=True,
        use_training_samples=use_training_samples,
    )


ENVIRONMENT_BENCHMARKS: dict[str, EnvironmentBenchmark] = {
    "hypergrid": EnvironmentBenchmark(
        key="hypergrid",
        label="HyperGrid",
        description="High-dimensional discrete grid.",
        color="#4a90e2",
        builder=_build_hypergrid_components,
        supported_flows=list(DEFAULT_FLOW_ORDER),
    ),
    "line": EnvironmentBenchmark(
        key="line",
        label="Line",
        description="Continuous 1D environment with Gaussian rewards.",
        color="#ffa600",
        builder=_build_line_components,
        supported_flows=list(DEFAULT_FLOW_ORDER),
    ),
    "bitseq_recurrent": EnvironmentBenchmark(
        key="bitseq_recurrent",
        label="BitSequence (Recurrent)",
        description="Recurrent sequence generation benchmark.",
        color="#2ca02c",
        builder=_build_bitsequence_recurrent_components,
        supported_flows=["tb"],  # Recurrent path supports TB only.
    ),
    "bitseq_mlp": EnvironmentBenchmark(
        key="bitseq_mlp",
        label="BitSequence (MLP)",
        description="Tabular/MLP bit sequence benchmark.",
        color="#17becf",
        builder=_build_bitsequence_mlp_components,
        supported_flows=["tb"],  # Only TB is stable for this environment.
    ),
    "diffusion": EnvironmentBenchmark(
        key="diffusion",
        label="Diffusion Sampler",
        description="Pinned Brownian motion diffusion sampling.",
        color="#a17be7",
        builder=_build_diffusion_components,
        supported_flows=list(DEFAULT_FLOW_ORDER),
    ),
    "graph_ring": EnvironmentBenchmark(
        key="graph_ring",
        label="Graph Ring",
        description="Graph-building environment for ring structures.",
        color="#ff7f0e",
        builder=_build_graph_ring_components,
        supported_flows=list(DEFAULT_FLOW_ORDER),
    ),
}


# ---------------------------------------------------------------------------
# Training + benchmarking utilities
# ---------------------------------------------------------------------------


def prepare_loss_fn(
    gflownet: PFBasedGFlowNet,
    compile_mode: CompileMode,
    variant: FlowVariant,
    args: argparse.Namespace,
) -> Callable[[Env, Any, bool], torch.Tensor]:
    def loss_wrapper(env: Env, training_batch: Any, recalc: bool) -> torch.Tensor:
        return gflownet.loss(
            env,
            training_batch,
            recalculate_all_logprobs=recalc,
        )

    if compile_mode.compile_loss:
        if variant.requires_logf:
            print(
                "[compile loss] Skipping torch.compile for Detailed/SubTB "
                "variants due to unsupported logF ops."
            )
            return loss_wrapper
        return torch.compile(
            loss_wrapper,
            mode=args.torch_compile_mode,
            fullgraph=args.compile_fullgraph,
        )
    return loss_wrapper


def maybe_compile_estimators(
    components: TrainingComponents,
    compile_mode: CompileMode,
    dynamo_mode: str,
) -> bool:
    if not compile_mode.compile_estimators:
        return False
    results = try_compile_gflownet(components.gflownet, mode=dynamo_mode)
    if not isinstance(results, dict):
        print("[compile estimators] try_compile_gflownet returned None/unknown result")
        return False
    joined = ", ".join(
        f"{name}:{'✓' if success else 'x'}" for name, success in results.items()
    )
    print(f"[compile estimators] {joined}")
    return any(results.values())


def sample_trajectories(
    components: TrainingComponents,
    batch_size: int,
) -> Any:
    kwargs = dict(components.sampler_kwargs)
    kwargs.setdefault("n", batch_size)
    if components.sampler is None:
        return components.gflownet.sample_trajectories(components.env, **kwargs)
    return components.sampler.sample_trajectories(components.env, **kwargs)


def training_loop(
    components: TrainingComponents,
    loss_fn: Callable[[Env, Any, bool], torch.Tensor],
    args: argparse.Namespace,
    *,
    n_iters: int,
    track_time: bool,
) -> tuple[float | None, Dict[str, list[float]]]:
    iterator: Iterable[int] = range(n_iters)
    iterator = tqdm(iterator, dynamic_ncols=True) if n_iters > 1 else iterator

    history = {"losses": [], "iter_times": []}
    start_time = time.perf_counter() if track_time else None

    for step in iterator:
        iter_start = time.perf_counter() if track_time else None
        trajectories = sample_trajectories(components, args.batch_size)
        training_batch = (
            components.gflownet.to_training_samples(trajectories)
            if components.use_training_samples
            else trajectories
        )

        components.optimizer.zero_grad()
        loss = loss_fn(components.env, training_batch, components.recalc_logprobs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(components.gflownet.parameters(), args.grad_clip)
        components.optimizer.step()

        history["losses"].append(loss.item())
        if iter_start is not None:
            history["iter_times"].append(time.perf_counter() - iter_start)

        if isinstance(iterator, tqdm):
            iterator.set_postfix({"loss": loss.item(), "iter": step + 1})

    elapsed = None
    if track_time:
        synchronize_if_needed(getattr(components.env, "device", torch.device("cpu")))
        elapsed = time.perf_counter() - start_time  # type: ignore[arg-type]
    return elapsed, history


def run_case(
    args: argparse.Namespace,
    device: torch.device,
    env_cfg: EnvironmentBenchmark,
    variant: FlowVariant,
    compile_mode: CompileMode,
) -> dict[str, Any]:
    set_seed(args.seed)
    components = env_cfg.builder(args, device, variant)
    loss_fn = prepare_loss_fn(components.gflownet, compile_mode, variant, args)
    estimator_compiled = maybe_compile_estimators(
        components, compile_mode, args.torch_compile_mode
    )

    if args.warmup_iters > 0:
        training_loop(
            components,
            loss_fn,
            args,
            n_iters=args.warmup_iters,
            track_time=False,
        )

    elapsed, history = training_loop(
        components,
        loss_fn,
        args,
        n_iters=args.n_iterations,
        track_time=True,
    )

    return {
        "env_key": env_cfg.key,
        "env_label": env_cfg.label,
        "gflownet_key": variant.key,
        "gflownet_label": variant.label,
        "compile_key": compile_mode.key,
        "label": compile_mode.label,
        "compile_description": compile_mode.description,
        "elapsed": elapsed or 0.0,
        "losses": history["losses"],
        "iter_times": history["iter_times"],
        "use_compile_estimator": estimator_compiled,
        "use_compile_loss": compile_mode.compile_loss,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (adapted from the earlier sketch)
# ---------------------------------------------------------------------------


def _summarize_iteration_times(times: list[float]) -> tuple[float, float]:
    if not times:
        return 0.0, 0.0
    mean_time = statistics.fmean(times)
    std_time = statistics.pstdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time


VARIANT_COLORS: dict[str, str] = {
    "tb": "#000000",
    "modified_dbg": "#1f77b4",
    "subtb": "#d62728",
}
LOSS_LINE_ALPHA = 0.5


def summarize_results(results: list[dict[str, Any]]) -> None:
    print("\nBenchmark summary:")
    grouped: Dict[tuple[str, str], list[dict[str, Any]]] = {}
    for res in results:
        grouped.setdefault((res["env_key"], res["gflownet_key"]), []).append(res)

    for (env_key, flow_key), entries in grouped.items():
        env_label = entries[0]["env_label"]
        flow_label = entries[0]["gflownet_label"]
        print(f"\n[{env_label}] {flow_label}")
        baseline = next((e for e in entries if e["compile_key"] == "eager"), entries[0])
        baseline_time = baseline["elapsed"] or 1.0
        for entry in entries:
            elapsed = entry["elapsed"]
            speedup = baseline_time / elapsed if elapsed else float("inf")
            print(
                f"  - {entry['label']:<18} {elapsed:.2f}s  "
                f"({speedup:.2f}x vs eager)  "
                f"compile_loss={entry['use_compile_loss']} "
                f"compile_estimators={entry['use_compile_estimator']}"
            )


def plot_benchmark(
    results: list[dict[str, Any]], output_path: str, run_label: str | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from exc

    env_keys = list({res["env_key"] for res in results})
    if not env_keys:
        print("No results collected; skipping plot.")
        return

    fig, axes = plt.subplots(len(env_keys), 3, figsize=(20, 5 * len(env_keys)))
    if run_label:
        fig.suptitle(f"Run label: {run_label}", fontsize=18)
    if len(env_keys) == 1:
        axes = [axes]  # type: ignore[list-item]

    palette = ["#6c757d", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for row_idx, env_key in enumerate(env_keys):
        env_results = [res for res in results if res["env_key"] == env_key]
        env_label = env_results[0]["env_label"]
        row_axes = axes[row_idx]

        labels = [f"{res['label']} [{res['gflownet_label']}]" for res in env_results]
        times = [res["elapsed"] for res in env_results]
        eager_baselines: dict[str, float] = {
            res["gflownet_key"]: res["elapsed"]
            for res in env_results
            if res["compile_key"] == "eager" and (res["elapsed"] or 0.0) > 0.0
        }
        fallback_baseline = min((t for t in times if t > 0), default=0.0)
        colors = [
            COMPILE_MODE_COLORS.get(res["compile_key"], palette[i % len(palette)])
            for i, res in enumerate(env_results)
        ]
        bars = row_axes[0].bar(labels, times, color=colors)
        row_axes[0].set_ylabel("Time (s)")
        row_axes[0].set_title(f"{env_label} | Total time")
        for bar, res in zip(bars, env_results):
            value = res["elapsed"]
            baseline = eager_baselines.get(res["gflownet_key"], fallback_baseline)
            rel = value / baseline if baseline and baseline > 0 else 0.0
            row_axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{rel:.2f}x",
                ha="center",
                va="bottom",
            )

        loss_ax = row_axes[1]
        for idx, res in enumerate(env_results):
            losses = res["losses"]
            if not losses:
                continue
            color = VARIANT_COLORS.get(res["gflownet_key"], palette[idx % len(palette)])
            loss_ax.plot(
                range(1, len(losses) + 1),
                losses,
                label=labels[idx],
                color=color,
                alpha=LOSS_LINE_ALPHA,
            )
        loss_ax.set_title(f"{env_label} | Loss curves")
        loss_ax.set_xlabel("Iteration")
        loss_ax.set_ylabel("Loss")
        if loss_ax.lines:
            loss_ax.legend(fontsize="small")

        iter_ax = row_axes[2]
        stats = [_summarize_iteration_times(res["iter_times"]) for res in env_results]
        means = [mean * 1000.0 for mean, _ in stats]
        stds = [std * 1000.0 for _, std in stats]
        iter_ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
        iter_ax.set_ylabel("Iteration time (ms)")
        iter_ax.set_title(f"{env_label} | Iter timing")

        for ax in row_axes:
            for label in ax.get_xticklabels():
                label.set_rotation(25)
                label.set_ha("right")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if run_label:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    if run_label:
        print(f"Saved benchmark plot to {output} [{run_label}]")
    else:
        print(f"Saved benchmark plot to {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    _apply_large_model_scaling(args)
    device = resolve_device(args.device)

    env_keys = _normalize_keys(args.environments, ENVIRONMENT_BENCHMARKS, "environment")
    flow_keys = _normalize_keys(args.gflownets, FLOW_VARIANTS, "GFlowNet variant")
    compile_keys = _normalize_keys(args.compile_modes, COMPILE_MODES, "compile mode")

    if not env_keys or not flow_keys or not compile_keys:
        raise ValueError("Nothing to benchmark—please specify envs, flows, and modes.")

    results: list[dict[str, Any]] = []

    for env_key in env_keys:
        env_cfg = ENVIRONMENT_BENCHMARKS[env_key]
        for flow_key in flow_keys:
            if flow_key not in env_cfg.supported_flows:
                print(
                    f"[skip] {env_cfg.label} does not support {FLOW_VARIANTS[flow_key].label}"
                )
                continue
            for compile_key in compile_keys:
                compile_mode = COMPILE_MODES[compile_key]
                print(
                    f"\n[run] Env={env_cfg.label}, Loss={FLOW_VARIANTS[flow_key].label}, "
                    f"Mode={compile_mode.label}"
                )
                result = run_case(
                    args,
                    device,
                    env_cfg,
                    FLOW_VARIANTS[flow_key],
                    compile_mode,
                )
                results.append(result)

    summarize_results(results)
    if not args.skip_plot:
        run_label = "large_model_run" if args.large_models else None
        plot_benchmark(results, args.benchmark_output, run_label=run_label)


if __name__ == "__main__":
    main()
