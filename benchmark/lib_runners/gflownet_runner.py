"""GFlowNet library (external) runner for benchmarking.

This runner uses the gflownet library from benchmark/gflownet/.
The library is tightly coupled with Hydra, so we use Hydra's compose API
to build the configuration and instantiate the GFlowNet agent.

Supports environments: hypergrid, ising, box (ccube).
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch

from benchmark.lib_runners.base import BenchmarkConfig, LibraryRunner

# Add gflownet to path
GFLOWNET_PATH = Path(__file__).parent.parent / "gflownet"
if str(GFLOWNET_PATH) not in sys.path:
    sys.path.insert(0, str(GFLOWNET_PATH))


class GFlowNetRunner(LibraryRunner):
    """Benchmark runner for the external gflownet library.

    This library uses Hydra for configuration, so we use Hydra's compose API
    to build the configuration programmatically.

    Supports environments:
    - hypergrid: Discrete grid navigation (uses grid env)
    - ising: Discrete Ising model environment
    - box: Continuous cube environment (uses ccube env)
    """

    name = "gflownet"

    def __init__(self):
        self.agent = None
        self.device = None
        self.config = None
        self._iteration = 0

    def setup(self, config: BenchmarkConfig, seed: int) -> None:
        """Initialize environment, model, and optimizer using Hydra compose."""
        import random

        import numpy as np
        from gflownet.utils.common import gflownet_from_config
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict

        self.config = config

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reset CUDA memory stats
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # Clear any existing Hydra state
        GlobalHydra.instance().clear()

        # Use Hydra compose to build Hydra configuration
        config_dir = str(GFLOWNET_PATH / "config")

        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            # Build environment-specific overrides
            overrides = self._get_env_overrides(config, seed)

            cfg = compose(config_name="train", overrides=overrides)

            # Set a dummy log directory
            with open_dict(cfg):
                cfg.logger = OmegaConf.create(
                    {
                        "_target_": "gflownet.utils.logger.Logger",
                        "do": {"online": False, "checkpoints": False, "times": False},
                        "logdir": {
                            "root": "/tmp/gflownet_benchmark",
                            "path": "/tmp/gflownet_benchmark",
                            "ckpts": "ckpts",
                            "overwrite": True,
                        },
                        "project_name": "benchmark",
                        "lightweight": True,
                        "progressbar": {"skip": True, "n_iters_mean": 100},
                        "debug": False,
                        "context": "0",
                        "notes": None,
                        "entity": None,
                        "run_id": None,
                        "is_resumed": False,
                        "run_name": None,
                        "run_name_date": False,
                        "run_name_job": False,
                    }
                )
                if not hasattr(cfg, "n_samples"):
                    cfg.n_samples = 0

            # Create the GFlowNet agent
            self.agent = gflownet_from_config(cfg)

        self._iteration = 0

    def _get_env_overrides(self, config: BenchmarkConfig, seed: int) -> List[str]:
        """Get Hydra overrides for the specified environment."""
        env_name = config.env_name

        # Common overrides for all environments
        common_overrides = [
            # Policy settings
            "policy=mlp",
            f"policy.forward.n_hid={config.hidden_dim}",
            f"policy.forward.n_layers={config.n_layers}",
            # Training settings
            f"gflownet.optimizer.batch_size.forward={config.batch_size}",
            f"gflownet.optimizer.lr={config.lr}",
            f"gflownet.optimizer.lr_z_mult={config.lr_logz / config.lr}",
            f"gflownet.optimizer.n_train_steps={config.n_iterations + config.n_warmup}",
            # Device and seed
            f"device={self.device.type}",
            f"seed={seed}",
            # Disable evaluation for benchmarking
            "evaluator.period=0",
        ]

        if env_name == "hypergrid":
            return self._get_hypergrid_overrides(config) + common_overrides
        elif env_name == "ising":
            return self._get_ising_overrides(config) + common_overrides
        elif env_name == "box":
            return self._get_ccube_overrides(config) + common_overrides
        else:
            raise ValueError(f"Unknown environment: {env_name}")

    def _get_hypergrid_overrides(self, config: BenchmarkConfig) -> List[str]:
        """Get Hydra overrides for hypergrid environment."""
        ndim = config.env_kwargs["ndim"]
        height = config.env_kwargs["height"]

        return [
            # Grid environment (default env is grid)
            f"env.n_dim={ndim}",
            f"env.length={height}",
            # Proxy - configure corners proxy inline
            "proxy._target_=gflownet.proxy.box.corners.Corners",
            "+proxy.mu=0.75",
            "+proxy.sigma=0.05",
            "+proxy.do_gaussians=true",
            "+proxy.do_threshold=false",
            "++proxy.reward_function=identity",
            "++proxy.reward_min=0.0",
            "++proxy.do_clip_rewards=false",
        ]

    def _get_ising_overrides(self, config: BenchmarkConfig) -> List[str]:
        """Get Hydra overrides for Ising environment."""
        L = config.env_kwargs["L"]

        return [
            # Ising environment
            "env=ising",
            "env.n_dim=2",
            f"env.length={L}",
            # Uniform proxy for Ising
            "proxy=uniform",
        ]

    def _get_ccube_overrides(self, config: BenchmarkConfig) -> List[str]:
        """Get Hydra overrides for continuous cube (box) environment."""
        n_dim = config.env_kwargs.get("n_dim", 2)
        delta = config.env_kwargs.get("delta", 0.25)

        return [
            # CCube environment
            "env=ccube",
            f"env.n_dim={n_dim}",
            "env.n_comp=5",
            f"env.min_incr={delta}",
            "env.beta_params_min=0.1",
            "env.beta_params_max=100.0",
            # Corners proxy for box/ccube
            "proxy=box/corners",
        ]

    def warmup(self, n_iters: int) -> None:
        """Run warmup iterations."""
        for _ in range(n_iters):
            self.run_iteration()
        self.synchronize()
        self._iteration = 0  # Reset iteration counter after warmup

    def run_iteration(self) -> None:
        """Run a single training iteration.

        This method is environment-agnostic. The bugfix for batch.proxy
        applies to all environments automatically.
        """
        # Sample batch
        batch, _ = self.agent.sample_batch(
            n_forward=self.agent.batch_size.forward,
            n_train=0,
            n_replay=0,
            collect_forwards_masks=True,
            collect_backwards_masks=self.agent.collect_backwards_masks,
        )

        # Workaround for gflownet bug: sample_batch creates an empty batch without
        # proxy, then merges sub-batches into it. The merge doesn't copy the proxy.
        # See gflownet.py line 594 vs 601.
        # This fix applies to ALL environments (hypergrid, ising, ccube).
        if batch.proxy is None:
            batch.proxy = self.agent.proxy

        # Compute loss
        losses = self.agent.loss.compute(batch, get_sublosses=True)

        # Backward and optimize
        if all([torch.isfinite(loss) for loss in losses.values()]):
            losses["all"].backward()
            if self.agent.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.agent.clip_grad_norm
                )
            self.agent.opt.step()
            self.agent.lr_scheduler.step()
            self.agent.opt.zero_grad()

        self._iteration += 1

    def synchronize(self) -> None:
        """Ensure all CUDA operations are complete."""
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def get_peak_memory(self) -> Optional[int]:
        """Return peak GPU memory usage in bytes."""
        if self.device is not None and self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device)
        return None

    def cleanup(self) -> None:
        """Release resources."""
        from hydra.core.global_hydra import GlobalHydra

        if self.agent is not None:
            del self.agent
            self.agent = None

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Clear Hydra state
        GlobalHydra.instance().clear()
