"""GFlowNet library (external) runner for benchmarking.

This runner uses the gflownet library from benchmark/gflownet/.
The library is tightly coupled with Hydra, so we use Hydra's compose API
to build the configuration and instantiate the GFlowNet agent.

Supports environments: hypergrid, ising, box (ccube).
"""

import sys
import time
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

    def __init__(self, patch_actions: bool = False, **kwargs):
        self.agent = None
        self.device = None
        self.config = None
        self._iteration = 0
        self._patch_actions = patch_actions

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

            # Monkeypatch CUDA device mismatches in upstream gflownet code:
            # several methods create tensors on CPU even when states are on GPU.
            self._patch_grid_states2policy()
            self._patch_cube_sample_actions()

            if self._patch_actions:
                self._patch_action_indexing()
                print(
                    "  [patch] Replaced O(batch*action_space) action indexing with O(batch) dict lookup"
                )

            # Create the GFlowNet agent
            self.agent = gflownet_from_config(cfg)

        self._iteration = 0

    @staticmethod
    def _patch_grid_states2policy():
        """Monkeypatch Grid.states2policy to put torch.arange on the correct device."""
        from gflownet.envs.grid import Grid

        def _states2policy_fixed(self, states):
            from gflownet.utils.common import tlong

            states = tlong(states, device=self.device)
            n_states = states.shape[0]
            cols = states + torch.arange(self.n_dim, device=self.device) * self.length
            rows = torch.repeat_interleave(
                torch.arange(n_states, device=self.device), self.n_dim
            )
            states_policy = torch.zeros(
                (n_states, self.length * self.n_dim),
                dtype=self.float,
                device=self.device,
            )
            states_policy[rows, cols.flatten()] = 1.0
            return states_policy

        Grid.states2policy = _states2policy_fixed

    @staticmethod
    def _patch_cube_sample_actions():
        """Monkeypatch Cube._sample_actions_batch_{forward,backward} to fix CUDA
        device mismatch: torch.zeros/ones are created on CPU when building action
        tensors that live on GPU (lines 1130, 1221, 1229 in cube.py)."""
        from gflownet.envs.cube import ContinuousCube as Cube
        from gflownet.utils.common import tbool, tfloat
        from torch.distributions import Bernoulli

        # --- Forward ---
        def _sample_forward_fixed(
            self,
            policy_outputs,
            mask=None,
            states_from=None,
            random_action_prob=0.0,
            temperature_logits=1.0,
        ):
            n_states = policy_outputs.shape[0]
            states_from_tensor = tfloat(
                states_from, float_type=self.float, device=self.device
            )
            is_eos = torch.zeros(n_states, dtype=torch.bool, device=self.device)
            is_source = ~mask[:, 1]
            is_eos_forced = mask[:, 0]
            is_eos[is_eos_forced] = True
            assert not torch.any(torch.logical_and(is_source, is_eos_forced))

            logits_sampling = self.randomize_and_temper_sampling_distribution(
                policy_outputs, random_action_prob, temperature_logits
            )

            do_eos = torch.logical_and(~is_source, ~is_eos_forced)
            if torch.any(do_eos):
                is_eos_sampled = torch.zeros_like(do_eos)
                logits_eos = self._get_policy_eos_logit(logits_sampling)[do_eos]
                distr_eos = Bernoulli(logits=logits_eos)
                is_eos_sampled[do_eos] = tbool(distr_eos.sample(), device=self.device)
                is_eos[is_eos_sampled] = True

            do_increments = ~is_eos
            if torch.any(do_increments):
                distr_increments = self._make_increments_distribution(
                    logits_sampling[do_increments]
                )
                increments = distr_increments.sample()
                is_relative = ~is_source[do_increments]
                states_from_rel = tfloat(
                    states_from_tensor[do_increments],
                    float_type=self.float,
                    device=self.device,
                )[is_relative]
                increments[is_relative] = self.relative_to_absolute_increments(
                    states_from_rel,
                    increments[is_relative],
                    is_backward=False,
                )

            actions_tensor = torch.full(
                (n_states, self.n_dim + 1),
                torch.inf,
                dtype=self.float,
                device=self.device,
            )
            if torch.any(do_increments):
                increments = self._mask_ignored_dimensions(
                    mask[do_increments], increments
                )
                actions_tensor[do_increments] = torch.cat(
                    (
                        increments,
                        torch.zeros(
                            (increments.shape[0], 1),
                            dtype=self.float,
                            device=self.device,
                        ),
                    ),
                    dim=1,
                )
            actions_tensor[is_source, -1] = 1
            return [tuple(a) for a in actions_tensor.tolist()]

        # --- Backward ---
        def _sample_backward_fixed(
            self,
            policy_outputs,
            mask=None,
            states_from=None,
            random_action_prob=0.0,
            temperature_logits=1.0,
        ):
            n_states = policy_outputs.shape[0]
            is_bts = torch.zeros(n_states, dtype=torch.bool, device=self.device)
            is_eos = ~mask[:, 2]
            is_bts_forced = ~mask[:, 1]
            is_bts[is_bts_forced] = True

            logits_sampling = self.randomize_and_temper_sampling_distribution(
                policy_outputs, random_action_prob, temperature_logits
            )

            do_bts = torch.logical_and(~is_bts_forced, ~is_eos)
            if torch.any(do_bts):
                is_bts_sampled = torch.zeros_like(do_bts)
                logits_bts = self._get_policy_source_logit(logits_sampling)[do_bts]
                distr_bts = Bernoulli(logits=logits_bts)
                is_bts_sampled[do_bts] = tbool(distr_bts.sample(), device=self.device)
                is_bts[is_bts_sampled] = True

            do_increments = torch.logical_and(~is_bts, ~is_eos)
            if torch.any(do_increments):
                distr_increments = self._make_increments_distribution(
                    logits_sampling[do_increments]
                )
                increments = distr_increments.sample()
                states_from_rel = tfloat(
                    states_from, float_type=self.float, device=self.device
                )[do_increments]
                increments = self.relative_to_absolute_increments(
                    states_from_rel,
                    increments,
                    is_backward=True,
                )

            actions_tensor = torch.zeros(
                (n_states, self.n_dim + 1), dtype=self.float, device=self.device
            )
            actions_tensor[is_eos] = tfloat(
                self.eos, float_type=self.float, device=self.device
            )
            if torch.any(do_increments):
                increments = self._mask_ignored_dimensions(
                    mask[do_increments], increments
                )
                actions_tensor[do_increments] = torch.cat(
                    (
                        increments,
                        torch.zeros(
                            (increments.shape[0], 1),
                            dtype=self.float,
                            device=self.device,
                        ),
                    ),
                    dim=1,
                )
            if torch.any(is_bts):
                actions_bts = tfloat(
                    states_from, float_type=self.float, device=self.device
                )[is_bts]
                actions_bts = torch.cat(
                    (
                        actions_bts,
                        torch.ones(
                            (actions_bts.shape[0], 1),
                            dtype=self.float,
                            device=self.device,
                        ),
                    ),
                    dim=1,
                )
                actions_tensor[is_bts] = actions_bts
                actions_tensor[is_bts, :-1] = self._mask_ignored_dimensions(
                    mask[is_bts], actions_tensor[is_bts, :-1]
                )
            return [tuple(a) for a in actions_tensor.tolist()]

        Cube._sample_actions_batch_forward = _sample_forward_fixed
        Cube._sample_actions_batch_backward = _sample_backward_fixed

    @staticmethod
    def _patch_action_indexing():
        """Replace O(batch × action_space) action indexing with O(batch) dict lookup.

        Patches two methods on GFlowNetEnv:
        1. actions2indices: tensor path — replaces expand-and-compare with
           per-element dict lookup via self._action2index.
        2. get_logprobs: list path — replaces Python list comprehension with
           a vectorized tensor construction using the same dict.

        Both paths are called during every trajectory step and loss computation,
        so this optimization reduces total work from O(B × A × T) to O(B × T)
        where B=batch, A=action_space_size, T=trajectory_length.
        """
        from gflownet.envs.base import GFlowNetEnv
        from gflownet.utils.common import tlong

        def _actions2indices_fast(self, actions):
            """O(batch) action indexing via dict lookup."""
            batch_size = actions.shape[0]
            indices = torch.empty(batch_size, dtype=torch.long, device=actions.device)
            for i in range(batch_size):
                action_tuple = tuple(actions[i].tolist())
                indices[i] = self._action2index[self.action2representative(action_tuple)]
            return indices

        def _get_logprobs_fast(
            self,
            policy_outputs,
            actions,
            mask=None,
            states_from=None,
            is_backward=False,
        ):
            """get_logprobs with O(batch) action indexing for both list and tensor paths."""
            device = policy_outputs.device
            ns_range = torch.arange(policy_outputs.shape[0], device=device)
            logits = policy_outputs.clone()
            if mask is not None:
                logits[mask] = -torch.inf

            if torch.is_tensor(actions):
                action_indices = tlong(self.actions2indices(actions), device=self.device)
            else:
                # Vectorized dict lookup — build tensor directly instead of
                # list comprehension + tlong conversion.
                action_indices = torch.tensor(
                    [self._action2index[self.action2representative(a)] for a in actions],
                    dtype=torch.long,
                    device=device,
                )

            logprobs = self.logsoftmax(logits)[ns_range, action_indices]
            return logprobs

        GFlowNetEnv.actions2indices = _actions2indices_fast
        GFlowNetEnv.get_logprobs = _get_logprobs_fast

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
        """Run a single training iteration with per-phase timing.

        This method is environment-agnostic. The bugfix for batch.proxy
        applies to all environments automatically.
        """
        # Sample batch
        t0 = time.perf_counter()
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
        t1 = time.perf_counter()

        # Compute loss
        losses = self.agent.loss.compute(batch, get_sublosses=True)
        t2 = time.perf_counter()

        # Backward and optimize
        if all([torch.isfinite(loss) for loss in losses.values()]):
            losses["all"].backward()
            if self.agent.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.agent.clip_grad_norm
                )
            t3 = time.perf_counter()

            self.agent.opt.step()
            self.agent.lr_scheduler.step()
            self.agent.opt.zero_grad()
            t4 = time.perf_counter()
        else:
            t3 = t2
            t4 = t2

        if hasattr(self, "_phase_times"):
            self._phase_times["sample"].append(t1 - t0)
            self._phase_times["loss"].append(t2 - t1)
            self._phase_times["backward"].append(t3 - t2)
            self._phase_times["optimizer"].append(t4 - t3)

        self._iteration += 1

    def synchronize(self) -> None:
        """Ensure all CUDA operations are complete."""
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def get_n_params(self) -> Optional[int]:
        """Return total number of trainable parameters."""
        if self.agent is not None:
            return sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        return None

    def get_logZ(self) -> Optional[float]:
        """Return current logZ value."""
        if self.agent is not None and hasattr(self.agent, "loss"):
            loss = self.agent.loss
            if hasattr(loss, "logZ") and loss.logZ is not None:
                return float(loss.logZ.mean().item())
        return None

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
