"""TorchGFN library runner for benchmarking.

Supports multiple environments: hypergrid, ising, box, bitseq.
"""

from typing import Optional

import torch

from benchmark.lib_runners.base import BenchmarkConfig, LibraryRunner
from gfn.utils.common import set_seed


class TorchGFNRunner(LibraryRunner):
    """Benchmark runner for the torchgfn library.

    Supports environments:
    - hypergrid: Discrete grid navigation
    - ising: Discrete EBM with Ising model energy
    - box: Continuous 2D box environment
    - bitseq: Bit sequence generation
    """

    name = "torchgfn"

    def __init__(self):
        self.env = None
        self.gflownet = None
        self.optimizer = None
        self.sampler = None
        self.device = None
        self.config = None
        self._env_type = None

    def setup(self, config: BenchmarkConfig, seed: int) -> None:
        """Initialize environment, model, and optimizer based on env_name."""
        set_seed(seed)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reset CUDA memory stats if on GPU
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        self._env_type = config.env_name

        if config.env_name == "hypergrid":
            self._setup_hypergrid(config)
        elif config.env_name == "ising":
            self._setup_ising(config)
        elif config.env_name == "box":
            self._setup_box(config)
        elif config.env_name == "bitseq":
            self._setup_bitseq(config)
        else:
            raise ValueError(f"Unknown environment: {config.env_name}")

    def _setup_hypergrid(self, config: BenchmarkConfig) -> None:
        """Setup HyperGrid environment with TB loss."""
        from gfn.estimators import DiscretePolicyEstimator
        from gfn.gflownet import TBGFlowNet
        from gfn.gym import HyperGrid
        from gfn.preprocessors import KHotPreprocessor
        from gfn.samplers import Sampler
        from gfn.utils.modules import MLP

        ndim = config.env_kwargs["ndim"]
        height = config.env_kwargs["height"]

        self.env = HyperGrid(
            ndim=ndim,
            height=height,
            reward_fn_str="original",
            reward_fn_kwargs={"R0": 0.1, "R1": 0.5, "R2": 2.0},
            device=self.device,  # type: ignore
            calculate_partition=False,
            store_all_states=False,
            debug=False,
        )

        preprocessor = KHotPreprocessor(height=self.env.height, ndim=self.env.ndim)

        module_PF = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=self.env.n_actions,
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_layers,
        )
        module_PB = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=self.env.n_actions - 1,
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_layers,
            trunk=module_PF.trunk,
        )

        pf_estimator = DiscretePolicyEstimator(
            module_PF, self.env.n_actions, preprocessor=preprocessor, is_backward=False
        )
        pb_estimator = DiscretePolicyEstimator(
            module_PB, self.env.n_actions, preprocessor=preprocessor, is_backward=True
        )

        self.gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
        self.gflownet = self.gflownet.to(self.device)

        self.optimizer = torch.optim.Adam(self.gflownet.pf_pb_parameters(), lr=config.lr)
        self.optimizer.add_param_group(
            {"params": self.gflownet.logz_parameters(), "lr": config.lr_logz}
        )

        self.sampler = Sampler(estimator=pf_estimator)

    def _setup_ising(self, config: BenchmarkConfig) -> None:
        """Setup Ising model environment with FM loss."""
        from gfn.estimators import DiscretePolicyEstimator
        from gfn.gflownet import FMGFlowNet
        from gfn.gym import DiscreteEBM
        from gfn.gym.discrete_ebm import IsingModel
        from gfn.utils.modules import MLP

        L = config.env_kwargs["L"]
        J_coupling = config.env_kwargs["J"]

        # Build Ising coupling matrix with periodic boundary conditions
        J = self._make_ising_J(L, J_coupling)

        N = L**2
        ising_energy = IsingModel(J)
        self.env = DiscreteEBM(
            N,
            alpha=1,
            energy=ising_energy,
            device=self.device,  # type: ignore
            debug=False,
        )

        pf_module = MLP(
            input_dim=self.env.ndim,
            output_dim=self.env.n_actions,
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_layers,
            activation_fn="relu",
        )

        pf_estimator = DiscretePolicyEstimator(
            pf_module, self.env.n_actions, is_backward=False
        )
        self.gflownet = FMGFlowNet(pf_estimator).to(self.device)
        self.optimizer = torch.optim.Adam(self.gflownet.parameters(), lr=config.lr)

        # FMGFlowNet samples trajectories directly
        self.sampler = None

    def _make_ising_J(self, L: int, coupling_constant: float) -> torch.Tensor:
        """Build Ising coupling matrix with periodic boundary conditions."""

        def ising_n_to_ij(L, n):
            i = n // L
            j = n - i * L
            return (i, j)

        N = L**2
        J = torch.zeros((N, N), device=self.device)
        for k in range(N):
            for m in range(k):
                x1, y1 = ising_n_to_ij(L, k)
                x2, y2 = ising_n_to_ij(L, m)
                if x1 == x2 and abs(y2 - y1) == 1:
                    J[k][m] = 1
                    J[m][k] = 1
                elif y1 == y2 and abs(x2 - x1) == 1:
                    J[k][m] = 1
                    J[m][k] = 1

        # Periodic boundary conditions
        for k in range(L):
            J[k * L][(k + 1) * L - 1] = 1
            J[(k + 1) * L - 1][k * L] = 1
            J[k][k + N - L] = 1
            J[k + N - L][k] = 1

        return coupling_constant * J

    def _setup_box(self, config: BenchmarkConfig) -> None:
        """Setup continuous Box environment with TB loss."""
        from gfn.gflownet import TBGFlowNet
        from gfn.gym import Box
        from gfn.gym.helpers.box_utils import (
            BoxPBEstimator,
            BoxPBMLP,
            BoxPFEstimator,
            BoxPFMLP,
        )
        from gfn.samplers import Sampler

        delta = config.env_kwargs.get("delta", 0.25)

        self.env = Box(
            delta=delta,
            epsilon=1e-10,
            device=self.device,  # type: ignore
            debug=False,
        )

        # Box environment uses specialized policy modules
        pf_module = BoxPFMLP(
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_layers,
            n_components=2,
            n_components_s0=4,
        )
        pb_module = BoxPBMLP(
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_layers,
            n_components=2,
            trunk=pf_module.trunk,  # Tied weights
        )

        pf_estimator = BoxPFEstimator(
            self.env,
            pf_module,
            n_components_s0=4,
            n_components=2,
            min_concentration=0.1,
            max_concentration=5.1,
        )
        pb_estimator = BoxPBEstimator(
            self.env,
            pb_module,
            n_components=2,
            min_concentration=0.1,
            max_concentration=5.1,
        )

        self.gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
        self.gflownet = self.gflownet.to(self.device)

        # Optimizer with separate learning rates
        self.optimizer = torch.optim.Adam(pf_module.parameters(), lr=config.lr)
        self.optimizer.add_param_group(
            {"params": pb_module.last_layer.parameters(), "lr": config.lr}
        )
        if "logZ" in dict(self.gflownet.named_parameters()):
            logZ = dict(self.gflownet.named_parameters())["logZ"]
            self.optimizer.add_param_group({"params": [logZ], "lr": config.lr_logz})

        self.sampler = Sampler(estimator=pf_estimator)

    def _setup_bitseq(self, config: BenchmarkConfig) -> None:
        """Setup BitSequence environment with TB loss."""
        from gfn.estimators import DiscretePolicyEstimator
        from gfn.gflownet import TBGFlowNet
        from gfn.gym import BitSequence
        from gfn.samplers import Sampler
        from gfn.utils.modules import MLP

        word_size = config.env_kwargs.get("word_size", 1)
        seq_size = config.env_kwargs.get("seq_size", 4)
        n_modes = config.env_kwargs.get("n_modes", 2)

        # Generate random mode set
        H = torch.randint(
            0, 2, (n_modes, seq_size), dtype=torch.long, device=self.device
        )
        self.env = BitSequence(word_size, seq_size, n_modes, H=H, debug=False)

        pf = MLP(
            self.env.words_per_seq, self.env.n_actions, hidden_dim=config.hidden_dim
        )
        pb = MLP(self.env.words_per_seq, self.env.n_actions - 1, trunk=pf.trunk)

        pf_estimator = DiscretePolicyEstimator(pf, n_actions=self.env.n_actions)
        pb_estimator = DiscretePolicyEstimator(
            pb, n_actions=self.env.n_actions, is_backward=True
        )

        self.gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0).to(
            self.device
        )

        non_logz_params = [
            v for k, v in dict(self.gflownet.named_parameters()).items() if k != "logZ"
        ]
        self.optimizer = torch.optim.Adam(non_logz_params, lr=config.lr)
        logz_params = [dict(self.gflownet.named_parameters())["logZ"]]
        self.optimizer.add_param_group({"params": logz_params, "lr": config.lr_logz})

        self.sampler = Sampler(estimator=pf_estimator)

    def warmup(self, n_iters: int) -> None:
        """Run warmup iterations for CUDA kernel caching."""
        for _ in range(n_iters):
            self.run_iteration()
        self.synchronize()

    def run_iteration(self) -> None:
        """Run a single training iteration."""
        if self._env_type == "ising":
            # FMGFlowNet uses direct sampling
            trajectories = self.gflownet.sample_trajectories(
                self.env,  # type: ignore
                n=self.config.batch_size,
                save_estimator_outputs=False,
                save_logprobs=False,
            )
            training_samples = self.gflownet.to_training_samples(trajectories)
            self.optimizer.zero_grad()
            loss = self.gflownet.loss(
                self.env,  # type: ignore
                training_samples,  # type: ignore
                recalculate_all_logprobs=True,
            )
            loss.backward()
            self.optimizer.step()
        else:
            # TBGFlowNet with Sampler
            trajectories = self.sampler.sample_trajectories(
                self.env,  # type: ignore
                n=self.config.batch_size,
                save_logprobs=self._env_type == "box",  # Box saves logprobs
                save_estimator_outputs=False,
                epsilon=0.0,
            )

            self.optimizer.zero_grad()
            loss = self.gflownet.loss_from_trajectories(
                self.env,  # type: ignore
                trajectories,
                recalculate_all_logprobs=(self._env_type != "box"),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gflownet.parameters(), 1.0)
            self.optimizer.step()

    def synchronize(self) -> None:
        """Ensure all CUDA operations are complete."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def get_peak_memory(self) -> Optional[int]:
        """Return peak GPU memory usage in bytes."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device)
        return None

    def cleanup(self) -> None:
        """Release resources."""
        del self.gflownet
        del self.optimizer
        del self.sampler
        del self.env

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.gflownet = None
        self.optimizer = None
        self.sampler = None
        self.env = None
