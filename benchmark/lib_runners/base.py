"""Base classes and configuration for library benchmarking."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark scenario.

    Environment-specific parameters are stored in env_kwargs and
    interpreted by each runner.
    """

    # Environment (type and parameters specified per scenario)
    env_name: str  # e.g., "hypergrid", "bitseq", etc.
    env_kwargs: dict  # Environment-specific parameters

    # Training
    n_iterations: int = 1000
    batch_size: int = 16
    lr: float = 1e-3
    lr_logz: float = 0.1

    # Network
    hidden_dim: int = 256
    n_layers: int = 2

    # Benchmark settings
    n_warmup: int = 50  # Warmup iterations (excluded from timing)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    library: str
    seed: int

    # Timing (seconds)
    total_time: float
    iter_times: List[float] = field(default_factory=list)

    # Memory (bytes)
    peak_memory: Optional[int] = None

    @property
    def mean_iter_time(self) -> float:
        """Mean iteration time in seconds."""
        if not self.iter_times:
            return 0.0
        return sum(self.iter_times) / len(self.iter_times)

    @property
    def std_iter_time(self) -> float:
        """Standard deviation of iteration time."""
        if len(self.iter_times) < 2:
            return 0.0
        mean = self.mean_iter_time
        variance = sum((t - mean) ** 2 for t in self.iter_times) / len(self.iter_times)
        return variance**0.5

    @property
    def throughput(self) -> float:
        """Throughput in iterations per second."""
        if self.total_time <= 0:
            return 0.0
        return len(self.iter_times) / self.total_time

    @property
    def peak_memory_mb(self) -> Optional[float]:
        """Peak memory in megabytes."""
        if self.peak_memory is None:
            return None
        return self.peak_memory / (1024 * 1024)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "library": self.library,
            "seed": self.seed,
            "total_time": self.total_time,
            "mean_iter_time": self.mean_iter_time,
            "std_iter_time": self.std_iter_time,
            "throughput_iters_per_sec": self.throughput,
            "peak_memory_mb": self.peak_memory_mb,
            "n_iterations": len(self.iter_times),
        }


class LibraryRunner(ABC):
    """Abstract base class for library-specific benchmark runners."""

    name: str  # e.g., "torchgfn", "gflownet", "gfnx"

    @abstractmethod
    def setup(self, config: BenchmarkConfig, seed: int) -> None:
        """Initialize environment, model, and optimizer.

        Called once per seed. This phase is not timed.

        Args:
            config: Benchmark configuration.
            seed: Random seed for reproducibility.
        """

    @abstractmethod
    def warmup(self, n_iters: int) -> None:
        """Run warmup iterations.

        Used to trigger JIT compilation (JAX) and CUDA kernel caching (PyTorch).
        These iterations are excluded from timing.

        Args:
            n_iters: Number of warmup iterations to run.
        """

    @abstractmethod
    def run_iteration(self) -> None:
        """Run a single training iteration.

        This should include: sampling trajectories, computing loss,
        and performing optimizer step.
        """

    @abstractmethod
    def synchronize(self) -> None:
        """Ensure all asynchronous operations are complete.

        For PyTorch: torch.cuda.synchronize() if on GPU
        For JAX: jax.block_until_ready() on outputs
        """

    @abstractmethod
    def get_peak_memory(self) -> Optional[int]:
        """Return peak memory usage in bytes.

        Returns:
            Peak memory in bytes, or None if not available (e.g., CPU-only).
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources and clean up.

        Called after benchmark completes.
        """
