"""Backward-compatibility shim for Box environment utilities.

Import from ``box_cartesian_utils`` or ``box_polar_utils`` directly for new code.
"""

from gfn.gym.helpers.box_cartesian_utils import (  # noqa: F401
    BoxCartesianDistribution,
    BoxCartesianPBDistribution,
    BoxCartesianPBEstimator,
    BoxCartesianPBMLP,
    BoxCartesianPFEstimator,
    BoxCartesianPFMLP,
    UniformBoxCartesianPBModule,
)
from gfn.gym.helpers.box_polar_utils import (  # noqa: F401
    BoxPBEstimator,
    BoxPBMLP,
    BoxPBUniform,
    BoxPFEstimator,
    BoxPFMLP,
    BoxStateFlowModule,
    DistributionWrapper,
    QuarterCircle,
    QuarterCircleWithExit,
    QuarterDisk,
    split_PF_module_output,
)

__all__ = [
    # Cartesian
    "BoxCartesianDistribution",
    "BoxCartesianPBDistribution",
    "BoxCartesianPBEstimator",
    "BoxCartesianPBMLP",
    "BoxCartesianPFEstimator",
    "BoxCartesianPFMLP",
    "UniformBoxCartesianPBModule",
    # Polar (legacy)
    "BoxPBEstimator",
    "BoxPBMLP",
    "BoxPBUniform",
    "BoxPFEstimator",
    "BoxPFMLP",
    "BoxStateFlowModule",
    "DistributionWrapper",
    "QuarterCircle",
    "QuarterCircleWithExit",
    "QuarterDisk",
    "split_PF_module_output",
]
