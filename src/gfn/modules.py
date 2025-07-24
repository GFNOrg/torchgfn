import warnings

warnings.warn(
    "'modules.py' is deprecated and will be removed in a future release. Please import "
    "from 'estimators.py' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gfn.estimators import (  # noqa: F401, E402
    ConditionalDiscretePolicyEstimator,
    ConditionalScalarEstimator,
    DiscreteGraphPolicyEstimator,
    DiscretePolicyEstimator,
    Estimator,
    ScalarEstimator,
)
