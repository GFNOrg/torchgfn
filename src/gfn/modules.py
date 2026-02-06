import logging

logger = logging.getLogger(__name__)

logger.warning(
    "'modules.py' is deprecated and will be removed in a future release. Please import "
    "from 'estimators.py' instead."
)

from gfn.estimators import (  # noqa: F401, E402
    ConditionalDiscretePolicyEstimator,
    ConditionalScalarEstimator,
    DiscreteGraphPolicyEstimator,
    DiscretePolicyEstimator,
    Estimator,
    ScalarEstimator,
)
