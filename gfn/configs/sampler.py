from dataclasses import dataclass
from typing import List, Optional, Tuple

from simple_parsing.helpers import JsonSerializable

from gfn.envs import Env
from gfn.parametrizations import (
    FMParametrization,
    Parametrization,
    PFBasedParametrization,
)
from gfn.parametrizations.forward_probs import (
    DBParametrization,
    TBParametrization,
    SubTBParametrization,
)
from gfn.samplers import (
    LogEdgeFlowsActionsSampler,
    LogitPFActionsSampler,
    StatesSampler,
    TrainingSampler,
    TrajectoriesSampler,
    TransitionsSampler,
)


@dataclass
class SamplerConfig(JsonSerializable):
    temperature: float = 1.0
    sf_temperature: float = 0.0

    def parse(
        self, env: Env, parametrization: Parametrization
    ) -> Tuple[TrainingSampler, TrajectoriesSampler]:

        if isinstance(parametrization, FMParametrization):
            actions_sampler_cls = LogEdgeFlowsActionsSampler
            estimator = parametrization.logF
            training_sampler_cls = StatesSampler
        elif isinstance(parametrization, PFBasedParametrization):
            actions_sampler_cls = LogitPFActionsSampler
            estimator = parametrization.logit_PF
            if isinstance(parametrization, DBParametrization):
                training_sampler_cls = TransitionsSampler
            elif isinstance(parametrization, TBParametrization) or isinstance(
                parametrization, SubTBParametrization
            ):
                training_sampler_cls = TrajectoriesSampler
            else:
                raise ValueError(f"Unknown parametrization {parametrization}")
        else:
            raise ValueError(f"Unknown parametrization {parametrization}")
        training_actions_sampler = actions_sampler_cls(
            estimator=estimator,
            temperature=self.temperature,
            sf_temperature=self.sf_temperature,
        )

        training_sampler: TrainingSampler = training_sampler_cls(
            env=env, actions_sampler=training_actions_sampler
        )

        return training_sampler
