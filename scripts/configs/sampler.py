from dataclasses import dataclass
from typing import Tuple

from simple_parsing.helpers import JsonSerializable

from gfn.envs import Env
from gfn.losses import FMParametrization, Parametrization, PFBasedParametrization
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler


@dataclass
class SamplerConfig(JsonSerializable):
    temperature: float = 1.0
    sf_bias: float = 0.0
    epsilon: float = 0.0

    def parse(
        self, env: Env, parametrization: Parametrization
    ) -> Tuple[TrajectoriesSampler, bool]:
        on_policy = (
            self.temperature == 1.0 and self.sf_bias == 0.0 and self.epsilon == 0.0
        )
        if isinstance(parametrization, FMParametrization):
            estimator = parametrization.logF
        elif isinstance(parametrization, PFBasedParametrization):
            estimator = parametrization.logit_PF
        else:
            raise ValueError(
                f"Cannot parse sampler for parametrization {parametrization}"
            )
        actions_sampler = DiscreteActionsSampler(
            estimator=estimator,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )

        trajectories_sampler = TrajectoriesSampler(
            env=env,
            actions_sampler=actions_sampler,
        )

        return trajectories_sampler, on_policy
