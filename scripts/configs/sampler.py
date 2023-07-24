import inspect
from dataclasses import dataclass
from typing import Tuple

from gfn.envs import Env
from gfn.losses import FMParametrization, Parametrization, PFBasedParametrization
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler


@dataclass
class SamplerConfig:
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
            sf_bias=self.sf_bias,
            epsilon=self.epsilon,
        )

        trajectories_sampler = TrajectoriesSampler(
            env=env,
            actions_sampler=actions_sampler,
        )

        return trajectories_sampler, on_policy


def make_sampler(
    config: dict, env: Env, parametrization: Parametrization
) -> Tuple[TrajectoriesSampler, bool]:
    name = config["sampler"]["name"]
    if not name:
        sampler_class = SamplerConfig
    else:
        raise ValueError(f"Invalid sampler name: {name}")

    args = inspect.getfullargspec(sampler_class.__init__).args
    sampler_config = {k: v for k, v in config["sampler"].items() if k in args}
    return sampler_class(**sampler_config).parse(env, parametrization)
