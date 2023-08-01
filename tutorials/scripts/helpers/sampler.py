import inspect
from dataclasses import dataclass
from typing import Tuple

from gfn.env import Env
from gfn.gflownet import FMGFlowNet, GFlowNet, PFBasedGFlowNet
from gfn.samplers import ActionsSampler, TrajectoriesSampler


@dataclass
class SamplerConfig:
    temperature: float = 1.0
    sf_bias: float = 0.0
    epsilon: float = 0.0

    def parse(
        self, env: Env, gflownet: GFlowNet
    ) -> Tuple[TrajectoriesSampler, bool]:
        on_policy = (
            self.temperature == 1.0 and self.sf_bias == 0.0 and self.epsilon == 0.0
        )
        if isinstance(gflownet, FMGFlowNet):
            estimator = gflownet.logF
        elif isinstance(gflownet, PFBasedGFlowNet):
            estimator = gflownet.logit_PF
        else:
            raise ValueError(
                f"Cannot parse sampler for gflownet {gflownet}"
            )
        actions_sampler = ActionsSampler(
            estimator=estimator,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )

        trajectories_sampler = TrajectoriesSampler(
            env=env,
            actions_sampler=actions_sampler,
        )

        return trajectories_sampler, on_policy


def make_sampler(
    config: dict, env: Env, gflownet: GFlowNet
) -> Tuple[TrajectoriesSampler, bool]:
    name = config["sampler"]["name"]
    if not name:
        sampler_class = SamplerConfig
    else:
        raise ValueError("Invalid sampler name: {}".format(name))

    args = inspect.getfullargspec(sampler_class.__init__).args
    sampler_config = {k: v for k, v in config["sampler"].items() if k in args}
    return sampler_class(**sampler_config).parse(env, gflownet)
