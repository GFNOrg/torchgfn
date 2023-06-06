from abc import ABC, abstractmethod

import torch.nn as nn
from torch.distributions import Categorical, Distribution
from torchtyping import TensorType as TT

from gfn.envs import DiscreteEnv, Env
from gfn.states import DiscreteStates, States


# TODO: Is is true that this is only ever used for Action probability distributions?
# TODO: Remove environment from here (instead accept n_actions or similar)?
# TODO
class FunctionEstimator(ABC):
    r"""Base class for modules mapping states to action probability distributions.

    Training a GFlowNet requires parameterizing one or more of the following functions:
    - $s \mapsto (\log F(s \rightarrow s'))_{s' \in Children(s)}$
    - $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$
    - $s' \mapsto (P_B(s \mid s'))_{s \in Parents(s')}$
    - $s \mapsto (\log F(s))_{s \in States}$

    This class is the base class for all such function estimators. The estimators need
    to encapsulate a `nn.Module`, which takes a a batch of preprocessed states as input
    and outputs a batch of outputs of the desired shape. When the goal is to represent
    a probability distribution, the outputs would correspond to the parameters of the
    distribution, e.g. logits for a categorical distribution for discrete environments.

    The preprocessor is also encapsulated in the estimator via the environment.
    These function estimators implement the `__call__` method, which takes `States`
    objects as inputs and calls the module on the preprocessed states.

    Attributes:
        env: the environment.
        module: The module to use. If the module is a Tabular module (from
            `gfn.utils.modules`), then the environment preprocessor needs to be an
            `EnumPreprocessor`.
        preprocessor: Preprocessor from the environment.
        output_dim_is_checked: Flag for tracking whether the output dimenions of
            the states (after being preprocessed and transformed by the modules) have
            been verified.
    """

    def __init__(self, env: Env, module: nn.Module) -> None:
        """Initalize the FunctionEstimator with an environment and a module.
        Args:
            env: the environment.
            module: The module to use. If the module is a Tabular module (from
                `gfn.utils.modules`), then the environment preprocessor needs to be an
                `EnumPreprocessor`.
        """
        self.env = env
        self.module = module
        self.preprocessor = env.preprocessor  # TODO: passed explicitly?
        self.output_dim_is_checked = False  # TODO: private?

    def __call__(self, states: States) -> TT["batch_shape", "output_dim", float]:
        out = self.module(self.preprocessor(states))
        if not self.output_dim_is_checked:
            self.check_output_dim(out)
            self.output_dim_is_checked = True

        return out

    @abstractmethod
    def check_output_dim(
        self, module_output: TT["batch_shape", "output_dim", float]
    ) -> None:
        """Check that the output of the module has the correct shape. Raises an error if not."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.env})"

    def named_parameters(self) -> dict:
        return dict(self.module.named_parameters())

    def load_state_dict(self, state_dict: dict):
        self.module.load_state_dict(state_dict)


# TODO: make it work for continuous environments.
class LogEdgeFlowEstimator(FunctionEstimator):
    r"""Container for estimators $(s \rightarrow s') \mapsto \log F(s \rightarrow s')$.

    The way it's coded is a function $s \mapsto (\log F(s \rightarrow (s + a)))_{a \in \mathbb{A}}$,
    where $s+a$ is the state obtained by performing action $a$ in state $s$.

    This estimator is used for the flow-matching loss, which only supports discrete
    environments.
    """
    def check_output_dim(self, module_output: TT["batch_shape", "output_dim", float]):
        if not isinstance(self.env, DiscreteEnv):
            raise ValueError(
                "LogEdgeFlowEstimator only supports discrete environments."
            )
        if module_output.shape[-1] != self.env.n_actions:
            raise ValueError(
                f"LogEdgeFlowEstimator output dimension should be {self.env.n_actions}, but is {module_output.shape[-1]}."
            )


class LogStateFlowEstimator(FunctionEstimator):
    r"""Container for estimators $s \mapsto \log F(s)$."""

    def check_output_dim(self, module_output: TT["batch_shape", "output_dim", float]):
        if module_output.shape[-1] != 1:
            raise ValueError(
                f"LogStateFlowEstimator output dimension should be 1, but is {module_output.shape[-1]}."
            )


class ProbabilityEstimator(FunctionEstimator, ABC):
    r"""Container for estimators of probability distributions.

    When calling (via `__call__`) such an estimator, an extra step is performed, which
    is to transform the output of the module into a probability distribution. This is
    done by applying the abstract `to_probability_distribution` method.

    The outputs of such an estimator are thus probability distributions, not the
    parameters of the distributions.
    """
    @abstractmethod
    def to_probability_distribution(
        self,
        states: States,
        module_output: TT["batch_shape", "output_dim", float],
    ) -> Distribution:
        """Transform the output of the module into a probability distribution."""
        pass

    def __call__(self, states: States) -> Distribution:
        return self.to_probability_distribution(states, super().__call__(states))


class LogEdgeFlowProbabilityEstimator(ProbabilityEstimator, LogEdgeFlowEstimator):
    r"""Container for Log Edge Flow Probability Estimator

    $(s \rightarrow s') \mapsto P_F(s' \mid s) = \frac{F(s \rightarrow s')}
        {\sum_{s' \in Children(s)} F(s \rightarrow s')}$.
    """
    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: TT["batch_shape", "output_dim", float],
    ) -> Distribution:
        logits = module_output
        logits[~states.forward_masks] = -float("inf")
        return Categorical(logits=logits)


class LogZEstimator:
    # TODO: should this be a FunctionEstimator with a nn.Module as well?
    r"""Container for the estimator $\log Z$."""

    def __init__(self, tensor: TT[0, float]) -> None:
        self.tensor = tensor
        assert self.tensor.shape == ()
        self.tensor.requires_grad = True

    def __repr__(self) -> str:
        return str(self.tensor.item())

    def named_parameters(self) -> dict:
        return {"logZ": self.tensor}

    def load_state_dict(self, state_dict: dict):
        self.tensor = state_dict["logZ"]
