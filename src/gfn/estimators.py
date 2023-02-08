from abc import ABC
from typing import Literal, Optional

from torchtyping import TensorType

from gfn.containers import States
from gfn.envs import Env
from gfn.envs.preprocessors.base import EnumPreprocessor
from gfn.modules import GFNModule, NeuralNet, Tabular, Uniform, ZeroGFNModule

# Typing
OutputTensor = TensorType["batch_shape", "output_dim", float]


class FunctionEstimator(ABC):
    """Base class for function estimators."""

    def __init__(
        self,
        env: Env,
        module: Optional[GFNModule] = None,
        output_dim: Optional[int] = None,
        module_name: Optional[
            Literal["NeuralNet", "Uniform", "Tabular", "Zero"]
        ] = None,
        **nn_kwargs,
    ) -> None:
        """Either module or (module_name, output_dim) must be provided.

        Args:
            env (Env): the environment.
            module (Optional[GFNModule], optional): The module to use. Defaults to None.
            output_dim (Optional[int], optional): Used only if module is None. Defines the output dimension of the module. Defaults to None.
            module_name (Optional[Literal[NeuralNet, Uniform, Tabular, Zero]], optional): Used only if module is None. What module to use. Defaults to None.
            **nn_kwargs: Keyword arguments to pass to the module, if module_name is NeuralNet.
        """
        self.env = env
        if module is None:
            assert module_name is not None and output_dim is not None
            if module_name == "NeuralNet":
                assert len(env.preprocessor.output_shape) == 1
                input_dim = env.preprocessor.output_shape[0]
                module = NeuralNet(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    **nn_kwargs,
                )
            elif module_name == "Uniform":
                module = Uniform(output_dim=output_dim)
            elif module_name == "Zero":
                module = ZeroGFNModule(output_dim=output_dim)
            elif module_name == "Tabular":
                module = Tabular(
                    n_states=env.n_states,
                    output_dim=output_dim,
                )
            else:
                raise ValueError(f"Unknown module_name {module_name}")
        self.module = module
        if isinstance(self.module, Tabular):
            self.preprocessor = EnumPreprocessor(env.get_states_indices)
        else:
            self.preprocessor = env.preprocessor

    def __call__(self, states: States) -> OutputTensor:
        return self.module(self.preprocessor(states))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.module})"

    def named_parameters(self) -> dict:
        return dict(self.module.named_parameters())

    def load_state_dict(self, state_dict: dict):
        self.module.load_state_dict(state_dict)


class LogEdgeFlowEstimator(FunctionEstimator):
    r"""Container for estimators $(s \rightarrow s') \mapsto \log F(s \rightarrow s')$.
    The way it's coded is a function $s \mapsto (\log F(s \rightarrow (s + a)))_{a \in \mathbb{A}}$,
    where $s+a$ is the state obtained by performing action $a$ in state $s$."""

    def __init__(
        self,
        env: Env,
        module: Optional[GFNModule] = None,
        module_name: Optional[
            Literal["NeuralNet", "Uniform", "Tabular", "Zero"]
        ] = None,
        **nn_kwargs,
    ) -> None:
        if module is not None:
            assert module.output_dim == env.n_actions
        super().__init__(
            env,
            module=module,
            output_dim=env.n_actions,
            module_name=module_name,
            **nn_kwargs,
        )


class LogStateFlowEstimator(FunctionEstimator):
    r"""Container for estimators $s \mapsto \log F(s)$."""

    def __init__(
        self,
        env: Env,
        module: Optional[GFNModule] = None,
        module_name: Optional[
            Literal["NeuralNet", "Uniform", "Tabular", "Zero"]
        ] = None,
        forward_looking=False,
        **nn_kwargs,
    ):
        if module is not None:
            assert module.output_dim == 1
        super().__init__(
            env, module=module, output_dim=1, module_name=module_name, **nn_kwargs
        )
        self.forward_looking = forward_looking

    def __call__(self, states: States) -> OutputTensor:
        out = super().__call__(states)
        if self.forward_looking:
            log_rewards = self.env.log_reward(states).unsqueeze(-1)
            out = out + log_rewards
        return out


class LogitPFEstimator(FunctionEstimator):
    r"""Container for estimators $s \mapsto (u(s + a \mid s))_{a \in \mathbb{A}}$,
    such that $P_F(s + a \mid s) = \frac{e^{u(s + a \mid s)}}{\sum_{a' \in \mathbb{A}} e^{u(s + a' \mid s)}}$."""

    def __init__(
        self,
        env: Env,
        module: Optional[GFNModule] = None,
        module_name: Optional[
            Literal["NeuralNet", "Uniform", "Tabular", "Zero"]
        ] = None,
        **nn_kwargs,
    ):
        if module is not None:
            assert module.output_dim == env.n_actions
        super().__init__(
            env,
            module=module,
            output_dim=env.n_actions,
            module_name=module_name,
            **nn_kwargs,
        )


class LogitPBEstimator(FunctionEstimator):
    r"""Container for estimators $s \mapsto (u(s' - a \mid s'))_{a \in \mathbb{A}}$,
    such that $P_B(s' - a \mid s') = \frac{e^{u(s' - a \mid s')}}{\sum_{a' \in \mathbb{A}} e^{u(s' - a' \mid s')}}$."""

    def __init__(
        self,
        env: Env,
        module: Optional[GFNModule] = None,
        module_name: Optional[
            Literal["NeuralNet", "Uniform", "Tabular", "Zero"]
        ] = None,
        **nn_kwargs,
    ):
        if module is not None:
            assert module.output_dim == env.n_actions - 1
        super().__init__(
            env,
            module=module,
            output_dim=env.n_actions - 1,
            module_name=module_name,
            **nn_kwargs,
        )


class LogZEstimator:
    r"""Container for the estimator $\log Z$."""

    def __init__(self, tensor: TensorType[0, float]) -> None:
        self.tensor = tensor
        assert self.tensor.shape == ()
        self.tensor.requires_grad = True

    def __repr__(self) -> str:
        return str(self.tensor.item())

    def named_parameters(self) -> dict:
        return {"logZ": self.tensor}

    def load_state_dict(self, state_dict: dict):
        self.tensor = state_dict["logZ"]
