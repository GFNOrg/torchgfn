from typing import Literal, Optional

from torchtyping import TensorType

from gfn.containers import States
from gfn.envs import Env
from gfn.envs.preprocessors.base import EnumPreprocessor
from gfn.modules import NeuralNet, Tabular, Uniform, ZeroGFNModule

# Typing
OutputTensor = TensorType["batch_shape", "output_dim", float]

module_names_to_clss = {
    "NeuralNet": NeuralNet,
    "Tabular": Tabular,
    "Uniform": Uniform,
    "Zero": ZeroGFNModule,
}


class DiscreteFunctionEstimator:
    def __init__(
        self,
        env: Env,
        output_dim: Optional[int],
        module_name: Literal["NeuralNet", "Uniform", "Tabular", "Zero"],
        **module_kwargs,
    ) -> None:
        module_cls = module_names_to_clss[module_name]
        module_kwargs["env"] = env
        self.input_shape = env.preprocessor.output_shape
        self.module_name = module_name
        if "module" in module_kwargs:
            self.module = module_kwargs["module"]
        else:
            self.module = module_cls(
                input_shape=self.input_shape, output_dim=output_dim, **module_kwargs
            )
        if module_name == "Tabular":
            self.preprocessor = EnumPreprocessor(env.get_states_indices)
        else:
            self.preprocessor = env.preprocessor

    def __call__(self, states: States) -> OutputTensor:
        return self.module(self.preprocessor(states))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.module})"


class LogEdgeFlowEstimator(DiscreteFunctionEstimator):
    def __init__(
        self,
        env: Env,
        module_name: Literal["NeuralNet", "Tabular", "Zero"] = "NeuralNet",
        **module_kwargs,
    ) -> None:
        super().__init__(
            env, output_dim=env.n_actions, module_name=module_name, **module_kwargs
        )


class LogStateFlowEstimator(DiscreteFunctionEstimator):
    def __init__(
        self,
        env: Env,
        module_name: Literal["NeuralNet", "Tabular", "Zero"] = "NeuralNet",
        **module_kwargs,
    ):
        super().__init__(env, output_dim=1, module_name=module_name, **module_kwargs)


class LogitPFEstimator(DiscreteFunctionEstimator):
    def __init__(
        self,
        env: Env,
        module_name: Literal["NeuralNet", "Tabular", "Uniform"] = "NeuralNet",
        **module_kwargs,
    ):
        super().__init__(
            env, output_dim=env.n_actions, module_name=module_name, **module_kwargs
        )


class LogitPBEstimator(DiscreteFunctionEstimator):
    def __init__(
        self,
        env: Env,
        module_name: Literal["NeuralNet", "Tabular", "Uniform"] = "NeuralNet",
        **module_kwargs,
    ):
        super().__init__(
            env, output_dim=env.n_actions - 1, module_name=module_name, **module_kwargs
        )


class LogZEstimator:
    def __init__(self, tensor: TensorType[0, float]) -> None:
        self.tensor = tensor
        assert self.tensor.shape == ()
        self.tensor.requires_grad = True

    def __repr__(self) -> str:
        return str(self.tensor.item())
