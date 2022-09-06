from torchtyping import TensorType

from gfn.containers import States
from gfn.modules import GFNModule, Tabular
from gfn.preprocessors.base import EnumPreprocessor, Preprocessor

# Typing
batch_shape = None
input_dim = None
output_dim = None
InputTensor = TensorType["batch_shape", "input_dim", float]
OutputTensor = TensorType["batch_shape", "output_dim", float]


class FunctionEstimator:
    def __init__(self, preprocessor: Preprocessor, module: GFNModule) -> None:
        assert module.input_dim is None or module.input_dim == preprocessor.output_dim
        assert module.output_type == "free"
        if isinstance(module, Tabular) and not isinstance(
            preprocessor, EnumPreprocessor
        ):
            raise ValueError("Tabular modules must use the EnumPreprocessor")
        self.preprocessor = preprocessor
        self.module = module

    def __call__(self, states: States) -> OutputTensor:
        return self.module(self.preprocessor(states))


class LogEdgeFlowEstimator(FunctionEstimator):
    def __init__(self, preprocessor: Preprocessor, module: GFNModule):
        super().__init__(preprocessor, module)
        assert module.output_dim == preprocessor.env.n_actions - 1


class LogStateFlowEstimator(FunctionEstimator):
    def __init__(self, preprocessor: Preprocessor, module: GFNModule):
        super().__init__(preprocessor, module)
        assert module.output_dim == 1


class LogitPFEstimator(FunctionEstimator):
    def __init__(self, preprocessor: Preprocessor, module: GFNModule):
        super().__init__(preprocessor, module)
        assert module.output_dim == preprocessor.env.n_actions


class LogitPBEstimator(FunctionEstimator):
    def __init__(self, preprocessor: Preprocessor, module: GFNModule):
        super().__init__(preprocessor, module)
        assert module.output_dim == preprocessor.env.n_actions - 1


class LogZEstimator:
    def __init__(self, tensor: TensorType[0, float]) -> None:
        self.tensor = tensor
        assert self.tensor.shape == ()
        self.tensor.requires_grad = True

    def __repr__(self) -> str:
        return str(self.tensor.item())
