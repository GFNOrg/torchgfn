from einops import rearrange
from torch.nn.functional import one_hot

from gfn.preprocessors.base import Preprocessor


class OneHotPreprocessor(Preprocessor):
    "Use One Hot Preprocessing for environment with enumerable states"
    name = "one_hot"

    @property
    def output_dim(self):
        return self.env.n_states

    def preprocess(self, states):
        state_indices = self.env.get_states_indices(states)
        return one_hot(state_indices, self.output_dim).float()


class KHotPreprocessor(Preprocessor):
    "Use K Hot Preprocessing for environment with enumerable states with a grid structure"
    name = "k_hot"

    @property
    def output_dim(self):
        if not hasattr(self.env, "height"):
            raise ValueError("The environment does not support K Hot preprocessing")
        output_dim = self.env.height * self.env.ndim
        return output_dim

    def preprocess(self, states):
        states_tensor = states.states
        assert states_tensor.equal(
            states_tensor.floor()
        ), "K Hot preprocessing only works for integer states"
        states_tensor = states_tensor.long()
        hot = one_hot(states_tensor, int(self.output_dim / self.env.ndim)).float()
        hot = rearrange(hot, "... a b -> ... (a b)")
        return hot
