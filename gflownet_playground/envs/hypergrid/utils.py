from ssl import OP_NO_RENEGOTIATION
import torch
from gflownet_playground.utils import Preprocessor
from torch.nn.functional import one_hot



def uniform_backwards_prob(traj):
    """
    :param traj: tensor of size T x state_dim representing a trajectory
    :return: In the HyperGrid, uniform prob over the parents of each state
    """
    return 1. / (traj[1:] > 0).sum(1)


class OneHotPreprocessor(Preprocessor):
    def __init__(self, ndim=2, H=8) -> None:
        self.ndim = ndim
        self.H = H
        self.output_dim = self.H ** self.ndim

    def indexify(self, states):
        assert states.shape[1] == self.ndim
        return get_states_indices(states, self.H)

    def preprocess(self, states):
        return one_hot(self.indexify(states), num_classes=self.output_dim).float()


class KHotPreprocessor(Preprocessor):
    def __init__(self, ndim=2, H=8) -> None:
        self.ndim = ndim
        self.H = H
        self.output_dim = self.H * self.ndim

    def preprocess(self, states):
        assert states.shape[1] == self.ndim
        return one_hot(states.long(), self.H).view(states.shape[0],-1).float()


def get_states_indices(states, H):
    ndim = states.shape[-1]
    canonical_base = H ** torch.arange(ndim - 1, -1, -1)
    return (canonical_base * states).sum(1).long()
    

if __name__ == '__main__':
    import torch
    from gflownet_playground.envs.hypergrid.hypergrid_env import HyperGrid

    ndim = 2
    H = 4

    env = HyperGrid(ndim, H)

    print('Testing OneHot and KHot Preprocessor')
    one_hot_preprocessor = OneHotPreprocessor(ndim, H)
    k_hot_preprocessor = KHotPreprocessor(ndim, H)

    explicit_grid = env.grid.view(-1, ndim)
    one_hot_grid = one_hot_preprocessor.preprocess(explicit_grid)
    k_hot_grid = k_hot_preprocessor.preprocess(explicit_grid)

    for i in range(H ** ndim):
        print(explicit_grid[i])
        print(one_hot_grid[i])
        print(k_hot_grid[i])
        print('')

    states = torch.randint(0, H, (10, ndim))
    print(states)
    print(get_states_indices(states, H))

