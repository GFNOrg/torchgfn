from gflownet_playground.preprocessors.base import IdentityPreprocessor
from gflownet_playground.preprocessors.hot import OneHotPreprocessor, KHotPreprocessor


if __name__ == '__main__':
    import torch
    from gflownet_playground.envs.hypergrid.hypergrid_env import HyperGrid

    ndim = 2
    H = 4

    env = HyperGrid(ndim, H)

    print('Testing The Preprocessors on HyperGrid')

    identity_preprocessor = IdentityPreprocessor(env)
    one_hot_preprocessor = OneHotPreprocessor(env)
    k_hot_preprocessor = KHotPreprocessor(env)

    explicit_grid = env.grid.view(-1, ndim)
    one_hot_grid = one_hot_preprocessor.preprocess(explicit_grid)
    k_hot_grid = k_hot_preprocessor.preprocess(explicit_grid)
    preprocessed_grid = identity_preprocessor.preprocess(explicit_grid)

    print('Looping through each state and showing its 3 preprocessed versions')
    for i in range(H ** ndim):
        print(explicit_grid[i])
        print(preprocessed_grid[i])
        print(one_hot_grid[i])
        print(k_hot_grid[i])
        print('')

    print('Doing the same, but without flattening the grid')
    explicit_grid = env.grid
    print("Shape of the grid:", explicit_grid.shape)
    preprocessed_grid = identity_preprocessor.preprocess(explicit_grid)
    print("Identity preprocessed grid is of shape {}:".format(
        preprocessed_grid.shape))
    print(preprocessed_grid)
    one_hot_grid = one_hot_preprocessor.preprocess(explicit_grid)
    print("OneHot preprocessed grid is of shape {}:".format(one_hot_grid.shape))
    print(one_hot_grid)
    k_hot_grid = k_hot_preprocessor.preprocess(explicit_grid)
    print("KHot preprocessed grid is of shape {}:".format(k_hot_grid.shape))
    print(k_hot_grid)
