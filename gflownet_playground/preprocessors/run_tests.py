from gflownet_playground.preprocessors.base import IdentityPreprocessor
from gflownet_playground.preprocessors.hot import OneHotPreprocessor, KHotPreprocessor


if __name__ == '__main__':
    import torch
    from gflownet_playground.envs.hypergrid import HyperGrid

    ndim = 2
    height = 4

    env = HyperGrid(ndim=ndim, height=height)

    print('Testing The Preprocessors on HyperGrid')

    identity_preprocessor = IdentityPreprocessor(env)
    one_hot_preprocessor = OneHotPreprocessor(env)
    k_hot_preprocessor = KHotPreprocessor(env)

    random_states = torch.randint(0, height, (10, ndim))
    random_states = env.StatesBatch(states=random_states)
    print('States to try: ', random_states)
    preprocessed_grid = identity_preprocessor.preprocess(random_states)
    print('Identity Preprocessor: ', preprocessed_grid)
    one_hot_grid = one_hot_preprocessor.preprocess(random_states)
    print('One Hot Preprocessor: ', one_hot_grid)
    k_hot_grid = k_hot_preprocessor.preprocess(random_states)
    print('K Hot Preprocessor: ', k_hot_grid)
