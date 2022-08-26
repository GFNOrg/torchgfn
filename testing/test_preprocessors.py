import pytest

from gfn.envs import HyperGrid
from gfn.preprocessors.base import IdentityPreprocessor
from gfn.preprocessors.hot import KHotPreprocessor, OneHotPreprocessor


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("height", [4, 8])
def test_preprocessors_hypergrid(ndim, height):
    env = HyperGrid(ndim=ndim, height=height)

    identity_preprocessor = IdentityPreprocessor(env)
    one_hot_preprocessor = OneHotPreprocessor(env)
    k_hot_preprocessor = KHotPreprocessor(env)

    random_states = env.reset(batch_shape=10, random_init=True)
    print("States to try: ", random_states)
    preprocessed_grid = identity_preprocessor.preprocess(random_states)
    print("Identity Preprocessor: ", preprocessed_grid)
    one_hot_grid = one_hot_preprocessor.preprocess(random_states)
    print("One Hot Preprocessor: ", one_hot_grid)
    k_hot_grid = k_hot_preprocessor.preprocess(random_states)
    print("K Hot Preprocessor: ", k_hot_grid)

    print("Testing The Preprocessors on HyperGrid with multi-dimensional batches")

    identity_preprocessor = IdentityPreprocessor(env)
    one_hot_preprocessor = OneHotPreprocessor(env)
    k_hot_preprocessor = KHotPreprocessor(env)

    random_states = env.reset(batch_shape=(4, 2), random_init=True)

    print("States to try: ", random_states)
    preprocessed_grid = identity_preprocessor.preprocess(random_states)
    print("Identity Preprocessor: ", preprocessed_grid)
    one_hot_grid = one_hot_preprocessor.preprocess(random_states)
    print("One Hot Preprocessor: ", one_hot_grid)
    k_hot_grid = k_hot_preprocessor.preprocess(random_states)
    print("K Hot Preprocessor: ", k_hot_grid)
