from gfn.preprocessors.base import IdentityPreprocessor
from gfn.preprocessors.hot import KHotPreprocessor, OneHotPreprocessor

if __name__ == "__main__":
    from gfn.envs import HyperGrid

    ndim = 2
    height = 4

    env = HyperGrid(ndim=ndim, height=height)

    print("Testing The Preprocessors on HyperGrid")

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
