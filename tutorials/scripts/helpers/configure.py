# unified version of all the data classes in other .py files.
# utlimately there will only be train.py parser.py and configure.py
# potentially ditching yamls too for clarity

from gfn.env import Env
from gfn.gym import DiscreteEBM, HyperGrid


def make_env(config: dict) -> Env:
    assert config["env"]["device"] in [
        "cpu",
        "cuda",
    ], "Invalid device: {}. Must be 'cpu' or 'cuda'".format(config["env"]["device"])

    name = config["env"]["name"]
    if name.lower() == "hypergrid".lower():
        processor_name = config["env"].get("preprocessor_name", "KHot")
        assert processor_name in {
            "KHot",
            "OneHot",
            "Identity",
        }, f"Invalid preprocessor name: {processor_name}"

        return HyperGrid(
            ndim=config["env"].get("ndim", 2),
            height=config["env"].get("height", 8),
            R0=config["env"].get("R0", 0.1),
            R1=config["env"].get("R1", 0.5),
            R2=config["env"].get("R2", 2.0),
            reward_cos=config["env"].get("reward_cos", False),
            device_str=config["env"]["device"],
            preprocessor_name=processor_name,
        )
    elif name.lower() == "discrete-ebm".lower():
        return DiscreteEBM(
            ndim=config["env"].get("ndim", 4),
            alpha=config["env"].get("alpha", 1.0),
            device_str=config["env"]["device"],
        )
    else:
        raise ValueError("Invalid env name: {}".format(name))
