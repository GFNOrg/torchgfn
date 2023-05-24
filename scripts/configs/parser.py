import ast
import collections
from argparse import ArgumentParser
from copy import deepcopy
from os.path import expandvars
from pathlib import Path
from typing import Any, Tuple, Union

from yaml import safe_load


def resolve(path: Union[str, Path]) -> Path:
    """
    Resolve a path to an absolute ``pathlib.Path``, expanding environment variables and
    user home directory.

    Args:
        path: The path to resolve.

    Returns:
        The resolved path.
    """
    return Path(expandvars(path)).expanduser().resolve()


def update(orig_dict: dict, new_dict: dict) -> dict:
    """
    Update a nested dictionary or similar mapping.
    Not in-place.

    .. code-block:: python

        >>> orig_dict = {'a': {'b': 1, 'c': 2}}
        >>> new_dict = {'a': {'b': 3, 'd': 4}}
        >>> update(orig_dict, new_dict)
        {'a': {'b': 3, 'c': 2, 'd': 4}}

    Args:
        orig_dict (dict): Dict to update
        new_dict (dict): Dict to update with

    Returns:
        dict: Deeply merged dict
    """
    orig_dict = deepcopy(orig_dict)
    for key, val in new_dict.items():
        if isinstance(val, collections.abc.Mapping):
            tmp = update(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = orig_dict.get(key, []) + val
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def parse_value(value: Any) -> Any:
    """
    Parse string as Python literal if possible and fallback to string.
    """
    try:
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Use as string if nothing else worked
        return value


def dict_set_recursively(dictionary, key_sequence, val):
    top_key = key_sequence.pop(0)
    if len(key_sequence) == 0:
        dictionary[top_key] = val
    else:
        if top_key not in dictionary:
            dictionary[top_key] = {}
        dict_set_recursively(dictionary[top_key], key_sequence, val)


def create_dict_from_args(args: list, sep: str = "."):
    """
    Create a (nested) dictionary from console arguments.
    Keys in different dictionary levels are separated by sep.
    """
    return_dict = {}
    for arg in args:
        arg = arg.strip("--")
        parts = arg.split("=") if "=" in arg else (arg, "True")
        if len(parts) == 2:
            keys_concat, val = parts
        elif len(parts) > 2:
            keys_concat, val = parts[0], "=".join(parts[1:])
        else:
            raise ValueError(f"Invalid argument {arg}")
        val = parse_value(val)
        key_sequence = keys_concat.split(sep)
        dict_set_recursively(return_dict, key_sequence, val)
    return return_dict


def parse_args_to_dict(parser: ArgumentParser) -> Tuple[dict, dict]:
    """
    Parse default arguments in a dictionnary,
    and arbitrary extra command line arguments to another dictionary.

        Returns:
            Tuple[dict, dict]: command-line args as dictionaries
    """
    # Parse args
    args, override_args = parser.parse_known_args()

    return vars(args), create_dict_from_args(override_args)


def load_named_config(namespace: str, value: str) -> dict:
    """
    Load a named config from a namespace.
    Starts with ./{namespace}/base.yaml and overrides with ./{namespace}/{value}.yaml.

    Args:
        namespace (str): Folder to look for configs in
        value (str): Name of the config to load in the namespace

    Raises:
        ValueError: If the config name does not exist in the folder.

    Returns:
        dict: Config for that namespace
    """
    base = resolve(Path(__file__)).parent
    config = safe_load((base / f"{namespace}/base.yaml").read_text())

    if value is None:
        return config

    value = value.replace(".yaml", "")
    value = base / f"{namespace}/{value}.yaml"

    if not value.exists():
        raise ValueError(f"Config {value.name} does not exist in {str(base)}.")

    return update(config, safe_load(value.read_text()))


def load_config(parser: ArgumentParser) -> dict:
    """
    Parse command line arguments and load config files.

    {namespace}/base.yaml is always loaded first and will be overridden by other config
    files specified from the command-line.

    Raises:
        ValueError: If the config file/name does not exist.

    Returns:
        dict: GFlowNet run config
    """
    config, cli = parse_args_to_dict(parser)
    namespaces = set(b.parent.name for b in Path(__file__).parent.glob("*/base.yaml"))
    for namespace in namespaces:
        value = config.get(namespace.replace(".yaml", ""))
        config[namespace] = load_named_config(namespace, value)
        config[namespace]["name"] = value
    return update(config, cli)
