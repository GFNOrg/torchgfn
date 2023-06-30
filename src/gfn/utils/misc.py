from pathlib import Path
import sys


def get_root() -> Path:
    """
    Returns the root directory of the project.
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def add_root_to_path():
    """
    Adds the root directory of the project to the python path.
    """
    sys.path.append(str(get_root()))
