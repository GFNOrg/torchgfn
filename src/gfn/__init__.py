import importlib.metadata as met
import logging

__version__ = met.version("torchgfn")

logging.getLogger(__name__).addHandler(logging.NullHandler())
