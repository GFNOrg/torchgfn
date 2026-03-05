"""This module contains all the environments implemented as Gym environments."""

from .bitSequence import BitSequence, BitSequencePlus
from .box import BoxPolar
from .box_cartesian import BoxCartesian
from .discrete_ebm import DiscreteEBM
from .graph_building import GraphBuilding, GraphBuildingOnEdges
from .hypergrid import ConditionalHyperGrid, HyperGrid
from .line import Line
from .perfect_tree import PerfectBinaryTree
from .set_addition import SetAddition

# Backward compat: Box = BoxCartesian (matches current test/tutorial usage)
Box = BoxCartesian

__all__ = [
    "Box",
    "BoxCartesian",
    "BoxPolar",
    "DiscreteEBM",
    "HyperGrid",
    "ConditionalHyperGrid",
    "Line",
    "BitSequence",
    "BitSequencePlus",
    "GraphBuilding",
    "GraphBuildingOnEdges",
    "PerfectBinaryTree",
    "SetAddition",
]
