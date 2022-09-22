"""Wrapper classes around multidimensional numpy matrices
for (dis)aggregation of data.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

__version__ = "0.5.1"
__title__ = "Data Disaggregation"
__author__ = "Christian Winger"
__email__ = "c.winger@oeko.de"
__copyright__ = "MIT"
__url__ = "https://github.com/wingechr/data-disaggregation"


from . import draw, exceptions
from .classes import (
    Dimension,
    DimensionLevel,
    Domain,
    ExtensiveScalar,
    ExtensiveVariable,
    IntensiveScalar,
    IntensiveVariable,
    Variable,
    Weight,
)

__all__ = [
    "Dimension",
    "DimensionLevel",
    "Domain",
    "ExtensiveScalar",
    "ExtensiveVariable",
    "IntensiveScalar",
    "IntensiveVariable",
    "Variable",
    "Weight",
]
