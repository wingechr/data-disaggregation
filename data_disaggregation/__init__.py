"""
Wrapper classes around multidimensional numpy matrices
for (dis)aggregation of data.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

__version__ = "0.1.3"
__title__ = "Data Disaggregation"
__author__ = "Christian Winger"
__email__ = "c.winger@oeko.de"
__copyright__ = "GPLv3+"
__url__ = "https://github.com/wingechr/data-disaggregation"

try:
    from .classes import (
        Dimension,
        ExtensiveScalar,
        ExtensiveVariable,
        IntensiveScalar,
        IntensiveVariable,
        Variable,
        Weight,
    )
except ImportError:
    # allow missing imports on install
    pass

__all__ = [
    "Dimension",
    "ExtensiveScalar",
    "ExtensiveVariable",
    "IntensiveScalar",
    "IntensiveVariable",
    "Variable",
    "Weight",
]
