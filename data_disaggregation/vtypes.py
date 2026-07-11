"""Type classes for data.
"""

from abc import ABC
from typing import TypeVar

from . import utils

F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")

SCALAR_DIM_NAME = "__SCALAR__"
# TODO: using None in pandas causes problems with autoconvert to nan
SCALAR_INDEX_KEY = "__SCALAR__"


class VariableType(ABC):
    @classmethod
    def weighted_aggregate(cls, data):
        """aggregate data

        Parameters
        ----------
        data : Iterable
            non empty list of (value, weight) pairs.
            weights must be numerical, positive, and sum up to 1.0.

        Returns
        -------
        aggregated value
        """
        raise NotImplementedError()


class VT_Nominal(VariableType):
    """Type class for nominal (categorical) data.

    - Aggregation method: mode (most commonly used)
    - Disaggregation method: keep value
    - Examples: regional codes
    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_mode(data)


class VT_Ordinal(VT_Nominal):
    """Type class for ordinal data (ranked categorical).

    - Aggregation method: median
    - Disaggregation method: keep value
    - Examples: Level of agreement
    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_median(data)


class VT_Numeric(VariableType):
    """Type class for numerical, intensive data

    An intensive variable is one which does not scale with the system size.

    - Aggregation method: weighted average
    - Disaggregation method: keep value
    - Examples: temperature, density, pressure
    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_sum(data)


class VT_NumericExt(VT_Numeric):
    """Type class for numerical, extensive data.

    An extensive variable is one which does scale with the system size
    (assuming an equal distribution).

    - Aggregation method: sum
    - Disaggregation method: distribute by weights
    - Examples: population, energy, total cost
    """

    pass
