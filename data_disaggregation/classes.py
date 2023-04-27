"""classes and types
"""

from abc import ABC
from typing import TypeVar

from . import utils

F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")

SCALAR_DIM_NAME = "SCALAR_DIM_NAME"
# using None causes problems with autoconvert to nan
SCALAR_INDEX_KEY = "SCALAR_INDEX_KEY"


class VT(ABC):
    @classmethod
    def weighted_aggregate(cls, data):
        """aggregation

        Args:
            data (list): non empty list of (value, weight) pairs

        Returns
            aggregated value
        """
        raise NotImplementedError()


class VT_Nominal(VT):
    """
    Examples: Regional Codes
    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_mode(data)


class VT_Ordinal(VT_Nominal):
    """
    Values can be sorted in a meaningful way
    Usually, that means using numerical codes that
    do not represent a metric distance, liek a likert scale

    Examples: [1 = "a little", 2 = "somewhat", 3 = "a lot"]

    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_median(data)


class VT_Numeric(VT):
    """
    * Values can be calculated by linear combinations
    * Examples: height, temperature, density
    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_sum(data)


class VT_NumericExt(VT_Numeric):
    """
    Values are extensive, i.e. they are can be transformed into intensive
    by dividing by domain size

    * Examples: population, energy production
    """

    pass
