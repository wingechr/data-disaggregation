"""classes and types
"""

from abc import ABC
from typing import TypeVar

from . import ext, utils

F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")


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

    @classmethod
    def disagg(
        cls,
        var,
        map,
        dim_out=None,
        dim_in=None,
        threshold: float = 0.0,
        as_int: bool = False,
    ):
        """
        Args:
            vtype: data type (impacts aggregation function)
            var: indexed data
            map: weight data
            size_t (optional):
            size_f (optional):
            threshold (optional):
            as_int (optional):

        """

        return ext.disagg(
            cls,
            var,
            map,
            dim_out=dim_out,
            dim_in=dim_in,
            threshold=threshold,
            as_int=as_int,
        )


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
