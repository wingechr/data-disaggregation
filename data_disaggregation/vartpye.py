from abc import ABC

from . import utils


class VarTypeBase(ABC):
    @classmethod
    def weighted_aggregate(cls, data):
        """aggregation

        Args:
            data (list): non empty list of (value, weight) pairs

        Returns
            aggregated value
        """
        raise NotImplementedError()


class VarTypeCategorical(VarTypeBase):
    """
    Examples: Regional Codes
    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_mode(data)


class VarTypeOrdinal(VarTypeCategorical):
    """
    Values can be sorted in a meaningful way
    Usually, that means using numerical codes that
    do not represent a metric distance, liek a likert scale

    Examples: [1 = "a little", 2 = "somewhat", 3 = "a lot"]

    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_median(data)


class VarTypeMetric(VarTypeBase):
    """
    * Values can be calculated by linear combinations
    * Examples: height, temperature, density
    """

    @classmethod
    def weighted_aggregate(cls, data):
        return utils.weighted_sum(data)


class VarTypeMetricExt(VarTypeMetric):
    """
    Values are extensive, i.e. they are can be transformed into intensive by dividing by domain size

    * Examples: population, energy production
    """

    pass
