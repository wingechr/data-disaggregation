"""Type classes for data."""

from abc import ABC

from pandas import Series

from .utils import (
    weighted_median_ds,
    weighted_mode_ds,
    weighted_sum_ds,
)


class VariableType(ABC):
    @classmethod
    def weighted_aggregate_ds(cls, ds_data: Series, ds_weights: Series):
        """aggregate data

        Parameters
        ----------
        ds_data: Series
            non empty list of (value, weight) pairs.
            weights must be numerical, positive, and sum up to 1.0.
        ds_weights: Series
            TODO

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
    def weighted_aggregate_ds(cls, ds_data: Series, ds_weights: Series):
        return weighted_mode_ds(ds_data, ds_weights)


class VT_Ordinal(VT_Nominal):
    @classmethod
    def weighted_aggregate_ds(cls, ds_data: Series, ds_weights: Series):
        return weighted_median_ds(ds_data, ds_weights)


class VT_Numeric(VariableType):
    """Type class for numerical, intensive data

    An intensive variable is one which does not scale with the system size.

    - Aggregation method: weighted average
    - Disaggregation method: keep value
    - Examples: temperature, density, pressure
    """

    @classmethod
    def weighted_aggregate_ds(cls, ds_data: Series, ds_weights: Series):
        """get sum product.

        Parameters
        ----------
        value_normweights : list
            non empty list of (value, weight) pairs.
            * values must be numerical.
            * weights must be numerical, positive, and sum up to 1.0.

        Returns
        -------
        : float

        """
        return weighted_sum_ds(ds_data, ds_weights)


class VT_NumericExt(VT_Numeric):
    """Type class for numerical, extensive data.

    An extensive variable is one which does scale with the system size
    (assuming an equal distribution).

    - Aggregation method: sum
    - Disaggregation method: distribute by weights
    - Examples: population, energy, total cost
    """

    pass
