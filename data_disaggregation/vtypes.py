"""Type classes for data."""

from abc import ABC
from typing import Any

from pandas import NA, DataFrame, Series

from .utils import (
    is_na,
    na_as_0,
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

    @classmethod
    def transform(
        cls,
        ds_data: Series[Any],  # incl. NA
        df_weight_map: DataFrame,  # >= 0
        ds_weights_from: Series,
        weight_rel_threshold: float = 0.0,
        na_dim_key: Any = None,
    ) -> Series:
        ds_data = ds_data.reindex(df_weight_map.index)

        def agg(weights: Series):
            weights_gt0 = weights.loc[weights > 0]
            ds_values_incl_na = ds_data.loc[weights_gt0.index]
            idx_val_na = ds_values_incl_na.isna()
            weights_val_na = weights_gt0.loc[idx_val_na]
            weights_val_not_na = weights_gt0.loc[~idx_val_na]
            values_not_na = ds_values_incl_na.loc[~idx_val_na]
            weights_sum = weights_gt0.sum()
            weights_val_na_sum = weights_val_na.sum()

            if (
                len(weights_val_not_na) == 0
                or (weights_val_na_sum / weights_sum) > weight_rel_threshold
            ):
                return NA

            result = cls.weighted_aggregate_ds(values_not_na, weights_val_not_na)
            return result

        ds_result: Series = df_weight_map.apply(agg)

        # remove if not used
        del ds_result[na_dim_key]

        return ds_result


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
        return ds_data.sum()

    @classmethod
    def transform(
        cls,
        ds_data: Series[Any],  # incl. NA
        df_weight_map: DataFrame,  # >= 0
        ds_weights_from: Series,
        weight_rel_threshold: float = 0.0,
        na_dim_key: Any = None,
    ) -> Series:
        # FIXME
        global sum_data_numeric_ext_unmapped

        # only for extensive: preserve values that are unmapped
        # and add them to NA output key
        sum_data_numeric_ext_unmapped = ds_data.loc[
            ~ds_data.index.isin(df_weight_map.index)
        ].sum()

        ds_data = ds_data.reindex(df_weight_map.index)

        def agg(weights: Series):
            weights_gt0 = weights.loc[weights > 0]
            ds_values_incl_na = ds_data.loc[weights_gt0.index]
            idx_val_na = ds_values_incl_na.isna()
            weights_val_na = weights_gt0.loc[idx_val_na]
            weights_val_not_na = weights_gt0.loc[~idx_val_na]
            values_not_na = ds_values_incl_na.loc[~idx_val_na]
            weights_sum = weights_gt0.sum()
            weights_val_na_sum = weights_val_na.sum()

            global sum_data_numeric_ext_unmapped

            values_not_na = values_not_na * weights_val_not_na / ds_weights_from

            if (
                len(weights_val_not_na) == 0
                or (weights_val_na_sum / weights_sum) > weight_rel_threshold
            ):
                sum_data_numeric_ext_unmapped += values_not_na.sum()
                return NA

            result = cls.weighted_aggregate_ds(values_not_na, weights_val_not_na)
            return result

        ds_result: Series = df_weight_map.apply(agg)

        if sum_data_numeric_ext_unmapped:
            ds_result[na_dim_key] = na_as_0(ds_result[na_dim_key]) + na_as_0(
                sum_data_numeric_ext_unmapped
            )

        # remove if not used
        if is_na(ds_result[na_dim_key]):
            del ds_result[na_dim_key]

        return ds_result
