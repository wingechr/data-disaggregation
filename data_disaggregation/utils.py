"""utility functions"""

import math
from typing import TypeVar

from pandas import NA, DataFrame, Series

F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")
K = TypeVar("K")
K2 = TypeVar("K2")
V = TypeVar("V")

SeriesDict = TypeVar("SeriesDict", Series, dict)
SeriesFrame = TypeVar("SeriesFrame", Series, DataFrame)

SCALAR_DIM_NAME = "__SCALAR__"
# TODO: using None in pandas causes problems with autoconvert to nan
SCALAR_INDEX_KEY = "__SCALAR__"


def weighted_sum_ds(ds_data: Series, ds_weights: Series) -> float:
    return (ds_data * ds_weights).sum() / ds_weights.sum()


def weighted_mode_ds(ds_data: Series, ds_weights: Series):
    """get most common value (but by weight)

    Parameters
    ----------
    value_normweights : list
        non empty list of (value, weight) pairs.
        * values must be anything sortable.
        * weights must be numerical, positive, and sum up to 1.0.

    Returns
    -------
    Any

    """
    return (
        _sum_weight_gruopby_values_ds(ds_data, ds_weights)
        .sort_values(ascending=False)
        .index[0]
    )


def _sum_weight_gruopby_values_ds(ds_data: Series, ds_weights: Series) -> Series:
    return ds_weights.set_axis(ds_data.values).groupby(level=0).sum()


def _weighted_percentile_ds(ds_data: Series, ds_weights: Series, p: float = 0.5):
    """get most median (but by weight)

    Parameters
    ----------
    value_normweights : list
        non empty list of (value, weight) pairs.
        * values must be anything sortable.
        * weights must be numerical, positive, and sum up to 1.0.
    p:
        threshold

    Returns
    -------
    Any

    """

    # normalize weights
    threshold = p * ds_weights.sum()
    # make values unique (sum weights)
    ds_grouped = _sum_weight_gruopby_values_ds(ds_data, ds_weights).sort_index()
    # find first index where cum sum >= p
    return ds_grouped.cumsum().ge(threshold).idxmax()


def weighted_median_ds(ds_data: Series, ds_weights: Series):
    """get most median (but by weight)

    Parameters
    ----------
    value_normweights : list
        non empty list of (value, weight) pairs.
        * values must be anything sortable.
        * weights must be numerical, positive, and sum up to 1.0.

    Returns
    -------
    Any

    """
    return _weighted_percentile_ds(ds_data, ds_weights, p=0.5)


def is_na(x) -> bool:
    return x is None or x is NA or (isinstance(x, float) and not math.isfinite(x))


def na_as_0(x: float) -> float:
    return x if math.isfinite(x) else 0


def is_scalar(x) -> bool:
    return isinstance(x, (str, int, float, bool)) or is_na(x)


def as_series(x: SeriesDict) -> Series:
    if isinstance(x, Series):
        return x
    return Series(x)
