"""utility functions"""

from collections.abc import Callable, Collection, Iterable, Mapping
import math
from typing import TypeVar

from pandas import DataFrame, Index, Series

from .vtypes import SCALAR_INDEX_KEY

K = TypeVar("K")
K2 = TypeVar("K2")
V = TypeVar("V")


def group_sum(key_vals: Iterable[tuple[K, V]]) -> Mapping[K, V]:
    """simple group sum.

    Parameters
    ----------
    key_vals: Mapping
        * keys can be anything hashable,
        * values must be numerical

    Returns
    -------
    : Mapping
        list of (unique key, sum of values) pairs

    """

    res = {}
    for k, v in key_vals:
        res[k] = res.get(k, 0) + v

    return res


def weighted_sum(value_normweights: Iterable[tuple[float, float]]) -> float:
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
    # TODO faster methods with numpy or ?
    return sum(v * w for v, w in value_normweights)


def weighted_sum_ds(ds_data: Series, ds_weights: Series) -> float:
    return (ds_data * ds_weights).sum()


def weighted_mode(value_normweights: Iterable[tuple]):
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
    # make values unique (sum weights)
    value_normweights = group_sum(value_normweights).items()
    # first element of item with highest value
    return sorted(value_normweights, key=lambda vw: vw[1], reverse=True)[0][0]


def ascending_values_sum_weights_ds(ds_data: Series, ds_weights: Series) -> Series:
    return (
        ds_weights.set_axis(ds_data.values)
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )


def weighted_mode_ds(ds_data: Series, ds_weights: Series):
    # first element of item with highest value
    return ascending_values_sum_weights_ds(ds_data, ds_weights).index[0]


def weighted_percentile(value_normweights: Iterable[tuple], p=0.5):
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
    # make values unique (sum weights)
    value_normweights = group_sum(value_normweights).items()
    # get cumulative weights, ordered by value
    wsum = 0
    for v, w in sorted(value_normweights, key=lambda vw: vw[0]):
        wsum += w
        if wsum >= p:
            return v
    raise ValueError()


def weighted_percentile_ds(ds_data: Series, ds_weights: Series, p: float = 0.5):
    # make values unique (sum weights)
    ds_grouped = ascending_values_sum_weights_ds(ds_data, ds_weights)
    # find first index where cum sum >= p
    return ds_grouped.cumsum().ge(p).idxmax()


def weighted_median(value_normweights: Iterable[tuple]):
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
    return weighted_percentile(value_normweights, p=0.5)


def weighted_median_ds(ds_data: Series, ds_weights: Series):
    return weighted_percentile_ds(ds_data, ds_weights, p=0.5)


def is_na(x) -> bool:
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def is_scalar(x) -> bool:
    return isinstance(x, (str, int, float, bool)) or is_na(x)


def is_list(x) -> bool:
    return isinstance(x, (list, tuple, set, Index))


def is_mapping(x) -> bool:
    return isinstance(x, (dict, Series, DataFrame))


def is_unique(x) -> bool:
    x = as_collection(x)
    return len(x) == len(set(x))


def is_subset(a, b):
    return set(as_collection(a)) <= set(as_collection(b))


def get_values(x) -> Collection:
    values = x.values
    if isinstance(values, Callable):
        values = values()
    return values


def get_keys(x) -> Collection:
    return x.keys()


def as_set(x) -> set:
    return set(as_collection(x))


def as_collection(x) -> Collection:
    # meaning: is index
    if is_list(x):
        return x
    elif is_mapping(x):
        if isinstance(x, (DataFrame, Series)):
            return x.index
        return list(x.keys())  # TODO maybe wrap in list
    raise TypeError(x)


def as_mapping(x, default_val=1) -> Mapping:
    if is_mapping(x):
        return x
    elif is_list(x):
        return dict.fromkeys(x, default_val)
    elif is_scalar(x):
        return {SCALAR_INDEX_KEY: x}
    raise TypeError(x)


def as_scalar(x):
    if as_scalar(x):
        return x
    elif is_mapping(x):
        assert set(x.keys()) == {SCALAR_INDEX_KEY}
        return x[SCALAR_INDEX_KEY]
    raise TypeError(x)


def is_map(x) -> bool:
    """TODO: this is slow"""
    return is_mapping(x) and all(len(k) == 2 for k in get_keys(x))
