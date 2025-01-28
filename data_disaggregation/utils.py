"""utility functions
"""

import math
from typing import Callable, List, Mapping, Tuple

from pandas import DataFrame, Index, Series

from . import vtypes


def group_sum(key_vals: Mapping, get_key: Callable = None) -> Mapping:
    """simple group sum.

    Parameters
    ----------
    key_vals: Mapping
        * keys can be anything hashable,
        * values must be numerical
    get_key: Callable = None

    Returns
    -------
    : Mapping
        list of (unique key, sum of values) pairs
    """

    res = {}
    if not get_key:
        for k, v in key_vals:
            res[k] = res.get(k, 0) + v
    else:
        # custom get key
        for k, v in key_vals.items():
            k = get_key(k)
            res[k] = res.get(k, 0) + v

    return res


def weighted_sum(value_normweights: Tuple[float]) -> float:
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


def weighted_mode(value_normweights: Tuple):
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


def weighted_percentile(value_normweights: Tuple, p=0.5):
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
    # make values unique (sum weights)
    value_normweights = group_sum(value_normweights).items()
    # get cumulative values
    wsum = 0
    for v, w in sorted(value_normweights, key=lambda vw: vw[0]):
        wsum += w
        if wsum >= p:
            return v
    raise ValueError()


def weighted_median(value_normweights: Tuple):
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


def group_idx_first(items: Mapping) -> Mapping:
    return group_sum(items, lambda k: k[0])


def group_idx_second(items: Mapping) -> Mapping:
    return group_sum(items, lambda k: k[1])


def is_na(x) -> bool:
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def is_scalar(x) -> bool:
    return isinstance(x, (str, int, float, bool)) or is_na(x)


def is_list(x) -> bool:
    return isinstance(x, (list, tuple, set, Index))


def is_mapping(x) -> bool:
    return isinstance(x, (dict, Series, DataFrame))


def is_unique(x) -> bool:
    x = as_list(x)
    return len(x) == len(set(x))


def is_subset(a, b):
    return set(as_list(a)) <= set(as_list(b))


def iter_values(x):
    for k in x.keys():
        yield x[k]


def as_set(x) -> set:
    return set(as_list(x))


def as_list(x) -> List:
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
        return dict((k, default_val) for k in x)
    elif is_scalar(x):
        return {vtypes.SCALAR_INDEX_KEY: x}
    raise TypeError(x)


def as_scalar(x):
    if as_scalar(x):
        return x
    elif is_mapping(x):
        assert set(x.keys()) == set([vtypes.SCALAR_INDEX_KEY])
        return x[vtypes.SCALAR_INDEX_KEY]
    raise TypeError(x)


def is_map(map) -> bool:
    """TODO: this is slow"""
    return is_mapping(map) and all(len(k) == 2 for k in map.keys())
