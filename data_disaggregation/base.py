"""
* a Variable is a (dict like) mapping from a Domain -> Value from a given Range
* a Transformation maps a Variable to a a new Variable in a different Domain
* Domains can be have multiple, nested dimensions

* DomainMap is a Variable that maps (Domain1 x Domain2) to a numerical value (size)
* DomainSize is a Variable that maps a Domain to a numerical value (size)
    this is only needed for Variables of type MetricExtVarType


Algorithm

* We want to map variable of U(Dom1) (of type T) to V(Dom2) (will also be of Type T)
* Inputs:
    * U(Dom1) and T
    * Dom1Size(Dom1)
    * Dom2Size(Dom2)
    * Dom1Dom2Map(Dom1 x Dom2)
* Output
    * V(Dom2) of type T
* Steps
    * start with Dom1Dom2Map, join U (this will replicate values!)
    * IF T==MetricExtVarType:
        * join Dom1Size and Dom2Size and rescale U: U' = U / Dom1Size * Dom2Size
    * GROUP BY Dom2 and use the respective aggregation functions of the type,
      with U being the value and  Dom1Dom2Map being the weight
* Optional Steps
    * if Type is numeric but limited to int: round values
    * if specified: gapfill missing values from Dom2 with na or a suitable default
    * if specified, a threshold for sum(weights)/size(dom2) can be set (usually 0.5)
      to drop elements from output

Helper to create the mapping
* Given a (multidim) input domain and a (multidim) output domain
* and a weight mapping over a (arbitrary) weight domain:
* organize into a unique list by shared dims: dims_in, dims_out
    => [dims_in_only] + [dims_shared] + [dims_out_only]
* weightdin MUST be a subset of this, but MAY have fewers

* Steps:
    * create super domain of cross product of all of those
    * join weights
    * create index pairs for result

"""

from abc import ABC
from itertools import product
from typing import List, Mapping, Optional, Tuple, TypeVar, Union

import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series

from . import utils
from .utils import group_idx_first, group_idx_second, is_na

F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")


# from .vartype import VarTypeBase


F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")


def get_groups(
    vtype: "VarTypeBase",
    var: Mapping[F, V],
    map: Mapping[Tuple[F, T], float],
    size_f: Mapping[F, float],
) -> Mapping[T, Tuple[V, float]]:
    groups = {}

    for (f, t), w in map.items():
        # get not na value
        v = var.get(f)
        if is_na(v):
            continue

        #  scale extensive => intensive
        if vtype == VarTypeMetricExt:
            v /= size_f[f]

        if t not in groups:
            groups[t] = []
        groups[t].append((v, w))

    return groups


def apply_map(
    vtype: "VarTypeBase",
    var: Mapping[F, V],
    map: Mapping[Tuple[F, T], float],
    size_f: Mapping[F, float] = None,
    size_t: Mapping[T, float] = None,
    threshold: float = 0.0,
    as_int: bool = False,
) -> Mapping[T, V]:
    # sanity check

    result = {}

    size_f = size_f or group_idx_first(map)
    size_t = size_t or group_idx_second(map)

    def _values(x):
        # TODO
        if isinstance(x, dict):
            return x.values()
        else:  # series
            return x.values

    assert all(v >= 0 for v in _values(map))
    assert all(v > 0 for v in _values(size_f))
    assert all(v > 0 for v in _values(size_t))

    groups = get_groups(vtype, var, map, size_f)

    for t, vws in groups.items():
        # weights sum
        sumw = sum(w for _, w in vws)

        # drop result?
        if threshold:
            if (sumw / size_t[t]) < threshold:
                continue

        # normalize weights
        vws = [(v, w / sumw) for v, w in vws]

        # aggregate
        v = vtype.weighted_aggregate(vws)

        #  re-scale intensive => extensive
        if vtype == VarTypeMetricExt:
            v *= size_t[t]

        result[t] = v

    # rounding
    if as_int and issubclass(vtype, VarTypeMetric):
        result = dict((t, round(v)) for t, v in result.items())

    return result


def is_multindex(x: Union[DataFrame, Series, Index, MultiIndex, float]) -> bool:
    if isinstance(x, (int, float)):
        return False
    if isinstance(x, (DataFrame, Series)):
        x = x.index
    return isinstance(x, MultiIndex)


def get_dimension_levels(
    x: Union[DataFrame, Series, Index, MultiIndex, float]
) -> List[Tuple[str, List]]:
    if isinstance(x, (int, float)):
        # scalar: dummy dimension
        return [(None, [None])]

    if isinstance(x, (DataFrame, Series)):
        x = x.index

    # names must be unique
    assert len(x.names) == len(set(x.names))

    if isinstance(x, MultiIndex):
        return list(zip(x.names, x.levels))
    else:
        return [(x.name, x.values)]


def get_idx_out(var: Union[DataFrame, Series, Index, MultiIndex, float], map: Series):
    map_levels = get_dimension_levels(map)

    from_levels = get_dimension_levels(var)
    from_level_names = [x[0] for x in from_levels]

    # determine out from difference (in - map)
    to_levels = [(k, v) for k, v in map_levels if k not in from_level_names]
    to_levels_names = [x[0] for x in to_levels]
    to_levels_values = [x[1] for x in to_levels]

    to_is_multindex = len(to_levels) > 1
    if not to_levels:  # aggregation to scalar
        to_levels = [(None, [None])]

    if to_is_multindex:
        return pd.MultiIndex.from_product(to_levels_values, names=to_levels_names)
    else:
        return pd.Index(to_levels_values[0], name=to_levels_names[0])


def align_map(
    var: Union[DataFrame, Series, Index, MultiIndex, float],
    map: Series,
    out: Optional[Union[DataFrame, Series, Index, MultiIndex, float]],
) -> Mapping[Tuple[F, T], float]:
    map_levels = get_dimension_levels(map)
    map_is_multindex = is_multindex(map)

    from_levels = get_dimension_levels(var)
    from_is_multindex = is_multindex(var)

    to_levels = get_dimension_levels(out)
    to_is_multindex = is_multindex(out)

    all_levels = []
    for n, items in from_levels:
        if n in dict(to_levels):
            continue
        all_levels.append((n, items))
    for n, items in to_levels:
        all_levels.append((n, items))

    # create indices mapping
    all_levels_names = [x[0] for x in all_levels]
    all_levels_values = [x[1] for x in all_levels]
    from_level_idcs = [all_levels_names.index(n) for n, _ in from_levels]
    to_level_idcs = [all_levels_names.index(n) for n, _ in to_levels]
    # this also ensures that map <= (from | to)
    map_level_idcs = [all_levels_names.index(n) for n, _ in map_levels]

    def get_key(row, indices, is_multindex):
        key = tuple(row[i] for i in indices)
        if not is_multindex:
            assert len(indices) == 1
            key = key[0]
        return key

    result = {}
    for row in product(*all_levels_values):
        key_map = get_key(row, map_level_idcs, map_is_multindex)
        val_map = map.get(key_map)
        if is_na(val_map):
            continue

        key_from = get_key(row, from_level_idcs, from_is_multindex)
        key_to = get_key(row, to_level_idcs, to_is_multindex)

        key = (key_from, key_to)
        result[key] = val_map

    result = pd.Series(result)
    result.index.names = ["dimensions_from", "dimensions_to"]

    return result


def apply_map_df(
    vtype,
    s_var,
    s_map,
    i_out=None,
    s_size_f=None,
    s_size_t=None,
    threshold=0,
    as_int=False,
):
    if i_out is None:
        i_out = get_idx_out(s_var, s_map)

    s_map_ft = align_map(s_var, s_map, i_out)

    result = apply_map(
        vtype=vtype,
        var=s_var,
        map=s_map_ft,
        size_f=s_size_f,
        size_t=s_size_t,
        threshold=threshold,
        as_int=as_int,
    )

    result = pd.Series(result)
    result.index.names = i_out.names

    return result


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

    @classmethod
    def apply_map(cls, *args, **kwargs):
        return apply_map(cls, *args, **kwargs)

    @classmethod
    def apply_map_df(cls, *args, **kwargs):
        return apply_map_df(cls, *args, **kwargs)


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
    Values are extensive, i.e. they are can be transformed into intensive
    by dividing by domain size

    * Examples: population, energy production
    """

    pass
