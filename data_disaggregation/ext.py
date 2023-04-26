from itertools import product
from typing import List, Mapping, Optional, Tuple, Union

import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series

from .base import apply_map
from .classes import VT, F, T, V
from .utils import (
    as_list,
    as_mapping,
    group_idx_first,
    group_idx_second,
    is_list,
    is_map,
    is_na,
    is_subset,
    is_unique,
    iter_values,
)


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


def get_idx_out(
    weights: Series, i_out: Union[DataFrame, Series, Index, MultiIndex, float]
):
    map_levels = get_dimension_levels(weights)

    from_levels = get_dimension_levels(i_out)
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


def create_map(
    weights: Series,
    i_in: Union[DataFrame, Series, Index, MultiIndex, float],
    i_out: Optional[Union[DataFrame, Series, Index, MultiIndex, float]] = None,
) -> Mapping[Tuple[F, T], float]:
    if is_list(weights):
        weights = pd.Series(1, index=weights)

    if i_out is None:
        i_out = get_idx_out(weights, i_in)

    map_levels = get_dimension_levels(weights)
    map_is_multindex = is_multindex(weights)

    from_levels = get_dimension_levels(i_in)
    from_levels_names = [x[0] for x in from_levels]
    from_is_multindex = is_multindex(i_in)

    to_levels = get_dimension_levels(i_out)
    to_levels_names = [x[0] for x in to_levels]
    to_is_multindex = is_multindex(i_out)

    all_levels = []
    for name, items in from_levels:
        if name in dict(to_levels):
            continue
        all_levels.append((name, items))
    for name, items in to_levels:
        all_levels.append((name, items))

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
        val_map = weights.get(key_map)
        if is_na(val_map):
            continue

        key_from = get_key(row, from_level_idcs, from_is_multindex)
        key_to = get_key(row, to_level_idcs, to_is_multindex)

        key = (key_from, key_to)
        result[key] = val_map

        from_name = (
            tuple(from_levels_names)
            if len(from_levels_names) > 1
            else from_levels_names[0]
        )
        to_name = (
            tuple(to_levels_names) if len(to_levels_names) > 1 else to_levels_names[0]
        )

    result = pd.Series(result).rename_axis([from_name, to_name])

    return result


def disagg(
    vtype: VT,
    var: Mapping[F, V],
    map: Union[Mapping[Tuple[F, T], float], Series],
    size_f: Mapping[F, float] = None,
    size_t: Mapping[T, float] = None,
    threshold: float = 0.0,
    as_int: bool = False,
) -> Mapping[T, V]:
    res_series_dtype = None
    res_series_names = None

    var = as_mapping(var)

    if not is_map(map):
        idx_f = as_list(size_f) if size_f is not None else as_list(var)
        idx_t = as_list(size_t) if size_t is not None else None
        map = create_map(map, idx_f, idx_t)

    # validate size_f
    if size_f is None:
        size_f = group_idx_first(map)
    size_f = as_mapping(size_f)
    assert is_unique(size_f)
    assert all(v > 0 for v in iter_values(size_f))

    # validate size_t
    if size_t is None:
        size_t = group_idx_second(map)
    size_t = as_mapping(size_t)
    assert is_unique(size_t)
    assert all(v > 0 for v in iter_values(size_t))

    # validate var
    assert is_unique(var)
    assert is_subset(var, size_f)

    # validate map
    assert is_unique(map)
    assert all(v >= 0 for v in iter_values(map))
    assert is_subset([x[0] for x in map.keys()], size_f)
    assert is_subset([x[1] for x in map.keys()], size_t)

    result = apply_map(
        vtype=vtype,
        var=var,
        map=map,
        size_f=size_f,
        size_t=size_t,
        threshold=threshold,
        as_int=as_int,
    )

    # todo remove checks at the end
    assert is_subset(result, size_t)
    assert is_unique(result)

    # result as series
    if res_series_names:
        result = pd.Series(result, dtype=res_series_dtype).rename_axis(res_series_names)

    # result as scalar
    if set(result.keys()) == set([None]):
        result = result[None]

    return result
