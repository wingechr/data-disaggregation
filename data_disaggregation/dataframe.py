from itertools import product
from typing import List, Mapping, Optional, Tuple, TypeVar, Union

import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series

from .base import apply_map
from .utils import is_na

F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")


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
