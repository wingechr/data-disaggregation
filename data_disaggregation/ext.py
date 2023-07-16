"""extended functions, especially for pandas Series
"""
from itertools import product
from typing import List, Mapping, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series

from .base import transform
from .classes import SCALAR_DIM_NAME, SCALAR_INDEX_KEY, VT, F, T
from .utils import is_na, is_scalar

SCALAR_INDEX = Index([SCALAR_INDEX_KEY], name=SCALAR_DIM_NAME)


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
        x = SCALAR_INDEX
    elif isinstance(x, (DataFrame, Series)):
        x = x.index

    # names must be unique
    assert len(x.names) == len(set(x.names))

    if isinstance(x, MultiIndex):
        return list(zip(x.names, x.levels))
    else:
        return [(x.name, x.values)]


def get_idx_out(i_weights: MultiIndex, i_in: MultiIndex, i_out: MultiIndex = None):
    """ """
    map_levels = get_dimension_levels(weights)

    from_levels = get_dimension_levels(i_in)
    from_level_names = [x[0] for x in from_levels]

    # determine out from difference (in - map)
    to_levels = [(k, v) for k, v in map_levels if k not in from_level_names]

    if not to_levels:  # aggregation to scalar
        to_levels = get_dimension_levels(SCALAR_INDEX)

    to_levels_names = [x[0] for x in to_levels]
    to_levels_values = [x[1] for x in to_levels]
    to_is_multindex = len(to_levels) > 1

    if to_is_multindex:
        return MultiIndex.from_product(to_levels_values, names=to_levels_names)
    else:
        return Index(to_levels_values[0], name=to_levels_names[0])


def as_index(x) -> Index:
    if isinstance(x, Index):
        return x
    elif isinstance(x, Series):
        return x.index

    raise TypeError(f"Must be of type Index instead of {type(x).__name__}")


def as_series(x, default_value=1.0) -> Series:
    if isinstance(x, Series):
        return x
    elif isinstance(x, Index):
        return Series(default_value, index=x)

    raise TypeError("Must be of type Series")


def ensure_multiindex(s: Series) -> Series:
    if not isinstance(s.index, MultiIndex):
        # replace index
        s = Series(s.values, MultiIndex.from_product([s.index]))
    return s


def ensure_scalar_dim(s: Series) -> Series:
    if not SCALAR_DIM_NAME in s.index.names:
        s = pd.concat([s], keys=[SCALAR_INDEX_KEY], names=[SCALAR_DIM_NAME])
    return s


def combine_weights(
    weights: Union[Index, Series, Tuple[Union[Index, Series]]]
) -> Series:
    """combine series by multiplying

    Args:
        weights (Union[Index, Series, Tuple[Union[Index, Series]]]): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        Series: _description_
    """
    if not isinstance(weights, (tuple, list)):
        weights = [weights]

    weights = [as_series(w) for w in weights]

    # all series must have MultiIndex ?
    weights = [ensure_multiindex(w) for w in weights]

    if len(weights) == 1:
        return weights[0]

    # all series must have scalar dimension (for joining)
    has_scalar_dim = any(SCALAR_DIM_NAME in w.index.names for w in weights)
    weights = [ensure_scalar_dim(w) for w in weights]

    # now multiply all

    result = weights[0]
    for w in weights[1:]:
        result = result * w

    # drop scalar dimension
    if not has_scalar_dim:
        result = Series(result.values, result.index.droplevel(SCALAR_DIM_NAME))

    return result


def create_weight_map(
    weights: Union[Index, Series, Tuple[Union[Index, Series]]],
    idx_in: Index,
    idx_out: Index = None,
) -> Mapping[Tuple[F, T], float]:
    weights = combine_weights(weights)
    idx_in = as_index(idx_in)

    idx_out = get_idx_out(weights.index, idx_in, idx_out)

    map_levels = get_dimension_levels(weights)
    map_is_multindex = is_multindex(weights)

    from_levels = get_dimension_levels(idx_in)
    from_levels_names = [x[0] for x in from_levels]
    from_is_multindex = is_multindex(idx_in)

    to_levels = get_dimension_levels(idx_out)
    to_levels_names = [x[0] for x in to_levels]
    to_is_multindex = is_multindex(idx_out)

    # from_levels AND to_levels (unique)
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
    try:
        map_level_idcs = [all_levels_names.index(n) for n, _ in map_levels]
    except Exception as e:
        raise ValueError(f"weight map cannot be created: {e}")

    def get_key(row, indices, is_multindex):
        key = tuple(row[i] for i in indices)
        if not is_multindex:
            assert len(indices) == 1
            key = key[0]
        return key

    result = {}

    # create cross product of all levels
    for row in product(*all_levels_values):
        key_map = get_key(row, map_level_idcs, map_is_multindex)
        val_map = weights.get(key_map)
        if is_na(val_map):
            continue

        # get in/out keys
        key_in = get_key(row, from_level_idcs, from_is_multindex)
        key_out = get_key(row, to_level_idcs, to_is_multindex)

        if key_in not in idx_in:
            continue
        if key_out not in idx_out:
            continue

        key = (key_in, key_out)
        result[key] = val_map

    from_name = (
        tuple(from_levels_names) if len(from_levels_names) > 1 else from_levels_names[0]
    )
    to_name = tuple(to_levels_names) if len(to_levels_names) > 1 else to_levels_names[0]

    result = Series(result).rename_axis([from_name, to_name])

    return result


def transform_pandas(
    vtype: VT,
    data: Union[Series, float],
    weights: Union[Index, Series, Tuple[Union[Index, Series]]],
    dim_in: Union[Index, Series] = None,
    dim_out: Union[Index, Series] = None,
    threshold: float = 0.0,
    validate: bool = True,
) -> Union[Series, float]:
    # TODO: apply same transformation all Series in DataFrame
    # to check: how are nan values handled if they are
    # different in the columns?

    if is_scalar(data):
        data = Series(data, index=SCALAR_INDEX)

    if isinstance(dim_in, Index):
        idx_in = dim_in
        size_in = None
    elif isinstance(dim_in, Series):
        idx_in = dim_in
        size_in = dim_in
    elif dim_in is None:
        idx_in = data.index
        size_in = None
    else:
        raise NotImplementedError()

    if isinstance(dim_out, Index):
        idx_out = dim_out
        size_out = None
    elif isinstance(dim_out, Series):
        idx_out = dim_out
        size_out = dim_out
    elif dim_out is None:
        idx_out = None
        size_out = None
    else:
        raise NotImplementedError()

    # ensure indices have no nans
    # assert #hasnans

    weight_map = create_weight_map(weights=weights, idx_in=idx_in, idx_out=idx_out)

    result = transform(
        vtype=vtype,
        data=data,
        weight_map=weight_map,
        size_in=size_in,
        size_out=size_out,
        threshold=threshold,
        validate=validate,
    )

    index_names = weight_map.index.names[1]

    # scalar
    if set(result.keys()) == set([SCALAR_INDEX_KEY]):
        result = result[SCALAR_INDEX_KEY]
    else:
        result = Series(result).rename_axis(index_names)

    return result
