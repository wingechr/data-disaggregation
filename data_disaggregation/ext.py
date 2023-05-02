"""extended functions, especially for pandas Series
"""
from itertools import product
from typing import List, Mapping, Tuple, Union

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


def get_idx_out(
    weights: Series, i_out: Union[DataFrame, Series, Index, MultiIndex, float]
):
    """ """
    map_levels = get_dimension_levels(weights)

    from_levels = get_dimension_levels(i_out)
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


def create_weight_map(
    weights: Series,
    idx_in: Index,
    idx_out: Index = None,
) -> Mapping[Tuple[F, T], float]:
    if idx_out is None:
        idx_out = get_idx_out(weights, idx_in)

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

        key = (key_in, key_out)
        result[key] = val_map

    from_name = (
        tuple(from_levels_names) if len(from_levels_names) > 1 else from_levels_names[0]
    )
    to_name = tuple(to_levels_names) if len(to_levels_names) > 1 else to_levels_names[0]

    result = Series(result).rename_axis([from_name, to_name])

    return result


def transform_ds(
    vtype: VT,
    data: Series,
    weights: Series,
    dim_in: Union[Index, Series] = None,
    dim_out: Union[Index, Series] = None,
    threshold: float = 0.0,
    validate=True,
) -> Series:
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
