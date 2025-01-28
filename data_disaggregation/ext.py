"""extended functions, especially for pandas Series
"""

from typing import List, Tuple, Union

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series

from .base import transform
from .utils import is_scalar
from .vtypes import SCALAR_DIM_NAME, SCALAR_INDEX_KEY, VariableType

IDX_SCALAR = MultiIndex.from_product([Index([SCALAR_INDEX_KEY], name=SCALAR_DIM_NAME)])
COL_WEIGHT = "__WEIGHT__"
COL_FROM = "__FROM__"
COL_TO = "__TO__"


def harmonize_input_data(data: Union[DataFrame, Series, float]) -> DataFrame:
    """
    return DataFrame with MultiIndex
    """
    if is_scalar(data):
        data = DataFrame({SCALAR_INDEX_KEY: data}, index=IDX_SCALAR)
    elif isinstance(data, Series):
        data = data.to_frame()
    # ensure multiindex
    data = ensure_multiindex(data)
    return data


def as_multiindex(index: Index) -> MultiIndex:
    if not isinstance(index, MultiIndex):
        index = MultiIndex.from_product([index])
    return index


def ensure_multiindex(item: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
    if not isinstance(item.index, MultiIndex):
        index = as_multiindex(item.index)
        item = item.copy()  # TODO: can we replace index without copy?
        item.index = index
    return item


def as_list_of_series_w_multiindex(
    items: Union[Index, Series, Tuple[Union[Index, Series]]]
) -> List[Series]:
    # make sure we have a list/tuple
    if not isinstance(items, (list, tuple)):
        items = [items]
    # make sure we have series:
    items = [it if isinstance(it, Series) else Series(1, index=it) for it in items]
    items = [ensure_multiindex(it) for it in items]
    return items


def merge_indices(items: List[Union[Series, Index]]) -> MultiIndex:
    """Create product of unions of indices"""
    # ensure items are multiindices
    items = [it if isinstance(it, Index) else it.index for it in items]
    items = [as_multiindex(it) for it in items]
    indices = {}
    for it in items:
        for idx in it.levels:
            if idx.name not in indices:
                indices[idx.name] = idx
            else:
                indices[idx.name] = indices[idx.name].union(idx)
    return MultiIndex.from_product(indices.values())


def combine_weights(
    weights: Union[Index, Series, Tuple[Union[Index, Series]]]
) -> Series:
    """multiply all weights series
    * join on overlapping columns (or all if none
    * if index and not series: use value 1

    Returns:
        Series with MultiIndex
    """
    # make sure we have series:
    weights = as_list_of_series_w_multiindex(weights)

    # merge indices
    idx = merge_indices(weights)

    # multiply all and drop nan
    result = Series(1, index=idx)
    for w in weights:
        # IMPORTANT: `result *= w` gives a different result, so DONT use it
        result = result * w

    # multiplications implicit join can change order:
    if isinstance(result.index, MultiIndex):
        result.index = result.index.reorder_levels(idx.names)

    # drop nan
    result.dropna(inplace=True)
    return result


def format_result(df, input_is_df, output_is_scalar, output_multiindex):
    if not output_multiindex:
        assert len(df.index.levels) == 1
        df = df.set_index(df.index.levels[0])
    if input_is_df:
        if output_is_scalar:
            # return series instead of Frame
            return df.iloc[0, :]
        else:
            return df.iloc[:, :]
    else:
        if output_is_scalar:
            # return scalar
            return df.iloc[0, 0]
        else:
            return df.iloc[:, 0]


def get_idx_out(idx_in: MultiIndex, idx_weights: MultiIndex) -> MultiIndex:
    idx_all = merge_indices([idx_in, idx_weights])
    idx_levels = dict(zip(idx_all.names, idx_all.levels))

    idx_names_only_in = set(idx_in.names) - set(idx_weights.names)
    idx_names_only_weights = set(idx_weights.names) - set(idx_in.names)
    # idx_names_both = set(idx_weights.names) & set(idx_in.names)

    # TODO give user options
    # symmetric difference
    idx_names_out = idx_names_only_in | idx_names_only_weights

    idx_out = MultiIndex.from_product([idx_levels[n] for n in idx_names_out])

    return idx_out


def remap_series_to_frame(s: Series, idx: MultiIndex, colname: str) -> DataFrame:
    assert set(s.index.names) <= set(idx.names)
    # create sub-index from idx: only columns that are in series
    # TODO: better way?
    df_result = DataFrame({colname: np.nan}, index=idx)
    # convert index into columns
    df_result.reset_index(inplace=True)
    # set index to same levels as `s`, but keep columns

    df_result.set_index(s.index.names, drop=False, inplace=True)
    df_result = ensure_multiindex(df_result)

    # join in data
    df_result[colname] = s
    df_result.reset_index(inplace=True, drop=True)

    return df_result


def create_weight_map(
    ds_weights: Series, idx_in: MultiIndex, idx_out: MultiIndex
) -> Series:
    """
    Returns weight Series
    Index is 2 dimensional (F, T), each part is a tuple from idx_in, idx_out
    for overlapping levels: left == right

    """
    idx_all = merge_indices([idx_in, idx_out])
    # expand index (TODO: check if weights are dropped??)
    df = remap_series_to_frame(ds_weights, idx_all, COL_WEIGHT)
    df[COL_FROM] = list(zip(*[df[n] for n in idx_in.names]))
    df[COL_TO] = list(zip(*[df[n] for n in idx_out.names]))

    # filter: TODO, faster way?
    df = df.loc[df[COL_FROM].isin(idx_in)]
    df = df.loc[df[COL_TO].isin(idx_out)]

    ds_weight_map = df.set_index([COL_FROM, COL_TO])[COL_WEIGHT]
    ds_weight_map = ds_weight_map.fillna(0)

    return ds_weight_map


def validate_multiindex(item: Union[Index, Series, DataFrame]):
    if isinstance(item, (Series, DataFrame)):
        item = item.index
    assert isinstance(item, MultiIndex)
    assert all(item.names)  # no empty names
    assert len(set(item.names)) == len(item.names)  # unique names
    assert item.is_unique  # unique values


def transform_pandas(
    vtype: VariableType,
    data: Union[DataFrame, Series, float],
    weights: Union[Index, Series, Tuple[Union[Index, Series]]],
    dim_in: Union[Index, Series] = None,
    dim_out: Union[Index, Series] = None,
    validate: bool = True,
) -> Union[DataFrame, Series, float]:
    """(dis-)aggregate data (pandas).

    Parameters
    ----------
    vtype : VariableType
        Variable type of input data, determines the aggregation method.
    data : Union[DataFrame, Series, float]
        Input data. Indices must be unique and have unique level names
    weights : Union[Index, Series, Tuple[Union[Index, Series]]]
        weights for combinations of input and output elements (must be positive).
    dim_in : Union[Index, Series], optional
        TODO
    dim_out : Union[Index, Series], optional
        TODO
    validate : bool, optional
        if True: run additional (but costly) validations of weights and data.

    Returns
    -------
    Union[DataFrame, Series, float]
        output data

    """

    # ensure data is DataFrame with MultiIndex
    df_data = harmonize_input_data(data)

    if validate:
        validate_multiindex(df_data)

    # combine weights into single Series with MultiIndex
    ds_weights = combine_weights(weights)

    if validate:
        validate_multiindex(ds_weights)

    # determine input index
    if dim_in is None:
        idx_in = df_data.index
        ds_size_in = None
    elif isinstance(dim_in, Index):
        idx_in = as_multiindex(dim_in)
        ds_size_in = None
    else:
        ds_size_in = ensure_multiindex(dim_in)
        idx_in = ds_size_in.index

    # determine output_index
    if dim_out is None:
        idx_out = get_idx_out(idx_in, ds_weights.index)
        ds_size_out = None
    elif isinstance(dim_out, Index):
        idx_out = as_multiindex(dim_out)
        ds_size_out = None
    else:
        ds_size_out = ensure_multiindex(dim_out)
        idx_out = ds_size_out.index

    if validate:
        validate_multiindex(idx_in)
        validate_multiindex(idx_out)

    # create weight map
    ds_weight_map = create_weight_map(ds_weights, idx_in, idx_out)

    if ds_size_in is None:
        ds_size_in = (
            ds_weight_map.reset_index().groupby(COL_FROM).sum(COL_WEIGHT)[COL_WEIGHT]
        )
        # fix index
        ds_size_in.index = MultiIndex.from_tuples(
            ds_size_in.index.values, names=idx_in.names
        )
        ds_size_in = ds_size_in.loc[ds_size_in > 0]

    if ds_size_out is None:
        ds_size_out = (
            ds_weight_map.reset_index().groupby(COL_TO).sum(COL_WEIGHT)[COL_WEIGHT]
        )
        # fix index
        ds_size_out.index = MultiIndex.from_tuples(
            ds_size_out.index.values, names=idx_out.names
        )
        ds_size_out = ds_size_out.loc[ds_size_out > 0]

    if validate:
        validate_multiindex(ds_size_in)
        validate_multiindex(ds_size_out)
        assert ds_size_in.index.isin(idx_in).all()
        assert ds_size_out.index.isin(idx_out).all()

    idx_in = ds_size_in.index
    idx_out = ds_size_out.index

    # apply base function
    df_result = DataFrame(index=idx_out)
    for name in df_data.columns:
        s_col = df_data[name]
        s_col = s_col.dropna()

        res_col = transform(
            vtype=vtype,
            data=s_col,
            weight_map=ds_weight_map,
            weights_from=ds_size_in,
            weights_to=ds_size_out,
            weight_rel_threshold=0.0,
            validate=False,  # we do validation in pandas
        )

        s_res_col = Series(res_col, name=s_col.name)
        df_result[s_col.name] = s_res_col

    return format_result(
        df_result,
        input_is_df=isinstance(data, DataFrame),
        output_is_scalar=idx_out.equals(IDX_SCALAR),
        output_multiindex=(
            True if dim_out is None else isinstance(idx_out, MultiIndex)
        ),
    )
