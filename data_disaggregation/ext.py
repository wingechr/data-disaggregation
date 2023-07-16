"""extended functions, especially for pandas Series
"""
from typing import List, Tuple, Union

from pandas import DataFrame, Index, MultiIndex, Series

from .base import transform
from .classes import SCALAR_DIM_NAME, SCALAR_INDEX_KEY, VT
from .utils import is_scalar

IDX_SCALAR = MultiIndex.from_product([Index([SCALAR_INDEX_KEY], name=SCALAR_DIM_NAME)])


def harmonize_input_data(data: Union[DataFrame, Series, float]) -> DataFrame:
    """
    return DataFrame with MultiIndex
    """
    if is_scalar(data):
        return DataFrame({SCALAR_INDEX_KEY: data}, index=IDX_SCALAR)
    if isinstance(data, Series):
        data = data.to_frame()
    # ensure multiindex
    data = ensure_multiindex(ensure_multiindex)
    return data


def as_multiindex(item: Index) -> MultiIndex:
    if not isinstance(item, MultiIndex):
        item = MultiIndex.from_product([item])
    return item


def ensure_multiindex(item: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
    if not isinstance(item.index, MultiIndex):
        item = item.set_index(as_multiindex(item.index))
    return item


def as_list_of_series(
    items: Union[Index, Series, Tuple[Union[Index, Series]]]
) -> List[Series]:
    # make sure we have a list/tuple
    if not isinstance(items, (list, tuple)):
        items = [items]
    # make sure we have series:
    items = [it if isinstance(it, Series) else Series(1, index=it) for it in items]
    return items


def merge_indices(items: List[Union[Series, Index]]) -> MultiIndex:
    # ensure items are indices
    items = [it if isinstance(it, Index) else it.index for it in items]
    indices = {}
    for it in items:
        for idx in it.index.levels:
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
    weights = as_list_of_series(weights)
    # merge indices
    idx = merge_indices(weights)
    # multiply all and drop nan
    result = Series(1, index=idx)
    for w in weights:
        result *= w
    # drop nan
    result = result.dropna()
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
            return df.iloca[:, :]
    else:
        if output_is_scalar:
            # return scalar
            return df.iloc[0, 0]
        else:
            return df.iloc[:, 0]


def get_idx_out(idx_in: MultiIndex, idx_weights: MultiIndex) -> MultiIndex:
    idx_all = merge_indices([idx_in, idx_weights])

    idx_names_only_in = set(idx_in.names) - set(idx_weights.names)
    idx_names_only_weights = set(idx_weights.names) - set(idx_in.names)
    # idx_names_both = set(idx_weights.names) & set(idx_in.names)

    # TODO give user options
    # symmetric difference
    idx_names_out = idx_names_only_in | idx_names_only_weights
    idx_out = MultiIndex.from_product([idx_all[n] for n in idx_names_out])

    return idx_out


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
    COL_VAL = "__VALUE__"
    COL_FROM = "__FROM__"
    COL_TO = "__TO__"
    df = ds_weights.reindex(idx_all).to_frame(COL_VAL)
    df[COL_FROM] = list(zip([df[n] for n in idx_in.names]))
    df[COL_TO] = list(zip([df[n] for n in idx_out.names]))
    ds_weight_map = df.set_index([COL_FROM, COL_TO])[COL_VAL]
    ds_weight_map = ds_weight_map.fillna(0)

    return ds_weight_map


def transform_pandas(
    vtype: VT,
    data: Union[DataFrame, Series, float],
    weights: Union[Index, Series, Tuple[Union[Index, Series]]],
    dim_in: Union[Index, Series] = None,
    dim_out: Union[Index, Series] = None,
    threshold: float = 0.0,
    validate: bool = True,
) -> Union[DataFrame, Series, float]:
    # ensure data is DataFrame with MultiIndex
    df_data = harmonize_input_data(data)

    # combine weights into single Series with MultiIndex
    ds_weights = combine_weights(weights)

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

    # create weight map
    ds_weight_map = create_weight_map(ds_weights, idx_in, idx_out)

    # apply base function
    df_result = DataFrame(index=ds_size_out.index)
    for col in df_data:
        df_result[col.name] = transform(
            vtype=vtype,
            data=col,
            weight_map=ds_weight_map,
            size_in=ds_size_in,
            size_out=ds_size_out,
            threshold=threshold,
            validate=validate,
        )

    return format_result(
        df_result,
        input_is_df=isinstance(data, DataFrame),
        output_is_scalar=ds_size_out.index.equals(IDX_SCALAR),
        output_multiindex=(
            True if dim_out is None else isinstance(dim_out.index, MultiIndex)
        ),
    )
