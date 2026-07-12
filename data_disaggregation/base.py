"""base code without dependencies.

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

from typing import Any

import pandas as pd
from pandas import NA, DataFrame, Index, Series

from .utils import SeriesDict, SeriesFrame, as_series
from .vtypes import VariableType, VT_NumericExt

NA_DIM_KEY = "__NA__"


def _assert_index_unique_no_na(index: Index, name: str):
    if not index.is_unique:
        raise Exception(f"index not unique in {name}: {set(index.duplicated())}")
    if any(index.isna()):
        raise Exception(f"index contains NA in {name}")
    if NA_DIM_KEY in index:
        raise Exception(f"index contains {NA_DIM_KEY} in {name}")


def _assert_all_gt0(data: SeriesFrame, name: str):
    if not all(data > 0):
        raise Exception(f"Not all values are > 0 in {name}")


def _assert_all_ge0(data: SeriesFrame, name: str):
    if not all(data >= 0):
        raise Exception(f"Not all values are >= 0 in {name}")


def _assert_index_subset(index: Index, ref_index: Index, name: str):
    err = set(index) - set(ref_index)
    if err:
        raise Exception(f"Unexepcted, additional elements in {name}: {err}")


def _create_full_weightmap(
    ds_data: Series[Any],
    ds_weight_map: Series[float],
    ds_weights_from: Series[float],
    ds_weights_to: Series[float],
) -> DataFrame:
    # convert weights to frame, rows = from, cols = to, realign
    df_weight_map = (
        ds_weight_map.unstack(level=1)
        .reindex(index=ds_weights_from.index, columns=ds_weights_to.index)
        .fillna(0)
    )

    # enforce single index
    df_weight_map.index = df_weight_map.index.get_level_values(0)
    df_weight_map.columns = df_weight_map.columns.get_level_values(0)
    ds_data.index = ds_data.index.get_level_values(0)
    ds_weights_from.index = ds_weights_from.index.get_level_values(0)
    ds_weights_to.index = ds_weights_to.index.get_level_values(0)

    # all indices must be unique and not NA
    _assert_index_unique_no_na(ds_data.index, "data")
    _assert_index_unique_no_na(df_weight_map.index, "weights (source)")
    _assert_index_unique_no_na(df_weight_map.columns, "weights (target)")
    _assert_index_unique_no_na(ds_weights_from.index, "weights (sum source)")
    _assert_index_unique_no_na(ds_weights_to.index, "weights (sum target)")
    _assert_all_gt0(ds_weights_from, "weights (sum source)")
    _assert_all_gt0(ds_weights_to, "weights (sum target)")
    _assert_all_ge0(df_weight_map, "weights")
    # weights have to be in ds_weights_from / ds_weights_to sums
    _assert_index_subset(df_weight_map.index, ds_weights_from.index, "weights (source)")
    _assert_index_subset(df_weight_map.columns, ds_weights_to.index, "weights (target)")

    # additional indices with data but no weights
    idx_data_no_mapping = list(set(ds_data.index) - set(ds_weights_from.index))

    # differences of sum weights and weigh sums
    ds_weights_from_rest = ds_weights_from - df_weight_map.sum(axis=1)
    _assert_all_ge0(ds_weights_from_rest, "col weights for NA")
    ds_weights_to_rest = ds_weights_to - df_weight_map.sum(axis=0)
    _assert_all_ge0(ds_weights_to_rest, "row weights for NA")

    if idx_data_no_mapping:
        # add rows
        df_weight_map = pd.concat(
            [
                df_weight_map,
                pd.DataFrame(
                    0, index=idx_data_no_mapping, columns=df_weight_map.columns
                ),
            ]
        )
        ds_1 = Series(1, index=idx_data_no_mapping)
        ds_weights_from_rest = pd.concat([ds_weights_from_rest, ds_1])
        ds_weights_from = pd.concat([ds_weights_from, ds_1])

    weights_to_rest_sum = ds_weights_to_rest.sum()
    if weights_to_rest_sum:
        # add additional row for NA in source
        df_weight_map = pd.concat(
            [df_weight_map, ds_weights_to_rest.to_frame().T.set_axis([NA_DIM_KEY])]
        )
        ds_weights_from_rest = pd.concat(
            [ds_weights_from_rest, Series(0, index=[NA_DIM_KEY])]
        )
        ds_weights_from = pd.concat([ds_weights_from, ds_weights_to_rest])

    # if required: add NA output col
    ds_weights_from_rest_sum = ds_weights_from_rest.sum()
    if ds_weights_from_rest_sum:
        df_weight_map = pd.concat(
            [df_weight_map, ds_weights_from_rest.rename(NA_DIM_KEY)], axis=1
        )

    return df_weight_map


def transform(
    vtype: type[VariableType],
    data: SeriesDict,
    weight_map: SeriesDict,
    weights_from: SeriesDict | None = None,
    weights_to: SeriesDict | None = None,
    weight_rel_threshold: float = 0.0,
) -> SeriesDict:
    """(dis-)aggregate data.

    Parameters
    ----------
    vtype: VariableType
        Variable type of input data, determines the aggregation method.
    data: Mapping[F, V]
        Input data: mapping (usually dict) from any keys (any hashable) to values.
    weight_map: Mapping[Tuple[F, T], float]
        weights for combinations of input and output elements (must be positive).
        Keys must tuples from input/output key pairs.
    weights_from: Mapping[F, float]
        optional weights of input elements (must be positive).
        If not specified, this will be calculated as a sum from `weight_map`.
    weights_to: Mapping[T, float]
        optional weights of output elements (must be positive).
        If not specified, this will be calculated as a sum from `weight_map`.
    weight_rel_threshold: float
        optional value between 0 and 1: all mappings are dropped
        if the sum of input weights / output weight is smaller than this threshold.
        For example, you may want to set it to 0.5 for geographical mappings with
        extensive data.

    Returns
    -------
    : Mapping[T, V]
        output data as a mapping from output keys (any hashable) to values.

    """

    ds_data = as_series(data)
    # harmonize NaN/None/NA and drop from data
    ds_data = ds_data.convert_dtypes().dropna()  # normalize and drop na

    ds_weight_map = as_series(weight_map)

    if weights_from is None:
        # group by index level 0
        ds_weights_from = ds_weight_map.groupby(level=0).sum()
    else:
        ds_weights_from = as_series(weights_from)

    if weights_to is None:
        # group by index level 1
        ds_weights_to = ds_weight_map.groupby(level=1).sum()
    else:
        ds_weights_to = as_series(weights_to)

    df_weight_map = _create_full_weightmap(
        ds_data, ds_weight_map, ds_weights_from, ds_weights_to
    )

    ds_result = _transform(
        vtype,
        ds_data,
        df_weight_map,
        weight_rel_threshold,
    )

    # FIXME: what todo with NA_DIM value - return separately?
    # currently, the output might have a an additional dimension than user expects
    # if NA_DIM_KEY in df_weight_map.columns:
    #    value_na_dim = ds_result.pop(NA_DIM_KEY)
    # assert tuple(ds_result.index) == tuple(ds_weights_to.index)

    return type(data)(ds_result)


def _transform(
    vtype: type[VariableType],
    ds_data: Series[Any],  # incl. NA
    df_weight_map: DataFrame,  # >= 0
    weight_rel_threshold: float = 0.0,
) -> Series:
    ds_data = ds_data.reindex(df_weight_map.index)

    #  scale extensive => intensive
    if vtype == VT_NumericExt:
        ds_weights_from = df_weight_map.sum(axis=1)
        ds_data = ds_data / ds_weights_from

    def agg(weights: Series):
        weights = weights.loc[weights > 0]
        weights_sum = weights.sum()
        if not weights_sum:
            # no values
            return NA
        ds_weights_rel = weights / weights_sum
        ds_values_incl_na = ds_data.loc[weights.index]

        idx_val_na = ds_values_incl_na.isna()

        sum_weights_rel_na = ds_weights_rel.loc[idx_val_na].sum()

        if sum_weights_rel_na > weight_rel_threshold or idx_val_na.all():
            return NA

        ds_values = ds_values_incl_na.loc[~idx_val_na]
        ds_weights_rel = ds_weights_rel.loc[~idx_val_na]

        return vtype.weighted_aggregate_ds(ds_values, ds_weights_rel)

    result = df_weight_map.apply(agg)

    #  re-scale intensive => extensive
    if vtype == VT_NumericExt:
        ds_weights_to = df_weight_map.sum(axis=0)
        result = result * ds_weights_to

    return result
