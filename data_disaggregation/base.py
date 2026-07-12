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

from __future__ import annotations  # Series[...] for older python/pandas

from typing import Any

import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series

from .utils import SeriesDict, SeriesFrame, as_series
from .vtypes import VariableType

_NA_DIM_KEY = "__NA__"


def get_NA_DIM_KEY(for_index: Index) -> Any:
    sample = for_index[0]
    if isinstance(sample, tuple):
        return tuple([_NA_DIM_KEY] * len(sample))
    return _NA_DIM_KEY


def _assert_index_unique_no_na(index: Index, name: str):
    if not index.is_unique:
        raise Exception(f"index not unique in {name}: {set(index.duplicated())}")
    for level in range(index.nlevels):
        if any(index.get_level_values(level).isna()):
            raise Exception(f"index contains NA in {name}")
    if get_NA_DIM_KEY(index) in index:
        raise Exception(f"index contains {get_NA_DIM_KEY(index)} in {name}")


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


def _check_initialize_weights(
    df_weight_map: DataFrame,
    ds_weights_from: Series[float],
    ds_weights_to: Series[float],
) -> tuple[DataFrame, Series, Series]:
    # convert weights to frame, rows = from, cols = to, realign
    df_weight_map = df_weight_map.reindex(
        index=ds_weights_from.index, columns=ds_weights_to.index
    ).fillna(0)

    # all indices must be unique and not NA
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

    # differences of sum weights and weigh sums
    ds_weights_from_rest = ds_weights_from - df_weight_map.sum(axis=1)
    _assert_all_ge0(ds_weights_from_rest, "col weights for NA")
    ds_weights_to_rest = ds_weights_to - df_weight_map.sum(axis=0)
    _assert_all_ge0(ds_weights_to_rest, "row weights for NA")

    # add additional row for NA in source
    df_weight_map = pd.concat(
        [
            df_weight_map,
            ds_weights_to_rest.to_frame().T.set_axis(
                [get_NA_DIM_KEY(df_weight_map.index)]
            ),
        ]
    )
    ds_weights_from_rest = pd.concat(
        [
            ds_weights_from_rest,
            Series(0, index=[get_NA_DIM_KEY(ds_weights_from_rest.index)]),
        ]
    )

    # add NA output col (temp)
    df_weight_map_w_na_column = pd.concat(
        [
            df_weight_map,
            ds_weights_from_rest.rename(
                get_NA_DIM_KEY(df_weight_map.columns)  # type: ignore (yes, NA should be col identifier)
            ),
        ],
        axis=1,
    )

    # df_weight_map.columns = ds_weights_to.index
    df_weight_map.columns.names = ds_weights_to.index.names

    ds_weights_from = df_weight_map_w_na_column.sum(axis=1)
    ds_weights_to = df_weight_map.sum(axis=0)

    return df_weight_map, ds_weights_from, ds_weights_to


class Transformer:
    def __init__(
        self,
        vtype: type[VariableType],
        df_weight_map: DataFrame,
        ds_weights_from: Series | None = None,
        ds_weights_to: Series | None = None,
        weight_rel_threshold: float = 0.0,
    ):
        ds_weights_from = (
            df_weight_map.sum(axis=1) if ds_weights_from is None else ds_weights_from
        )
        ds_weights_to = (
            df_weight_map.sum(axis=0) if ds_weights_to is None else ds_weights_to
        )
        df_weight_map, ds_weights_from, ds_weights_to = _check_initialize_weights(
            df_weight_map, ds_weights_from, ds_weights_to
        )

        self.df_weight_map: DataFrame = df_weight_map
        self.ds_weights_from: Series = ds_weights_from
        self.ds_weights_to: Series = ds_weights_to
        self.vtype: type[VariableType] = vtype
        self.weight_rel_threshold: float = weight_rel_threshold

    def __call__(self, ds_data: Series) -> Series:
        _assert_index_unique_no_na(ds_data.index, str(ds_data.name))
        return (
            self.vtype.transform(
                ds_data=ds_data,
                df_weight_map=self.df_weight_map,
                ds_weights_from=self.ds_weights_from,
                # ds_weights_to=self.ds_weights_to,
                weight_rel_threshold=self.weight_rel_threshold,
            )
            .rename(ds_data.name)  # type:ignore
            .rename_axis(index=self.df_weight_map.columns.names)
        )  # type:ignore


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
    df_weight_map = ds_weight_map.unstack(level=1)
    # fix problem with multiindex
    if isinstance(ds_data.index, MultiIndex) and not isinstance(
        df_weight_map.index, MultiIndex
    ):
        df_weight_map.index = pd.MultiIndex.from_tuples(df_weight_map.index)

    ds_weights_from = None if weights_from is None else as_series(weights_from)
    ds_weights_to = None if weights_to is None else as_series(weights_to)

    transformer = Transformer(
        vtype=vtype,
        df_weight_map=df_weight_map,
        ds_weights_from=ds_weights_from,
        ds_weights_to=ds_weights_to,
        weight_rel_threshold=weight_rel_threshold,
    )

    ds_result = transformer(ds_data)

    result = type(data)(ds_result)

    return result
