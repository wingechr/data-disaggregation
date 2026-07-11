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

from collections.abc import Mapping

from pandas import Series

from .utils import (
    as_set,
    get_keys,
    get_values,
    is_map,
    is_mapping,
    is_subset,
    is_unique,
)
from .vtypes import F, T, V, VariableType, VT_NumericExt

VALIDATE_EQ_REL_TOLERANCE = 1e-10


def _validate(data, weight_map, weights_from, weights_to):
    # validate size_f
    assert is_mapping(weights_from)
    assert is_unique(weights_from)
    assert all(v > 0 for v in get_values(weights_from))

    # validate size_t
    assert is_mapping(weights_to)
    assert is_unique(weights_to)
    assert all(v > 0 for v in get_values(weights_to))

    # validate var
    assert is_mapping(data)
    assert is_unique(data)

    if not is_subset(data, weights_from):
        err = as_set(data) - as_set(weights_from)
        raise Exception(
            f"Variable index is not a subset of input dimension subset: {err}"
        )

    # validate map
    assert is_map(weight_map)
    assert is_unique(weight_map)
    assert all(v >= 0 for v in get_values(weight_map))
    assert is_subset([x[0] for x in get_keys(weight_map)], weights_from)
    assert is_subset([x[1] for x in get_keys(weight_map)], weights_to)
    # assert all(isinstance(v, (float, int)) for v in iter_values(weight_map))


def transform(
    vtype: type[VariableType],
    data: Mapping[F, V],
    weight_map: Mapping[tuple[F, T], float],
    weights_from: Mapping[F, float] | None = None,
    weights_to: Mapping[T, float] | None = None,
    weight_rel_threshold: float = 0.0,
    validate: bool = True,
) -> Mapping[T, V]:
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
    validate: bool
        if True: run additional (but costly) validations of weights and data.

    Returns
    -------
    : Mapping[T, V]
        output data as a mapping from output keys (any hashable) to values.

    """
    data = Series(data)
    weight_map = Series(weight_map)
    weights_from = Series(weights_from) if weights_from is not None else None
    weights_to = Series(weights_to) if weights_to is not None else None

    return _transform(
        vtype,
        data,
        weight_map,
        weights_from,
        weights_to,
        weight_rel_threshold,
        validate,
    )


def _transform(
    vtype: type[VariableType],
    data: Series,
    weight_map: Series,  # must have multiindex
    weights_from: Series | None = None,
    weights_to: Series | None = None,
    weight_rel_threshold: float = 0.0,
    validate: bool = False,
) -> Mapping[T, V]:
    if weights_from is None:
        # group by index level 0
        weights_from = weight_map.groupby(level=0).sum()

    if weights_to is None:
        # group by index level 1
        weights_to = weight_map.groupby(level=1).sum()

    if validate:
        _validate(data, weight_map, weights_from, weights_to)

    # filter nan in data
    data = data.dropna()

    #  scale extensive => intensive
    if vtype == VT_NumericExt:
        data = data / weights_from.loc[data.index]

    # filter unused in weight_map: input:
    weight_map = weight_map[weight_map.index.get_level_values(0).isin(data.index)]
    # weight_map = {
    #    (f, t): w for (f, t), w in weight_map.items() if weights_to.get(t, 0) > 0
    # }
    weight_map = weight_map.loc[
        weights_to.reindex(weight_map.index.get_level_values(1))
        .set_axis(weight_map.index)
        .fillna(0)
        > 0
    ]

    # filter unused in weight_map: output

    # init groups
    # group data by output keys

    # _groups = defaultdict(list)
    # for (f, t), w in weight_map.items():
    #    _groups[t].append((data[f], w))

    # group to second level
    df_weight_map = weight_map.unstack(level=0)

    # create weight sums
    # group_sumw = {t: sum(w for _, w in vws) for t, vws in groups.items()}
    # group_sumw = weight_map.groupby(level=1).sum()
    group_sumw = df_weight_map.sum(axis=1)

    # drop groups under threshold
    if weight_rel_threshold:
        # sumw_rel = {t: sumw / weights_to[t] for t, sumw in group_sumw.items()}
        sumw_rel = group_sumw / weights_to.reindex(group_sumw.index)
        # NOTE: sumw_rel,index is also df_weight_map.index

        _filter = sumw_rel >= weight_rel_threshold
        df_weight_map = df_weight_map.loc[_filter]

        # _groups = {
        #    t: vws for t, vws in _groups.items() if sumw_rel[t] >= weight_rel_threshold
        # }

        # _groups = {
        #    idx: [(data.loc[i], v) for i, v in weights.items()]
        #    for idx, weights in df_weight_map.iterrows()
        # }

    # aggregate
    # result = {
    #    t: vtype.weighted_aggregate([(v, w / group_sumw[t]) for v, w in vws])
    #    for t, vws in _groups.items()
    # }

    def agg_r(row_weights):
        t = row_weights.name
        sumw = group_sumw[t]
        f_weights_rel = row_weights.dropna() / sumw
        vals = data.loc[f_weights_rel.index]
        return vtype.weighted_aggregate_ds(vals, f_weights_rel)

    result = df_weight_map.apply(agg_r, axis=1)

    #  re-scale intensive => extensive
    if vtype == VT_NumericExt:
        # result = {t: v * weights_to[t] for t, v in result.items()}
        result = result * weights_to.reindex(result.index)

    return result
