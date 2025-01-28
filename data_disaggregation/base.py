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

from typing import Mapping, Tuple

from .utils import (
    as_set,
    group_idx_first,
    group_idx_second,
    is_map,
    is_mapping,
    is_na,
    is_subset,
    is_unique,
    iter_values,
)
from .vtypes import F, T, V, VariableType, VT_NumericExt

VALIDATE_EQ_REL_TOLERANCE = 1e-10


def transform(
    vtype: VariableType,
    data: Mapping[F, V],
    weight_map: Mapping[Tuple[F, T], float],
    weights_from: Mapping[F, float] = None,
    weights_to: Mapping[T, float] = None,
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
    validate bool:
        if True: run additional (but costly) validations of weights and data.

    Returns
    -------
    : Mapping[T, V]
        output data as a mapping from output keys (any hashable) to values.

    """
    if weights_from is None:
        weights_from = group_idx_first(weight_map)

    if weights_to is None:
        weights_to = group_idx_second(weight_map)

    if validate:
        # validate size_f
        assert is_mapping(weights_from)
        assert is_unique(weights_from)
        assert all(v > 0 for v in iter_values(weights_from))

        # validate size_t
        assert is_mapping(weights_to)
        assert is_unique(weights_to)
        assert all(v > 0 for v in iter_values(weights_to))

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
        assert all(v >= 0 for v in iter_values(weight_map))
        assert is_subset([x[0] for x in weight_map.keys()], weights_from)
        assert is_subset([x[1] for x in weight_map.keys()], weights_to)
        # assert all(isinstance(v, (float, int)) for v in iter_values(weight_map))

    # filter nan in data
    data = dict((f, v) for f, v in data.items() if not is_na(v))

    #  scale extensive => intensive
    if vtype == VT_NumericExt:
        data = dict((f, v / weights_from[f]) for f, v in data.items())

    # filter unused in weight_map: input:
    weight_map = dict(((f, t), w) for (f, t), w in weight_map.items() if f in data)

    # filter unused in weight_map: output
    weight_map = dict(
        ((f, t), w) for (f, t), w in weight_map.items() if weights_to.get(t, 0) > 0
    )

    # init groups
    groups = dict((t, []) for t in set(_t for (_, _t) in weight_map.keys()))
    # group data by output keys
    for (f, t), w in weight_map.items():
        v = data[f]
        groups[t].append((v, w))

    # create weight sums
    group_sumw = dict((t, sum(w for _, w in vws)) for t, vws in groups.items())

    # drop groups under threshold
    if weight_rel_threshold:
        sumw_rel = dict((t, sumw / weights_to[t]) for t, sumw in group_sumw.items())
        groups = dict(
            (t, vws) for t, vws in groups.items() if sumw_rel[t] >= weight_rel_threshold
        )

    result = {}
    for t, vws in groups.items():
        # normalize weights
        vws = [(v, w / group_sumw[t]) for v, w in vws]
        # aggregate
        result[t] = vtype.weighted_aggregate(vws)

    #  re-scale intensive => extensive
    if vtype == VT_NumericExt:
        for t, v in result.items():
            result[t] = v * weights_to[t]

    return result
