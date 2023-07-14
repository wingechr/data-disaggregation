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

from .classes import VT, F, T, V, VT_NumericExt
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

VALIDATE_EQ_REL_TOLERANCE = 1e-10


def get_groups(
    vtype: VT,
    data: Mapping[F, V],
    weight_map: Mapping[Tuple[F, T], float],
    size_in: Mapping[F, float],
) -> Mapping[T, Tuple[V, float]]:
    # filter nan in data
    data = dict((f, v) for f, v in data.items() if not is_na(v))

    #  scale extensive => intensive
    if vtype == VT_NumericExt:
        data = dict((f, v / size_in[f]) for f, v in data.items())

    # filter unused in weight_map:
    weight_map = dict(((f, t), w) for (f, t), w in weight_map.items() if f in data)

    # init results
    result = dict((t, []) for t in set(_t for (_, _t) in weight_map.keys()))

    # group data
    for (f, t), w in weight_map.items():
        v = data[f]
        result[t].append((v, w))

    return result


def transform(
    vtype: VT,
    data: Mapping[F, V],
    weight_map: Mapping[Tuple[F, T], float],
    size_in: Mapping[F, float] = None,
    size_out: Mapping[T, float] = None,
    threshold: float = 0.0,
    validate=True,
) -> Mapping[T, V]:
    if size_in is None:
        size_in = group_idx_first(weight_map)

    if size_out is None:
        size_out = group_idx_second(weight_map)

    if validate:
        # validate size_f
        assert is_mapping(size_in)
        assert is_unique(size_in)
        assert all(v > 0 for v in iter_values(size_in))
        # assert all(isinstance(v, (float, int)) for v in iter_values(size_in))

        # validate size_t
        assert is_mapping(size_out)
        assert is_unique(size_out)
        assert all(v > 0 for v in iter_values(size_out))
        # assert all(isinstance(v, (float, int)) for v in iter_values(size_out))

        # validate var
        assert is_mapping(data)
        assert is_unique(data)

        if not is_subset(data, size_in):
            err = as_set(data) - as_set(size_in)
            raise Exception(
                f"Variable index is not a subset of input dimension subset: {err}"
            )

        # validate map
        assert is_map(weight_map)
        assert is_unique(weight_map)
        assert all(v >= 0 for v in iter_values(weight_map))
        assert is_subset([x[0] for x in weight_map.keys()], size_in)
        assert is_subset([x[1] for x in weight_map.keys()], size_out)
        # assert all(isinstance(v, (float, int)) for v in iter_values(weight_map))

    groups = get_groups(vtype=vtype, data=data, weight_map=weight_map, size_in=size_in)

    # create weight sums
    group_sumw = dict((t, sum(w for _, w in vws)) for t, vws in groups.items())

    if validate or threshold:
        sumw_rel = dict((t, sumw / size_out[t]) for t, sumw in group_sumw.items())

    # sumw always <= size_out ==> sumw_rel <= 1
    # TODO: maybe drop this check, this should never really happen
    if validate:
        scomp = 1 + VALIDATE_EQ_REL_TOLERANCE
        assert all(s <= scomp for s in sumw_rel.values()), sumw_rel

    # drop groups under threshold
    if threshold:
        groups = dict((t, vws) for t, vws in groups.items() if sumw_rel[t] >= threshold)

    result = {}
    for t, vws in groups.items():
        # normalize weights
        vws = [(v, w / group_sumw[t]) for v, w in vws]

        # aggregate
        result[t] = vtype.weighted_aggregate(vws)

    #  re-scale intensive => extensive
    if vtype == VT_NumericExt:
        for t, v in result.items():
            result[t] = v * size_out[t]

    if validate:
        # todo remove checks at the end
        assert is_subset(result, size_out)
        assert is_unique(result)

    return result
