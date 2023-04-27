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

from .classes import VT, F, T, V, VT_Numeric, VT_NumericExt
from .utils import (
    group_idx_first,
    group_idx_second,
    is_map,
    is_mapping,
    is_na,
    is_subset,
    is_unique,
    iter_values,
)


def get_groups(
    vtype: VT,
    var: Mapping[F, V],
    map: Mapping[Tuple[F, T], float],
    size_f: Mapping[F, float],
) -> Mapping[T, Tuple[V, float]]:
    result = {}

    for (f, t), w in map.items():
        # get not na value
        v = var.get(f)
        if is_na(v):
            continue

        #  scale extensive => intensive
        if vtype == VT_NumericExt:
            v /= size_f[f]

        if t not in result:
            result[t] = []
        result[t].append((v, w))

    return result


def apply_map(
    vtype: VT,
    var: Mapping[F, V],
    map: Mapping[Tuple[F, T], float],
    size_t: Mapping[T, float] = None,
    size_f: Mapping[F, float] = None,
    threshold: float = 0.0,
    as_int: bool = False,
    validate=True,
) -> Mapping[T, V]:
    if size_f is None:
        size_f = group_idx_first(map)

    if size_t is None:
        size_t = group_idx_second(map)

    if validate:
        # validate size_f
        assert is_mapping(size_f)
        assert is_unique(size_f)
        assert all(v > 0 for v in iter_values(size_f))

        # validate size_t
        assert is_mapping(size_t)
        assert is_unique(size_t)
        assert all(v > 0 for v in iter_values(size_t))

        # validate var
        assert is_mapping(var)
        assert is_unique(var)

        assert is_subset(
            var, size_f
        ), "Variable index is not a subset of input dimension subset"

        # validate map
        assert is_map(map)
        assert is_unique(map)
        assert all(v >= 0 for v in iter_values(map))
        assert is_subset([x[0] for x in map.keys()], size_f)
        assert is_subset([x[1] for x in map.keys()], size_t)

    result = {}

    groups = get_groups(vtype, var, map, size_f)

    for t, vws in groups.items():
        # weights sum
        sumw = sum(w for _, w in vws)
        # TODO drop test
        sumw <= size_t[t]

        # drop result?
        if threshold:
            if (sumw / size_t[t]) < threshold:
                continue

        # normalize weights
        vws = [(v, w / sumw) for v, w in vws]

        # aggregate
        v = vtype.weighted_aggregate(vws)

        #  re-scale intensive => extensive
        if vtype == VT_NumericExt:
            v *= size_t[t]

        result[t] = v

    # rounding
    if as_int:
        assert issubclass(vtype, VT_Numeric)
        result = dict((t, round(v)) for t, v in result.items())

    if validate:
        # todo remove checks at the end
        assert is_subset(result, size_t)
        assert is_unique(result)

    return result
