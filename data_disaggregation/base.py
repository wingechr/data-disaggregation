"""
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

from typing import Mapping, Tuple, TypeVar

from . import vartype
from .utils import group_idx_first, group_idx_second, is_na
from .vartype import VarTypeBase

F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")


def get_groups(
    vtype: VarTypeBase,
    var: Mapping[F, V],
    map: Mapping[Tuple[F, T], float],
    size_f: Mapping[F, float],
) -> Mapping[T, Tuple[V, float]]:
    groups = {}

    for (f, t), w in map.items():
        # get not na value
        v = var.get(f)
        if is_na(v):
            continue

        #  scale extensive => intensive
        if vtype == vartype.VarTypeMetricExt:
            v /= size_f[f]

        if t not in groups:
            groups[t] = []
        groups[t].append((v, w))

    return groups


def apply_map(
    vtype: VarTypeBase,
    var: Mapping[F, V],
    map: Mapping[Tuple[F, T], float],
    size_f: Mapping[F, float] = None,
    size_t: Mapping[T, float] = None,
    threshold: float = 0.0,
    as_int: bool = False,
) -> Mapping[T, V]:
    # sanity check

    result = {}

    size_f = size_f or group_idx_first(map)
    size_t = size_t or group_idx_second(map)

    def _values(x):
        # TODO
        if isinstance(x, dict):
            return x.values()
        else:  # series
            return x.values

    assert all(v >= 0 for v in _values(map))
    assert all(v > 0 for v in _values(size_f))
    assert all(v > 0 for v in _values(size_t))

    groups = get_groups(vtype, var, map, size_f)

    for t, vws in groups.items():
        # weights sum
        sumw = sum(w for _, w in vws)

        # drop result?
        if threshold:
            if (sumw / size_t[t]) < threshold:
                continue

        # normalize weights
        vws = [(v, w / sumw) for v, w in vws]

        # aggregate
        v = vtype.weighted_aggregate(vws)

        #  re-scale intensive => extensive
        if vtype == vartype.VarTypeMetricExt:
            v *= size_t[t]

        result[t] = v

    # rounding
    if as_int and issubclass(vtype, vartype.VarTypeMetric):
        result = dict((t, round(v)) for t, v in result.items())

    return result
