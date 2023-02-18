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

__version__ = "0.7.0"


from . import vartpye
from .utils import group_sum, is_na


def get_groups(variable, var_type, dom1_dom2_weights, size1=None):

    groups = {}
    for (d_f, d_t), w in dom1_dom2_weights.items():
        # get not na value
        v = variable.get(d_f)
        if is_na(v):
            continue

        # extensive scaling
        if var_type == vartpye.VarTypeMetricExt:
            # get size of dom1
            size1 = size1 or group_sum(dom1_dom2_weights.items(), lambda dd: dd[0])
            v /= size1[d_f]

        if d_t not in groups:
            groups[d_t] = []
        groups[d_t].append((v, w))

    return groups


def minimal_example(
    variable,
    var_type,
    dom1_dom2_weights,
    size1=None,
    size2=None,
    threshold=0,
    as_int=False,
):
    """TODO: speed up by using dataframes or even numpy matrix?"""

    groups = get_groups(variable, var_type, dom1_dom2_weights, size1=size1)

    result = {}
    for d_t, vws in groups.items():
        # weights sum
        sumw = sum(w for _, w in vws)

        # drop result?
        if threshold:
            size2 = size2 or group_sum(dom1_dom2_weights.items(), lambda dd: dd[1])
            if (sumw / size2[d_t]) < threshold:
                continue

        # normalize weights
        vws = [(v, w / sumw) for v, w in vws]

        # aggregate
        v = var_type.weighted_aggregate(vws)

        # extensive scaling
        if var_type == vartpye.VarTypeMetricExt:
            size2 = size2 or group_sum(dom1_dom2_weights.items(), lambda dd: dd[1])
            v *= size2[d_t]

        result[d_t] = v

        if issubclass(var_type, vartpye.VarTypeMetric) and as_int:
            result = dict((d, round(v)) for d, v in result.items())

    return result
