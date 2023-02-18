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


"""

__version__ = "0.7.0"


from . import vartpye
from .utils import is_na


def minimal_example(dom1, dom2, dom1_dom2, variable, var_type, threshold=0):

    d2_items = dict((d, []) for d in dom2)
    for (d1, d2), w in dom1_dom2.items():
        # get not na value
        u = variable.get(d1)
        if is_na(u):
            continue

        # extensive scaling
        if var_type == vartpye.VarTypeMetricExt:
            u /= dom1[d1]

        d2_items[d2].append((u, w))

    result = {}
    for d2, data in d2_items.items():
        if not data:
            continue

        # weights sum
        sumw = sum(x[1] for x in data)

        # drop result?
        if (sumw / dom2[d2]) < threshold:
            continue

        # normalize weights
        data = [(u, w / sumw) for u, w in data]

        # aggregate
        v = var_type.weighted_aggregate(data)

        # extensive scaling
        if var_type == vartpye.VarTypeMetricExt:
            v *= dom2[d2]

        result[d2] = v

    return result
