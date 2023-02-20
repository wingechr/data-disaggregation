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

import logging
from itertools import product

import pandas as pd

from . import vartpye
from .utils import (
    as_index,
    as_multi_index,
    as_single_index,
    get_levels_dict,
    group_sum,
    is_na,
)


def get_i_out(i_var, i_map):

    i_map = as_index(i_map)
    i_var = as_index(i_var)
    # get input levels: name -> elements
    common_level_names = set(i_map.names) & set(i_var.names)
    # get output levels: name -> elements
    levels = get_levels_dict(i_var) | get_levels_dict(i_map)
    lvls = dict((k, v) for k, v in levels.items() if k not in common_level_names)

    return pd.MultiIndex.from_product(lvls.values(), names=lvls.keys())


def align_map(i_var, s_map, i_out):
    # ------------------------------
    # create map
    # ------------------------------

    i_var = as_index(i_var)

    # get input levels: name -> elements
    input_levels = get_levels_dict(i_var)

    # get output levels: name -> elements
    output_levels = get_levels_dict(i_out)

    # get mapping levels: name -> elements
    # TODO: mapping levels must be non empty subset of all_levels
    mapping_levels = get_levels_dict(s_map.index)

    # combine input and output levels (output higher priority) to unique set
    all_levels = input_levels | output_levels

    input_levels_idx = [list(all_levels).index(n) for n in input_levels]
    output_levels_idx = [list(all_levels).index(n) for n in output_levels]
    mapping_levels_idx = [list(all_levels).index(n) for n in mapping_levels]

    map_dict = {}
    for row in product(*all_levels.values()):
        key_map = tuple(row[i] for i in mapping_levels_idx)
        val_map = s_map.get(key_map)
        if is_na(val_map):
            continue

        key_in = tuple(row[i] for i in input_levels_idx)
        key_out = tuple(row[i] for i in output_levels_idx)

        key = (key_in, key_out)
        map_dict[key] = val_map

    result = pd.Series(map_dict)
    result.index.names = list(mapping_levels)

    return result


def get_groups(s_var, vtype, s_map_ft, s_size_f):

    groups = {}
    for (d_f, d_t), w in s_map_ft.items():
        # get not na value
        v = s_var.get(d_f)
        if is_na(v):
            continue

        # extensive scaling
        if vtype == vartpye.VarTypeMetricExt:
            # get size of dom1
            v /= s_size_f[d_f]

        if d_t not in groups:
            groups[d_t] = []
        groups[d_t].append((v, w))

    return groups


def group_idx_from(items):
    return group_sum(items, lambda k: k[0])


def group_idx_to(items):
    return group_sum(items, lambda k: k[1])


def minimal_example(
    s_var,
    vtype,
    s_map,
    i_out=None,
    s_size_f=None,
    s_size_t=None,
    threshold=0,
    as_int=False,
):
    """TODO: speed up by using dataframes or even numpy matrix?"""

    logging.debug("=======================")

    s_var = as_multi_index(s_var)
    logging.debug(f"\ns_var:\n{s_var}\n{vtype}")

    if isinstance(s_map, pd.Index):
        # all value 1
        s_map = pd.Series(1, index=s_map)

    logging.debug(f"\ns_map:\n{s_map}")

    if i_out is None:
        i_out = get_i_out(s_var, s_map)

    i_out = as_multi_index(i_out)

    logging.debug(f"\ni_out:\n{i_out}")

    s_map_ft = align_map(s_var, s_map, i_out)

    logging.debug(f"\ns_map_ft:\n{s_map_ft}")

    if s_size_f is None:
        s_size_f = group_idx_from(s_map_ft)
    s_size_f = as_multi_index(s_size_f)
    logging.debug(f"\ns_size_f:\n{s_size_f}")

    if s_size_t is None:
        s_size_t = group_idx_to(s_map_ft)
    s_size_t = as_multi_index(s_size_t)

    logging.debug(f"\ns_size_t:\n{s_size_t}")

    groups = get_groups(s_var, vtype, s_map_ft, s_size_f)

    logging.debug(f"\ngroups:\n{groups}")

    result = {}
    for d_t, vws in groups.items():
        # weights sum
        sumw = sum(w for _, w in vws)

        # drop result?
        if threshold:
            if (sumw / s_size_t[d_t]) < threshold:
                continue

        # normalize weights
        vws = [(v, w / sumw) for v, w in vws]

        # aggregate
        v = vtype.weighted_aggregate(vws)

        # extensive scaling
        if vtype == vartpye.VarTypeMetricExt:
            logging.debug("rescale VarTypeMetricExt")
            v *= s_size_t[d_t]

        result[d_t] = v

    logging.debug(f"\nresult:\n{result}")

    if issubclass(vtype, vartpye.VarTypeMetric) and as_int:
        result = dict((d, round(v)) for d, v in result.items())

    result = pd.Series(result)
    result.index.names = i_out.names

    logging.debug(f"\nresult:\n{result}")

    if len(result.index.names) == 1:
        result = as_single_index(result)

    logging.debug(f"\nresult:\n{result}")

    return result
