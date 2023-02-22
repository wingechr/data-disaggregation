import logging
from itertools import product
from typing import TypeVar

import pandas as pd

from .base import apply_map, group_idx_from, group_idx_to
from .utils import as_index, as_multi_index, get_levels_dict, is_na

F = TypeVar("F")
T = TypeVar("T")
V = TypeVar("V")


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


def apply_map_df(
    vtype,
    s_var,
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

    result = apply_map(
        vtype=vtype,
        var=s_var,
        map=s_map_ft,
        size_f=s_size_f,
        size_t=s_size_t,
        threshold=threshold,
        as_int=as_int,
    )

    result = pd.Series(result)
    result.index.names = i_out.names

    return result
