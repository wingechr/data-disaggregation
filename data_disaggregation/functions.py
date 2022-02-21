import math
from argparse import ArgumentError
from collections import OrderedDict

import numpy as np


def create_group_matrix(group_sizes):
    """
    Examples:
    >>> create_group_matrix([1, 2])
    array([[1., 0.],
           [0., 1.],
           [0., 1.]])
    """
    return create_weighted_group_matrix(
        [[1] * n for n in group_sizes], on_group_sum_ne_1="ignore"
    )


def create_weighted_group_matrix(
    grouped_weights, on_group_sum_ne_1="error", rel_tol=1e-09, abs_tol=0.0
):
    """
    Examples:
    >>> create_weighted_group_matrix([[1], [0.6, 0.4]])
    array([[1. , 0. ],
           [0. , 0.6],
           [0. , 0.4]])

    >>> create_weighted_group_matrix([[1], [6, 4]], on_group_sum_ne_1="rescale")
    array([[1. , 0. ],
           [0. , 0.6],
           [0. , 0.4]])
    """
    if on_group_sum_ne_1 not in ("error", "ignore", "rescale"):
        raise ArgumentError(
            'on_group_sum_ne_1 must be one of ("error", "ignore", "rescale")'
        )

    n_rows = sum(len(gw) for gw in grouped_weights)
    n_columns = len(grouped_weights)
    matrix = np.zeros(shape=(n_rows, n_columns))
    row_i_start = 0
    for col_i, weights in enumerate(grouped_weights):
        # check weights
        sum_weights = sum(weights)
        if not math.isclose(1, sum_weights, rel_tol=rel_tol, abs_tol=abs_tol):
            if on_group_sum_ne_1 == "error":
                raise ValueError("Sum of weights != 1")
            elif on_group_sum_ne_1 == "rescale":
                if not sum_weights:
                    raise ValueError("Sum of weights == 0")
                weights = np.array(weights) / sum_weights
        n_rows = len(weights)
        row_i_end = row_i_start + n_rows
        matrix[row_i_start:row_i_end, col_i] = weights
        row_i_start = row_i_end
    return matrix


def group(items):
    result_lists = OrderedDict()
    result_sets = OrderedDict()
    for key, val in items:
        if key not in result_lists:
            result_lists[key] = []
            result_sets[key] = set()
        if val in result_sets[key]:
            raise KeyError("Duplicate key: %s" % val)
        result_sets[key].add(val)
        result_lists[key].append(val)
    return result_lists
