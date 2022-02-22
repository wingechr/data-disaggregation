"""Helper functions and tools"""

import math
from argparse import ArgumentError
from collections import OrderedDict

import numpy as np


def create_group_matrix(group_sizes):
    """Create a grouping matrix.

    Args:
        group_sizes: list of integers describing
            the groups sizes

    Returns:
        numpy matrix (2 dimensions) of 1 and 0


    Examples:
    >>> create_group_matrix([1, 2])
    array([[1., 0.],
           [0., 1.],
           [0., 1.]])
    """
    return create_weighted_group_matrix([[1] * n for n in group_sizes], on_err="ignore")


def create_weighted_group_matrix(
    grouped_weights, on_err="error", rel_tol=1e-09, abs_tol=0.0
):
    """Create a grouping matrix with weights.
    Weights should add up to 1.0 in each group

    Args:
        grouped_weights: list of list of values
        on_err(str, optional): what to do if groups don't all sum up to 1
            error (default): raise Exception
            rescale: try to rescale groups to sum of 1.0
            ignore: do nothing
        rel_tol(float): relative tolerance for compare float
        abs_tol(float): absolute tolerance for compare float

    Returns:
        numpy matrix (2 dimensions)


    Examples:
    >>> create_weighted_group_matrix([[1], [0.6, 0.4]])
    array([[1. , 0. ],
           [0. , 0.6],
           [0. , 0.4]])

    >>> create_weighted_group_matrix([[1], [6, 4]], on_err="rescale")
    array([[1. , 0. ],
           [0. , 0.6],
           [0. , 0.4]])
    """
    if on_err not in ("error", "ignore", "rescale"):
        raise ArgumentError('on_err must be one of ("error", "ignore", "rescale")')

    n_rows = sum(len(gw) for gw in grouped_weights)
    n_columns = len(grouped_weights)
    matrix = np.zeros(shape=(n_rows, n_columns))
    row_i_start = 0
    for col_i, weights in enumerate(grouped_weights):
        # check weights
        sum_weights = sum(weights)
        if not math.isclose(1, sum_weights, rel_tol=rel_tol, abs_tol=abs_tol):
            if on_err == "error":
                raise ValueError("Sum of weights != 1")
            elif on_err == "rescale":
                if not sum_weights:
                    raise ValueError("Sum of weights == 0")
                weights = np.array(weights) / sum_weights
        n_rows = len(weights)
        row_i_end = row_i_start + n_rows
        matrix[row_i_start:row_i_end, col_i] = weights
        row_i_start = row_i_end
    return matrix


def group_unique_values(items):
    """group items (pairs) into dict of lists.
    Values in each group stay in the original order and must be unique

    Args:
        items: iterable of key-value pairs

    Returns:
        dict of key -> lists (of unique values)

    Raises:
        ValueError if value in a group is a duplicate
    """

    result_lists = OrderedDict()
    result_sets = OrderedDict()
    for key, val in items:
        if key not in result_lists:
            result_lists[key] = []
            result_sets[key] = set()
        if val in result_sets[key]:
            raise ValueError("Duplicate value: %s" % val)
        result_sets[key].add(val)
        result_lists[key].append(val)
    return result_lists
