"""Helper functions and tools"""

import math
from argparse import ArgumentError
from collections import OrderedDict

import numpy as np

from .exceptions import DimensionStructureError


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


def get_path_up_down(path_source, path_target):
    """paths for up/down

    NOTE: both lists always show the LOWER level element
      even for the so for path up, it shows the source,
      for path down the target!

    Args:
        path_source(list)
        path_target(list)

    """
    # find common part of path
    path_shared = []
    for pu, pd in zip(path_source, path_target):
        if pu != pd:
            break
        path_shared.append(pu)
    n = len(path_shared)  # root is always shared
    peak = path_shared[-1]
    path_down = path_target[n:]
    path_up = list(reversed(path_source[n:]))
    return path_up, peak, path_down


def normalize_groups(
    group_matrix: np.array, data_matrix1_d: np.array, on_group_0: str = "error"
) -> np.array:
    """Normalize values in each group (sums should be 1)

    Args:
        group_matrix(np.array): (n, m) matrix
            with n = number of elemens in this level
            and m = number of elemens in parent level (usually smaller than n)
        data_matrix1_d(np.array): only works for one-dimensional data,
            so shape must be (n,)
        on_group_0(str, optional): how to handle group sums of 0: must be one of
            * 'error': the default, raises an Error
            * 'zero': sets resulting weights to 0.
               this can lead to a loss of values on disaggregation
            * 'equal': sets resulting weights to 1/n

    returns:
        array  in  shape (n,)
    """

    if on_group_0 not in ("error", "zero", "equal"):
        raise ValueError("invalid value for on_group_0")

    if len(data_matrix1_d.shape) != 1:
        raise DimensionStructureError(
            "can only normalize one dimensional data, but shape is %s"
            % data_matrix1_d.shape
        )
    n = data_matrix1_d.shape[0]

    if len(group_matrix.shape) != 2 or group_matrix.shape[0] != n:
        raise DimensionStructureError(
            "Invalid group matrix, shape is %s" % group_matrix.shape
        )
    m = group_matrix.shape[1]

    # all values must be >= 0
    if not np.all(data_matrix1_d >= 0):
        raise ValueError("Not all values >= 0 before normalization")

    # EXAMPLE: n=5, m=2
    # group_matrix=[[1, 0],[1, 0],  [0, 1],[0, 1],[0, 1]]
    # data_matrix1_d=[[1], [2],   [3], [4], [5]]

    # create sum for groups: (m, n) * (n,) = (m,)
    sums = group_matrix.transpose().dot(data_matrix1_d)

    # EXAMPLE: [[3], [12]]

    # check that none is 0 (because we want to divide by it)
    if not np.all(sums != 0):
        if on_group_0 == "error":
            raise ValueError("some groups add up to 0")
        # replace 0 with (number of group members). this works, because
        # it's only 0 if all elements are 0 ==> division by number = 0
        n_members = np.sum(group_matrix, axis=0)
        group_is_0 = sums == 0
        sums += n_members * group_is_0
        if on_group_0 == "equal":
            # set all elements in groups to 1
            # by dividing by group size, we get equal distribution
            # first: repeat group_is_0: (m,) => (n, m)
            group_is_0 = np.reshape(group_is_0, (1, m))

            # EXAMPLE: [[1, 0]]
            group_is_0 = np.repeat(group_is_0, n, axis=0)
            # EXAMPLE: [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]

            # transform with group matrix and sum over axis: (n, m) => (n,)
            group_is_0 = group_is_0 * group_matrix
            # EXAMPLE: [[1, 0], [1, 0],   [0, 0], [0, 0], [0, 0]]
            group_is_0 = np.sum(group_is_0, axis=1)
            # EXAMPLE: [[1], [1],   [0], [0], [0]]

            # add this to data
            data_matrix1_d += group_is_0

    # create inverse and repeat, sum: (m,) => (1, m) => (n, m)
    sums = np.reshape(1 / sums, (1, m))
    # EXAMPLE: [[1/3,   1/12]]
    sums = np.repeat(sums, n, axis=0)
    # EXAMPLE: [[1/3, 1/12], [1/3, 1/12],   [1/3, 1/12], [1/3, 1/12], [1/3, 1/12]]

    # transform with group matrix and sum over axis: (n, m) => (n,)
    sums = sums * group_matrix
    # EXAMPLE: [[1/3, 0], [1/3, 0],   [0, 1/12], [0, 1/12], [0, 1/12]]
    sums = np.sum(sums, axis=1)
    # EXAMPLE: [[1/3], [1/3],   [1/12], [1/12], [1/12]]

    # transform with data_matrix: (n, ) => (n,)
    data = sums * data_matrix1_d
    # EXAMPLE: [[1/3], [2/3],   [3/12], [4/12], [5/12]]

    return data
