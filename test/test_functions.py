import unittest
from functools import partial

import numpy as np
from numpy.testing import assert_almost_equal

from data_disaggregation.functions import (
    create_group_matrix,
    create_weighted_group_matrix,
    get_path_up_down,
    group_unique_values,
    normalize_groups,
)


class TestFunctions(unittest.TestCase):
    def test_group(self):
        # basic result
        res = group_unique_values([("a", "a1"), ("a", "a2"), (("b", "b1"))])
        exp_res = {"a": ["a1", "a2"], "b": ["b1"]}
        self.assertEqual(res, exp_res)

        # error on duplicates
        res = partial(group_unique_values, [("a", "a1"), ("a", "a1")])
        self.assertRaises(ValueError, res)

        # error on wrong structure
        res = partial(group_unique_values, [1])
        self.assertRaises(TypeError, res)

    def test_create_weighted_group_matrix(self):
        # basic working example
        res = create_weighted_group_matrix([[1], [0.6, 0.4]])
        expected_res = np.array([[1.0, 0.0], [0.0, 0.6], [0.0, 0.4]])
        assert_almost_equal(res, expected_res)

        # rescale
        res = create_weighted_group_matrix([[1], [6, 4]], on_err="rescale")
        expected_res = np.array([[1.0, 0.0], [0.0, 0.6], [0.0, 0.4]])
        assert_almost_equal(res, expected_res)

        # wrong arguments
        res = partial(create_weighted_group_matrix, [[[1]]])
        self.assertRaises(TypeError, res)

    def test_create_group_matrix(self):
        # basic working example
        res = create_group_matrix([1, 2])
        expected_res = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        assert_almost_equal(res, expected_res)

    def test_get_path_up_down(self):
        """
        NOTE: both lists always show the LOWER level element
        even for the so for path up, it shows the source,
        for path down the target!
        """
        # no change
        pu, p, pd = get_path_up_down([1, 2, 3], [1, 2, 3])
        path = tuple(pu) + tuple(pd)
        self.assertEqual(path, ())
        self.assertEqual(p, 3)

        # only down
        pu, p, pd = get_path_up_down([1], [1, 2, 3])
        path = tuple(pu) + tuple(pd)
        self.assertEqual(path, (2, 3))
        self.assertEqual(p, 1)

        # only up
        pu, p, pd = get_path_up_down([1, 2, 3], [1])
        path = tuple(pu) + tuple(pd)
        self.assertEqual(path, (3, 2))
        self.assertEqual(p, 1)

        # up + down
        pu, p, pd = get_path_up_down([1, 2, 3], [1, 4, 5])
        path = tuple(pu) + tuple(pd)
        self.assertEqual(path, (3, 2, 4, 5))
        self.assertEqual(p, 1)

    def test_normalize_groups(self):
        group_matrix = np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
            ]
        )

        res = normalize_groups(
            group_matrix=group_matrix,
            data_matrix1_d=np.array([1, 2, 3, 4, 5]).transpose(),
        )
        assert_almost_equal(
            res, np.array([1 / 3, 2 / 3, 3 / 12, 4 / 12, 5 / 12]).transpose()
        )

        # throw exception if group sum 0
        res = partial(
            normalize_groups,
            group_matrix=group_matrix,
            data_matrix1_d=np.array([0, 0, 3, 4, 5]).transpose(),
        )
        self.assertRaises(ValueError, res)

        res = normalize_groups(
            group_matrix=group_matrix,
            data_matrix1_d=np.array([0, 0, 3, 0, 5]).transpose(),
            on_group_0="nan",
        )
        assert_almost_equal(
            res, np.array([np.nan, np.nan, 3 / 8, 0 / 8, 5 / 8]).transpose()
        )

        res = normalize_groups(
            group_matrix=group_matrix,
            data_matrix1_d=np.array([0, 0, 3, 4, 5]).transpose(),
            on_group_0="equal",
        )
        assert_almost_equal(
            res, np.array([0.5, 0.5, 3 / 12, 4 / 12, 5 / 12]).transpose()
        )

        # throw exception if any value < 0
        res = partial(
            normalize_groups,
            group_matrix=group_matrix,
            data_matrix1_d=np.array([-1, 2, 3, 4, 5]).transpose(),
        )
        self.assertRaises(ValueError, res)
