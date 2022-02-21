import unittest
from functools import partial

import numpy as np
from numpy.testing import assert_array_equal

from data_disaggregation.functions import (
    create_group_matrix,
    create_weighted_group_matrix,
    group,
)


class TestFunctions(unittest.TestCase):
    def test_group(self):
        # basic result
        res = group([("a", "a1"), ("a", "a2"), (("b", "b1"))])
        exp_res = {"a": ["a1", "a2"], "b": ["b1"]}
        self.assertEqual(res, exp_res)

        # error on duplicates
        res = partial(group, [("a", "a1"), ("a", "a1")])
        self.assertRaises(KeyError, res)

        # error on wrong structure
        res = partial(group, [1])
        self.assertRaises(TypeError, res)

    def test_create_weighted_group_matrix(self):
        # basic working example
        res = create_weighted_group_matrix([[1], [0.6, 0.4]])
        expected_res = np.array([[1.0, 0.0], [0.0, 0.6], [0.0, 0.4]])
        assert_array_equal(res, expected_res)

        # rescale
        res = create_weighted_group_matrix([[1], [6, 4]], on_group_sum_ne_1="rescale")
        expected_res = np.array([[1.0, 0.0], [0.0, 0.6], [0.0, 0.4]])
        assert_array_equal(res, expected_res)

        # wrong arguments
        res = partial(create_weighted_group_matrix, [[[1]]])
        self.assertRaises(TypeError, res)

    def test_create_group_matrix(self):
        # basic working example
        res = create_group_matrix([1, 2])
        expected_res = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        assert_array_equal(res, expected_res)
