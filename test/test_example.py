import unittest

from numpy.testing import assert_array_equal

from data_disaggregation.classes import Dimension, Variable


class TestExample(unittest.TestCase):
    def test_example(self):
        time = Dimension("time")
        hour = time.add_level("hour", [1, 2, 3])

        space = Dimension("space")
        region = space.add_level("region", ["r1", "r2"])
        subregion = region.add_level(
            "subregion", {"r1": ["sr1_1", "sr1_2"], "r2": ["sr2_1"]}
        )

        v1 = Variable(
            name="v1",
            data={
                (1, "sr1_1"): 2,
                (1, "sr1_2"): 3,
                (2, "sr1_2"): 4,
                (2, "sr2_1"): 5,
            },
            domain=[hour, subregion],
            vartype="extensive",
        )
        v2 = v1.transform(domain=[region])
        assert_array_equal(v2._data_matrix, [9, 5])
