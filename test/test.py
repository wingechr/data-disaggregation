import logging
from unittest import TestCase

from data_disaggregation import minimal_example, vartpye
from data_disaggregation.utils import (
    group_sum,
    weighted_median,
    weighted_mode,
    weighted_sum,
)

logging.basicConfig(
    format="[%(asctime)s %(levelname)7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class TestUtils(TestCase):
    def test_groupsum(self):
        res = group_sum([["a", 1], ("a", 2), (3, 4), ((0, 0), 5), ((0, 0), 5)])
        res_d = dict(res)

        self.assertEqual(len(res), len(res_d))
        self.assertEqual(res_d["a"], 3)
        self.assertEqual(res_d[3], 4)
        self.assertEqual(res_d[(0, 0)], 10)

    def test_weighted_mode(self):
        res = weighted_mode([(3, 0.4), (2, 0.25), [1, 0.35]])
        self.assertEqual(res, 3)

    def test_weighted_median(self):
        res = weighted_median([(3, 0.4), (2, 0.25), [1, 0.35]])
        self.assertEqual(res, 2)

    def test_weighted_sum(self):
        res = weighted_sum([(3, 0.4), (2, 0.25), [1, 0.35]])
        self.assertAlmostEqual(res, 2.05)


class TestExample(TestCase):
    """"""

    def get_example(self, var_type):
        return minimal_example(
            dom1={"a": 1, "b": 2, "c": 3},
            dom2={"D": 1, "E": 2, "F": 3},
            dom1_dom2={
                ("a", "F"): 1,
                ("b", "D"): 1,
                ("b", "F"): 1,
                ("c", "E"): 2,
                ("c", "F"): 1,
            },
            variable={"a": 5, "b": 10, "c": 30},
            var_type=var_type,
        )

    def test_example(self):
        res = self.get_example(vartpye.VarTypeCategorical)
        for k, v in {"D": 10, "E": 30, "F": 5}.items():
            self.assertEqual(v, res[k])

        res = self.get_example(vartpye.VarTypeOrdinal)
        for k, v in {"D": 10, "E": 30, "F": 10}.items():
            self.assertEqual(v, res[k])

        res = self.get_example(vartpye.VarTypeMetric)
        for k, v in {"D": 10, "E": 30, "F": 15}.items():
            self.assertAlmostEqual(v, res[k])

        res = self.get_example(vartpye.VarTypeMetricExt)
        for k, v in {"D": 5, "E": 20, "F": 20}.items():
            self.assertAlmostEqual(v, res[k])
