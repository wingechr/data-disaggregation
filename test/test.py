import logging
from unittest import TestCase

import pandas as pd

from data_disaggregation import vartype
from data_disaggregation.base import apply_map
from data_disaggregation.dataframe import apply_map_df
from data_disaggregation.utils import (
    as_multi_index,
    as_single_index,
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

    def test_as_multi_index(self):
        res = as_multi_index(pd.Index(["a"]))
        self.assertEqual(res[0], ("a",))

        res = as_multi_index(pd.MultiIndex.from_product([["a"]]))
        self.assertEqual(res[0], ("a",))

        res = as_multi_index(pd.Series({"a": 1}))
        self.assertEqual(res.index[0], ("a",))

        res = as_multi_index(pd.Series({("a",): 1}))
        self.assertEqual(res.index[0], ("a",))

    def test_as_single_index(self):
        res = as_single_index(pd.Index(["a"]))
        self.assertEqual(res[0], "a")

        res = as_single_index(pd.MultiIndex.from_product([["a"]]))
        self.assertEqual(res[0], "a")

        res = as_single_index(pd.Series({"a": 1}))
        self.assertEqual(res.index[0], "a")

        res = as_single_index(pd.Series({("a",): 1}))
        self.assertEqual(res.index[0], "a")


class TestBase(TestCase):
    """"""

    def get_example(self, vtype):
        """
        M | D  E  F | S | V
        ====================
        a |       2 | 2 |  5
        b | 1     2 | 3 | 10
        c |    2  1 | 3 | 30
        ====================
        S | 1  2  5 | 8 |

        Groups: (value, normweight, rescale)
        D: [(b, 10, 1/1, 1/3)]
        E: [(c, 30, 2/2, 2/3)]
        F: [
            (a,  5, 2/5, 5/2),
            (b, 10, 2/5, 5/3),
            (c, 30, 1/5, 5/3)
        ]

        """
        map = {
            ("a", "F"): 2,
            ("b", "D"): 1,
            ("b", "F"): 2,
            ("c", "E"): 2,
            ("c", "F"): 1,
        }

        var = {"a": 5, "b": 10, "c": 30}

        return apply_map(vtype=vtype, var=var, map=map, as_int=True)

    def test_example_type_categorical(self):
        res = self.get_example(vartype.VarTypeCategorical)
        for k, v in {"D": 10, "E": 30, "F": 5}.items():
            self.assertEqual(v, res[k])

    def test_example_type_ordinal(self):
        res = self.get_example(vartype.VarTypeOrdinal)
        for k, v in {"D": 10, "E": 30, "F": 10}.items():
            self.assertEqual(v, res[k])

    def test_example_type_metric(self):
        res = self.get_example(vartype.VarTypeMetric)
        for k, v in {"D": 10, "E": 30, "F": 12}.items():
            self.assertEqual(v, res[k])

    def test_example_type_metric_ext(self):
        res = self.get_example(vartype.VarTypeMetricExt)
        for k, v in {"D": round(3.33333333), "E": 20, "F": round(21.6666667)}.items():
            self.assertEqual(v, res[k])


class TestDataframe(TestCase):
    """"""

    def get_example(self, vtype):
        """
        M | D  E  F | S | V
        ====================
        a |       2 | 2 |  5
        b | 1     2 | 3 | 10
        c |    2  1 | 3 | 30
        ====================
        S | 1  2  5 | 8 |

        Groups: (value, normweight, rescale)
        D: [(b, 10, 1/1, 1/3)]
        E: [(c, 30, 2/2, 2/3)]
        F: [
            (a,  5, 2/5, 5/2),
            (b, 10, 2/5, 5/3),
            (c, 30, 1/5, 5/3)
        ]

        """
        s_map = pd.Series(
            {
                ("a", "F"): 2,
                ("b", "D"): 1,
                ("b", "F"): 2,
                ("c", "E"): 2,
                ("c", "F"): 1,
            }
        )
        s_map.index.names = ["d1", "d2"]

        s_var = pd.Series({"a": 5, "b": 10, "c": 30})
        s_var.index.names = ["d1"]

        return apply_map_df(
            s_map=s_map,
            s_var=s_var,
            vtype=vtype,
            as_int=True,
        )

    def test_example_type_categorical(self):
        res = self.get_example(vartype.VarTypeCategorical)
        for k, v in {"D": 10, "E": 30, "F": 5}.items():
            self.assertEqual(v, res[k])

    def test_example_type_ordinal(self):
        res = self.get_example(vartype.VarTypeOrdinal)
        for k, v in {"D": 10, "E": 30, "F": 10}.items():
            self.assertEqual(v, res[k])

    def test_example_type_metric(self):
        res = self.get_example(vartype.VarTypeMetric)
        for k, v in {"D": 10, "E": 30, "F": 12}.items():
            self.assertEqual(v, res[k])

    def test_example_type_metric_ext(self):
        res = self.get_example(vartype.VarTypeMetricExt)
        for k, v in {"D": round(3.333), "E": 20, "F": round(21.667)}.items():
            self.assertEqual(v, res[k])

    def test_x(self):
        d_A = pd.Index(["a1", "a2", "a3"], name="A")
        d_B = pd.Index(["b1", "b2"], name="B")
        d_C = pd.Index(["c1", "c2"], name="C")
        d_AB = pd.MultiIndex.from_product([d_A, d_B])
        # d_AC = pd.MultiIndex.from_product([d_A, d_C])
        d_BC = pd.MultiIndex.from_product([d_B, d_C])
        # d_ABC = pd.MultiIndex.from_product([d_A, d_B, d_C])

        d_BC_ = pd.MultiIndex.from_tuples(
            [x for x in d_BC if x != ("b1", "c2")], names=d_BC.names
        )

        v_AB = pd.Series(
            {
                ("a1", "b1"): 1,
                ("a1", "b2"): 2,
                ("a2", "b1"): 3,
                ("a2", "b2"): 4,
                ("a3", "b1"): 5,
                ("a3", "b2"): 6,
            },
            index=d_AB,
        )

        m_BC = pd.Series(  # noqa
            {
                ("b1", "c1"): 1,
                ("b1", "c2"): 2,
                ("b2", "c1"): 3,
                ("b2", "c2"): 4,
            },
            index=d_BC,
        )

        apply_map_df(
            s_var=v_AB,
            s_map=d_BC_,
            vtype=vartype.VarTypeCategorical,
        )
