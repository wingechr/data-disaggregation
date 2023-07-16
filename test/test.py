import logging
from unittest import TestCase

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series

from data_disaggregation.base import transform
from data_disaggregation.classes import (
    SCALAR_INDEX_KEY,
    VT_Nominal,
    VT_Numeric,
    VT_NumericExt,
    VT_Ordinal,
)
from data_disaggregation.utils import (
    as_mapping,
    group_idx_first,
    group_sum,
    is_list,
    is_mapping,
    is_na,
    is_scalar,
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

    def test_is_na(self):
        for x, y in [
            (1, False),
            (0, False),
            ("a", False),
            ("", False),
            (False, False),
            (None, True),
            (float("inf"), True),
            (float("nan"), True),
            (np.nan, True),
            (np.NAN, True),
        ]:
            self.assertEqual(is_na(x), y)

    def test_is_scalar(self):
        for x in [1, None, "xyz", True]:
            res = (is_scalar(x), is_list(x), is_mapping(x))
            self.assertEqual(res, (True, False, False), x)

    def test_is_list(self):
        for x in [
            [],
            (1, 2, 3),
            MultiIndex.from_product([[1, 2]]),
            Index(["a", "b"]),
            set([1, 2]),
        ]:
            res = (is_scalar(x), is_list(x), is_mapping(x))
            self.assertEqual(res, (False, True, False), x)

    def test_is_mapping(self):
        for x in [
            {},
            Series(dtype=float),
            Series({1: 1}),
            DataFrame(dtype=float),
            DataFrame({"a": [1, 2, 3]}),
        ]:
            res = (is_scalar(x), is_list(x), is_mapping(x))
            self.assertEqual(res, (False, False, True), x)

    def test_as_mapping(self):
        # is mapping
        res = as_mapping({"a": 1, "b": 1})
        self.assertDictEqual(res, {"a": 1, "b": 1})

        # is list
        res = as_mapping(["a", "b"])
        self.assertDictEqual(res, {"a": 1, "b": 1})

        # is scalar
        res = as_mapping(99)
        self.assertDictEqual(res, {SCALAR_INDEX_KEY: 99})


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

        return transform(vtype=vtype, data=var, weight_map=map)

    def test_example_type_categorical(self):
        res = self.get_example(VT_Nominal)
        for k, v in {"D": 10, "E": 30, "F": 5}.items():
            self.assertEqual(v, res[k])

    def test_example_type_ordinal(self):
        res = self.get_example(VT_Ordinal)
        for k, v in {"D": 10, "E": 30, "F": 10}.items():
            self.assertEqual(v, res[k])

    def test_example_type_metric(self):
        res = self.get_example(VT_Numeric)
        for k, v in {"D": 10, "E": 30, "F": 12}.items():
            self.assertEqual(v, res[k])

    def test_example_type_metric_ext(self):
        res = self.get_example(VT_NumericExt)
        for k, v in {
            "D": 3.333333333333333333,
            "E": 20,
            "F": 21.6666666666666667,
        }.items():
            self.assertAlmostEqual(v, res[k])


class TestBasePandasSeries(TestCase):
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
        s_map = Series(
            {
                ("a", "F"): 2,
                ("b", "D"): 1,
                ("b", "F"): 2,
                ("c", "E"): 2,
                ("c", "F"): 1,
            },
            dtype=float,
        )
        s_map.index.names = ["d1", "d2"]

        s_var = Series({"a": 5, "b": 10, "c": 30})
        s_var.index.names = ["d1"]

        return transform(weight_map=s_map, data=s_var, vtype=vtype)

    def test_example_type_categorical(self):
        res = self.get_example(VT_Nominal)
        for k, v in {"D": 10, "E": 30, "F": 5}.items():
            self.assertEqual(v, res[k])

    def test_example_type_ordinal(self):
        res = self.get_example(VT_Ordinal)
        for k, v in {"D": 10, "E": 30, "F": 10}.items():
            self.assertEqual(v, res[k])

    def test_example_type_metric(self):
        res = self.get_example(VT_Numeric)
        for k, v in {"D": 10, "E": 30, "F": 12}.items():
            self.assertEqual(v, res[k])

    def test_example_type_metric_ext(self):
        res = self.get_example(VT_NumericExt)
        for k, v in {"D": 3.333333333333333, "E": 20, "F": 21.66666666666667}.items():
            self.assertAlmostEqual(v, res[k])


class TestBaseExamples(TestCase):
    def test_aggregate_ext(self):
        res = transform(
            vtype=VT_NumericExt,
            data={1: 1, 2: 1, 3: 10},
            # size does not matter
            weight_map={
                (1, SCALAR_INDEX_KEY): 10,
                (2, SCALAR_INDEX_KEY): 20,
                (3, SCALAR_INDEX_KEY): 30,
            },
            threshold=0.5,
        )
        self.assertAlmostEqual(res[SCALAR_INDEX_KEY], 1 + 1 + 10)

    def test_aggregate_int(self):
        res = transform(
            vtype=VT_Numeric,
            data={1: 1, 2: 1, 3: 10},
            # size does not matter
            weight_map={
                (1, SCALAR_INDEX_KEY): 10,
                (2, SCALAR_INDEX_KEY): 20,
                (3, SCALAR_INDEX_KEY): 30,
            },
        )
        self.assertAlmostEqual(
            res[SCALAR_INDEX_KEY], 1 * 10 / 60 + 1 * 20 / 60 + 10 * 30 / 60
        )

    def test_disaggregate_thres(self):
        """case: rasterize a shape to partially overlapping cells"""
        res = transform(
            vtype=VT_NumericExt,
            data={SCALAR_INDEX_KEY: 100},
            weight_map={
                (SCALAR_INDEX_KEY, "00"): 0.51,
                (SCALAR_INDEX_KEY, "01"): 0.49,  # cell will be dropped
                (SCALAR_INDEX_KEY, "11"): 0.99,  # cell will be dropped (size = 2)
                (SCALAR_INDEX_KEY, "10"): 1.1,  # (size = 2)
            },
            size_in={SCALAR_INDEX_KEY: 5},
            size_out={"00": 1, "01": 1, "11": 2, "10": 2},
            threshold=0.5,
        )

        self.assertAlmostEqual(res["00"], 100 / 5)
        self.assertAlmostEqual(res["10"], 100 / 5 * 2)
        self.assertAlmostEqual(res.get("11", 0), 0)
        self.assertAlmostEqual(res.get("01", 0), 0)

    def test_scalar_key_none(self):
        """when using None as ley for scalars,
        pandas series index converts it no nan.
        because nan != nan, group sum no longer works
        """
        d = {(SCALAR_INDEX_KEY, 1): 1, (SCALAR_INDEX_KEY, 2): 2}
        s = Series(d)
        g = group_idx_first(s)
        self.assertEqual(len(g), 1, "None should be grouped (but nan is not)")

    def test_todo(self):
        """case: split a variable differently in different years"""
        res = transform(
            vtype=VT_NumericExt,
            data={
                ("v1", "t1"): 10,
                ("v2", "t1"): 11,
                ("v1", "t2"): 12,
                ("v2", "t2"): 13,
            },
            weight_map={
                # normalized in t1
                (("v1", "t1"), ("u1", "t1")): 0.7,
                (("v1", "t1"), ("u2", "t1")): 0.3,
                (("v2", "t1"), ("u3", "t1")): 0.2,
                (("v2", "t1"), ("u4", "t1")): 0.8,
                # not normalized in t2
                (("v1", "t2"), ("u1", "t2")): 0.7 * 10,
                (("v1", "t2"), ("u2", "t2")): 0.3 * 10,
                (("v2", "t2"), ("u3", "t2")): 11,
                (("v2", "t2"), ("u4", "t2")): 99,
            },
        )

        self.assertAlmostEqual(res[("u1", "t1")], 10 * 0.7)
        self.assertAlmostEqual(res[("u3", "t2")], 13 / (99 + 11) * 11)
        self.assertAlmostEqual(sum(v for k, v in res.items() if k[1] == "t1"), 10 + 11)
        self.assertAlmostEqual(sum(v for k, v in res.items() if k[1] == "t2"), 12 + 13)
