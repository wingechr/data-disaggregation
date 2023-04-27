import logging
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import Index, MultiIndex, Series

from data_disaggregation.classes import (
    SCALAR_DIM_NAME,
    SCALAR_INDEX_KEY,
    VT_Nominal,
    VT_Numeric,
    VT_NumericExt,
    VT_Ordinal,
)
from data_disaggregation.ext import (
    create_map,
    disagg,
    get_dimension_levels,
    is_multindex,
)
from data_disaggregation.utils import (
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

    def test_df_utils(self):
        s = 10  # scalar
        self.assertEqual(is_multindex(s), False)
        lvls = dict(get_dimension_levels(s))
        self.assertEqual(lvls[SCALAR_DIM_NAME][0], SCALAR_INDEX_KEY)

        d1 = pd.Index([10], name="d1")
        self.assertEqual(is_multindex(d1), False)
        lvls = dict(get_dimension_levels(d1))
        self.assertEqual(lvls["d1"][0], 10)

        s1 = pd.Series(index=d1, dtype="float")
        self.assertEqual(is_multindex(s1), False)
        lvls = dict(get_dimension_levels(s1))
        self.assertEqual(lvls["d1"][0], 10)

        dm = pd.MultiIndex.from_product([d1])
        self.assertEqual(is_multindex(dm), True)
        lvls = dict(get_dimension_levels(dm))
        self.assertEqual(lvls["d1"][0], (10,))

        sm = pd.Series(index=dm, dtype="float")
        self.assertEqual(is_multindex(sm), True)
        lvls = dict(get_dimension_levels(sm))
        self.assertEqual(lvls["d1"][0], (10,))

        # names must be unique

        xn = pd.MultiIndex.from_product([[1], [2]], names=["d", "d"])
        self.assertRaises(Exception, get_dimension_levels, xn)

    def test_align_map_todo(self):
        d0 = pd.Index([SCALAR_INDEX_KEY], name=SCALAR_DIM_NAME)
        d1 = pd.Index([1], name="d1")
        d1m = pd.MultiIndex.from_product([d1])
        d2 = pd.Index([2, 3], name="d2")
        d2m = pd.MultiIndex.from_product([d2])
        d12 = pd.MultiIndex.from_product([d1, d2])
        d3 = pd.Index([4], name="d3")
        d23 = pd.MultiIndex.from_product([d2, d3])

        res = create_map(pd.Series(1, index=d12), d1, d2)
        self.assertEqual(res[(1, 2)], 1)

        res = create_map(pd.Series(1, index=d12), d1m, d2)
        self.assertEqual(res[((1,), 2)], 1)

        res = create_map(pd.Series(1, index=d12), d1m, d2m)
        self.assertEqual(res[((1,), (2,))], 1)

        res = create_map(pd.Series(1, index=d23), d12, d23)
        self.assertEqual(res[((1, 2), (2, 4))], 1)

        res = create_map(pd.Series(1, index=d1), 0, d1)
        self.assertEqual(res[(SCALAR_INDEX_KEY, 1)], 1)

        res = create_map(pd.Series(1, index=d1), d1, d0)
        self.assertEqual(res[(1, SCALAR_INDEX_KEY)], 1)

    def test_is_scalar(self):
        for x in [1, None, "xyz", True]:
            res = (is_scalar(x), is_list(x), is_mapping(x))
            self.assertEqual(res, (True, False, False), x)

    def test_is_list(self):
        for x in [
            [],
            (1, 2, 3),
            pd.MultiIndex.from_product([[1, 2]]),
            pd.Index(["a", "b"]),
            set([1, 2]),
        ]:
            res = (is_scalar(x), is_list(x), is_mapping(x))
            self.assertEqual(res, (False, True, False), x)

    def test_is_mapping(self):
        for x in [
            {},
            pd.Series(dtype=float),
            pd.Series({1: 1}),
            pd.DataFrame(dtype=float),
            pd.DataFrame({"a": [1, 2, 3]}),
        ]:
            res = (is_scalar(x), is_list(x), is_mapping(x))
            self.assertEqual(res, (False, False, True), x)


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

        return disagg(
            vtype=vtype, data=var, weights=map, as_int=issubclass(vtype, VT_Numeric)
        )

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

        return disagg(
            weights=s_map,
            data=s_var,
            vtype=vtype,
            as_int=issubclass(vtype, VT_Numeric),
        )

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
        for k, v in {"D": round(3.333), "E": 20, "F": round(21.667)}.items():
            self.assertEqual(v, res[k])


class TestBaseExamples(TestCase):
    def test_aggregate_ext(self):
        res = disagg(
            vtype=VT_NumericExt,
            data={1: 1, 2: 1, 3: 10},
            # size does not matter
            weights={
                (1, SCALAR_INDEX_KEY): 10,
                (2, SCALAR_INDEX_KEY): 20,
                (3, SCALAR_INDEX_KEY): 30,
            },
            threshold=0.5,
        )
        self.assertAlmostEqual(res, 1 + 1 + 10)

    def test_aggregate_int(self):
        res = disagg(
            vtype=VT_Numeric,
            data={1: 1, 2: 1, 3: 10},
            # size does not matter
            weights={
                (1, SCALAR_INDEX_KEY): 10,
                (2, SCALAR_INDEX_KEY): 20,
                (3, SCALAR_INDEX_KEY): 30,
            },
        )
        self.assertAlmostEqual(res, 1 * 10 / 60 + 1 * 20 / 60 + 10 * 30 / 60)

    def test_disaggregate_thres(self):
        """case: rasterize a shape to partially overlapping cells"""
        res = disagg(
            vtype=VT_NumericExt,
            data={SCALAR_INDEX_KEY: 100},
            weights={
                (SCALAR_INDEX_KEY, "00"): 0.51,
                (SCALAR_INDEX_KEY, "01"): 0.49,  # cell will be dropped
                (SCALAR_INDEX_KEY, "11"): 0.99,  # cell will be dropped (size = 2)
                (SCALAR_INDEX_KEY, "10"): 1.1,  # (size = 2)
            },
            dim_in={SCALAR_INDEX_KEY: 5},
            dim_out={"00": 1, "01": 1, "11": 2, "10": 2},
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
        s = pd.Series(d)
        g = group_idx_first(s)
        self.assertEqual(len(g), 1, "None should be grouped (but nan is not)")

    def test_todo(self):
        """case: split a variable differently in different years"""
        res = disagg(
            vtype=VT_NumericExt,
            data={
                ("v1", "t1"): 10,
                ("v2", "t1"): 11,
                ("v1", "t2"): 12,
                ("v2", "t2"): 13,
            },
            weights={
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

        self.assertEqual(res[("u1", "t1")], 10 * 0.7)
        self.assertEqual(res[("u3", "t2")], 13 / (99 + 11) * 11)
        self.assertEqual(sum(v for k, v in res.items() if k[1] == "t1"), 10 + 11)
        self.assertEqual(sum(v for k, v in res.items() if k[1] == "t2"), 12 + 13)

    def test_ex_1(self):
        # create dimensions as pandas Index
        idx_month = pd.Index(["1", "2"], name="month")
        idx_hour = pd.Index(["11", "21", "22"], name="hour")
        idx_region = pd.Index(["a", "b"], name="region")

        # create multi dimensions as pandas MultiIndex
        idx_region_month = pd.MultiIndex.from_product([idx_region, idx_month])
        # idx_region_hour = pd.MultiIndex.from_product([idx_region, idx_hour])

        # create data as pandas series over dimensions
        # example: series over region
        s_region = pd.Series({"a": 10, "b": 30}, index=idx_region)
        # or just rename the index
        # s_region = pd.Series({"a": 10, "b": 30}, dtype="pint[meter]").rename_axis(["region"])  # noqa
        s_region = pd.Series({"a": 10, "b": 30}).rename_axis(["region"])

        # create (weighted) map between dimensions (in the simple case, all weights = 1)
        # as a series, with all dimensions in the index
        s_month = pd.Series({"1": 2, "2": 3}, index=idx_month)

        # replicate weight map to a second dimension
        s_month = s_month * pd.Series(1, index=idx_region_month)

        # auto aggregation => output dimension is only month
        res = VT_NumericExt.disagg(s_region, s_month)
        self.assertAlmostEqual(res["1"], 16)

        # use size_t with a series (probably wrongly)
        res = VT_NumericExt.disagg(
            s_region, s_month, dim_out=pd.Series(1, index=idx_region_month)
        )
        self.assertAlmostEqual(res[("a", "1")], 2)

        # use size_t with index only:
        res = VT_NumericExt.disagg(s_region, s_month, dim_out=idx_region_month)
        self.assertAlmostEqual(res[("a", "1")], 4)

        self.assertRaises(
            Exception, VT_NumericExt.disagg, s_region, s_month, size_t=idx_hour
        )

    def test_ex_2(self):
        # although not required, we use pandas Series for data and weights
        # using named Index/MultiIndex
        # We start by setting up some dimensions (2 spatial, 1 temporal)
        # using named Index

        dim_region = Index(["r1", "r2"], name="region")
        dim_subregion = Index(["r11", "r12", "r21", "r22"], name="subregion")
        dim_time = Index(["t1", "t2", "t3"], name="time")

        # We can use MultiIndex to create cross products:
        dim_region_subregion = MultiIndex.from_product([dim_region, dim_subregion])
        dim_region_time = MultiIndex.from_product([dim_region, dim_time])

        # now we create Series for data and weights (which also includes
        # relationships between dimensions)
        # using a value of 1 here because all the subregions have
        # the same weight relatively
        w_region_subregion = Series(
            {("r1", "r11"): 1, ("r1", "r12"): 1, ("r2", "r21"): 1, ("r2", "r22"): 1},
            index=dim_region_subregion,
        )

        # define some data on the regional level
        d_region = Series({"r1": 100, "r2": 200}, index=dim_region)

        # use extensive disaggregation:
        d_subregion = VT_NumericExt.disagg(d_region, w_region_subregion)
        self.assertEqual(d_subregion.index.name, "subregion")
        self.assertEqual(
            set(d_subregion.items()),
            set([("r11", 50), ("r12", 50), ("r21", 100), ("r22", 100)]),
        )

        # applying the same weight map aggregates it back.
        d_region2 = VT_NumericExt.disagg(d_subregion, w_region_subregion)
        self.assertEqual(d_region2.index.name, "region")
        self.assertEqual(
            set(d_region.items()),
            set(d_region2.items()),
        )

        # using Intensive distribution, the values for the regions
        # in the disaggregation are duplicated
        d_region2 = VT_Numeric.disagg(d_subregion, w_region_subregion)
        self.assertEqual(
            set(d_region2.items()),
            set([("r1", 50), ("r2", 100)]),
        )

        # distribute over a new dimension (time)
        w_time = Series({"t1": 2, "t2": 3, "t3": 5}, index=dim_time)
        s_region_time = VT_NumericExt.disagg(d_region, w_time, dim_out=dim_region_time)
        self.assertEqual(tuple(s_region_time.index.names), ("region", "time"))
        self.assertEqual(
            set(s_region_time.items()),
            set(
                [
                    (("r1", "t1"), 20),
                    (("r1", "t2"), 30),
                    (("r1", "t3"), 50),
                    (("r2", "t1"), 40),
                    (("r2", "t2"), 60),
                    (("r2", "t3"), 100),
                ]
            ),
        )

        # and what about scalar?
        s_time = VT_NumericExt.disagg(100, w_time)
        self.assertEqual(s_time.index.name, "time")
        self.assertEqual(
            set(s_time.items()),
            set(
                [
                    ("t1", 20),
                    ("t2", 30),
                    ("t3", 50),
                ]
            ),
        )

        # ... and back
        s = VT_NumericExt.disagg(s_time, w_time)
        self.assertEqual(s, 100)
