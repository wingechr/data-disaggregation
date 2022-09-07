import logging
from functools import partial
from unittest import TestCase

import pandas as pd

from data_disaggregation import transform
from data_disaggregation.utils import get_map_dims, norm_map

logging.basicConfig(
    format="[%(asctime)s %(levelname)7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class Tests(TestCase):
    def test_norm_map(self):
        d = {"a": ["a1", "a1", "a2"], "b": ["b1", "b2", "b2"], "v": [1, 3, 2]}
        mp = pd.DataFrame(d).set_index(["a", "b"])["v"]
        res = norm_map(mp, "a").to_dict()
        self.assertDictEqual(
            res, {("a1", "b1"): 0.25, ("a1", "b2"): 0.75, ("a2", "b2"): 1.0}
        )

        res = norm_map(mp, "b").to_dict()
        self.assertDictEqual(
            res, {("a1", "b1"): 1.0, ("a1", "b2"): 0.6, ("a2", "b2"): 0.4}
        )

        mp = pd.Series({"a1": 1, "a2": 3}).rename_axis(index="a")
        res = norm_map(mp, "a").to_dict()
        self.assertDictEqual(res, {"a1": 1.0, "a2": 1.0})

        res = norm_map(mp, None).to_dict()
        self.assertDictEqual(res, {"a1": 0.25, "a2": 0.75})

    def test_examples(self):
        df1 = {"s": 12, "x": 16}
        mp_a = pd.Series({"a1": 1, "a2": 3}).rename_axis(index="a")
        mp_b = pd.Series({"b1": 1, "b2": 3}).rename_axis(index="b")
        d = {"c": ["c1", "c1", "c2"], "b": ["b1", "b2", "b2"], "m": 1}
        mp_bc = pd.DataFrame(d).set_index(["c", "b"])["m"]
        df2 = transform(df1, mp_a)
        res = df2.to_dict("index")
        self.assertDictEqual(
            res, {"a1": {"s": 3.0, "x": 4.0}, "a2": {"s": 9.0, "x": 12.0}}
        )

        res = transform(df2, mp_a)
        self.assertDictEqual(res, {"s": 12.0, "x": 16.0})
        df4 = transform(df2, mp_b)
        res = df4.to_dict("index")
        self.assertDictEqual(
            res,
            {
                ("a1", "b1"): {"s": 0.75, "x": 1.0},
                ("a1", "b2"): {"s": 2.25, "x": 3.0},
                ("a2", "b1"): {"s": 2.25, "x": 3.0},
                ("a2", "b2"): {"s": 6.75, "x": 9.0},
            },
        )
        df5 = transform(df4, mp_bc)
        res = df5.to_dict("index")
        self.assertDictEqual(
            res,
            {
                ("a1", "c1"): {"s": 1.875, "x": 2.5},
                ("a1", "c2"): {"s": 1.125, "x": 1.5},
                ("a2", "c1"): {"s": 5.625, "x": 7.5},
                ("a2", "c2"): {"s": 3.375, "x": 4.5},
            },
        )
        df6 = transform(df5, mp_a)
        res = df6.to_dict("index")
        self.assertDictEqual(
            res, {"c1": {"s": 7.5, "x": 10.0}, "c2": {"s": 4.5, "x": 6.0}}
        )

    def test_examples_intensive(self):
        df1 = {"s": 12, "x": 16}
        mp_a = pd.Series({"a1": 1, "a2": 3}).rename_axis(index="a")
        mp_b = pd.Series({"b1": 1, "b2": 3}).rename_axis(index="b")
        d = {"c": ["c1", "c1", "c2"], "b": ["b1", "b2", "b2"], "m": 1}
        mp_bc = pd.DataFrame(d).set_index(["c", "b"])["m"]
        df2 = transform(df1, mp_a, intensive=True)
        res = df2.to_dict("index")
        self.assertDictEqual(res, {"a1": {"s": 12, "x": 16}, "a2": {"s": 12, "x": 16}})
        res = transform(df2, mp_a, intensive=True)
        self.assertDictEqual(res, {"s": 12.0, "x": 16.0})
        df4 = transform(df2, mp_b, intensive=True)
        res = df4.to_dict("index")
        self.assertDictEqual(
            res,
            {
                ("a1", "b1"): {"s": 12, "x": 16},
                ("a1", "b2"): {"s": 12, "x": 16},
                ("a2", "b1"): {"s": 12, "x": 16},
                ("a2", "b2"): {"s": 12, "x": 16},
            },
        )
        df5 = transform(df4, mp_bc, intensive=True)
        res = df5.to_dict("index")
        self.assertDictEqual(
            res,
            {
                ("a1", "c1"): {"s": 12.0, "x": 16.0},
                ("a1", "c2"): {"s": 12.0, "x": 16.0},
                ("a2", "c1"): {"s": 12.0, "x": 16.0},
                ("a2", "c2"): {"s": 12.0, "x": 16.0},
            },
        )
        df6 = transform(df5, mp_a, intensive=True)
        res = df6.to_dict("index")
        self.assertDictEqual(
            res, {"c1": {"s": 12.0, "x": 16.0}, "c2": {"s": 12.0, "x": 16.0}}
        )

    def test_map_via_get_map_dims(self):
        #  map must be 1 or 2-dim index or series
        # with unique names and values >= 2

        # non unique
        self.assertRaises(
            KeyError,
            partial(get_map_dims, pd.Series(1, index=pd.Index(["a1", "a1"], name="a"))),
        )

        # no name
        self.assertRaises(
            Exception, partial(get_map_dims, pd.Series(1, index=pd.Index(["a1", "a2"])))
        )

        # too many dims
        self.assertRaises(
            KeyError,
            partial(
                get_map_dims,
                pd.Series(
                    1,
                    index=pd.MultiIndex.from_product(
                        [["a1"], ["b1"], ["c1"]], names=["a", "b", "c"]
                    ),
                ),
            ),
        )

        # unique names
        self.assertRaises(
            KeyError,
            partial(
                get_map_dims,
                pd.Series(
                    1,
                    index=pd.MultiIndex.from_product(
                        [["a1"], ["a2"]], names=["a", "a"]
                    ),
                ),
            ),
        )

        # works
        mp, dims = get_map_dims(pd.Series(1, index=pd.Index(["a"], name="a")))
        self.assertIsInstance(mp, pd.Series)
        self.assertEqual(set(dims), set(["a"]))

    def test_transform_errors(self):
        d = pd.DataFrame([{"a": "a1", "v": 1}, {"a": "a2", "v": 2}]).set_index("a")["v"]
        m1 = pd.DataFrame(
            [{"a": "a1", "b": "b1", "m": 1}, {"a": "a1", "b": "b2", "m": 2}]
        ).set_index(["a", "b"])["m"]
        m2 = pd.DataFrame(
            [{"a": "a1", "b": "b1", "m": 4}, {"a": "a2", "b": "b1", "m": 5}]
        ).set_index(["a", "b"])["m"]

        # a2 cannot be mapped
        self.assertRaises(KeyError, partial(transform, d, m1))
        # but now it can be
        res = transform(d, m2)
        self.assertEqual(res.b1, 3)
        self.assertEqual(res.sum(), 3)

        res = transform(d, m2, intensive=True)


class TestExample(TestCase):
    def test_example(self):
        # we start with a single value for the country
        v = 120

        # the country consists of 3 regions of differenz size
        reg_size = pd.Series({"r1": 2, "r2": 3, "r3": 5}).rename_axis(index="reg")

        # distribute value via size
        reg_v = transform(v, reg_size)

        # regions are subdivided into subregions, and we have data of
        # population for each intersection
        subreg_reg_pop = pd.Series(
            {
                ("r1", "sr1a"): 2,
                ("r1", "sr1b"): 2,
                ("r2", "sr2a"): 3,
                ("r3", "sr3a"): 1,
                ("r3", "sr3b"): 2,
                ("r3", "sr3c"): 3,
            }
        ).rename_axis(index=["reg", "subreg"])

        # use population to further subdivide
        subreg_v = transform(reg_v, subreg_reg_pop)

        subreg = pd.Series(
            1,
            index=pd.Index(
                ["sr1a", "sr1b", "sr2a", "sr3a", "sr3b", "sr3c"], name="subreg"
            ),
        )
        v2 = transform(subreg_v, subreg)

        self.assertEqual(v, v2)
