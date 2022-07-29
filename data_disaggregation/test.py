from unittest import TestCase

import pandas as pd

from . import transform
from .utils import norm_map


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