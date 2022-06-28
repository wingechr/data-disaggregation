import unittest
from functools import partial

import pandas as pd

from data_disaggregation.classes import (
    Dimension,
    DimensionLevel,
    Domain,
    FrozenMap,
    Variable,
)


class TestFrozenMap(unittest.TestCase):
    """immutable, ordered map of unique hashables mapping to variables of a type"""

    def test_frozenmap(self):
        # wrong value type
        self.assertRaises(TypeError, partial(FrozenMap, [(1, "val")], int))

        # unique
        self.assertRaises(KeyError, partial(FrozenMap, [(1, 1), (1, 1)], int))

        # hashable keys
        self.assertRaises(TypeError, partial(FrozenMap, [([1], 1)], int))

        fm = FrozenMap([(1, 1), (2, 1)], int)
        fm[1]  # access works

        # immutable
        def modify():
            fm[1] = 1

        self.assertRaises(TypeError, modify)


class TestDimensionLevel(unittest.TestCase):
    """wrapper around FrozenMap of elements"""

    @classmethod
    def setUpClass(cls):
        cls.dim1 = Dimension("d1")
        cls.dim1_lev1 = DimensionLevel("dl1", cls.dim1, ["a", "b"])
        cls.dim1_lev2 = DimensionLevel("dl2", cls.dim1, ["X", "Y", "Z"])

    def test_uniquenames(self):
        self.assertRaises(KeyError, partial(DimensionLevel, "d1", self.dim1, []))

    def test_dimensionlevel(self):
        # wrong value type
        self.assertEqual(self.dim1_lev1.size, 2)
        self.assertEqual(tuple(self.dim1_lev1.keys()), ("a", "b"))

    def test_domain_scalar(self):
        sd = Domain([])
        self.assertEqual(sd.size, 0)
        self.assertEqual(sd.shape, ())
        self.assertEqual(tuple(sd.indices.keys()), ())
        _v = Variable(10, sd, None)  # noqa

    def test_domain_linear(self):
        sd = Domain(self.dim1_lev1)
        self.assertEqual(sd.size, 1)
        self.assertEqual(sd.shape, (2,))
        self.assertEqual(
            tuple(sd.indices.keys()), ("a", "b")
        )  # keys are not tuples but values
        data = {"a": 1, "b": 2}
        _v = Variable(data, sd, None)  # noqa
        _v = Variable(pd.Series(data), sd, None)  # noqa

    def test_domain_multi(self):
        sd = Domain([self.dim1_lev1, self.dim1_lev2.alias("d1b")])
        self.assertEqual(sd.size, 2)
        self.assertEqual(sd.shape, (2, 3))
        self.assertEqual(
            tuple(sd.indices.keys()),
            (("a", "X"), ("a", "Y"), ("a", "Z"), ("b", "X"), ("b", "Y"), ("b", "Z")),
        )
        data = [
            {"d1": "a", "d2": "X", "v": 1},
            {"d1": "a", "d2": "Y", "v": 2},
            {"d1": "b", "d2": "Z", "v": 3},
        ]
        series = pd.DataFrame(data).set_index(["d1", "d2"])["v"]
        dct = dict(series.items())
        _v = Variable(dct, sd, None)  # noqa
        _v = Variable(series, sd, None)  # noqa

    def test_equality(self):
        self.assertEqual(
            Domain([self.dim1_lev1, self.dim1_lev2.alias("d1b")]),
            Domain([self.dim1_lev1.alias("d1b"), self.dim1_lev2]),
        )

    def test_transpose(self):
        dom = [self.dim1_lev1, self.dim1_lev1.alias("d1b")]
        v = Variable({("a", "a"): 1, ("a", "b"): 2, ("b", "a"): 3}, dom, None)
        self.assertRaises(KeyError, partial(v.transpose, ["x"]))
        vt = v.transpose(["d1b", "d1"])
        data = dict(kv for kv in vt.items() if kv[1])
        self.assertDictEqual(data, {("a", "a"): 1, ("a", "b"): 3, ("b", "a"): 2})

    def test_dom_insert_squeeze(self):
        v = Variable({}, self.dim1_lev1, None)
        v = v.expand(self.dim1.alias("d1b"))
        self.assertEqual(tuple(v.domain.keys()), ("d1", "d1b"))
        v = v.transpose(("d1b", "d1"))
        self.assertRaises(Exception, partial(v.squeeze))
        v = v.transpose(("d1", "d1b"))
        v = v.squeeze()
        self.assertEqual(tuple(v.domain.keys()), ("d1",))
