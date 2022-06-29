import unittest
from functools import partial

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

    def test_domain_linear(self):
        sd = Domain(self.dim1_lev1)
        self.assertEqual(sd.size, 1)
        self.assertEqual(sd.shape, (2,))
        self.assertEqual(
            tuple(sd.indices.keys()), ("a", "b")
        )  # keys are not tuples but values

    def test_domain_multi(self):
        sd = Domain([self.dim1_lev1, self.dim1_lev2.alias("d1b")])
        self.assertEqual(sd.size, 2)
        self.assertEqual(sd.shape, (2, 3))
        self.assertEqual(
            tuple(sd.indices.keys()),
            (("a", "X"), ("a", "Y"), ("a", "Z"), ("b", "X"), ("b", "Y"), ("b", "Z")),
        )

    def test_equality(self):
        self.assertEqual(
            Domain([self.dim1_lev1, self.dim1_lev2.alias("d1b")]),
            Domain([self.dim1_lev1.alias("d1b"), self.dim1_lev2]),
        )

    def test_transpose(self):
        dom = [self.dim1_lev1, self.dim1_lev1.alias("d1b")]
        v = Variable({("a", "a"): 1, ("a", "b"): 2, ("b", "a"): 3}, dom, None, True)
        self.assertRaises(KeyError, partial(v.transpose, ["x"]))
        vt = v.transpose(["d1b", "d1"])
        data = dict(kv for kv in vt.items() if kv[1])
        self.assertDictEqual(data, {("a", "a"): 1, ("a", "b"): 3, ("b", "a"): 2})

    def test_dom_insert_squeeze(self):
        v = Variable({}, self.dim1_lev1, None, True)
        v = v._expand(self.dim1.alias("d1b"))
        self.assertEqual(tuple(v.domain.keys()), ("d1", "d1b"))
        v = v.transpose(("d1b", "d1"))
        self.assertRaises(Exception, partial(v._squeeze))
        v = v.transpose(("d1", "d1b"))
        v = v._squeeze()
        self.assertEqual(tuple(v.domain.keys()), ("d1",))

    def test_mult_add(self):
        dom1 = self.dim1_lev1
        v1 = Variable({"a": 2, "b": 2}, dom1, None, True)

        dom2 = [self.dim1_lev1.alias("d1b"), self.dim1_lev2]
        v2 = Variable(
            {("a", "X"): 1, ("a", "Y"): 2, ("b", "Y"): 3, ("b", "Z"): 3},
            dom2,
            None,
            True,
        )

        self.assertRaises(Exception, partial(v2.transform, v1))
        v3 = v1.transform(v2)
        data = dict(kv for kv in v3.items() if kv[1])
        self.assertDictEqual(data, {"X": 2 * 1, "Y": 2 * 2 + 2 * 3, "Z": 2 * 3})

    def test_normalize(self):
        # check 2d only
        self.assertRaises(
            TypeError, partial(Variable({}, self.dim1_lev1, None).normalize)
        )

        dom = [self.dim1_lev1.alias("d1b"), self.dim1_lev2]
        v = Variable({("a", "X"): 1, ("b", "Z"): 2}, dom, None, True)
        v.normalize()

        # check sum zero
        self.assertRaises(ValueError, partial(v.normalize, transposed=True))

    def test_to_unit(self):
        v = Variable(10, None, "meter")
        v2 = v.to_unit("cm")
        self.assertEqual(v2.data, 1000)
        # to dimensionless hould fail
        self.assertRaises(Exception, partial(v.to_unit, None))

    def test_transform(self):
        # start with scalar
        v = Variable(10, None, None, True)

        # TODO: ugly hack: we want to test if we can create
        # a combined transform variable from None -> None,
        # but cannot use None as name twice

        none1 = None
        none2 = ""

        # auto expand via transform
        vt1 = Variable(
            {(None, "a"): 0.6, (None, "b"): 0.4},
            [self.dim1.alias(none1), self.dim1_lev1],
            None,
            True,
        )
        v1 = v.transform(vt1)
        self.assertDictEqual(dict(v1.items()), {"a": 6, "b": 4})

        # change level
        vt2 = Variable(
            {("a", "X"): 1, ("b", "Y"): 1},
            [self.dim1_lev1, self.dim1_lev2.alias("d1b")],
            None,
            True,
        )
        v2 = v1.transform(vt2)
        self.assertDictEqual(dict(v2.items()), {"X": 6, "Y": 4, "Z": 0})

        # auto squeeze
        vt3 = Variable(
            {("X", None): 0.1, ("Y", None): 0.2},
            [self.dim1_lev2.alias("d1b"), self.dim1.alias(none2)],
            None,
            True,
        )
        v3 = v2.transform(vt3)
        self.assertEqual(v3.data, 6 * 0.1 + 4 * 0.2)

        # do it in one step
        vt = vt1.transform(vt2).transform(vt3, autosqueeze=False)
        v4 = v.transform(vt)
        self.assertEqual(v3.data, v4.data)
