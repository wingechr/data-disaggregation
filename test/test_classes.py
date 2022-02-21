import unittest
from functools import partial

from data_disaggregation.classes import Dimension, Domain, Variable
from data_disaggregation.exceptions import DimensionStructureError, DuplicateNameError

try:
    import pandas as pd
except ImportError:
    pd = None


class TestClasses(unittest.TestCase):
    def test_dimension(self):
        # dimensions can have same name (although, why?)
        d = Dimension(name="D")
        Dimension(name="D")

        # dimension levels must be unique in a dimension
        res = partial(d.add_level, name="D", grouped_elements=["d1", "d2"])
        self.assertRaises(DuplicateNameError, res)

        # elements must be grouped ...
        d1a = d.add_level(name="D1a", grouped_elements={None: ["d1", "d2"]})
        # ... in first level: can also just be a group
        d.add_level(name="D1b", grouped_elements=["d1", "d2"])
        if pd:
            d.add_level(name="D1c", grouped_elements=pd.Series(["d1", "d2"]))

        # elements must be unique and hashable
        res = partial(d.add_level, name="D1", grouped_elements=["d1", "d1"])
        self.assertRaises(DuplicateNameError, res)
        res = partial(d.add_level, name="D1", grouped_elements=[["d1"]])
        self.assertRaises(TypeError, res)

        # all parents must have at least one child
        d2a = d1a.add_level(
            name="D2a", grouped_elements={"d1": ["d11", "d12"], "d2": ["d21"]}
        )

        res = partial(
            d1a.add_level, name="D2b", grouped_elements={"d1": ["d11"], "d2": []}
        )
        self.assertRaises(DimensionStructureError, res)
        res = partial(d1a.add_level, name="D2b", grouped_elements={"d1": ["d11"]})
        self.assertRaises(DimensionStructureError, res)
        res = partial(d1a.add_level, name="D2b", grouped_elements={"d3": ["d31"]})
        self.assertRaises(DimensionStructureError, res)

        # check path
        self.assertEqual(tuple(d2a.path), ("D", "D1a", "D2a"))

    def test_domain(self):
        # cannot duplicate dimensions
        d1 = Dimension(name="D1")
        d2 = Dimension(name="D2")
        d1a = d1.add_level(name="D1a", grouped_elements=["d11", "d12"])
        d2a = d2.add_level(name="D2a", grouped_elements=["d21", "d22"])

        res = partial(Domain, [d1, d1a])
        self.assertRaises(DuplicateNameError, res)

        dom = Domain([d1a, d2a])
        # get keys
        self.assertEqual(
            tuple(dom.keys),
            (("d11", "d21"), ("d11", "d22"), ("d12", "d21"), ("d12", "d22")),
        )

    def test_variable(self):
        # cannot duplicate dimensions
        d1 = Dimension(name="D1")
        d2 = Dimension(name="D2")
        d1a = d1.add_level(name="D1a", grouped_elements=["d11", "d12"])
        d2a = d2.add_level(name="D2a", grouped_elements=["d21", "d22"])

        dom = Domain([d1a, d2a])
        var = Variable(name="v", data=1, domain=None, vartype="extensive")
        # TODO ... test case
        assert dom
        assert var
