import logging
import unittest
from functools import partial

import numpy as np
from numpy.testing import assert_array_equal

from data_disaggregation.classes import (
    Dimension,
    ExtensiveVariable,
    IntensiveScalar,
    Variable,
    Weight,
)
from data_disaggregation.draw import draw_transform
from data_disaggregation.exceptions import AggregationError, ProgramNotFoundError

LOGGING_DATE_FMT = "%Y-%m-%d %H:%M:%S"
LOGGING_FMT = "[%(asctime)s %(levelname)7s] %(message)s"

logging.basicConfig(format=LOGGING_FMT, datefmt=LOGGING_DATE_FMT, level=logging.DEBUG)


class TestExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time = Dimension("time")
        cls.year_hour = cls.time.add_level("year_hour", [1, 2, 3, 4, 5])
        cls.day = cls.time.add_level("day", ["mo", "di"])
        cls.day_hour = cls.day.add_level(
            "day_hour", {"mo": ["01", "02"], "di": ["03", "04", "05"]}
        )
        cls.space = Dimension("space")
        cls.region = cls.space.add_level("region", ["r1", "r2"])
        cls.subregion = cls.region.add_level(
            "subregion", {"r1": ["sr1_1", "sr1_2"], "r2": ["sr2_1"]}
        )

    def test_transform1(self):
        v1 = Variable(
            data={
                (1, "sr1_1"): 2,
                (1, "sr1_2"): 3,
                (2, "sr1_2"): 4,
                (2, "sr2_1"): 5,
            },
            domain=[self.year_hour, self.subregion],
            vartype="extensive",
        )
        v2 = v1.transform(domain=[self.region])
        assert_array_equal(v2._data_matrix, [9, 5])

        v1sum = np.sum(v1._data_matrix)
        v2sum = np.sum(v2._data_matrix)
        self.assertAlmostEqual(v1sum, v2sum)

    def test_extensive(self):
        v1 = Variable(
            data={1: 10, 2: 20, 3: 30, 4: 40, 5: 50},
            domain=self.year_hour,
            vartype="extensive",
        )
        v2 = v1.transform(self.time)  # should work

        # auto disaggregate for extensive does not work
        res = partial(v2.transform, self.year_hour)
        self.assertRaises(AggregationError, res)
        # you need weights
        v3 = v2.transform(self.year_hour, {"year_hour": v1})

        v1sum = np.sum(v1._data_matrix)
        v2sum = np.sum(v2._data_matrix)
        v3sum = np.sum(v3._data_matrix)
        self.assertAlmostEqual(v1sum, v2sum)
        self.assertAlmostEqual(v1sum, v3sum)

    def test_intensive(self):
        v1 = IntensiveScalar(10)
        v2 = v1.transform(self.day_hour)
        self.assertEqual(set([10]), set(v2.to_dict().values()))

        # auto aggregate  for extensive does not work
        res = partial(v2.transform, self.time)
        self.assertRaises(AggregationError, res)
        v3 = v2.transform(self.year_hour, {"day_hour": v1, "day": v1})

        v2sum = np.sum(v2._data_matrix)
        v3sum = np.sum(v3._data_matrix)
        self.assertAlmostEqual(v2sum, v3sum)
        self.assertAlmostEqual(v2sum, 10 * 5)

    def test_weights(self):
        # this should work
        Weight(
            data={"01": 0.8, "02": 0.2, "03": 0.5, "05": 0.5},
            dimension_level=self.day_hour,
        )

        # these should fail
        res = partial(
            Weight,
            data={"01": 1, "03": 1, "05": 1},
            dimension_level=self.day_hour,
        )
        self.assertRaises(ValueError, res)

    def test_transform_steps(self):
        v1 = Variable(
            data={
                (1, "sr1_1"): 2,
                (1, "sr1_2"): 3,
                (2, "sr1_2"): 4,
                (2, "sr2_1"): 5,
            },
            domain=[self.year_hour, self.subregion],
            vartype="extensive",
        )
        steps_dct = v1.get_transform_steps(domain=[self.region])

        steps = tuple(
            (
                dim.name,
                tuple(
                    (
                        f.name if f else None,
                        t.name if t else None,
                        a,
                        w.name if w else None,
                    )
                    for f, t, a, w in stps
                ),
            )
            for dim, stps in steps_dct.items()
        )

        self.assertEqual(
            steps,
            (
                ("space", (("subregion", "region", "aggregate", None),)),
                (
                    "time",
                    (
                        ("year_hour", "time", "aggregate", None),
                        ("time", None, "squeeze", None),
                    ),
                ),
            ),
        )

    def test_to_series(self):
        v1 = Variable(
            data={1: 10},
            domain=self.year_hour,
            vartype="extensive",
        )
        try:
            series = v1.to_series()
        except ImportError:
            return  # pandas not installed
        self.assertEqual(10, int(series[1]))

    def test_draw_get_image_bytes(self):
        v1 = Variable(
            data={
                (1, "sr1_1"): 2,
                (1, "sr1_2"): 3,
                (2, "sr1_2"): 4,
                (2, "sr2_1"): 5,
            },
            domain=[self.year_hour, self.subregion],
            vartype="extensive",
        )
        steps = v1.get_transform_steps(domain=[self.region])
        try:
            image_bytes = draw_transform(steps, filetype="svg")
        except ProgramNotFoundError:
            return  # dot not in PATH
        self.assertTrue(image_bytes.decode().startswith("<?xml"))

    def test_math(self):
        domain1 = [self.year_hour, self.region]
        v1 = ExtensiveVariable(
            {
                (1, "r1"): 1,
                (1, "r2"): 2,
            },
            domain1,
        )
        v2 = ExtensiveVariable(
            {
                (1, "r1"): 3,
                (2, "r2"): 4,
            },
            domain1,
        )

        # test add
        v_add = v1 + v2
        self.assertDictEqual(
            v_add.to_dict(skip_0=True),
            {
                (1, "r1"): 4,
                (1, "r2"): 2,
                (2, "r2"): 4,
            },
        )

        # test subtract
        v_sub = v1 - v2
        self.assertDictEqual(
            v_sub.to_dict(skip_0=True),
            {
                (1, "r1"): -2,
                (1, "r2"): 2,
                (2, "r2"): -4,
            },
        )

        # test mult
        v_mult = v1 * v2
        self.assertDictEqual(v_mult.to_dict(skip_0=True), {(1, "r1"): 3})

        # test truediv
        # NOTE: division by 0 creates na, so we use fillna
        v_div = v_mult / v2
        self.assertDictEqual(v_div.fillna().to_dict(skip_0=True), {(1, "r1"): 1})

        # test neg
        v_neg = -v_mult
        self.assertDictEqual(v_neg.to_dict(skip_0=True), {(1, "r1"): -3})

        # ----------------------------------------------
        # test scalar
        # ----------------------------------------------

        v_add = v1 + 1
        self.assertDictEqual(
            v_add.to_dict(skip_0=True),
            {
                (1, "r1"): 2,
                (1, "r2"): 3,
                (2, "r1"): 1,
                (2, "r2"): 1,
                (3, "r1"): 1,
                (3, "r2"): 1,
                (4, "r1"): 1,
                (4, "r2"): 1,
                (5, "r1"): 1,
                (5, "r2"): 1,
            },
        )

        # rmul

        v_rmul = 2 * v1
        self.assertDictEqual(v_rmul.to_dict(skip_0=True), {(1, "r1"): 2, (1, "r2"): 4})

        # rdiv
        # NOTE: will create inf for x/0

        rdiv = 12 / (v1 + 1) - 12
        self.assertDictEqual(
            rdiv.to_dict(skip_0=True),
            {(1, "r1"): -6, (1, "r2"): -8},  # 12 / (1 + 1) - 12  # 12 / (2 + 1) - 12
        )
