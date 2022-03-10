import logging
import unittest

from data_disaggregation.classes import Dimension, Variable
from data_disaggregation.draw import draw_transform

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
            name="v1",
            data={
                (1, "sr1_1"): 2,
                (1, "sr1_2"): 3,
                (2, "sr1_2"): 4,
                (2, "sr2_1"): 5,
            },
            domain=[self.year_hour, self.subregion],
            vartype="extensive",
        )

        dim_steps = v1.get_transform_steps(domain=[self.region])
        res = draw_transform(dim_steps)
        self.assertEqual(type(res), bytes)