import unittest


# coding: utf-8
import unittest
import logging

logging.basicConfig(
    format="[%(asctime)s %(levelname)7s] %(message)s", level=logging.INFO
)


class TestTemplate(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    # EXAMPLE
    def test_TEMPLATE(self):
        pass
