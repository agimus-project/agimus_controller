import unittest

from agimus_controller.warm_start_base import WarmStartBase


class TestWarmStartBase(unittest.TestCase):
    def test_abstract_class_instantiation(self):
        with self.assertRaises(TypeError):
            WarmStartBase()
