import unittest

from agimus_controller.ocp_base import OCPBase


class TestOCPBase(unittest.TestCase):
    def test_abstract_class_instantiation(self):
        with self.assertRaises(TypeError):
            OCPBase()


if __name__ == "__main__":
    unittest.main()
