import unittest

from app.api_interface import ApiInterface


class TestInferenceRunner(unittest.TestCase):
    def test_that_creating_api_interface_without_parameters_fails(self):
        self.assertRaises(ApiInterface(verbosity=1, parameters=dict()), KeyError)
