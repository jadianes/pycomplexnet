from unittest import TestCase
from pycomplexnet import generators

class TestGenerator(TestCase):
    def test_is_disconnected(self):
        net = generators.disconnected_network(100)
        self.assertTrue(net.get_in_degrees().sum()==0.0)
        self.assertTrue(net.get_out_degrees().sum()==0.0)
