from unittest import TestCase
from pycomplexnet import Network
import pandas as pd

class TestConstructors(TestCase):
    def test_from_pandas(self):
        net = Network.from_pandas(
            pd.DataFrame({
                'from':['A','B','C'],
                'to':['B','C','B']
            })
        )
        self.assertEqual(net.node_labels,{0: 'A', 1: 'B', 2: 'C'})
        # self.assertTrue(np.array_equal(
        #     net.toarray(),
        #     np.array(
        #         [[0., 0., 1.],
        #          [1., 0., 0.],
        #          [1., 0., 0.]]
        #     )
        # ))
        # self.assertTrue(net.node_labels=={0: 'B', 1: 'A', 2: 'C'})