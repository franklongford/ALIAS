from unittest import TestCase

import numpy as np

from alias.src.pivot_density import mol_exchange


class TestPivotDensity(TestCase):

    def setUp(self):

        pass

    def test_mol_exchange(self):
        pivots_1 = np.array([
           [1, 2, 3, 4], [1, 3, 4, 5], [1, 2, 3, 5]
        ])
        pivots_2 = np.array([
            [1, 5, 6, 7], [2, 6, 7, 8], [1, 4, 3, 7]
        ])

        ex_1, ex_2 = mol_exchange(pivots_1, pivots_2)

        self.assertEqual(0.25, ex_1)
        self.assertEqual(0.625, ex_2)
