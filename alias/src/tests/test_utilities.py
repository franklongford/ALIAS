from unittest import TestCase

import numpy as np

from alias.src.utilities import (
    unit_vector, numpy_remove,
    bubble_sort
)


class TestUtilities(TestCase):

    def setUp(self):
        self.test_data = np.arange(50)

    def test_unit_vector(self):

        vector = [-3, 2, 6]

        self.assertTrue(
            np.allclose(
                np.array([-0.42857143, 0.28571429, 0.85714286]),
                unit_vector(vector)
            )
        )

        vector_array = [[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]]

        u_vector_array = unit_vector(vector_array)

        self.assertEqual((4, 3), u_vector_array.shape)

    def test_numpy_remove(self):

        self.assertTrue(
            np.allclose(
                np.arange(20),
                numpy_remove(self.test_data, self.test_data + 20)
            )
        )

    def test_bubble_sort(self):
        array = np.array([0, 4, 3, 2, 7, 8, 1, 5, 6])
        key = np.array([0, 6, 3, 2, 1, 7, 8, 4, 5])

        bubble_sort(array, key)
        
        self.assertTrue(np.allclose(
            np.array([0, 7,2, 3, 5, 6, 4, 8, 1]),
            array))
        self.assertTrue(np.allclose(key, np.arange(9)))
