from unittest import TestCase

import numpy as np


class AliasTestCase(TestCase):

    def assertArrayAlmostEqual(self, array1, array2):
        return self.assertTrue(
            np.allclose(array1, array2))
