import numpy as np

from alias.src.wave_function import (
    check_uv,
    wave_function,
    d_wave_function,
    dd_wave_function,
    cos_sin_indices,
    wave_arrays
)
from alias.tests.alias_test_case import AliasTestCase


class TestWaveFunction(AliasTestCase):

    def setUp(self):

        self.lx = 10

    def test_check_uv(self):

        self.assertEqual(4, check_uv(0, 0))
        self.assertEqual(2, check_uv(1, 0))
        self.assertEqual(2, check_uv(0, 1))
        self.assertEqual(1, check_uv(1, 1))

    def test_wave_function(self):

        self.assertEqual(
            1.0, wave_function(0, 0, self.lx))
        self.assertEqual(
            1.0, wave_function(0, self.lx, self.lx))
        self.assertEqual(
            1.0, wave_function(self.lx, 0, self.lx))
        self.assertEqual(
            1.0, wave_function(self.lx, self.lx, self.lx))

        self.assertEqual(
            -1.0, wave_function(1, self.lx / 2, self.lx))
        self.assertEqual(
            1.0, wave_function(2, self.lx / 2, self.lx))

    def test_d_wave_function(self):

        self.assertAlmostEqual(
            0.0, d_wave_function(0, 0, self.lx))
        self.assertAlmostEqual(
            0.0, d_wave_function(0, self.lx, self.lx))
        self.assertAlmostEqual(
            0.0, d_wave_function(self.lx, 0, self.lx))
        self.assertAlmostEqual(
            0.0, d_wave_function(self.lx, self.lx, self.lx))

        self.assertAlmostEqual(
            0.0, d_wave_function(1, self.lx / 2, self.lx))
        self.assertAlmostEqual(
            0.0, d_wave_function(2, self.lx / 2, self.lx))

        self.assertAlmostEqual(
            - np.pi / 2, d_wave_function(1, self.lx / 4, self.lx))
        self.assertAlmostEqual(
            np.pi / 2, d_wave_function(3, self.lx / 4, self.lx))

    def test_dd_wave_function(self):

        self.assertAlmostEqual(
            0.0, dd_wave_function(0, 0, self.lx))
        self.assertAlmostEqual(
            0.0, dd_wave_function(self.lx, 0, self.lx))

        self.assertAlmostEqual(
            0.0, dd_wave_function(self.lx / 2, 0.5, self.lx))

    def test_wave_indices(self):

        u_array = np.arange(-4, 5)

        cos_indices, sin_indices = cos_sin_indices(u_array)
        self.assertListEqual(
            [[0], [1], [2], [3]], sin_indices.tolist())
        self.assertListEqual(
            [[4], [5], [6], [7], [8]], cos_indices.tolist())

    def test_wave_arrays(self):

        u_array, v_array = wave_arrays(1)
        self.assertArrayAlmostEqual(
            np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]),
            u_array
        )
        self.assertArrayAlmostEqual(
            np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1]),
            v_array
        )

        u_array, v_array = wave_arrays(2)
        self.assertEqual((25,), u_array.shape)
        self.assertEqual((25,), v_array.shape)
