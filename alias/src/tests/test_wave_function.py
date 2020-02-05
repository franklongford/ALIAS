from unittest import TestCase

import numpy as np

from alias.src.wave_function import (
    check_uv,
    wave_function,
    d_wave_function,
    dd_wave_function,
    wave_indices
)


class TestWaveFunction(TestCase):

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

        cos_indices, sin_indices = wave_indices(u_array)
        self.assertListEqual(
            [[0], [1], [2], [3]], sin_indices.tolist())
        self.assertListEqual(
            [[4], [5], [6], [7], [8]], cos_indices.tolist())

