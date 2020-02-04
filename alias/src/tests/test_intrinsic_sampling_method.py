from unittest import TestCase

import numpy as np

from alias.src.intrinsic_surface import xi, dxy_dxi, ddxy_ddxi


class TestISM(TestCase):

    def setUp(self):
        self.qm = 8
        self.qu = 5
        self.n_waves = 2 * self.qm + 1
        self.coeff = np.ones(self.n_waves ** 2) * 0.5
        self.dim = [10., 12.]

        self.pos = np.arange(10)
        self.u_array = np.arange(-self.qm, self.qm + 1)

    def test_xi(self):

        xi_array = xi(
            self.pos, self.pos, self.coeff,
            self.qm, self.qu, self.dim
        )

        self.assertEqual((10,), xi_array.shape)

        for index, x in enumerate(self.pos):
            array = xi(x, x, self.coeff, self.qm, self.qu, self.dim)
            self.assertTrue(np.allclose(array, xi_array[index]))

    def test_dxy_dxi(self):

        dx_dxi_array, dy_dxi_array = dxy_dxi(
            self.pos, self.pos, self.coeff, self.qm, self.qu, self.dim
        )

        self.assertEqual((10,), dx_dxi_array.shape)
        self.assertEqual((10,), dy_dxi_array.shape)

        for index, x in enumerate(self.pos):
            dx_dxi, dy_dxi = dxy_dxi(
                x, x, self.coeff, self.qm, self.qu, self.dim)

            self.assertTrue(np.allclose(dx_dxi, dx_dxi_array[index]))
            self.assertTrue(np.allclose(dy_dxi, dy_dxi_array[index]))

    def test_ddxy_ddxi(self):

        ddx_ddxi_array, ddy_ddxi_array = ddxy_ddxi(
            self.pos, self.pos, self.coeff, self.qm, self.qu, self.dim
        )

        self.assertEqual((10,), ddx_ddxi_array.shape)
        self.assertEqual((10,), ddy_ddxi_array.shape)

        for index, x in enumerate(self.pos):
            ddx_ddxi, ddy_ddxi = ddxy_ddxi(
                x, x, self.coeff, self.qm, self.qu, self.dim)

            self.assertTrue(np.allclose(ddx_ddxi, ddx_ddxi_array[index]))
            self.assertTrue(np.allclose(ddy_ddxi, ddy_ddxi_array[index]))
