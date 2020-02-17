import numpy as np

from alias.src.intrinsic_analysis import (
    coeff_slice
)
from alias.src.surface_reconstruction import (
    H_xy, H_var_mol)
from alias.tests.alias_test_case import AliasTestCase


class TestIA(AliasTestCase):

    def setUp(self):
        self.qm = 8
        self.qu = 5
        self.n_waves = 2 * self.qm + 1
        self.coeff = np.ones(self.n_waves ** 2) * 0.5
        self.dim = [10., 12.]

        self.pos = np.arange(10)
        self.u_array = np.arange(-self.qm, self.qm + 1)

    def test_H_xy(self):

        H_array = H_xy(
            self.pos, self.pos, self.coeff, self.qm, self.qu, self.dim
        )

        self.assertEqual((10,), H_array.shape)

        for index, x in enumerate(self.pos):
            H = H_xy(x, x, self.coeff, self.qm, self.qu, self.dim)

            self.assertArrayAlmostEqual(H, H_array[index])

    def test_H_var_mol(self):

        pos = np.arange(100)
        H_array = H_xy(
            pos, pos, self.coeff, self.qm, self.qu, self.dim)

        H_var = H_var_mol(
            pos, pos, self.coeff, self.qm, self.qu, self.dim)

        self.assertTrue(np.allclose(H_var, np.var(H_array), 0.07))

    def test_coeff_slice(self):
        qm = 5
        qu = 3

        n_waves_qm = 2 * qm + 1
        n_waves_qu = 2 * qu + 1

        u_array_qm = np.array(np.arange(n_waves_qm ** 2) / n_waves_qm, dtype=int) - qm
        v_array_qm = np.array(np.arange(n_waves_qm ** 2) % n_waves_qm, dtype=int) - qm

        u_array_qu = np.array(np.arange(n_waves_qu ** 2) / n_waves_qu, dtype=int) - qu
        v_array_qu = np.array(np.arange(n_waves_qu ** 2) % n_waves_qu, dtype=int) - qu

        q_array_qm = np.sqrt(u_array_qm ** 2 + v_array_qm ** 2)
        q_array_qu = np.sqrt(u_array_qu ** 2 + v_array_qu ** 2)

        q_array_qu_2 = coeff_slice(q_array_qm, qm, qu)

        self.assertTrue(np.allclose(q_array_qu_2, q_array_qu))
