from unittest import TestCase

import numpy as np

from alias.src.positions import (
    molecules, molecular_positions
)


class TestPositions(TestCase):

    def setUp(self):

        self.simple_coord = np.array([[0, 0, 0],
                                      [1, 1, 1],
                                      [4, 4, 4],
                                      [5, 5, 5],
                                      [2, 0, 2]])
        self.simple_masses = np.array([1, 2, 1, 2])

        self.large_coord = np.array(
            [[2.741, 7.518, 3.306], [3.075, 7.604, 3.104],
             [3.410, 7.690, 2.901], [3.744, 7.775, 2.699],
             [2.516, 0.583, 1.985], [2.551, 0.953, 2.135],
             [2.586, 1.322, 2.285], [2.621, 1.691, 2.435],
             [6.715, 2.014, 3.789], [6.999, 1.741, 3.721],
             [7.283, 1.467, 3.654], [7.567, 1.194, 3.587]])
        self.large_masses = np.array(
            [10, 5, 5, 5] * 3
        )

    def test_simple_molecular_positions(self):

        coord = self.simple_coord[:-1]

        molecules = molecular_positions(
            coord, 2, self.simple_masses)

        self.assertEqual((2, 3), molecules.shape)
        self.assertTrue(
            np.allclose(np.array([[0.666667, 0.666667, 0.666667],
                                  [4.666667, 4.666667, 4.666667]]),
                        molecules)
        )

        molecules = molecular_positions(
            coord, 2, self.simple_masses, mode='sites',
            com_sites=0)

        self.assertEqual((2, 3), molecules.shape)
        self.assertTrue(
            np.allclose(np.array([[0, 0, 0],
                                  [4, 4, 4]]),
                        molecules)
        )

    def test_large_molecular_positions(self):

        # Test centre of mass for whole molecule
        molecules = molecular_positions(
            self.large_coord, 4, self.large_masses)
        self.assertEqual((3, 3), molecules.shape)
        self.assertTrue(
            np.allclose(np.array([[3.1422, 7.621, 3.0632],
                                  [2.558, 1.0264, 2.165],
                                  [7.0558, 1.686, 3.708]]),
                        molecules)
        )

        # Test include atom as site
        molecules = molecular_positions(
            self.large_coord, 4, self.large_masses,
            mode='sites', com_sites=0
        )
        self.assertEqual((3, 3), molecules.shape)
        self.assertTrue(
            np.allclose(np.array([[2.741, 7.518, 3.306],
                                  [2.516, 0.583, 1.985],
                                  [6.715, 2.014, 3.789]]),
                        molecules)
        )

        # Test centre of mass for first 3 atoms
        molecules = molecular_positions(
            self.large_coord, 4, self.large_masses,
            mode='sites', com_sites=[0, 1, 2]
        )
        self.assertEqual((3, 3), molecules.shape)
        self.assertTrue(
            np.allclose(np.array([[2.99175, 7.5825, 3.15425],
                                  [2.54225, 0.86025, 2.0975],
                                  [6.928, 1.809, 3.73825]]),
                        molecules)
        )

    def test_invalid_mode(self):

        with self.assertRaisesRegex(
                AssertionError,
                "Argument mode==invalid must be either"
                " 'molecule' or 'sites'"):
            molecular_positions(
                self.simple_coord, 2, self.simple_masses,
                mode='invalid')

    def test_invalid_com_sites(self):

        with self.assertRaisesRegex(
                AssertionError,
                "Argument com_sites must have a length "
                r"\(3\) less than n_sites \(2\)"):
            molecular_positions(
                self.simple_coord, 2, self.simple_masses,
                mode='sites', com_sites=[0, 1, 2])

    def test_molecules(self):
        xat = np.array(
            [20.3155606, 20.3657056, 19.7335474,
             20.2454104, 23.1171728, 23.0142095,
             23.7594160, 23.1883006])
        yat = np.array(
            [29.0287238, 28.0910350, 29.3759130,
             28.9508404, 35.2457050, 34.8579738,
             34.6865613, 35.1208178])
        zat = np.array(
            [58.6756206, 58.8612466, 59.3516029,
             58.7892616, 63.1022910, 63.9713681,
             62.6651254, 63.1592576])
        mol_M = np.array(
            [16.0, 1.008, 1.008,
             0.0, 16.0, 1.008,
             1.008, 0.0])

        answer_xmol = np.array([20.3155606, 23.1171728])
        answer_ymol = np.array([29.0287238, 35.2457050])
        answer_zmol = np.array([58.6756206, 63.1022910])

        xmol, ymol, zmol = molecules(xat, yat, zat, 2, 4, mol_M, [0])

        self.assertEqual((2,), xmol.shape)
        self.assertEqual((2,), ymol.shape)
        self.assertEqual((2,), zmol.shape)

        self.assertTrue(np.allclose(xmol, answer_xmol))
        self.assertTrue(np.allclose(ymol, answer_ymol))
        self.assertTrue(np.allclose(zmol, answer_zmol))

        answer_xmol = np.array([20.28580243, 23.14734565])
        answer_ymol = np.array([28.99568519, 35.19272710])
        answer_zmol = np.array([58.72382781, 63.12645656])

        xmol, ymol, zmol = molecules(xat, yat, zat, 2, 4, mol_M, 'COM')

        self.assertTrue(np.allclose(xmol, answer_xmol))
        self.assertTrue(np.allclose(ymol, answer_ymol))
        self.assertTrue(np.allclose(zmol, answer_zmol))
