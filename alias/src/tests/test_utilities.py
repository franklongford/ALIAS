from unittest import TestCase
import tempfile
import tables

import numpy as np

from alias.src.utilities import (
    unit_vector, numpy_remove,
    make_checkfile, read_checkfile, update_checkfile,
    load_npy, save_hdf5, load_hdf5, make_hdf5,
    shape_check_hdf5, molecules, bubble_sort
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

    def test_checkfile(self):

        with tempfile.NamedTemporaryFile() as tmp_file:

            make_checkfile(tmp_file.name)

            checkfile = read_checkfile(tmp_file.name)

            self.assertFalse(checkfile)

            checkfile = update_checkfile(
                tmp_file.name, 'M', [16.0, 1.008, 1.008, 0]
            )

        self.assertEqual(4, len(checkfile['M']))
        self.assertAlmostEqual(
            18.016,
            np.sum(np.array(checkfile['M']))
        )

    def test_load_npy(self):

        with tempfile.NamedTemporaryFile() as tmp_file:

            np.save(tmp_file.name, self.test_data)
            load_data = load_npy(tmp_file.name)

            self.assertTrue(np.allclose(self.test_data, load_data))

            new_test_data = self.test_data[:10]
            load_data = load_npy(tmp_file.name, frames=range(10))

            self.assertTrue(np.allclose(new_test_data, load_data))

    def test_load_save_hdf5(self):

        with tempfile.NamedTemporaryFile() as tmp_file:

            make_hdf5(tmp_file.name, self.test_data.shape, tables.Int64Atom())
            save_hdf5(tmp_file.name, self.test_data, 0)

            self.assertEqual(
                (1,) + self.test_data.shape,
                shape_check_hdf5(tmp_file.name))

            load_data = load_hdf5(tmp_file.name, 0)

            self.assertTrue(np.allclose(self.test_data, load_data))

            new_test_data = self.test_data * 20
            save_hdf5(tmp_file.name, new_test_data, 0, mode='r+')
            load_data = load_hdf5(tmp_file.name, 0)

            self.assertTrue(np.allclose(new_test_data, load_data))

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

    def test_bubble_sort(self):
        array = np.array([0, 4, 3, 2, 7, 8, 1, 5, 6])
        key = np.array([0, 6, 3, 2, 1, 7, 8, 4, 5])

        bubble_sort(array, key)
        
        self.assertTrue(np.allclose(
            np.array([0, 7,2, 3, 5, 6, 4, 8, 1]),
            array))
        self.assertTrue(np.allclose(key, np.arange(9)))
