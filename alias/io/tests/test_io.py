from unittest import TestCase
import tempfile
import tables

import numpy as np

from alias.io.hdf5_io import (
    make_hdf5, load_hdf5, save_hdf5, shape_check_hdf5)
from alias.io.numpy_io import load_npy
from alias.io.checkfile_io import (
    make_checkfile, load_checkfile, update_checkfile)


class TestIO(TestCase):

    def setUp(self):
        self.test_data = np.arange(50)

    def test_checkfile(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            make_checkfile(tmp_file.name)

            checkfile = load_checkfile(tmp_file.name)

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
