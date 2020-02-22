import numpy as np
import mdtraj as md

from alias.src.positions import (
    molecular_positions,
    minimum_image,
    coordinate_arrays,
    orientation,
    batch_coordinate_loader
)
from alias.tests.alias_test_case import AliasTestCase
from alias.tests.fixtures import (
    amber_topology,
    amber_trajectory
)
from alias.tests.probe_classes import ProbeSurfaceParameters


class TestPositions(AliasTestCase):

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

        self.cell_dim = np.array([8., 8., 8.])
        self.parameters = ProbeSurfaceParameters()
        self.traj = md.load(
            amber_trajectory, top=amber_topology)

    def test_coordinate_arrays(self):

        traj = self.traj.atom_slice(self.parameters.atom_indices)[:5]
        masses = np.repeat(self.parameters.masses, self.parameters.n_mols)

        mol_coord = coordinate_arrays(
            traj, self.parameters.atoms, masses)

        self.assertEqual((5, 4, 3), mol_coord.shape)

        mol_coord = coordinate_arrays(
            traj, self.parameters.atoms, masses,
            mode='sites', com_sites=['N'])

        self.assertEqual((5, 4, 3), mol_coord.shape)

        mol_traj = coordinate_arrays(
            traj, self.parameters.atoms, masses,
            mode='sites', com_sites=['N', 'H'])

        self.assertEqual((5, 4, 3), mol_traj.shape)

    def test_orientation(self):

        traj = self.traj.atom_slice(self.parameters.atom_indices)[:5]

        mol_vec = orientation(
            traj, self.parameters.center_atom,
            self.parameters.vector_atoms)

        self.assertEqual((5, 4, 3), mol_vec.shape)

    def test_batch_coordinate_loader(self):

        (mol_traj, com_traj,
         cell_dim, mol_vec) = batch_coordinate_loader(
            amber_trajectory, self.parameters,
            topology=amber_topology)

        self.assertEqual((10, 4, 3), mol_traj.shape)
        self.assertEqual((10, 4, 3), mol_vec.shape)
        self.assertEqual((10, 3), cell_dim.shape)
        self.assertEqual((10, 3), com_traj.shape)

    def test_simple_molecular_positions(self):

        coord = self.simple_coord[:-1]

        molecules = molecular_positions(
            coord, ['A', 'B'], self.simple_masses,)

        self.assertEqual((2, 3), molecules.shape)
        self.assertTrue(
            np.allclose(np.array([[0.666667, 0.666667, 0.666667],
                                  [4.666667, 4.666667, 4.666667]]),
                        molecules)
        )

        molecules = molecular_positions(
            coord, ['A', 'B'], self.simple_masses,
            mode='sites', com_sites='A')

        self.assertEqual((2, 3), molecules.shape)
        self.assertTrue(
            np.allclose(np.array([[0, 0, 0],
                                  [4, 4, 4]]),
                        molecules)
        )

    def test_large_molecular_positions(self):

        # Test centre of mass for whole molecule
        molecules = molecular_positions(
            self.large_coord, ['A', 'B', 'C', 'D'],
            self.large_masses)
        self.assertEqual((3, 3), molecules.shape)
        self.assertTrue(
            np.allclose(np.array([[3.1422, 7.621, 3.0632],
                                  [2.558, 1.0264, 2.165],
                                  [7.0558, 1.686, 3.708]]),
                        molecules)
        )

        # Test include atom as site
        molecules = molecular_positions(
            self.large_coord, ['A', 'B', 'C', 'D'],
            self.large_masses, mode='sites',
            com_sites='A'
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
            self.large_coord, ['A', 'B', 'C', 'D'],
            self.large_masses, mode='sites',
            com_sites=['A', 'B', 'C']
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
                self.simple_coord, self.parameters.atoms,
                self.simple_masses,
                mode='invalid')

    def test_invalid_com_sites(self):

        com_sites = ['C'] * (len(self.parameters.atoms) + 1)

        with self.assertRaisesRegex(
                AssertionError,
                "Argument com_sites must have a length "
                r"\(25\) less than n_sites \(24\)"):
            molecular_positions(
                self.simple_coord, self.parameters.atoms,
                self.simple_masses,
                mode='sites',
                com_sites=com_sites
            )

    def test_minimum_image(self):
        d_coord = np.array([[[0, 0, 0],
                             [1, 7, 1]],
                            [[-1, -7, -1],
                             [0, 0, 0]]], dtype=float)

        minimum_image(d_coord, self.cell_dim)

        self.assertTrue(np.allclose(
            np.array([[[0, 0, 0],
                       [1, -1, 1]],
                      [[-1, 1, -1],
                       [0, 0, 0]]]), d_coord)
        )

        with self.assertRaises(AssertionError):
            minimum_image(d_coord, self.cell_dim[:2])
