from unittest import mock, TestCase

from alias.tests.probe_classes import ProbeSurfaceParameters


class TestSurfaceParameters(TestCase):

    def setUp(self):

        self.parameters = ProbeSurfaceParameters()

    def test_load_traj(self):

        self.assertIsNotNone(self.parameters._traj)
        self.assertEqual(1, self.parameters._traj.n_frames)
        self.assertEqual(13, self.parameters._traj.n_residues)
        self.assertEqual(220, self.parameters._traj.n_atoms)

    def test_select_residue(self):

        with mock.patch('alias.src.surface_parameters.input',
                        return_value='TRP'):
            self.parameters.select_residue()

        self.assertEqual('TRP', self.parameters.molecule)
        self.assertListEqual(
            ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3',
             'CG', 'CD1', 'HD1', 'NE1', 'HE1', 'CE2',
             'CZ2', 'HZ2', 'CH2', 'HH2', 'CZ3', 'HZ3',
             'CE3', 'HE3', 'CD2', 'C', 'O'],
            self.parameters.atoms
        )
        self.assertEqual(96, self.parameters.n_atoms)
        self.assertEqual(4, self.parameters.n_mols)
        self.assertEqual(24, self.parameters.n_sites)

    def test_select_masses(self):

        with mock.patch('alias.src.surface_parameters.input',
                        return_value='TRP'):
            self.parameters.select_residue()

        self.assertEqual(0, len(self.parameters.masses))

        with mock.patch('alias.src.surface_parameters.input',
                        return_value='Y'):
            self.parameters.select_masses()

        self.assertEqual(
            24, len(self.parameters.masses))
        self.assertAlmostEqual(
            186.21092, self.parameters.masses.sum())