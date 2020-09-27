import logging

import numpy as np

from alias.io.checkfile_io import load_checkfile
from alias.io.command_line_input import (
    ask_question,
    instruction
)
from alias.src.pivot_density import optimise_pivot_diffusion
from alias.src.utilities import load_traj_frame

log = logging.getLogger(__name__)


class SurfaceParameters:

    #: Attributes to be serialised as JSON
    json_attributes = [
        'molecule', 'mol_sigma', 'masses', 'com_mode',
        'com_sites', 'center_atom', 'vector_atoms',
        'pivot_density', 'n_frames', 'cell_dim', 'v_lim',
        'n_cube', 'tau', 'max_r', 'phi', 'recon']

    def __init__(self, molecule=None, mol_sigma=None, masses=None, v_lim=3,
                 n_cube=3, tau=0.5, max_r=1.5, phi=5E-8, com_mode='molecule',
                 com_sites=None, center_atom=None, vector_atoms=None,
                 n_frames=None, pivot_density=None, cell_dim=None,
                 recon=False):
        """Initialise parameters for a Intrinsic surface

        Parameters
        ----------
        molecule: str
            Symbol in trajectory file representing molecular
            residue to build surface from
        atoms: list of str
            List of symbols representing atoms in molecule
            to calculate centre of mass from.
        """

        self.molecule = molecule

        if masses is None:
            self.masses = []
        else:
            self.masses = masses

        self.mol_sigma = mol_sigma
        self.v_lim = v_lim
        self.n_cube = n_cube
        self.tau = tau
        self.max_r = max_r
        self.phi = phi

        self.com_mode = com_mode
        if com_sites is None:
            self.com_sites = []
        else:
            self.com_sites = com_sites

        self.center_atom = center_atom
        if vector_atoms is None:
            self.vector_atoms = []
        else:
            self.vector_atoms = vector_atoms

        self.pivot_density = pivot_density
        self.n_frames = n_frames
        self.cell_dim = cell_dim
        self.recon = recon

        self._traj = None

    @property
    def atoms(self):
        """List of atoms in residue"""
        return [atom.name for atom in self._residue.atoms]

    @property
    def atom_indices(self):
        """List of indices in trajectory that refer to atoms in residue"""
        return [
            atom.index for atom in self._traj.topology.atoms
            if (atom.residue.name == self.molecule)
        ]

    @property
    def mol_indices(self):
        """List of indices in trajectory that refer to atoms in residue"""
        return [
            molecule.index for molecule in self._traj.topology.residues
            if (molecule.name == self.molecule)
        ]

    @property
    def n_atoms(self):
        """Number of atoms in trajectory assigned to residue"""
        return len(self.atom_indices)

    @property
    def n_mols(self):
        """Number of molecules in each trajectory to be included
        in the surface"""
        return len(self.mol_indices)

    @property
    def n_sites(self):
        """List of atoms in each residue"""
        return self._residue.n_atoms

    @property
    def _residue(self):
        index = self.mol_indices[0]
        return self._traj.topology.residue(index)

    @property
    def area(self):
        """Area of xy plane in simulation cell """
        return self.cell_dim[0] * self.cell_dim[1]

    @property
    def l_slice(self):
        """Length of each slice in histogram distribution
        along the z axis"""
        return 0.05 * self.mol_sigma

    @property
    def n_slice(self):
        """Number of bins in histogram distribution along
        z axis"""
        return self.cell_dim[2] / self.l_slice

    @property
    def q_max(self):
        """Maximum frequency in Fourier sum representing
        intrinsic surface"""
        return 2 * np.pi / self.mol_sigma

    @property
    def q_min(self):
        """Minimum frequency in Fourier sum representing
        intrinsic surface"""
        return 2 * np.pi / self.area

    @property
    def q_m(self):
        """Maximum number of wave frequencies allowed in
        Fourier sum representing intrinsic surface"""
        return int(self.q_max / self.q_min)

    @property
    def n_waves(self):
        """Number of waves in each dimension of Fourier sum
        representing intrinsic surface"""
        return 2 * self.q_m + 1

    @property
    def n_pivots(self):
        """Number of pivot molecules used to fit intrinsic
        surface"""
        return int(self.area * self.pivot_density / self.mol_sigma ** 2)

    def _standard_masses(self):
        """Return standard masses for each element from mdtraj
         library"""
        return [
            atom.element.mass
            for atom in self._residue.atoms
        ]

    def _infer_lipid_vectors(self):

        if self.n_atoms == 134:
            self.center_atom = 'P'
            self.vector_atoms = ['C218', 'C316']

        elif self.n_atoms == 138:
            self.center_atom = 'P'
            self.vector_atoms = ['C218', 'C318']

        else:
            self.center_atom = 'PO4'
            self.vector_atoms = ['C4A', 'C4B']

    def wavelength(self, q_u):
        """Wavelength of given integer upper bound frequency"""
        return self.q_max / (q_u * self.q_min)

    def load_traj_parameters(self, trajectory, topology=None):
        """Load in and parse trajectory details

        Parameters
        ---------
        trajectory: str
            File path of trajectory file to load
        topology: str, optional
            File path of topology file to load if required
        """
        self._traj = load_traj_frame(trajectory, topology)

    def select_residue(self):
        """Select molecular residue for building surface"""

        if self.molecule:
            if ask_question(
                    f"Use current residue? {self.molecule}"):
                return

        residues = [
            molecule.name
            for molecule in self._traj.topology.residues]
        set_residues = set(residues)

        print("List of residues found: {}".format(set_residues))

        if len(set_residues) > 1:
            self.molecule = instruction(
                "Choose residue to use for surface identification")
        else:
            self.molecule = residues[0].name

        log.info(f"Using residue {self.molecule} "
                 f"for surface identification")

    def select_masses(self):
        """Select masses for each site in chosen molecule"""

        if len(self.masses) == self.n_sites:
            if ask_question(
                    f"Use current elemental masses? {self.masses} g mol-1"):
                return

        if ask_question("Use standard elemental masses?"):
            self.masses = self._standard_masses()
        else:
            self.masses = []
            for index in range(self.n_sites):
                self.masses.append(float(
                    instruction(f"  Enter mass for site "
                                f"{self.atoms[index]} g mol-1")
                ))

        print(f"Using atomic site masses: {self.masses} g mol-1")
        print(f"Molar mass: {sum(self.masses)}  g mol-1")

    def select_center_of_mass(self):
        """Select method to calculate center of mass for chosen
        molecule"""

        if self.com_mode is not None:
            if ask_question(
                    "Use current centre of molecular mass? "
                    f"{self.com_mode}: {self.com_sites}"):
                return

        if ask_question("Use atomic sites as centre of molecular mass?"):
            print(f"Atomic sites: {self.atoms}")
            self.com_mode = 'sites'
            self.com_sites = instruction("  Site names: ").split()
        else:
            self.com_mode = 'molecule'

    def select_orientation_vector(self):

        if self.n_sites < 2:
            self.center_atom = None
            self.vector_atoms = []
            return

        elif self.n_sites > 1:
            response = input("Measure molecular orientation? (Y/N):")
            if response.upper() != 'Y':
                self.center_atom = None
                self.vector_atoms = []
                return

        print(f"Atomic sites: {self.atoms}")

        response = input(
            f"\nUse current orientational parameters? "
            f"{self.center_atom}: {self.vector_atoms} (Y/N): ")
        if response.upper() == 'Y':
            return

        response = input(
            'Use atomic site for center of mass? (Y/N): ')

        if response.upper() == 'Y':
            self.center_atom = input(" Site name: ")
        else:
            self.center_atom = 'molecule'

        self.vector_atoms = input("  Site names: ").split()

    def select_mol_sigma(self):
        """Select radius for pivot molecule"""
        if self.mol_sigma is not None:
            if ask_question(
                    "Use current molecular radius? "
                    f"{self.mol_sigma} Angstroms"):
                return

        self.mol_sigma = float(
            instruction("Enter molecular radius: (Angstroms)"))

    def select_pivot_density(self, file_name, data_dir):

        if self.pivot_density is not None:
            response = input(
                f"\nUse surface pivot number found in checkfile? "
                f"{self.n_pivots} pivots (Y/N): ")

            if response.upper() == 'Y':
                return

        response = input(
            f"\nManually enter in new surface pivot number? "
            f"(search will commence otherwise) (Y/N): ")

        if response.upper() == 'Y':
            n_pivots = int(
                input("\nEnter number of surface pivots: "))
            self.pivot_density = n_pivots * self.mol_sigma ** 2 / self.area
        else:
            print("\n-------OPTIMISING SURFACE DENSITY-------\n")
            optimise_pivot_diffusion(file_name, data_dir, self)

    def serialize(self):
        """Convert state of SurfaceParameters instance to a
        dictionary suitable for a JSON file dump"""

        state = {
            attr: getattr(self, attr)
            for attr in self.json_attributes
        }

        return state

    @classmethod
    def from_json(cls, file_path):
        """Create instance of SurfaceParameters from a
        suitable JSON file"""

        parameters = load_checkfile(file_path)

        return cls(**parameters)
