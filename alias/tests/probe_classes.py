from alias.src.surface_parameters import SurfaceParameters

from .fixtures import amber_trajectory, amber_topology


class ProbeSurfaceParameters(SurfaceParameters):

    def __init__(self, molecule='TRP'):
        super().__init__(molecule)

        self.load_traj_parameters(
            amber_trajectory, amber_topology)
        self.masses = self._standard_masses()
        self.center_atom = 'C'
        self.vector_atoms = ['N', 'O']
