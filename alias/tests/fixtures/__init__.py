import os


path = os.path.dirname(__file__)

gromacs_coordinate = os.path.join(path, 'gromacs_coordinate.gro')
gromacs_trajectory = os.path.join(path, 'gromacs_trajectory.nx')
amber_trajectory = os.path.join(path, 'amber_trajectory.nc')
amber_topology = os.path.join(path, 'amber_topology.prmtop')
