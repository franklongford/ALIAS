import os
import sys

import mdtraj as md
import numpy as np

from alias.src.utilities import unit_vector


def molecular_positions(
        atom_coord,
        n_site,
        masses,
        mode='molecule',
        com_sites=None):
    """
    Returns XYZ array of molecular positions from array of atoms

    Parameters
    ----------
    atom_coord:  array_like of floats
        Positions of particles in 3 dimensions
    n_site:  int
        Number of atomic sites per molecule
    masses:  array_like of float
        Masses of all atomic sites in g mol-1
    mode: str, optional, default: 'molecule'
        Mode of calculation, either 'molecule' or 'sites':
        if `molecule`, molecular centre of mass is used. Otherwise, if
        'sites', only atoms with corresponding indices given by com_sites
        are used.
    com_sites: int or list of int, optional
        List of atomic sites to use in center of mass calculation

    Returns
    -------
    mol_coord:  array_like of floats
        Positions of molecules in 3 dimensions
    """

    # Calculate the expected number of molecules in mol_coord
    n_mol = atom_coord.shape[0] // n_site

    # Create an empty array containing molecular coordinates
    mol_coord = np.zeros((n_mol, 3))

    assert mode in ['molecule', 'sites'], (
        f"Argument mode=={mode} must be either 'molecule' or 'sites'"
    )

    # Use centre of mass of molecule as molecular position
    if mode == 'molecule':
        for i in range(3):
            mol_coord[:, i] = np.sum(
                np.reshape(
                    atom_coord[:, i] * masses, (n_mol, n_site)
                ), axis=1
            )
            mol_coord[:, i] *= n_mol / masses.sum()

        return mol_coord

    # Convert integer com_sites input into a list
    if isinstance(com_sites, int):
        com_sites = [com_sites]

    assert len(com_sites) < n_site, (
        f"Argument com_sites must have a length ({len(com_sites)}) "
        f"less than n_sites ({n_site})"
    )

    # Use single atom as molecular position
    if len(com_sites) == 1:
        mol_list = np.arange(n_mol) * n_site + int(com_sites[0])
        for i in range(3):
            mol_coord[:, i] = atom_coord[mol_list, i]

    # Use centre of mass of a group of atoms within molecule as
    # molecular position
    elif len(com_sites) > 1:
        mol_list = np.arange(n_mol) * n_site
        mol_list = mol_list.repeat(len(com_sites))
        mol_list += np.tile(com_sites, n_mol)

        for i in range(3):
            mol_coord[:, i] = np.sum(
                np.reshape(
                    atom_coord[mol_list, i] * masses[mol_list],
                    (n_mol, len(com_sites))
                ), axis=1
            )
            mol_coord[:, i] *= n_mol / masses[mol_list].sum()

    return mol_coord


def molecules(xat, yat, zat, nmol, nsite, mol_M, mol_com):
    """
    Returns XYZ arrays of molecular positions"

    Parameters
    ----------

    xat:  float, array_like; shape=(natom)
        Coordinates of atoms in x dimension
    yat:  float, array_like; shape=(natom)
        Coordinates of atoms in y dimension
    zat:  float, array_like; shape=(natom)
        Coordinates of atoms in z dimension
    nmol:  int
        Number of molecules in simulation
    nsite:  int
        Number of atomic sites per molecule
    mol_M:  float, array_like; shape=(natom)
        Masses of atomic sites in g mol-1
    mol_com:
        Mode of calculation: if 'COM', centre of mass is used, otherwise atomic site index is used

    Returns
    -------

    xmol:  float, array_like; shape=(nmol)
        Coordinates of molecules in x dimension
    ymol:  float, array_like; shape=(nmol)
        Coordinates of molecules in y dimension
    zmol:  float, array_like; shape=(nmol)
        Coordinates of molecules in z dimension

    """
    if mol_com == 'COM':
        "USE COM of MOLECULE AS MOLECULAR POSITION"
        xmol = np.sum(np.reshape(xat * mol_M, (nmol, nsite)), axis=1) * nmol / mol_M.sum()
        ymol = np.sum(np.reshape(yat * mol_M, (nmol, nsite)), axis=1) * nmol / mol_M.sum()
        zmol = np.sum(np.reshape(zat * mol_M, (nmol, nsite)), axis=1) * nmol / mol_M.sum()

    elif len(mol_com) > 1:
        "USE COM of GROUP OF ATOMS WITHIN MOLECULE MOLECULAR POSITION"
        mol_list = np.arange(nmol) * nsite
        mol_list = mol_list.repeat(len(mol_com)) + np.tile(mol_com, nmol)

        xmol = np.sum(np.reshape(xat[mol_list] * mol_M[mol_list],
                (nmol, len(mol_com))), axis=1) * nmol / mol_M[mol_list].sum()
        ymol = np.sum(np.reshape(yat[mol_list] * mol_M[mol_list],
                (nmol, len(mol_com))), axis=1) * nmol / mol_M[mol_list].sum()
        zmol = np.sum(np.reshape(zat[mol_list] * mol_M[mol_list],
                (nmol, len(mol_com))), axis=1) * nmol / mol_M[mol_list].sum()

    elif len(mol_com) == 1:
        "USE SINGLE ATOM AS MOLECULAR POSITION"
        mol_list = np.arange(nmol) * nsite + int(mol_com[0])
        xmol = xat[mol_list]
        ymol = yat[mol_list]
        zmol = zat[mol_list]

    return xmol, ymol, zmol


def make_mol_com(traj_file, top_file, directory, file_name, natom, nmol, AT, at_index, nsite, mol_M, sys_M, mol_com, nframe=0):
    """
    make_mol_com(traj, directory, file_name, natom, nmol, at_index, nframe, dim, nsite, M, mol_com)

    Generates molecular positions and centre of mass for each frame

    Parameters
    ----------

    traj:  mdtraj obj
        Mdtraj trajectory object
    directory:  str
        File path of directory of alias analysis.
    file_name:  str
        File name of trajectory being analysed
    natom:  int
        Number of atoms in simulation
    nmol:  int
        Number of molecules in simulation
    at_index:  int, array_like; shape=(nsite*nmol)
        Indicies of atoms that are in molecules selected to determine intrinsic surface
    nframe:  int
        Number of frames in simulation trajectory
    nsite:  int
        Number of atomic sites in molecule
    M:  float, array_like; shape=(nsite)
        Masses of atomic sites in molecule
    mol_com:
        Mode of calculation: if 'COM', centre of mass is used, otherwise atomic site index is used

    """
    print("\n-----------CREATING POSITIONAL FILES------------\n")

    pos_dir = directory + 'pos/'
    if not os.path.exists(pos_dir): os.mkdir(pos_dir)

    file_name_pos = file_name + '_{}'.format(nframe)

    if not os.path.exists(pos_dir + file_name_pos + '_zvec.npy'):

        dim = np.zeros((0, 3))
        xmol = np.zeros((0, nmol))
        ymol = np.zeros((0, nmol))
        zmol = np.zeros((0, nmol))
        COM = np.zeros((0, 3))
        zvec = np.zeros((0, nmol))

        mol_M = np.array(mol_M * nmol)
        sys_M = np.array(sys_M)

        #XYZ = np.moveaxis(traj.xyz, 1, 2) * 10

        chunk = 500

        #for frame in xrange(nframe):
        for i, traj in enumerate(md.iterload(traj_file, chunk=chunk, top=top_file)):

            lengths = np.array(traj.unitcell_lengths)
            zvec = np.concatenate((zvec, orientation(traj, AT)[:,:,2]))
            chunk_index = np.arange(i*chunk, i*chunk + traj.n_frames)

            XYZ = np.moveaxis(traj.xyz, 1, 2) * 10

            for j, frame in enumerate(chunk_index):
                sys.stdout.write("PROCESSING IMAGE {} \r".format(frame))
                sys.stdout.flush()

                COM = np.concatenate((COM, [traj.xyz[j].astype('float64').T.dot(sys_M / sys_M.sum()) * 10]))
                dim = np.concatenate((dim, [lengths[j] * 10]))

                xat = XYZ[j][0][at_index]
                yat = XYZ[j][1][at_index]
                zat = XYZ[j][2][at_index]

                if nsite > 1:
                    xyz_mol = molecules(xat, yat, zat, nmol, nsite, mol_M, mol_com)
                    xmol = np.concatenate((xmol, [xyz_mol[0]]))
                    ymol = np.concatenate((ymol, [xyz_mol[1]]))
                    zmol = np.concatenate((zmol, [xyz_mol[2]]))

                else:
                    xmol = np.concatenate((xmol, [xat]))
                    ymol = np.concatenate((ymol, [yat]))
                    zmol = np.concatenate((zmol, [zat]))

        nframe = zmol.shape[0]
        file_name_pos = file_name + '_{}'.format(nframe)

        print('\nSAVING OUTPUT MOLECULAR POSITION FILES\n')

        np.save(pos_dir + file_name_pos + '_dim.npy', dim)
        np.save(pos_dir + file_name_pos + '_xmol.npy', xmol)
        np.save(pos_dir + file_name_pos + '_ymol.npy', ymol)
        np.save(pos_dir + file_name_pos + '_zmol.npy', zmol)
        np.save(pos_dir + file_name_pos + '_com.npy', COM)
        np.save(pos_dir + file_name_pos + '_zvec.npy', zvec)

    return nframe


def orientation(traj, AT):
    """
    Calculates orientational unit vector for lipid models, based on vector between phosphorus group
    and carbon backbone.
    """

    if len(AT) == 134:
        P_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'P')]
        C1_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'C218')]
        C2_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'C316')]

    elif len(AT) == 138:
        P_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'P')]
        C1_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'C218')]
        C2_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'C318')]
    else:
        P_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'PO4')]
        C1_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'C4A')]
        C2_atoms = [atom.index for atom in traj.topology.atoms if (atom.name == 'C4B')]

    XYZ = traj.xyz * 10
    dim = traj.unitcell_lengths * 10
    u_vectors = np.zeros((XYZ.shape[0], len(P_atoms), 3))

    for j in range(XYZ.shape[0]):
        midpoint = (XYZ[j][C1_atoms] + XYZ[j][C2_atoms]) / 2
        vector = XYZ[j][P_atoms] - midpoint
        for i, l in enumerate(dim[j]): vector[:, i] -= l * np.array(2 * vector[:, i] / l, dtype=int)
        u_vectors[j] = unit_vector(vector)

    return u_vectors


def check_pbc(xmol, ymol, zmol, pivots, dim, max_r=30):
    """
    Check periodic boundary conditions of molecule positions to ensure most
    appropriate position along is used wrt each surface.

    Parameters
    ----------
    xmol:  float, array_like; shape=(nmol)
        Molecular coordinates in x dimension
    ymol:  float, array_like; shape=(nmol)
        Molecular coordinates in y dimension
    zmol:  float, array_like; shape=(nmol)
        Molecular coordinates in z dimension
    surf_0: float, array_like; shape=(2)
        Intial guess for mean position of surface along z axis
    Lz:  float
        Cell dimension of z axis

    Returns
    -------
    zmol:  float, array_like; shape=(nmol)
        Molecular coordinates in z dimension using most appropriate PBC
    """

    # Create pivot map
    for index_i, pivot in enumerate(pivots):
        p_map = np.isin(np.arange(zmol.size), pivot)

        for check in range(2):
            for index_j, n in enumerate(pivot):

                dxyz = np.stack(
                    (xmol[p_map] - xmol[n],
                     ymol[p_map] - ymol[n],
                     zmol[p_map] - zmol[n])
                )
                for index_k, l in enumerate(dim[:2]):
                    dxyz[index_k] -= l * np.array(2 * dxyz[index_k] / l, dtype=int)

                dr2 = np.sum(dxyz**2, axis=0)
                neighbour_count = np.count_nonzero(dr2 < max_r**2)

                dxyz[2] += dim[2] * np.array([-1, 1])[index_i]
                dr2 = np.sum(dxyz**2, axis=0)
                neighbour_count_flip = np.count_nonzero(dr2 < max_r**2)

                if neighbour_count_flip > neighbour_count:
                    zmol[n] += dim[2] * np.array([1, -1])[index_i]

    return zmol
