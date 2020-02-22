"""
*************** INTRINSIC SAMPLING METHOD MODULE *******************

Performs intrinsic sampling analysis on a set of interfacial 
simulation configurations

********************************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 27/2/2018 by Frank Longford
"""
import os
import sys
import time
import tables

import numpy as np

from alias.io.hdf5_io import (
    make_hdf5, load_hdf5, save_hdf5, shape_check_hdf5)
from alias.io.numpy_io import load_npy
from alias.src.linear_algebra import update_A_b, lu_decomposition
from alias.src.self_consistent_cycle import (
    self_consistent_cycle,
    initialise_surface
)
from alias.src.spectra import intrinsic_area
from alias.src.utilities import create_surface_file_path

from .positions import check_pbc
from .surface_reconstruction import surface_reconstruction
from .utilities import numpy_remove


def build_surface(xmol, ymol, zmol, dim, qm, n0, phi, tau, max_r,
                  ncube=3, vlim=3, recon=0, surf_0=[0, 0], zvec=None):

    """
    build_surface(xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, tau, max_r, ncube=3, vlim=3, recon=0, surf_0=[0, 0])

    Create coefficients for Fourier sum representing intrinsic surface.

    Parameters
    ----------

    xmol:  float, array_like; shape=(nmol)
        Molecular coordinates in x dimension
    ymol:  float, array_like; shape=(nmol)
        Molecular coordinates in y dimension
    zmol:  float, array_like; shape=(nmol)
        Molecular coordinates in z dimension
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    mol_sigma:  float
        Radius of spherical molecular interaction sphere
    qm:  int
        Maximum number of wave frequencies in Fourier Sum representing intrinsic surface
    n0:  int
        Maximum number of molecular pivot in intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface optimisation function
    tau:  float
        Tolerance along z axis either side of existing intrinsic surface for selection of new pivot points
    max_r:  float
        Maximum radius for selection of vapour phase molecules
    ncube:	int (optional)
        Grid size for initial pivot molecule selection
    vlim:  int (optional)
        Minimum number of molecular meighbours within radius max_r required for molecular NOT to be considered in vapour region
    recon: bool (optional)
        Whether to peform surface reconstruction routine
    surf_0: float, array-like; shape=(2) (optional)
        Initial guesses for surface plane positions

    Returns
    -------
    coeff:	array_like (float); shape=(2, n_waves**2)
        Optimised surface coefficients
    pivot:  array_like (int); shape=(2, n0)
        Indicies of pivot molecules in molecular position arrays

    """

    nmol = len(xmol)
    mol_list = np.arange(nmol)

    start = time.time()

    surf_param = initialise_surface(qm, phi, dim, recon)

    if recon == 1:
        psi = phi * dim[0] * dim[1]
        coeff, A, b, area_diag, curve_matrix, H_var = surf_param
    else:
        coeff, A, b, area_diag = surf_param

    "Remove molecules from vapour phase and assign an initial grid of pivots furthest away from centre of mass"
    print('Lx = {:5.3f}   Ly = {:5.3f}   qm = {:5d}\nphi = {}   n_piv = {:5d}   vlim = {:5d}   max_r = {:5.3f}'.format(dim[0], dim[1], qm, phi, n0, vlim, max_r))
    print('Surface plane initial guess = {} {}'.format(surf_0[0], surf_0[1]))

    pivot_search = (ncube > 0)

    if not pivot_search:

        print("\n{:^74s} | {:^21s} | {:^21s}".format('TIMINGS (s)', 'PIVOTS', 'INT AREA'))
        print(' {:20s} {:20s} {:20s} {:10s} | {:10s} {:10s} | {:10s} {:10s}'.format(
            'Pivot selection', 'Matrix Formation', 'LU Decomposition', 'TOTAL', 'n_piv1', 'n_piv2', 'surf1', 'surf2'))
        print("_" * 120)

        start1 = time.time()

        if zvec is not None:
            "Separate pivots based on orientational vector"
            piv_n1 = np.argwhere(zvec < 0).flatten()
            piv_n2 = np.argwhere(zvec >= 0).flatten()

        else:
            "Separate pivots based on position"
            piv_n1 = np.argwhere(zmol < 0).flatten()
            piv_n2 = np.argwhere(zmol >= 0).flatten()

        pivot = [piv_n1, piv_n2]

        if not (len(pivot[0]) == n0) * (len(pivot[1]) == n0):
            #ut.view_surface(coeff, pivot, qm, qm, xmol, ymol, zmol, 2, dim)
            zmol, pivot = pivot_swap(xmol, ymol, zmol, pivot, dim, max_r, n0)

        zmol = check_pbc(xmol, ymol, zmol, pivot, dim)
        pivot = np.array(pivot, dtype=int)

        end1 = time.time()

        "Update A matrix and b vector"
        temp_A, temp_b, fuv = update_A_b(xmol, ymol, zmol, dim, qm, pivot)

        A += temp_A
        b += temp_b

        end2 = time.time()

        "Perform LU decomosition to solve Ax = b"
        coeff[0] = lu_decomposition(A[0] + area_diag, b[0])
        coeff[1] = lu_decomposition(A[1] + area_diag, b[1])

        if recon:
            coeff[0], _ = surface_reconstruction(coeff[0], A[0], b[0], area_diag, curve_matrix, H_var, qm, pivot[0].size, psi)
            coeff[1], _ = surface_reconstruction(coeff[1], A[1], b[1], area_diag, curve_matrix, H_var, qm, pivot[1].size, psi)

        end3 = time.time()

        "Calculate surface areas excess"
        area1 = intrinsic_area(coeff[0], qm, qm, dim)
        area2 = intrinsic_area(coeff[1], qm, qm, dim)

        end = time.time()

        print(' {:20.3f} {:20.3f} {:20.3f} {:10.3f} | {:10d} {:10d} | {:10.3f} {:10.3f}'.format(
            end1 - start1, end2 - end1, end3 - end2, end - start1, len(pivot[0]), len(pivot[1]), area1, area2))

    #ut.view_surface(coeff, pivot, qm, qm, xmol, ymol, zmol, 50, dim)

    else:
        piv_n1 = np.arange(ncube**2)
        piv_n2 = np.arange(ncube**2)
        piv_z1 = np.zeros(ncube**2)
        piv_z2 = np.zeros(ncube**2)
        vapour_list = []
        new_piv1 = []
        new_piv2 = []

        dxyz = np.reshape(np.tile(np.stack((xmol, ymol, zmol)), (1, nmol)), (3, nmol, nmol))
        dxyz = np.transpose(dxyz, axes=(0, 2, 1)) - dxyz
        for i, l in enumerate(dim[:2]): dxyz[i] -= l * np.array(2 * dxyz[i] / l, dtype=int)
        dr2 = np.sum(dxyz**2, axis=0)

        vapour_list = np.where(np.count_nonzero(dr2 < max_r**2, axis=1) < vlim)
        print('Removing {} vapour molecules'.format(vapour_list[0].size))
        mol_list = numpy_remove(mol_list, vapour_list)
        del dxyz, dr2

        print('Selecting initial {} pivots'.format(ncube**2))
        index_x = np.array(xmol * ncube / dim[0], dtype=int) % ncube
        index_y = np.array(ymol * ncube / dim[1], dtype=int) % ncube

        for n in mol_list:
            if zmol[n] < piv_z1[ncube*index_x[n] + index_y[n]]:
                piv_n1[ncube*index_x[n] + index_y[n]] = n
                piv_z1[ncube*index_x[n] + index_y[n]] = zmol[n]
            elif zmol[n] > piv_z2[ncube*index_x[n] + index_y[n]]:
                piv_n2[ncube*index_x[n] + index_y[n]] = n
                piv_z2[ncube*index_x[n] + index_y[n]] = zmol[n]

        "Update molecular and pivot lists"
        mol_list = numpy_remove(mol_list, piv_n1)
        mol_list = numpy_remove(mol_list, piv_n2)

        new_piv1 = piv_n1
        new_piv2 = piv_n2

        assert np.sum(np.isin(vapour_list, mol_list)) == 0
        assert np.sum(np.isin(piv_n1, mol_list)) == 0
        assert np.sum(np.isin(piv_n2, mol_list)) == 0

        print('Initial {} pivots selected: {:10.3f} s'.format(ncube**2, time.time() - start))

        "Split molecular position lists into two volumes for each surface"
        mol_list1 = mol_list
        mol_list2 = mol_list

        assert piv_n1 not in mol_list1
        assert piv_n2 not in mol_list2

        coeff, pivot = self_consistent_cycle(
            coeff, A, b, dim, qm, tau, xmol, ymol, zmol,
            [piv_n1, piv_n2], mol_list1, mol_list2, phi, n0,
            new_piv1=[], new_piv2=[], recon=False)

    print('\n')

    return coeff, pivot


def create_intrinsic_surfaces(directory, file_name, dim, qm, n0, phi, mol_sigma, nframe, recon=False, ncube=3, vlim=3, tau=0.5, max_r=1.5, ow_coeff=False, ow_recon=False):
    """
    create_intrinsic_surfaces(directory, file_name, dim, qm, n0, phi, mol_sigma, nframe, recon=False, ow_coeff=False, ow_recon=False)

    Routine to find optimised pivot density coefficient ns and pivot number n0 based on lowest pivot diffusion rate

    Parameters
    ----------

    directory:  str
        File path of directory of alias analysis.
    file_name:  str
        File name of trajectory being analysed.
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    n0:  int
        Maximum number of molecular pivots in intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface optimisation function
    mol_sigma:  float
        Radius of spherical molecular interaction sphere
    nframe:  int
        Number of frames in simulation trajectory
    recon:  bool (optional)
        Whether to perform surface reconstruction routine (default=True)
    ow_coeff:  bool (optional)
        Whether to overwrite surface coefficients (default=False)
    ow_recon:  bool (optional)
        Whether to overwrite reconstructed surface coefficients (default=False)

    """

    print("\n--- Running Intrinsic Surface Routine ---\n")

    surf_dir = directory + 'surface/'
    pos_dir = directory + 'pos/'

    if not os.path.exists(surf_dir):
        os.mkdir(surf_dir)

    n_waves = 2 * qm + 1
    max_r *= mol_sigma
    tau *= mol_sigma

    coeff_file_name = create_surface_file_path(
        file_name, surf_dir, qm, n0, phi, nframe, recon
    )

    "Make coefficient and pivot files"
    if not os.path.exists(coeff_file_name + '_coeff.hdf5'):
        make_hdf5(coeff_file_name + '_coeff', (2, n_waves**2), tables.Float64Atom())
        make_hdf5(coeff_file_name + '_pivot', (2, n0), tables.Int64Atom())
        file_check = False
    elif not ow_coeff:
        "Checking number of frames in current coefficient files"
        try:
            file_check = (shape_check_hdf5(coeff_file_name + '_coeff') == (nframe, 2, n_waves**2))
            file_check *= (shape_check_hdf5(coeff_file_name + '_pivot') == (nframe, 2, n0))
        except:
            file_check = False
    else:
        file_check = False

    if not file_check:
        print("IMPORTING GLOBAL POSITION DISTRIBUTIONS\n")
        mol_traj = load_npy(pos_dir + file_name + '_{}_mol_traj'.format(nframe))
        mol_vec = load_npy(pos_dir + file_name + '_{}_mol_vec'.format(nframe))
        com_traj = load_npy(pos_dir + file_name + '_{}_com'.format(nframe))

        n_mols = mol_traj.shape[1]
        com_tile = np.moveaxis(
            np.tile(com_traj, (n_mols, 1, 1)),
            [0, 1, 2], [2, 1, 0])[2]
        mol_traj[:, :, 2] -= com_tile

        for frame in range(nframe):

            "Checking number of frames in coeff and pivot files"
            frame_check_coeff = (shape_check_hdf5(coeff_file_name + '_coeff')[0] <= frame)
            frame_check_pivot = (shape_check_hdf5(coeff_file_name + '_pivot')[0] <= frame)

            if frame_check_coeff:
                mode_coeff = 'a'
            elif ow_coeff:
                mode_coeff = 'r+'
            else:
                mode_coeff = False

            if frame_check_pivot:
                mode_pivot = 'a'
            elif ow_coeff:
                mode_pivot = 'r+'
            else:
                mode_pivot = False

            if not mode_coeff and not mode_pivot:
                pass
            else:
                sys.stdout.write("Optimising Intrinsic Surface coefficients: frame {}\n".format(frame))
                sys.stdout.flush()

                if frame == 0:
                    surf_0 = [-dim[2]/4, dim[2]/4]
                else:
                    index = (2 * qm + 1)**2 / 2
                    coeff = load_hdf5(coeff_file_name + '_coeff', frame-1)
                    surf_0 = [coeff[0][index], coeff[1][index]]

                coeff, pivot = build_surface(
                    mol_traj[frame, :, 0], mol_traj[frame, :, 1], mol_traj[frame, :, 2],
                    dim, qm, n0, phi, tau, max_r,
                    ncube=ncube, vlim=vlim, recon=recon, surf_0=surf_0, zvec=mol_vec[frame])

                save_hdf5(coeff_file_name + '_coeff', coeff, frame, mode_coeff)
                save_hdf5(coeff_file_name + '_pivot', pivot, frame, mode_pivot)


def pivot_swap(xmol, ymol, zmol, pivots, dim, max_r, n0):

    assert (pivots[0].size + pivots[1].size == 2 * n0), (pivots[0].size + pivots[1].size, 2 * n0)

    "Check equal number of pivots exists for each surface"
    while pivots[0].size != n0 or pivots[1].size != n0:

        "Identify the overloaded surface"
        surf_g = int(pivots[0].size < pivots[1].size)
        surf_l = int(pivots[0].size > pivots[1].size)
        piv_g = pivots[surf_g]

        "Calculate radial distances between each pivot in the surface"
        dxyz = np.reshape(np.tile(np.stack((xmol[piv_g], ymol[piv_g], zmol[piv_g])),
                                  (1, pivots[surf_g].size)),
                          (3, pivots[surf_g].size, pivots[surf_g].size))
        dxyz = np.transpose(dxyz, axes=(0, 2, 1)) - dxyz
        for i, l in enumerate(dim[:2]):
            dxyz[i] -= l * np.array(2 * dxyz[i] / l, dtype=int)
        dr2 = np.sum(dxyz**2, axis=0)

        "Compose a nearest neighbour list, ordered by number of neighbours"
        vapour_list = np.argsort(np.count_nonzero(dr2 < max_r**2, axis=1))[:1]
        piv = pivots[surf_g][vapour_list]

        "Swap the pivot molecule with the smallest number of neighbours to the other surface"
        pivots[surf_l] = np.concatenate((pivots[surf_l], piv))
        pivots[surf_g] = np.delete(pivots[surf_g], vapour_list)

    return zmol, pivots