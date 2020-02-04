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
import scipy as sp

from alias.io.hdf5_io import make_hdf5, load_hdf5, save_hdf5, shape_check_hdf5
from alias.io.numpy_io import load_npy

from .intrinsic_surface import xi
from .positions import check_pbc
from .surface_reconstruction import surface_reconstruction
from .wave_function import check_uv, wave_function
from .utilities import numpy_remove, bubble_sort


vcheck = np.vectorize(check_uv)


def update_A_b(xmol, ymol, zmol, dim, qm, new_pivot):
    """
    Update A matrix and b vector for new pivot selection

    Paramters
    ---------
    xmol:  float, array_like; shape=(nmol)
        Molecular coordinates in x dimension
    ymol:  float, array_like; shape=(nmol)
        Molecular coordinates in y dimension
    zmol:  float, array_like; shape=(nmol)
        Molecular coordinates in z dimension
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    n_waves:  int
        Number of coefficients / waves in surface
    new_pivot:  int, array_like
        Indices of new pivot molecules for both surfaces

    Returns
    -------
    A:  float, array_like; shape=(2, n_waves**2, n_waves**2)
        Matrix containing wave product weightings f(x, u1, Lx).f(y, v1, Ly).f(x, u2, Lx).f(y, v2, Ly)
        for each coefficient in the linear algebra equation Ax = b for both surfaces
    b:  float, array_like; shape=(2, n_waves**2)
        Vector containing solutions z.f(x, u, Lx).f(y, v, Ly) to the linear algebra equation Ax = b
        for both surfaces

    """
    n_waves = 2 * qm +1

    u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
    v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

    A = np.zeros((2, n_waves**2, n_waves**2))
    b = np.zeros((2, n_waves**2))

    fuv1 = np.zeros((n_waves**2, len(new_pivot[0])))
    fuv2 = np.zeros((n_waves**2, len(new_pivot[1])))

    for j in range(n_waves**2):
        fuv1[j] = wave_function(xmol[new_pivot[0]], u_array[j], dim[0]) * wave_function(ymol[new_pivot[0]], v_array[j], dim[1])
        b[0][j] += np.sum(zmol[new_pivot[0]] * fuv1[j])
        fuv2[j] = wave_function(xmol[new_pivot[1]], u_array[j], dim[0]) * wave_function(ymol[new_pivot[1]], v_array[j], dim[1])
        b[1][j] += np.sum(zmol[new_pivot[1]] * fuv2[j])

    A[0] += np.dot(fuv1, fuv1.T)
    A[1] += np.dot(fuv2, fuv2.T)

    return A, b, fuv1, fuv2


def LU_decomposition(A, b):
    """
    LU_decomposition(A, b)

    Perform lower-upper decomposition to solve equation Ax = b using scipy linalg lover-upper solver

    Parameters
    ----------

    A:  float, array_like; shape=(2, n_waves**2, n_waves**2)
        Matrix containing wave product weightings f(x, u1, Lx).f(y, v1, Ly).f(x, u2, Lx).f(y, v2, Ly)
        for each coefficient in the linear algebra equation Ax = b for both surfaces
    b:  float, array_like; shape=(2, n_waves**2)
        Vector containing solutions z.f(x, u, Lx).f(y, v, Ly) to the linear algebra equation Ax = b
        for both surfaces

    Returns
    -------

    coeff:	array_like (float); shape=(n_waves**2)
        Optimised surface coefficients

    """
    lu, piv  = sp.linalg.lu_factor(A)
    coeff = sp.linalg.lu_solve((lu, piv), b)

    return coeff


def make_zeta_list(xmol, ymol, zmol, dim, mol_list, coeff, qm, qu):
    """
    zeta_list(xmol, ymol, dim, mol_list, coeff, qm)

    Calculate dz (zeta) between molecular sites and intrinsic surface for resolution qu"

    Parameters
    ----------

    xmol:  float, array_like; shape=(nmol)
        Molecular coordinates in x dimension
    ymol:  float, array_like; shape=(nmol)
        Molecular coordinates in y dimension
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    mol_list:  int, array_like; shape=(n0)
        Indices of molcules available to be slected as pivots
    coeff:	array_like (float); shape=(n_waves**2)
        Optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface

    Returns
    -------

    zeta_list:  float, array_like; shape=(n0)
        Array of dz (zeta) between molecular sites and intrinsic surface

    """

    "Calculate shortest z distance between molecules and surface, zeta"
    zeta_list = xi(xmol[mol_list], ymol[mol_list], coeff, qm, qu, dim)

    zeta_list = zmol[mol_list] - zeta_list
    zeta_list -= dim[2] * np.array(2 * zeta_list / dim[2], dtype=int)
    zeta_list = abs(zeta_list)

    return zeta_list


def pivot_selection(mol_list, zeta_list, piv_n, tau, n0):
    """
    pivot_selection(mol_list, zeta_list, piv_n, tau, n0)

    Search through zeta_list for values within tau threshold and add to pivot list

    Parameters
    ----------

    mol_list:  int, array_like; shape=(n0)
        Indices of molcules available to be selected as pivots
    zeta_list:  float, array_like; shape=(n0)
        Array of dz (zeta) between molecular sites and intrinsic surface
    piv_n:  int, array_like; shape=(n0)
        Molecular pivot indices
    tau:  float
        Threshold length along z axis either side of existing intrinsic surface for selection of new pivot points
    n0:  int
        Maximum number of molecular pivots in intrinsic surface

    Returns
    -------

    mol_list:  int, array_like; shape=(n0)
        Updated indices of molcules available to be selected as pivots
    new_piv:  int, array_like
        Indices of new pivot molecules just selected
    piv_n:  int, array_like; shape=(n0)
        Updated molecular pivot indices

    """

    "Find new pivots based on zeta <= tau"
    new_piv = mol_list[zeta_list <= tau]
    dz_new_piv = zeta_list[zeta_list <= tau]

    "Order pivots by zeta (shortest to longest)"
    bubble_sort(new_piv, dz_new_piv)

    "Add new pivots to pivoy list and check whether max n0 pivots are selected"
    piv_n = np.concatenate((piv_n, new_piv))

    if piv_n.shape[0] > n0:
        new_piv = new_piv[:-piv_n.shape[0]+n0]
        piv_n = piv_n[:n0]

    far_tau = 6.0 * tau

    "Remove pivots far from molecular search list"
    far_piv = mol_list[zeta_list > far_tau]
    if len(new_piv) > 0: mol_list = numpy_remove(mol_list, np.concatenate((new_piv, far_piv)))

    assert np.sum(np.isin(new_piv, mol_list)) == 0

    return mol_list, new_piv, piv_n


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


def initialise_surface(qm, phi, dim, recon=False):
    """
    initialise_surface(qm, phi, dim, recon=False)

    Calculate initial parameters for ISM and reconstructed ISM fitting proceedure

    Parameters
    ----------

    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface optimisation function
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    recon:  bool (optional)
        Surface reconstruction

    Returns
    -------

    coeff:	array_like (float); shape=(n_waves**2)
        Optimised surface coefficients
    A:  float, array_like; shape=(n_waves**2, n_waves**2)
        Matrix containing wave product weightings f(x, u1, Lx).f(y, v1, Ly).f(x, u2, Lx).f(y, v2, Ly)
        for each coefficient in the linear algebra equation Ax = b for both surfaces
    b:  float, array_like; shape=(n_waves**2)
        Vector containing solutions z.f(x, u, Lx).f(y, v, Ly) to the linear algebra equation Ax = b
        for both surfaces
    area_diag: float, array_like; shape=(n_waves**2)
        Surface area diagonal terms for A matrix
    curve_matrix: float, array_like; shape=(n_waves**2, n_waves**2)
        Surface curvature terms for A matrix
    H_var: float, array_like; shape=(n_waves**2)
        Diagonal terms for global variance of mean curvature
    """

    n_waves = 2*qm+1

    "Form the diagonal xi^2 terms"
    u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
    v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
    uv_check = vcheck(u_array, v_array)

    "Make diagonal terms of A matrix"
    area_diag = phi * (u_array**2 * dim[1] / dim[0] + v_array**2 * dim[0] / dim[1])
    area_diag = 4 * np.pi**2 * np.diagflat(area_diag * uv_check)

    "Create empty A matrix and b vector for linear algebra equation Ax = b"
    A = np.zeros((2, n_waves**2, n_waves**2))
    b = np.zeros((2, n_waves**2))
    coeff = np.zeros((2, n_waves**2))

    if recon:
        u_matrix = np.tile(u_array, (n_waves**2, 1))
        v_matrix = np.tile(v_array, (n_waves**2, 1))

        H_var = 4 * np.pi**4 * uv_check * (u_array**4 / dim[0]**4 + v_array**4 / dim[1]**4 + 2 * (u_array * v_array)**2 / np.prod(dim**2))

        curve_matrix = 16 * np.pi**4 * ((u_matrix * u_matrix.T)**2 / dim[0]**4 + (v_matrix * v_matrix.T)**2 / dim[1]**4 +
                                        ((u_matrix * v_matrix.T)**2 + (u_matrix.T * v_matrix)**2) / np.prod(dim**2))

        return coeff, A, b, area_diag, curve_matrix, H_var

    return coeff, A, b, area_diag


def intrinsic_area(coeff, qm, qu, dim):
    """
    intrinsic_area(coeff, qm, qu, dim)

    Calculate the intrinsic surface area from coefficients at resolution qu

    Parameters
    ----------

    coeff:	float, array_like; shape=(n_waves**2)
        Optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    int_A:  float
        Relative size of intrinsic surface area, compared to cell cross section XY
    """

    n_waves = 2 * qm +1

    u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
    v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
    wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
    indices = np.argwhere(wave_check).flatten()

    q2 = np.pi**2  * vcheck(u_array[indices], v_array[indices]) * (u_array[indices]**2 / dim[0]**2 + v_array[indices]**2 / dim[1]**2)
    int_A = q2 * coeff[indices]**2
    int_A = 1 + 0.5 * np.sum(int_A)

    return int_A


def self_consistent_cycle(
        coeff, A, b, dim, qm, tau, xmol, ymol, zmol,
        pivot, mol_list1, mol_list2, phi, n0,
        new_piv1=[], new_piv2=[], recon=False):

    start = time.time()

    print("{:^77s} | {:^43s} | {:^21s} | {:^21s}".format('TIMINGS (s)', 'PIVOTS', 'TAU', 'INT AREA'))
    print(' {:20s}  {:20s}  {:20s}  {:10s} | {:10s} {:10s} {:10s} {:10s} | {:10s} {:10s} | {:10s} {:10s}'.format(
        'Matrix Formation', 'LU Decomposition', 'Pivot selection', 'TOTAL', 'n_piv1',
        '(new)', 'n_piv2', '(new)', 'surf1', 'surf2', 'surf1', 'surf2'))
    print("_" * 170)

    tau1 = tau
    tau2 = tau
    inc = 0.1 * tau

    surf_param = initialise_surface(qm, phi, dim, recon)

    if recon == 1:
        psi = phi * dim[0] * dim[1]
        coeff, A, b, area_diag, curve_matrix, H_var = surf_param
    else: coeff, A, b, area_diag = surf_param

    building_surface = True
    build_surf1 = True
    build_surf2 = True

    while building_surface:

        start1 = time.time()

        "Update A matrix and b vector"
        temp_A, temp_b, fuv1, fuv2 = update_A_b(xmol, ymol, zmol, dim, qm, [new_piv1, new_piv2])

        A += temp_A
        b += temp_b

        end1 = time.time()

        "Perform LU decomosition to solve Ax = b"
        if build_surf1: coeff[0] = LU_decomposition(A[0] + area_diag, b[0])
        if build_surf2: coeff[1] = LU_decomposition(A[1] + area_diag, b[1])

        if recon:
            if build_surf1:
                coeff[0], _ = surface_reconstruction(coeff[0], A[0], b[0], area_diag, curve_matrix, H_var, qm, len(pivot[0]), psi)
            if build_surf2:
                coeff[1], _ = surface_reconstruction(coeff[1], A[1], b[1], area_diag, curve_matrix, H_var, qm, len(pivot[1]), psi)

        end2 = time.time()

        #ut.view_surface(coeff, [piv_n1, piv_n2], qm, qm, xmol, ymol, zmol, 30, dim)

        "Calculate surface areas excess"
        area1 = intrinsic_area(coeff[0], qm, qm, dim)
        area2 = intrinsic_area(coeff[1], qm, qm, dim)

        "Check whether more pivots are needed"
        if len(pivot[0]) == n0:
            build_surf1 = False
            new_piv1 = []
        if len(pivot[1]) == n0:
            build_surf2 = False
            new_piv2 = []

        if build_surf1 or build_surf2:
            finding_pivots = True
            piv_search1 = True
            piv_search2 = True
        else:
            finding_pivots = False
            building_surface = False
            print("ENDING SEARCH")

        "Calculate distance between molecular z positions and intrinsic surface"
        if build_surf1: zeta_list1 = make_zeta_list(xmol, ymol, zmol, dim, mol_list1, coeff[0], qm, qm)
        if build_surf2: zeta_list2 = make_zeta_list(xmol, ymol, zmol, dim, mol_list2, coeff[1], qm, qm)

        "Search for more molecular pivot sites"

        while finding_pivots:
            "Perform pivot selectrion"
            if piv_search1 and build_surf1:
                mol_list1, new_piv1, pivot[0] = pivot_selection(mol_list1, zeta_list1, pivot[0], tau1, n0)
            if piv_search2 and build_surf2:
                mol_list2, new_piv2, pivot[1] = pivot_selection(mol_list2, zeta_list2, pivot[1], tau2, n0)

            "Check whether threshold distance tau needs to be increased"
            if len(new_piv1) == 0 and len(pivot[0]) < n0:
                tau1 += inc
            else:
                piv_search1 = False

            if len(new_piv2) == 0 and len(pivot[1]) < n0:
                tau2 += inc
            else:
                piv_search2 = False

            if piv_search1 or piv_search2: finding_pivots = True
            else: finding_pivots = False

        end = time.time()

        print(' {:20.3f}  {:20.3f}  {:20.3f}  {:10.3f} | {:10d} {:10d} {:10d} {:10d} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f}'.format(
            end1 - start1, end2 - end1, end - end2, end - start1, len(pivot[0]), len(new_piv1),
            len(pivot[1]), len(new_piv2), tau1, tau2, area1, area2))

    print('\nTOTAL time: {:7.2f} s \n'.format(end - start))

    pivot = np.array(pivot, dtype=int)

    return coeff, pivot


def build_surface(xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, tau, max_r, ncube=3, vlim=3, recon=0, surf_0=[0, 0], zvec=None):

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
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
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
    else: coeff, A, b, area_diag = surf_param

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

        if (zvec != None).any():
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
        temp_A, temp_b, fuv1, fuv2 = update_A_b(xmol, ymol, zmol, dim, qm, pivot)

        A += temp_A
        b += temp_b

        end2 = time.time()

        "Perform LU decomosition to solve Ax = b"
        coeff[0] = LU_decomposition(A[0] + area_diag, b[0])
        coeff[1] = LU_decomposition(A[1] + area_diag, b[1])

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

    if not os.path.exists(surf_dir): os.mkdir(surf_dir)

    file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
    n_waves = 2 * qm + 1
    max_r *= mol_sigma
    tau *= mol_sigma

    if recon: file_name_coeff += '_r'

    "Make coefficient and pivot files"
    if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_coeff)):
        make_hdf5(surf_dir + file_name_coeff + '_coeff', (2, n_waves**2), tables.Float64Atom())
        make_hdf5(surf_dir + file_name_coeff + '_pivot', (2, n0), tables.Int64Atom())
        file_check = False
    elif not ow_coeff:
        "Checking number of frames in current coefficient files"
        try:
            file_check = (shape_check_hdf5(surf_dir + file_name_coeff + '_coeff') == (nframe, 2, n_waves**2))
            file_check *= (shape_check_hdf5(surf_dir + file_name_coeff + '_pivot') == (nframe, 2, n0))
        except: file_check = False
    else: file_check = False

    if not file_check:
        print("IMPORTING GLOBAL POSITION DISTRIBUTIONS\n")
        xmol = load_npy(pos_dir + file_name + '_{}_xmol'.format(nframe))
        ymol = load_npy(pos_dir + file_name + '_{}_ymol'.format(nframe))
        zmol = load_npy(pos_dir + file_name + '_{}_zmol'.format(nframe))
        zvec = load_npy(pos_dir + file_name + '_{}_zvec'.format(nframe))
        COM = load_npy(pos_dir + file_name + '_{}_com'.format(nframe))
        nmol = xmol.shape[1]
        com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
        zmol = zmol - com_tile

        for frame in range(nframe):

            "Checking number of frames in coeff and pivot files"
            frame_check_coeff = (shape_check_hdf5(surf_dir + file_name_coeff + '_coeff')[0] <= frame)
            frame_check_pivot = (shape_check_hdf5(surf_dir + file_name_coeff + '_pivot')[0] <= frame)

            if frame_check_coeff: mode_coeff = 'a'
            elif ow_coeff: mode_coeff = 'r+'
            else: mode_coeff = False

            if frame_check_pivot: mode_pivot = 'a'
            elif ow_coeff: mode_pivot = 'r+'
            else: mode_pivot = False

            if not mode_coeff and not mode_pivot: pass
            else:
                sys.stdout.write("Optimising Intrinsic Surface coefficients: frame {}\n".format(frame))
                sys.stdout.flush()

                if frame == 0: surf_0 = [-dim[2]/4, dim[2]/4]
                else:
                    index = (2 * qm + 1)**2 / 2
                    coeff = load_hdf5(surf_dir + file_name_coeff + '_coeff', frame-1)
                    surf_0 = [coeff[0][index], coeff[1][index]]

                coeff, pivot = build_surface(xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma,
                                             qm, n0, phi, tau, max_r, ncube=ncube, vlim=vlim, recon=recon, surf_0=surf_0, zvec=zvec[frame])

                save_hdf5(surf_dir + file_name_coeff + '_coeff', coeff, frame, mode_coeff)
                save_hdf5(surf_dir + file_name_coeff + '_pivot', pivot, frame, mode_pivot)
