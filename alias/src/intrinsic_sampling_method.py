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
import scipy.constants as con

from .utilities import (
    numpy_remove, bubble_sort, load_npy, make_hdf5, load_hdf5,
    shape_check_hdf5, save_hdf5
)


def check_uv(u, v):
    """
    check_uv(u, v)

    Returns weightings for frequencies u and v for anisotropic surfaces

    """

    if abs(u) + abs(v) == 0:
        return 4.
    elif u * v == 0:
        return 2.
    return 1.


vcheck = np.vectorize(check_uv)


def wave_function(x, u, Lx):
    """
    wave_function(x, u, Lx)

    Wave in Fouier sum

    """

    if u >= 0:
        return np.cos(2 * np.pi * u * x / Lx)
    return np.sin(2 * np.pi * abs(u) * x / Lx)


def d_wave_function(x, u, Lx):
    """
    d_wave_function(x, u, Lx)

    Derivative of wave in Fouier sum wrt x

    """

    coeff = 2 * np.pi / Lx

    if u >= 0:
        return - coeff * u * np.sin(coeff * u * x)
    return coeff * abs(u) * np.cos(coeff * abs(u) * x)


def dd_wave_function(x, u, Lx):
    """
    dd_wave_function(x, u, Lx)

    Second derivative of wave in Fouier sum wrt x

    """

    return - 4 * np.pi**2 * u**2 / Lx**2 * wave_function(x, u, Lx)


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
        for i, l in enumerate(dim[:2]): dxyz[i] -= l * np.array(2 * dxyz[i] / l, dtype=int)
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


def H_var_values(coeff, A, curve_matrix, H_var, qm, n0):
    """
    H_var_values(coeff, A, curve_matrix, H_var, qm, n0)

    Returns values involved in iterative optimisation of mean curvature variance

    Parameters
    ----------

    coeff:	array_like (float); shape=(n_waves**2)
        Optimised surface coefficients
    A:  float, array_like; shape=(n_waves**2, n_waves**2)
        Matrix containing wave product weightings f(x, u1, Lx).f(y, v1, Ly).f(x, u2, Lx).f(y, v2, Ly)
        for each coefficient in the linear algebra equation Ax = b for both surfaces
    curve_matrix: float, array_like; shape=(n_waves**2, n_waves**2)
        Surface curvature terms for A matrix
    H_var: float, array_like; shape=(n_waves**2)
        Diagonal elements coressponding to variance of mean curvature
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    n0:  int
        Maximum number of molecular pivots in intrinsic surface

    Returns
    -------

    H_var_coeff: float
        Global variance of mean curvature
    H_var_piv:  float
        Variance of mean curvature at surface pivot sites
    H_var_func:
        Function to minimise, difference between H_var_coeff and H_var_piv
    """

    n_waves = 2*qm+1

    "Calculate variance of curvature across entire surface from coefficients"
    H_var_coeff = np.sum(H_var * coeff**2)
    "Calculate variance of curvature at pivot sites only"
    coeff_matrix = np.tile(coeff, (n_waves**2, 1))
    H_var_piv = np.sum(coeff_matrix * coeff_matrix.T * A * curve_matrix / n0)
    "Calculate optimisation function (diff between coeff and pivot variance)"
    H_var_func = abs(H_var_coeff - H_var_piv)

    return H_var_coeff, H_var_piv, H_var_func


def surface_reconstruction(coeff, A, b, area_diag, curve_matrix, H_var, qm, n0, psi, precision=1E-3, max_step=20):
    """
    surface_reconstruction(coeff, A, b, diag, curve_matrix, H_var, qm, n0, psi, precision=1E-3, max_step=20)

    Iterative algorithm to perform surface reconstruction routine. Solves Ax = b general matrix equation until
    solution found where global variance of mean curvature H is equivalent to curvature at positions of surface
    molecules H_var.

    Parameters
    ----------

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
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    n0:  int
        Maximum number of molecular pivots in intrinsic surface
    psi:  float
        Initial value of weighting factor for surface reconstruction routine
    precision:  float (optional)
        Threshold value determining target similarity of global and sample curvatures
    max_step:  int
        Maximum iterative steps without solution until algorithm is restarted with a reduced psi

    Returns
    -------

    A_recon:  float, array_like; shape=(n_waves**2, n_waves**2)
        Reconstructed A matrix
    coeff_recon:  array_like (float); shape=(2, n_waves**2)
        Reconstructed surface coefficients
    """

    reconstructing = True
    psi_array = np.array([0, psi])
    step = 0
    weight = 0.8

    H_var_coeff = np.zeros(2)
    H_var_piv = np.zeros(2)
    H_var_func = np.zeros(2)
    H_var_grad = np.zeros(2)

    H_var_coeff[0], H_var_piv[0], H_var_func[0] = H_var_values(coeff, A, curve_matrix, H_var, qm, n0)
    H_var_grad[0] = 1

    #print " {:^11s} | {:^13s} {:^13s}".format( 'PSI', 'VAR(coeff_H)', 'VAR(piv_H)' )
    #print( ' {:10.8f} {:10.4f} {:10.4f}'.format(psi_array[1],  H_var_coeff[0], H_var_piv[0]))

    "Amend psi weighting coefficient until H_var == H_piv_var"
    while reconstructing:
        "Update A matrix and b vector"
        A_recon = A * (1. + curve_matrix * psi_array[1] / n0)

        "Update coeffs by performing LU decomosition to solve Ax = b"
        coeff_recon = LU_decomposition(A_recon + area_diag, b)

        H_var_coeff[1], H_var_piv[1], H_var_func[1] = H_var_values(coeff_recon, A_recon, curve_matrix, H_var, qm, n0)

        "Recalculate gradient of optimistation function wrt psi"
        H_var_grad[1] = (H_var_func[1] - H_var_func[0]) / (psi_array[1] - psi_array[0])

        if abs(H_var_func[1]) <= precision: reconstructing = False
        else:
            step += 1

            if step >= max_step or psi_array[1] < 0:
                "Reconstruction routine failed to find minimum. Restart using smaller psi"
                psi_array[0] = 0
                psi_array[1] = psi * weight

                "Calculate original values of Curvature variances"
                H_var_coeff[0], H_var_piv[0], H_var_func[0] = H_var_values(coeff, A, curve_matrix, H_var, qm, n0)
                H_var_grad[0] = 1

                "Decrease psi weighting for next run"
                weight *= 0.9
                "Reset number of steps"
                step = 0
            else:
                gamma = H_var_func[1] / H_var_grad[1]
                psi_array[0] = psi_array[1]
                psi_array[1] -= gamma

                H_var_coeff[0] = H_var_coeff[1]
                H_var_piv[0] = H_var_piv[1]
                H_var_func[0] = H_var_func[1]
                H_var_grad[0] = H_var_grad[1]

    #print( ' {:10.8f} {:10.4f} {:10.4f}'.format(psi_array[1],  H_var_coeff[0], H_var_piv[0]))

    return coeff_recon, A_recon


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


def self_consistent_cycle(coeff, A, b, dim, qm, tau, xmol, ymol, zmol, pivot, mol_list1, mol_list2, new_piv1=[], new_piv2=[], recon=False):

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
            if piv_search1 and build_surf1: mol_list1, new_piv1, pivot[0] = pivot_selection(mol_list1, zeta_list1, pivot[0], tau1, n0)
            if piv_search2 and build_surf2: mol_list2, new_piv2, pivot[1] = pivot_selection(mol_list2, zeta_list2, pivot[1], tau2, n0)

            "Check whether threshold distance tau needs to be increased"
            if len(new_piv1) == 0 and len(pivot[0]) < n0: tau1 += inc
            else: piv_search1 = False

            if len(new_piv2) == 0 and len(pivot[1]) < n0: tau2 += inc
            else: piv_search2 = False

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

        coeff, pivot = self_consistent_cycle(coeff, A, b, dim, qm, tau, xmol, ymol, zmol, piv_n1, piv_n2, mol_list1, mol_list2, new_piv1=[], new_piv2=[], recon=False)

    print('\n')

    return coeff, pivot


def wave_function_array(x, u_array, Lx):
    """
    wave_function_array(x, u_array, Lx)

    Returns numpy array of all waves in Fouier sum

    """

    q = 2 * np.pi * np.abs(u_array) * x / Lx

    cos_indicies = np.argwhere(u_array >= 0)
    sin_indicies = np.argwhere(u_array < 0)
    f_array = np.zeros(u_array.shape)
    f_array[cos_indicies] += np.cos(q[cos_indicies])
    f_array[sin_indicies] += np.sin(q[sin_indicies])

    return f_array


def d_wave_function_array(x, u_array, Lx):
    """
    d_wave_function_array(x, u_array, Lx)

    Returns numpy array of all derivatives of waves in Fouier sum

    """

    q = 2 * np.pi * np.abs(u_array) * x / Lx

    cos_indicies = np.argwhere(u_array >= 0)
    sin_indicies = np.argwhere(u_array < 0)
    f_array = np.zeros(u_array.shape)
    f_array[cos_indicies] -= np.sin(q[cos_indicies])
    f_array[sin_indicies] += np.cos(q[sin_indicies])
    f_array *= 2 * np.pi * np.abs(u_array) / Lx

    return f_array


def dd_wave_function_array(x, u_array, Lx):
    """
    dd_wave_function_array(x, u_array, Lx)
    Returns numpy array of all second derivatives of waves in Fouier sum

    """
    return - 4 * np.pi**2 * u_array**2 / Lx**2 * wave_function_array(x, u_array, Lx)


def xi(x, y, coeff, qm, qu, dim):
    """
    xi(x, y, coeff, qm, qu, dim)

    Function returning position of intrinsic surface at position (x,y)

    Parameters
    ----------

    x:  float, array_like; shape=(nmol)
        Coordinate in x dimension
    y:  float, array_like; shape=(nmol)
        Coordinate in y dimension
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

    xi_z:  float, array_like; shape=(nmol)
        Positions of intrinsic surface in z dimension

    """

    n_waves = 2 * qm + 1

    if np.isscalar(x):
        u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
        v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
        wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
        indices = np.argwhere(wave_check).flatten()

        fuv = wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
        xi_z = np.sum(fuv * coeff[indices])
    else:
        xi_z = np.zeros(x.shape)
        for u in range(-qu, qu+1):
            for v in range(-qu, qu+1):
                j = (2 * qm + 1) * (u + qm) + (v + qm)
                xi_z += wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
    return xi_z


def dxy_dxi(x, y, coeff, qm, qu, dim):
    """
    dxy_dxi(x, y, qm, qu, coeff, dim)

    Function returning derivatives of intrinsic surface at position (x,y) wrt x and y

    Parameters
    ----------

    x:  float
        Coordinate in x dimension
    y:  float
        Coordinate in y dimension
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

    dx_dxi:  float
        Derivative of intrinsic surface in x dimension
    dy_dxi:  float
        Derivative of intrinsic surface in y dimension

    """

    n_waves = 2 * qm + 1

    if np.isscalar(x):
        u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
        v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
        wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
        indices = np.argwhere(wave_check).flatten()

        dx_dxi = d_wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
        dy_dxi = wave_function_array(x, u_array[indices], dim[0]) * d_wave_function_array(y, v_array[indices], dim[1])

        dx_dxi = np.sum(dx_dxi * coeff[indices])
        dy_dxi = np.sum(dy_dxi * coeff[indices])

    else:
        dx_dxi = np.zeros(x.shape)
        dy_dxi = np.zeros(x.shape)
        for u in range(-qu, qu+1):
            for v in range(-qu, qu+1):
                j = (2 * qm + 1) * (u + qm) + (v + qm)
                dx_dxi += d_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
                dy_dxi += wave_function(x, u, dim[0]) * d_wave_function(y, v, dim[1]) * coeff[j]

    return dx_dxi, dy_dxi


def ddxy_ddxi(x, y, coeff, qm, qu, dim):
    """
    ddxy_ddxi(x, y, coeff, qm, qu, dim)

    Function returning second derivatives of intrinsic surface at position (x,y) wrt x and y

    Parameters
    ----------

    x:  float
        Coordinate in x dimension
    y:  float
        Coordinate in y dimension
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

    ddx_ddxi:  float
        Second derivative of intrinsic surface in x dimension
    ddy_ddxi:  float
        Second derivative of intrinsic surface in y dimension

    """

    n_waves = 2 * qm + 1

    if np.isscalar(x):
        u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
        v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
        wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
        indices = np.argwhere(wave_check).flatten()

        ddx_ddxi = dd_wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
        ddy_ddxi = wave_function_array(x, u_array[indices], dim[0]) * dd_wave_function_array(y, v_array[indices], dim[1])

        ddx_ddxi = np.sum(ddx_ddxi * coeff[indices])
        ddy_ddxi = np.sum(ddy_ddxi * coeff[indices])

    else:
        ddx_ddxi = np.zeros(x.shape)
        ddy_ddxi = np.zeros(x.shape)
        for u in range(-qu, qu+1):
            for v in range(-qu, qu+1):
                j = (2 * qm + 1) * (u + qm) + (v + qm)
                ddx_ddxi += dd_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
                ddy_ddxi += wave_function(x, u, dim[0]) * dd_wave_function(y, v, dim[1]) * coeff[j]

    return ddx_ddxi, ddy_ddxi


def optimise_ns_diff(directory, file_name, nmol, nframe, qm, phi, dim, mol_sigma, start_ns, step_ns=0.05, recon=False,
                     nframe_ns = 20, ncube=3, vlim=3, tau=0.5, max_r=1.5, precision=5E-4, gamma=0.5):
    """
    optimise_ns(directory, file_name, nmol, nframe, qm, phi, ncube, dim, mol_sigma, start_ns, step_ns=0.05, nframe_ns=20, vlim=3)

    Routine to find optimised pivot density coefficient ns and pivot number n0 based on lowest pivot diffusion rate

    Parameters
    ----------

    directory:  str
        File path of directory of alias analysis.
    file_name:  str
        File name of trajectory being analysed.
    nmol:  int
        Number of molecules in simulation
    nframe: int
        Number of trajectory frames in simulation
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface optimisation function
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    mol_sigma:  float
        Radius of spherical molecular interaction sphere
    start_ns:  float
        Initial value for pivot density optimisation routine
    step_ns:  float
        Search step difference between each pivot density value ns
    nframe_ns:  (optional) int
        Number of trajectory frames to perform optimisation with

    Returns
    -------

    opt_ns: float
        Optimised surface pivot density parameter
    opt_n0: int
        Optimised number of pivot molecules

    """

    pos_dir = directory + 'pos/'
    surf_dir = directory + 'surface/'

    if not os.path.exists(surf_dir): os.mkdir(surf_dir)

    mol_ex_1 = []
    mol_ex_2 = []
    NS = []
    derivative = []

    n_waves = 2 * qm + 1
    max_r *= mol_sigma
    tau *= mol_sigma

    xmol = load_npy(pos_dir + file_name + '_{}_xmol'.format(nframe), frames=range(nframe_ns))
    ymol = load_npy(pos_dir + file_name + '_{}_ymol'.format(nframe), frames=range(nframe_ns))
    zmol = load_npy(pos_dir + file_name + '_{}_zmol'.format(nframe), frames=range(nframe_ns))
    COM = load_npy(pos_dir + file_name + '_{}_com'.format(nframe), frames=range(nframe_ns))
    zvec = load_npy(pos_dir + file_name + '_{}_zvec'.format(nframe), frames=range(nframe_ns))

    com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
    zmol = zmol - com_tile

    if nframe < nframe_ns: nframe_ns = nframe
    ns = start_ns
    optimising = True

    print("Surface pivot density precision = {}".format(precision))

    while optimising:

        NS.append(ns)
        n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)

        print("Density Coefficient = {}".format(ns))
        print("Using pivot number = {}".format(n0))

        tot_piv_n1 = np.zeros((nframe_ns, n0), dtype=int)
        tot_piv_n2 = np.zeros((nframe_ns, n0), dtype=int)

        file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)

        if recon: file_name_coeff += '_r'

        if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_coeff)):
            make_hdf5(surf_dir + file_name_coeff + '_coeff', (2, n_waves**2), tables.Float64Atom())
            make_hdf5(surf_dir + file_name_coeff + '_pivot', (2, n0), tables.Int64Atom())

        for frame in range(nframe_ns):
            "Checking number of frames in coeff and pivot files"
            frame_check_coeff = (shape_check_hdf5(surf_dir + file_name_coeff + '_coeff')[0] <= frame)
            frame_check_pivot = (shape_check_hdf5(surf_dir + file_name_coeff + '_pivot')[0] <= frame)

            if frame_check_coeff: mode_coeff = 'a'
            else: mode_coeff = False

            if frame_check_pivot: mode_pivot = 'a'
            else: mode_pivot = False

            if not mode_coeff and not mode_pivot:
                pivot = load_hdf5(surf_dir + file_name_coeff + '_pivot', frame)
            else:
                sys.stdout.write("Optimising Intrinsic Surface coefficients: frame {}\n".format(frame))
                sys.stdout.flush()

                if frame == 0: surf_0 = [-dim[2]/4, dim[2]/4]
                else:
                    index = (2 * qm + 1)**2 / 2
                    coeff = load_hdf5(surf_dir + file_name_coeff + '_coeff', frame-1)
                    surf_0 = [coeff[0][index], coeff[1][index]]

                coeff, pivot = build_surface(xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma,
                                             qm, n0, phi, tau, max_r, ncube=ncube, vlim=vlim, surf_0=surf_0,
                                             recon=recon, zvec=zvec[frame])

                save_hdf5(surf_dir + file_name_coeff + '_coeff', coeff, frame, mode_coeff)
                save_hdf5(surf_dir + file_name_coeff + '_pivot', pivot, frame, mode_pivot)

            tot_piv_n1[frame] += pivot[0]
            tot_piv_n2[frame] += pivot[1]

        ex_1, ex_2 = mol_exchange(tot_piv_n1, tot_piv_n2, nframe_ns, n0)

        mol_ex_1.append(ex_1)
        mol_ex_2.append(ex_2)

        av_mol_ex = (np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.
        print("Average Pivot Diffusion Rate = {} mol / frame".format(av_mol_ex[-1]))

        if len(av_mol_ex) > 1:
            step_size = (av_mol_ex[-1] - av_mol_ex[-2])
            derivative.append(step_size / (NS[-1] - NS[-2]))
            #min_arg = np.argsort(abs(NS[-1] - np.array(NS)))
            #min_derivative = (av_mol_ex[min_arg[0]] - av_mol_ex[min_arg[1]]) / (NS[min_arg[0]] - NS[min_arg[1]])
            #derivative.append(min_derivative)

            check = abs(step_size) <= precision
            if check: optimising = False
            else:
                #if len(derivative) > 1: gamma = (NS[-1] - NS[-2]) / (derivative[-1] - derivative[-2])
                ns -= gamma * derivative[-1]
                print("Optimal pivot density not found.\nSurface density coefficient step size = |{}| > {}\n".format(step_size, precision))

        else:
            ns += step_ns
            print("Optimal pivot density not found.\nSurface density coefficient step size = |{}| > {}\n".format(step_ns / gamma, precision))


    opt_ns = NS[np.argmin(av_mol_ex)]
    opt_n0 = int(dim[0] * dim[1] * opt_ns / mol_sigma**2)

    print("Optimal pivot density found = {}\nSurface density coefficient step size = |{}| < {}\n".format(opt_ns, step_size, precision))
    print("Optimal number of pivots = {}".format(opt_n0))

    for ns in NS:
        if ns != opt_ns:
            n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)
            file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
            if recon: file_name_coeff += '_r'
            os.remove(surf_dir + file_name_coeff + '_coeff.hdf5')
            os.remove(surf_dir + file_name_coeff + '_pivot.hdf5')

    return opt_ns, opt_n0


def mol_exchange(piv_1, piv_2, nframe, n0):
    """
    mol_exchange(piv_1, piv_2, nframe, n0)

    Calculates average diffusion rate of surface pivot molecules between frames

    Parameters
    ----------

    piv_1:  float, array_like; shape=(nframe, n0)
        Molecular pivot indicies of upper surface at each frame
    piv_2:  float, array_like; shape=(nframe, n0)
        Molecular pivot indicies of lower surface at each frame
    nframe:  int
        Number of frames to sample over
    n0:  int
        Number of pivot molecules in each surface

    Returns
    -------

    diff_rate1: float
        Diffusion rate of pivot molecules in mol frame^-1 of upper surface
    diff_rate2: float
        Diffusion rate of pivot molecules in mol frame^-1 of lower surface

    """
    n_1 = 0
    n_2 = 0

    for frame in range(nframe-1):

        n_1 += len(set(piv_1[frame]) - set(piv_1[frame+1]))
        n_2 += len(set(piv_2[frame]) - set(piv_2[frame+1]))

    diff_rate1 = n_1 / (n0 * float(nframe-1))
    diff_rate2 = n_2 / (n0 * float(nframe-1))

    return diff_rate1, diff_rate2



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
