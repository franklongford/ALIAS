import time

import numpy as np

from alias.src.intrinsic_surface import xi
from alias.src.linear_algebra import update_A_b, lu_decomposition
from alias.src.spectra import intrinsic_area
from alias.src.surface_reconstruction import surface_reconstruction
from alias.src.utilities import bubble_sort, numpy_remove
from alias.src.wave_function import wave_arrays, vcheck


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
    else:
        coeff, A, b, area_diag = surf_param

    building_surface = True
    build_surf1 = True
    build_surf2 = True

    while building_surface:

        start1 = time.time()

        "Update A matrix and b vector"
        temp_A, temp_b, fuv = update_A_b(xmol, ymol, zmol, dim, qm, [new_piv1, new_piv2])

        A += temp_A
        b += temp_b

        end1 = time.time()

        "Perform LU decomosition to solve Ax = b"
        if build_surf1:
            coeff[0] = lu_decomposition(A[0] + area_diag, b[0])
        if build_surf2:
            coeff[1] = lu_decomposition(A[1] + area_diag, b[1])

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
        if build_surf1:
            zeta_list1 = make_zeta_list(xmol, ymol, zmol, dim, mol_list1, coeff[0], qm, qm)
        if build_surf2:
            zeta_list2 = make_zeta_list(xmol, ymol, zmol, dim, mol_list2, coeff[1], qm, qm)

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
    if len(new_piv) > 0:
        mol_list = numpy_remove(mol_list, np.concatenate((new_piv, far_piv)))

    assert np.sum(np.isin(new_piv, mol_list)) == 0

    return mol_list, new_piv, piv_n


def initialise_surface(qm, phi, dim, recon=False):
    """
    Calculate initial parameters for ISM and reconstructed ISM fitting procedure

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
    u_array, v_array = wave_arrays(qm)
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