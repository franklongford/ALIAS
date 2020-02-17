import numpy as np

from alias.src.linear_algebra import LU_decomposition
from alias.src.wave_function import (
    wave_function_array,
    wave_function,
    vcheck
)


def surface_reconstruction(coeff, A, b, area_diag, curve_matrix, H_var, qm, n0, psi, precision=1E-3, max_step=20):
    """
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

    n_waves = 2 * qm + 1

    "Calculate variance of curvature across entire surface from coefficients"
    H_var_coeff = np.sum(H_var * coeff**2)
    "Calculate variance of curvature at pivot sites only"
    coeff_matrix = np.tile(coeff, (n_waves**2, 1))
    H_var_piv = np.sum(coeff_matrix * coeff_matrix.T * A * curve_matrix / n0)
    "Calculate optimisation function (diff between coeff and pivot variance)"
    H_var_func = abs(H_var_coeff - H_var_piv)

    return H_var_coeff, H_var_piv, H_var_func


def H_xy(x, y, coeff, qm, qu, dim):
    """
    H_xy(x, y, coeff, qm, qu, dim)

    Calculation of mean curvature at position (x,y) at resolution qu

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

    H:  float
        Mean curvature of intrinsic surface at point x,y
    """

    n_waves = 2 * qm + 1

    if np.isscalar(x) and np.isscalar(y):
        u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
        v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
        wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
        indices = np.argwhere(wave_check).flatten()

        fuv = wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
        H = -4 * np.pi**2 * np.sum((u_array[indices]**2 / dim[0]**2 + v_array[indices]**2 / dim[1]**2) * fuv * coeff[indices])
    else:
        H_array = np.zeros(x.shape)
        for u in range(-qu, qu+1):
            for v in range(-qu, qu+1):
                j = (2 * qm + 1) * (u + qm) + (v + qm)
                H_array += wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * (u ** 2 / dim[0] ** 2 + v ** 2 / dim[1] ** 2) * coeff[j]
        H = -4 * np.pi**2 * H_array

    return H


def H_var_coeff(coeff, qm, qu, dim):
    """
    H_var_coeff(coeff, qm, qu, dim)

    Variance of mean curvature H across surface determined by coeff at resolution qu

    Parameters
    ----------

    coeff:	float, array_like; shape=(n_frame, n_waves**2)
        Optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    H_var:  float
        Variance of mean curvature H across whole surface

    """

    if qu == 0: return 0


    nframe = coeff.shape[0]
    n_waves = 2 * qm +1

    u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
    v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
    wave_filter = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
    indices = np.argwhere(wave_filter).flatten()
    Psi = vcheck(u_array, v_array)[indices] / 4.

    coeff_filter = coeff[:,:,indices]
    av_coeff_2 = np.mean(coeff_filter**2, axis=(0, 1)) * Psi

    H_var_array = vcheck(u_array[indices], v_array[indices]) * av_coeff_2[indices]
    H_var_array *= (u_array[indices]**4 / dim[0]**4 + v_array[indices]**4 / dim[1]**4 + 2 * u_array[indices]**2 * v_array[indices]**2 / (dim[0]**2 * dim[1]**2))
    H_var = 16 * np.pi**4 * np.sum(H_var_array)

    return H_var


def H_var_mol(xmol, ymol, coeff, qm, qu, dim):
    """
    H_var_mol(xmol, ymol, coeff, pivot, qm, qu, dim)

    Variance of mean curvature H at molecular positions determined by coeff at resolution qu

    Parameters
    ----------

    xmol:  float, array_like; shape=(nmol)
        Molecular coordinates in x dimension
    ymol:  float, array_like; shape=(nmol)
        Molecular coordinates in y dimension
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

    H_var:  float
        Variance of mean curvature H at pivot points

    """

    if qu == 0: return 0

    n_waves = 2 * qm +1
    nmol = xmol.shape[0]

    "Create arrays of wave frequency indicies u and v"
    u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
    v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
    wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
    indices = np.argwhere(wave_check).flatten()

    "Create matrix of wave frequency indicies (u,v)**2"
    u_matrix = np.tile(u_array[indices], (len([indices]), 1))
    v_matrix = np.tile(v_array[indices], (len([indices]), 1))

    "Make curvature diagonal terms of A matrix"
    curve_diag = 16 * np.pi**4 * (u_matrix**2 * u_matrix.T**2 / dim[0]**4 + v_matrix**2 * v_matrix.T**2 / dim[1]**4 +
                                  (u_matrix**2 * v_matrix.T**2 + u_matrix.T**2 * v_matrix**2) / (dim[0]**2 * dim[1]**2))

    "Form the diagonal xi^2 terms and b vector solutions"
    fuv = np.zeros((n_waves**2, nmol))
    for u in range(-qu, qu+1):
        for v in range(-qu, qu+1):
            j = (2 * qm + 1) * (u + qm) + (v + qm)
            fuv[j] = wave_function(xmol, u_array[j], dim[0]) * wave_function(ymol, v_array[j], dim[1])
    ffuv = np.dot(fuv[indices], fuv[indices].T)

    coeff_matrix = np.tile(coeff[indices], (len([indices]), 1))
    H_var = np.sum(coeff_matrix * coeff_matrix.T * ffuv * curve_diag / nmol)

    return H_var
