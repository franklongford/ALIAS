import numpy as np
import scipy as sp

from alias.src.wave_function import wave_arrays, wave_function


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

    u_array, v_array = wave_arrays(qm)

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
