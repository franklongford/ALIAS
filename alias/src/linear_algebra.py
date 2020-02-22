import numpy as np
import scipy as sp

from alias.src.wave_function import wave_arrays, wave_function


def update_A_b(xmol, ymol, zmol, dim, qm, new_pivot):
    """
    Update A matrix and b vector for new pivot selection

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
    qm:  int
        Maximum number of wave frequencies in Fouier Sum
        representing intrinsic surface
    n_waves:  int
        Number of coefficients / waves in surface
    new_pivot:  int, array_like
        Indices of new pivot molecules for both surfaces

    Returns
    -------
    A:  float, array_like; shape=(2, n_waves**2, n_waves**2)
        Matrix containing wave product weightings
        f(x, u1, Lx).f(y, v1, Ly).f(x, u2, Lx).f(y, v2, Ly)
        for each coefficient in the linear algebra equation
        Ax = b for both surfaces
    b:  float, array_like; shape=(2, n_waves**2)
        Vector containing solutions z.f(x, u, Lx).f(y, v, Ly)
        to the linear algebra equation Ax = b
        for both surfaces

    """
    n_waves = 2 * qm + 1

    u_array, v_array = wave_arrays(qm)

    A = np.zeros((2, n_waves**2, n_waves**2))
    b = np.zeros((2, n_waves**2))

    fuv = np.zeros((2, n_waves**2, len(new_pivot[0])))

    for surf in range(2):
        for index in range(n_waves**2):
            wave_x = wave_function(
                xmol[new_pivot[surf]], u_array[index], dim[0])
            wave_y = wave_function(
                ymol[new_pivot[surf]], v_array[index], dim[1])
            fuv[surf][index] = wave_x * wave_y
            b[surf][index] += np.sum(
                zmol[new_pivot[index]] * fuv[surf][index])

        A[surf] += np.dot(fuv[surf], fuv[surf].T)

    return A, b, fuv


def lu_decomposition(A, b):
    """
    Perform lower-upper decomposition to solve equation Ax = b
    using scipy linalg lover-upper solver

    Parameters
    ----------
    A:  float, array_like; shape=(2, n_waves**2, n_waves**2)
        Matrix containing wave product weightings
        f(x, u1, Lx).f(y, v1, Ly).f(x, u2, Lx).f(y, v2, Ly)
        for each coefficient in the linear algebra equation
        Ax = b for both surfaces
    b:  float, array_like; shape=(2, n_waves**2)
        Vector containing solutions z.f(x, u, Lx).f(y, v, Ly)
        to the linear algebra equation Ax = b
        for both surfaces

    Returns
    -------
    coeff:	array_like (float); shape=(n_waves**2)
        Optimised surface coefficients

    """
    lu, piv = sp.linalg.lu_factor(A)
    coeff = sp.linalg.lu_solve((lu, piv), b)

    return coeff
