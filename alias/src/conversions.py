import numpy as np

from alias.src.wave_function import vcheck, wave_arrays
from alias.src.spectra import calculate_frequencies


def coeff_to_fourier(coeff, qm, dim):
    """
    Returns Fouier coefficients for Fouier series representing
    intrinsic surface from linear algebra coefficients

    Parameters
    ----------
    coeff:	float, array_like; shape=(n_waves**2)
        Optimised linear algebra surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum
        representing intrinsic surface

    Returns
    -------
    f_coeff:  float, array_like; shape=(n_waves**2)
        Optimised Fouier surface coefficients

    """

    n_waves = 2 * qm + 1
    u_array, v_array = wave_arrays(qm)
    frequencies, _ = calculate_frequencies(u_array, v_array, dim)

    amplitudes = np.zeros(coeff.shape, dtype=complex)

    for u in range(-qm, qm+1):
        for v in range(-qm, qm+1):
            index = n_waves * (u + qm) + (v + qm)

            j1 = n_waves * (abs(u) + qm) + (abs(v) + qm)
            j2 = n_waves * (-abs(u) + qm) + (abs(v) + qm)
            j3 = n_waves * (abs(u) + qm) + (-abs(v) + qm)
            j4 = n_waves * (-abs(u) + qm) + (-abs(v) + qm)

            if abs(u) + abs(v) == 0:
                value = coeff[j1]

            elif v == 0:
                value = (coeff[j1] - np.sign(u) * 1j * coeff[j2]) / 2.
            elif u == 0:
                value = (coeff[j1] - np.sign(v) * 1j * coeff[j3]) / 2.

            elif u < 0 and v < 0:
                value = (
                    coeff[j1] + 1j * (coeff[j2] + coeff[j3]) - coeff[j4]) / 4.
            elif u > 0 and v > 0:
                value = (
                    coeff[j1] - 1j * (coeff[j2] + coeff[j3]) - coeff[j4]) / 4.

            elif u < 0:
                value = (
                    coeff[j1] + 1j * (coeff[j2] - coeff[j3]) + coeff[j4]) / 4.
            elif v < 0:
                value = (
                    coeff[j1] - 1j * (coeff[j2] - coeff[j3]) + coeff[j4]) / 4.

            amplitudes[index] = value

    return amplitudes, frequencies


def coeff_to_fourier_2(coeff_2, qm, dim):
    """Converts square coefficients to square Fourier
    coefficients"""

    n_waves = 2 * qm + 1
    u_mat, v_mat = np.meshgrid(
        np.arange(-qm, qm+1),
        np.arange(-qm, qm+1))
    x_mat, y_mat = np.meshgrid(
        np.linspace(0, 1 / dim[0], n_waves),
        np.linspace(0, 1 / dim[1], n_waves))

    Psi = vcheck(u_mat.flatten(), v_mat.flatten()) / 4.

    frequencies = np.pi * 2 * (u_mat * x_mat + y_mat * v_mat) / n_waves
    amplitudes_2 = np.reshape(Psi * coeff_2, (n_waves, n_waves))

    A = np.zeros((n_waves, n_waves))

    for i in range(n_waves):
        for j in range(n_waves):
            A[i][j] += (
                amplitudes_2 * np.exp(
                    -2 * np.pi * 1j / n_waves * (
                        u_mat * x_mat[i][j] + y_mat[i][j] * v_mat
                    )
                )
            ).sum()

    return A, frequencies
