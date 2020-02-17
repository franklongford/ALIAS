import numpy as np


def check_uv(u, v):
    """
    Returns weightings for frequencies u and v
    for anisotropic surfaces
    """

    if abs(u) + abs(v) == 0:
        return 4.
    elif u * v == 0:
        return 2.
    return 1.


vcheck = np.vectorize(check_uv)


def wave_function(x, u, Lx):
    """
    Wave in Fourier sum
    """

    coeff = 2 * np.pi / Lx

    if u >= 0:
        return np.cos(coeff * u * x)
    return np.sin(coeff * abs(u) * x)


def d_wave_function(x, u, Lx):
    """
    First derivative of wave in Fourier sum wrt x
    """

    coeff = 2 * np.pi / Lx

    if u >= 0:
        return - coeff * u * np.sin(coeff * u * x)
    return coeff * abs(u) * np.cos(coeff * abs(u) * x)


def dd_wave_function(x, u, Lx):
    """
    Second derivative of wave in Fouier sum wrt x
    """

    coeff = 2 * np.pi / Lx

    return - coeff ** 2 * u ** 2 * wave_function(x, u, Lx)


def cos_sin_indices(u_array):
    """Return indices of wave function arrays for
    both cos and sin functions"""

    cos_indices = np.argwhere(u_array >= 0)
    sin_indices = np.argwhere(u_array < 0)

    return cos_indices, sin_indices


def wave_function_array(x, u_array, Lx):
    """
    Returns numpy array of all waves in Fourier sum
    """

    coeff = 2 * np.pi / Lx
    q = coeff * np.abs(u_array) * x
    cos_indices, sin_indices = cos_sin_indices(u_array)

    f_array = np.zeros(u_array.shape)
    f_array[cos_indices] += np.cos(q[cos_indices])
    f_array[sin_indices] += np.sin(q[sin_indices])

    return f_array


def d_wave_function_array(x, u_array, Lx):
    """
    d_wave_function_array(x, u_array, Lx)

    Returns numpy array of all derivatives of waves
    in Fourier sum

    """

    coeff = 2 * np.pi / Lx
    q = coeff * np.abs(u_array) * x
    cos_indices, sin_indices = cos_sin_indices(u_array)

    f_array = np.zeros(u_array.shape)
    f_array[cos_indices] -= np.sin(q[cos_indices])
    f_array[sin_indices] += np.cos(q[sin_indices])
    f_array *= coeff * np.abs(u_array)

    return f_array


def dd_wave_function_array(x, u_array, Lx):
    """Returns numpy array of all second derivatives
    of waves in Fourier sum"""

    coeff = 2 * np.pi / Lx
    f_array = wave_function_array(x, u_array, Lx)

    return - coeff ** 2 * u_array ** 2 * f_array


def wave_arrays(qm):
    """Return full arrays of each (u, v) 2D wave frequency
    combination for a given maximum frequency, `qm`"""
    n_waves = 2 * qm + 1
    wave_range = np.arange(n_waves ** 2)

    u_array = np.array(wave_range / n_waves, dtype=int) - qm
    v_array = np.array(wave_range % n_waves, dtype=int) - qm

    return u_array, v_array


def wave_indices(qu, u_array, v_array):
    """Return indices of both u_array and v_array that contain
    waves resulting from truncation of `qu` upper bound
    frequency"""

    wave_mask = (
        (u_array >= -qu) * (u_array <= qu)
        * (v_array >= -qu) * (v_array <= qu)
    )
    indices = np.argwhere(wave_mask).flatten()

    return indices
