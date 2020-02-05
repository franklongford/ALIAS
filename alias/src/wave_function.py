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


def wave_function(x, u, Lx):
    """
    Wave in Fouier sum
    """

    coeff = 2 * np.pi / Lx

    if u >= 0:
        return np.cos(coeff * u * x)
    return np.sin(coeff * abs(u) * x)


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

    coeff = 2 * np.pi / Lx

    return - coeff ** 2 * u ** 2 * wave_function(x, u, Lx)


def wave_indices(u_array):
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
    cos_indices, sin_indices = wave_indices(u_array)

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
    cos_indices, sin_indices = wave_indices(u_array)

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
